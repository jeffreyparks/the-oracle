"""Corpus storage: the only writer into ``resource``, ``resource_review``, ``item``.

Two rules hold this module together.

**Ids are content-derived.** A resource id is a hash of the objective id plus
the normalised URL, or of the objective id plus the lesson body. Re-running the
pipeline over the same objective therefore cannot produce a second row for the
same material - the insert collapses onto the id that already exists. Nothing
here depends on a unique constraint that ``store/models.py`` does not have.

**No new columns.** Anything the table does not model - source rank, snippet,
word count, fetch status, the structured lesson - goes into ``Resource.meta``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from sqlalchemy import Engine
from sqlmodel import Session, select

from the_oracle.store import models
from the_oracle.store.db import get_engine, session_scope

__all__ = [
    "accepted_for",
    "has_enough",
    "normalise_url",
    "put_items",
    "put_lesson",
    "put_resource",
    "record_review",
    "resource_id_for",
    "tokens_spent",
]

_TRACKING_PREFIXES = ("utm_", "fbclid", "gclid", "mc_cid", "mc_eid", "ref_src")


def normalise_url(url: str) -> str:
    """Canonical form of a URL, for deduplication only.

    Lowercases scheme and host, drops a default port, drops ``www.``, strips a
    fragment and common tracking parameters, and removes a trailing slash.
    """
    raw = (url or "").strip()
    if not raw:
        return ""
    parts = urlsplit(raw if "//" in raw else f"https://{raw}")
    scheme = (parts.scheme or "https").lower()
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    port = parts.port
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        host = f"{host}:{port}"
    path = parts.path.rstrip("/")
    query = "&".join(
        piece
        for piece in sorted(parts.query.split("&"))
        if piece and not piece.lower().startswith(_TRACKING_PREFIXES)
    )
    return urlunsplit((scheme, host, path, query, ""))


def _digest(*parts: str) -> str:
    joined = "\x1f".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def resource_id_for(objective_id: str, *, url: str | None = None, body: str | None = None) -> str:
    """Deterministic resource id. Same input, same id, on every machine.

    A found resource keys on the normalised URL. An authored resource keys on
    the lesson body, so re-authoring identical prose reuses the row instead of
    stacking near-duplicates under one objective.
    """
    normalised = normalise_url(url) if url else ""
    if normalised:
        return "res_" + _digest(objective_id, "url", normalised)[:32]
    return "res_" + _digest(objective_id, "body", body or "")[:32]


def _dump(value: Any) -> Any:
    """Best-effort JSON form. Works for pydantic models, dataclasses, and dicts."""
    if value is None:
        return None
    if isinstance(value, dict | list | str | int | float | bool):
        return value
    dumper = getattr(value, "model_dump", None)
    if callable(dumper):
        return dumper(mode="json")
    if hasattr(value, "__dict__"):
        return json.loads(json.dumps(vars(value), default=str))
    return str(value)


def put_resource(
    objective_id: str,
    *,
    kind: str = "found",
    url: str | None = None,
    title: str | None = None,
    body: str | None = None,
    meta: dict[str, Any] | None = None,
    engine: Engine | None = None,
) -> str:
    """Insert or refresh one resource. Returns its deterministic id.

    Calling this twice with the same objective and URL updates the existing row
    and returns the same id. It never creates a second row.
    """
    if not objective_id:
        raise ValueError("objective_id is required")
    resource_id = resource_id_for(objective_id, url=url, body=body)
    kind_value = models.ResourceKind(str(kind))
    payload = dict(meta or {})
    with session_scope(engine or get_engine()) as session:
        row = session.get(models.Resource, resource_id)
        if row is None:
            session.add(
                models.Resource(
                    id=resource_id,
                    objective_id=objective_id,
                    kind=kind_value,
                    url=url,
                    title=title,
                    body=body,
                    meta=payload,
                )
            )
            return resource_id
        row.kind = kind_value
        if url is not None:
            row.url = url
        if title is not None:
            row.title = title
        if body is not None:
            row.body = body
        if payload:
            merged = dict(row.meta or {})
            merged.update(payload)
            row.meta = merged
        session.add(row)
    return resource_id


def record_review(resource_id: str, review: Any, engine: Engine | None = None) -> None:
    """Store one verdict against one resource. Idempotent.

    ``review`` is duck-typed: anything with ``verdict``, ``score``, ``rubric``
    and ``notes`` works, which keeps this module independent of the Reviewer's
    import graph.
    """
    verdict = models.ReviewVerdict(str(getattr(review, "verdict", "reject")))
    score = float(getattr(review, "score", 0.0) or 0.0)
    rubric = _dump(getattr(review, "rubric", None)) or {}
    if not isinstance(rubric, dict):
        rubric = {"value": rubric}
    notes = getattr(review, "notes", None)
    hard = getattr(review, "hard_reject_reason", None)
    if hard:
        rubric = {**rubric, "hard_reject_reason": hard}

    with session_scope(engine or get_engine()) as session:
        existing = session.exec(
            select(models.ResourceReview).where(models.ResourceReview.resource_id == resource_id)
        ).all()
        for row in existing:
            if row.verdict == verdict and abs(row.score - score) < 1e-9:
                return  # the same verdict is already on record
        session.add(
            models.ResourceReview(
                resource_id=resource_id,
                verdict=verdict,
                score=score,
                rubric=rubric,
                notes=notes,
            )
        )


def _latest_verdicts(
    session: Session, resource_ids: list[str]
) -> dict[str, models.ReviewVerdict]:
    if not resource_ids:
        return {}
    rows = session.exec(
        select(models.ResourceReview).where(
            models.ResourceReview.resource_id.in_(resource_ids)  # type: ignore[attr-defined]
        )
    ).all()
    latest: dict[str, tuple[Any, int, models.ReviewVerdict]] = {}
    for row in rows:
        stamp = (row.created_at, row.id or 0)
        current = latest.get(row.resource_id)
        if current is None or stamp > (current[0], current[1]):
            latest[row.resource_id] = (row.created_at, row.id or 0, row.verdict)
    return {rid: value[2] for rid, value in latest.items()}


def accepted_for(objective_id: str, engine: Engine | None = None) -> list[models.Resource]:
    """Resources for one objective whose most recent review said ``accept``."""
    with Session(engine or get_engine()) as session:
        resources = list(
            session.exec(
                select(models.Resource)
                .where(models.Resource.objective_id == objective_id)
                .order_by(models.Resource.created_at, models.Resource.id)  # type: ignore[arg-type]
            ).all()
        )
        verdicts = _latest_verdicts(session, [r.id for r in resources])
        return [r for r in resources if verdicts.get(r.id) == models.ReviewVerdict.ACCEPT]


def has_enough(objective_id: str, *, minimum: int = 2, engine: Engine | None = None) -> bool:
    """True when this objective already has teachable, accepted material.

    Either ``minimum`` accepted resources, or one accepted authored lesson. The
    authored lesson counts on its own because that is the fallback the pipeline
    pays for when the web does not supply enough - re-paying for it on the next
    run would defeat the point.
    """
    accepted = accepted_for(objective_id, engine)
    if any(r.kind == models.ResourceKind.AUTHORED for r in accepted):
        return True
    return len(accepted) >= max(1, int(minimum))


def render_lesson(lesson: Any) -> str:
    """Markdown body for an authored lesson. Deterministic, so the id is stable."""
    parts: list[str] = []
    explainer = getattr(lesson, "explainer", "") or ""
    if explainer:
        parts.append(explainer.strip())
    examples = list(getattr(lesson, "worked_examples", []) or [])
    if examples:
        parts.append("## Worked examples")
        parts.extend(f"{i}. {text}".strip() for i, text in enumerate(examples, start=1))
    code = getattr(lesson, "code", None)
    if code:
        parts.append("## Code\n\n```\n" + code.strip() + "\n```")
    exercises = list(getattr(lesson, "exercises", []) or [])
    if exercises:
        parts.append("## Exercises")
        for i, exercise in enumerate(exercises, start=1):
            parts.append(f"{i}. {getattr(exercise, 'prompt', '')}".strip())
    summary = getattr(lesson, "review_summary", "") or ""
    if summary:
        parts.append("## Review summary\n\n" + summary.strip())
    return "\n\n".join(parts).strip()


def put_lesson(objective_id: str, lesson: Any, engine: Engine | None = None) -> str:
    """Store an authored lesson as a ``kind=authored`` resource. Returns its id.

    The structured lesson lives in ``meta``; ``body`` holds the rendered
    markdown a reader sees.
    """
    body = render_lesson(lesson)
    meta: dict[str, Any] = {"lesson": _dump(lesson), "authored": True}
    citations = list(getattr(lesson, "citations", []) or [])
    if citations:
        meta["citations"] = citations
    title = (getattr(lesson, "review_summary", "") or "").strip().splitlines()
    return put_resource(
        objective_id,
        kind="authored",
        url=None,
        title=(title[0][:120] if title else f"Authored lesson for {objective_id}"),
        body=body,
        meta=meta,
        engine=engine,
    )


def put_items(objective_id: str, exercises: Any, engine: Engine | None = None) -> list[str]:
    """Store lesson exercises as shared assessment items. Returns their ids.

    Ids are derived from the objective plus the stem, so re-storing the same
    exercise updates one row instead of adding another.
    """
    ids: list[str] = []
    rows = list(exercises or [])
    if not rows:
        return ids
    with session_scope(engine or get_engine()) as session:
        for exercise in rows:
            stem = (getattr(exercise, "prompt", "") or "").strip()
            if not stem:
                continue
            item_id = "item_" + _digest(objective_id, "stem", stem)[:32]
            try:
                bloom = models.Bloom(str(getattr(exercise, "bloom", "") or "understand"))
            except ValueError:
                bloom = models.Bloom.UNDERSTAND
            answer_key: dict[str, Any] = {
                "answer": getattr(exercise, "answer", ""),
                "wrong_answer_feedback": _dump(getattr(exercise, "wrong_answer_feedback", {})) or {},
            }
            existing = session.get(models.Item, item_id)
            if existing is None:
                session.add(
                    models.Item(
                        id=item_id,
                        objective_id=objective_id,
                        bloom=bloom,
                        stem=stem,
                        answer_key=answer_key,
                    )
                )
            else:
                existing.bloom = bloom
                existing.answer_key = answer_key
                session.add(existing)
            ids.append(item_id)
    return ids


def tokens_spent(engine: Engine | None = None, *, since: Any = None) -> int:
    """Total tokens recorded in the idempotency cache, optionally since a time.

    The pipeline never counts tokens itself. Every model call goes through
    :class:`the_oracle.agents.base.Agent`, which writes one ``agent_call`` row,
    so this is the honest number rather than an estimate.
    """
    statement = select(models.AgentCall)
    if since is not None:
        statement = statement.where(models.AgentCall.created_at >= since)
    with Session(engine or get_engine()) as session:
        return sum(int(row.total_tokens or 0) for row in session.exec(statement).all())
