"""Scout: find candidate web resources for one objective.

Search only. No model call, no state write. The orchestrator decides what to do
with the candidates; this module just proposes them.
"""

from __future__ import annotations

import os
from typing import Any, Final, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit

from pydantic import BaseModel, Field

SERPER_URL: Final[str] = "https://google.serper.dev/search"
SERPER_ENV_VAR: Final[str] = "SERPER_API_KEY"
USER_AGENT: Final[str] = "the-oracle/0.1 (+https://github.com/jeffreyparks/the-oracle)"

#: Query-string keys that never change the page. Stripped before dedupe.
TRACKING_PARAMS: Final[frozenset[str]] = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "utm_id",
        "utm_reader",
        "gclid",
        "fbclid",
        "mc_cid",
        "mc_eid",
        "ref",
        "ref_src",
        "source",
        "spm",
        "igshid",
        "_hsenc",
        "_hsmi",
    }
)

#: Bloom verb -> cognitive rank. Unknown verbs sit in the middle.
BLOOM_RANK: Final[dict[str, int]] = {
    "remember": 1,
    "understand": 2,
    "apply": 3,
    "analyse": 4,
    "analyze": 4,
    "evaluate": 5,
    "create": 6,
}


class ScoutUnavailableError(RuntimeError):
    """Search cannot run: no key, or the search backend refused."""


class Candidate(BaseModel):
    """One search hit. Not yet fetched, not yet judged."""

    url: str
    title: str
    snippet: str = ""
    source: str = ""
    rank: int = 0


class ScoutRequest(BaseModel):
    """What the scout needs to know about the objective it is shopping for."""

    objective_id: str
    title: str
    description: str
    bloom: str
    difficulty: int
    tags: list[str] = Field(default_factory=list)


def normalise_url(url: str) -> str:
    """Collapse the harmless differences between two URLs for the same page.

    Scheme, ``www.``, a trailing slash, a fragment, and tracking parameters all
    vanish. Two URLs with the same normal form are the same resource.
    """
    raw = (url or "").strip()
    if not raw:
        return ""
    if "//" not in raw.split("?", 1)[0][:8]:
        raw = f"//{raw}"
    parts = urlsplit(raw, scheme="https")
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if parts.port and parts.port not in (80, 443):
        host = f"{host}:{parts.port}"
    path = parts.path or "/"
    if len(path) > 1:
        path = path.rstrip("/") or "/"
    query = urlencode(
        sorted(
            (k, v)
            for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if k.lower() not in TRACKING_PARAMS
        )
    )
    return f"{host}{path}?{query}" if query else f"{host}{path}"


def level_band(bloom: str, difficulty: int) -> str:
    """``intro``, ``working``, or ``deep``, from the bloom verb and difficulty.

    A ``remember``-level difficulty-1 objective wants an introduction. An
    ``evaluate``-level difficulty-5 objective wants depth.
    """
    rank = BLOOM_RANK.get((bloom or "").strip().lower(), 3)
    score = rank + max(1, min(5, int(difficulty or 1)))
    if score <= 4:
        return "intro"
    if score <= 7:
        return "working"
    return "deep"


#: Per band, the query flavours. First is the plain search, then two shaped ones.
_BAND_SUFFIXES: Final[dict[str, tuple[str, str, str]]] = {
    "intro": (
        "introduction",
        "explained simply for beginners",
        "beginner tutorial with examples",
    ),
    "working": (
        "tutorial",
        "worked example step by step",
        "practice problems with solutions",
    ),
    "deep": (
        "in depth",
        "advanced treatment derivation proof",
        "critique limitations when it fails",
    ),
}


def build_queries(req: ScoutRequest, *, max_queries: int = 3) -> list[str]:
    """Two or three query variants: plain, level-shaped, and tag-shaped."""
    title = " ".join((req.title or "").split())
    if not title:
        raise ValueError("ScoutRequest.title is required to build a query")
    tags = [t.strip() for t in req.tags if t and t.strip()]
    band = level_band(req.bloom, req.difficulty)
    plain, shaped, practice = _BAND_SUFFIXES[band]

    lead_tags = " ".join(tags[:2])
    queries = [
        " ".join(x for x in (title, lead_tags, plain) if x),
        " ".join(x for x in (title, shaped) if x),
        " ".join(x for x in (title, " ".join(tags[:3]) or lead_tags, practice) if x),
    ]

    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(q)
    return unique[: max(1, max_queries)]


def _api_key() -> str:
    key = (os.environ.get(SERPER_ENV_VAR) or "").strip()
    if not key:
        raise ScoutUnavailableError(
            f"{SERPER_ENV_VAR} is not set. Web search is unavailable; set the key or "
            "author the lesson instead."
        )
    return key


def serper_search(query: str, *, api_key: str, num: int = 10, timeout: float = 15.0) -> list[dict[str, Any]]:
    """One Serper call. Returns the raw ``organic`` list.

    Kept tiny and separate so tests can replace it without touching the network.
    """
    import httpx  # local import: the offline path must not need it

    try:
        response = httpx.post(
            SERPER_URL,
            headers={
                "X-API-KEY": api_key,
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
            },
            json={"q": query, "num": max(1, min(20, num))},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # network, auth, quota, or malformed JSON
        raise ScoutUnavailableError(f"Serper search failed for {query!r}: {exc}") from exc

    organic = payload.get("organic") if isinstance(payload, dict) else None
    return [hit for hit in (organic or []) if isinstance(hit, dict)]


def _to_candidate(hit: dict[str, Any], rank: int) -> Candidate | None:
    url = str(hit.get("link") or hit.get("url") or "").strip()
    if not url or not url.lower().startswith(("http://", "https://")):
        return None
    host = urlsplit(url).hostname or ""
    return Candidate(
        url=url,
        title=str(hit.get("title") or "").strip() or url,
        snippet=str(hit.get("snippet") or "").strip(),
        source=host[4:] if host.lower().startswith("www.") else host,
        rank=rank,
    )


def search_candidates(req: ScoutRequest, *, limit: int = 10) -> list[Candidate]:
    """Search the live web for resources that could teach this objective.

    Runs two or three query variants, deduplicates by normalised URL, keeps the
    best rank seen for a page, and returns at most ``limit`` candidates.

    Raises :class:`ScoutUnavailableError` when there is no key or search fails.
    An empty list means "searched and found nothing", never "could not search".
    """
    if limit <= 0:
        return []
    key = _api_key()
    queries = build_queries(req)

    by_url: dict[str, Candidate] = {}
    order: list[str] = []
    failures: list[str] = []
    for query in queries:
        try:
            hits = serper_search(query, api_key=key, num=limit)
        except ScoutUnavailableError as exc:
            failures.append(str(exc))
            continue
        for position, hit in enumerate(hits, start=1):
            candidate = _to_candidate(hit, position)
            if candidate is None:
                continue
            norm = normalise_url(candidate.url)
            if not norm:
                continue
            existing = by_url.get(norm)
            if existing is None:
                by_url[norm] = candidate
                order.append(norm)
            elif candidate.rank < existing.rank:
                by_url[norm] = candidate

    if failures and not by_url:
        raise ScoutUnavailableError("; ".join(failures))

    position_of = {norm: i for i, norm in enumerate(order)}
    ranked = sorted(order, key=lambda n: (by_url[n].rank, position_of[n]))
    return [by_url[n] for n in ranked[:limit]]


__all__: Sequence[str] = (
    "Candidate",
    "ScoutRequest",
    "ScoutUnavailableError",
    "build_queries",
    "level_band",
    "normalise_url",
    "search_candidates",
    "serper_search",
)
