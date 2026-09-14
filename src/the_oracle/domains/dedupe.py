"""Dedupe: decide whether a drafted objective already exists in the library.

This is the highest-risk decision in the engine. Mastery is keyed by
``(learner_id, objective_id)`` with no domain, so a **false merge silently
corrupts mastery in both directions**: credit for one skill leaks into another.
A false split only costs one duplicate objective and a little repeated study.

So the rule is: **bias to split.** Reuse needs strong evidence.

Three bands (PLAN.md section 2):

* ``score >= REUSE_THRESHOLD``      -> :attr:`Decision.REUSE`, no LLM call
* ``score >= ADJUDICATE_THRESHOLD`` -> :attr:`Decision.ADJUDICATE`, ask the Critic
* otherwise                         -> :attr:`Decision.MINT`, a new objective

Two guards sit on top of raw cosine similarity:

1. **Weak-evidence raise.** When the lexical fallback is the embedder, both
   thresholds go up by :data:`LEXICAL_THRESHOLD_RAISE`. Lexical overlap is
   weaker evidence than a dense vector. A real dense model - local or API -
   uses the base thresholds.
2. **Depth penalty.** The same words at a different Bloom level or difficulty
   are not the same skill. The gap between the two objectives subtracts from
   the score, so a survey treatment never merges into an expert one even when
   an embedder finds the wording near-identical.

No subject matter lives in this file.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from the_oracle.domains import embed as embedding
from the_oracle.domains.embed import Embedder, cosine, is_lexical

if TYPE_CHECKING:  # pragma: no cover
    from the_oracle.domains.schema import Objective

log = logging.getLogger(__name__)


class Decision(StrEnum):
    """What to do with one drafted objective."""

    REUSE = "reuse"
    ADJUDICATE = "adjudicate"
    MINT = "mint"


#: Strong evidence only. Above this the two texts are the same skill.
REUSE_THRESHOLD = 0.92

#: Middle band floor. Between the two the Critic decides.
ADJUDICATE_THRESHOLD = 0.78

#: Added to both thresholds when the lexical fallback is in use.
LEXICAL_THRESHOLD_RAISE = 0.05

#: How many near misses to keep on a :class:`Match`, for the "what was reused"
#: report the learner sees.
RUNNERS_UP = 3

#: A Critic verdict below this confidence is not strong enough to merge.
JUDGE_MIN_CONFIDENCE = 0.6

#: Bloom levels, ordered. Distance between two levels is a depth gap.
BLOOM_ORDER: tuple[str, ...] = (
    "remember",
    "understand",
    "apply",
    "analyze",
    "evaluate",
    "create",
)

#: Score subtracted per Bloom step and per difficulty step between two
#: otherwise similar objectives, capped by :data:`MAX_DEPTH_PENALTY`.
DEPTH_PENALTY_PER_BLOOM_STEP = 0.06
DEPTH_PENALTY_PER_DIFFICULTY_STEP = 0.04
MAX_DEPTH_PENALTY = 0.25


@dataclass(frozen=True)
class Match:
    """The dedupe verdict for one candidate.

    ``score`` is the final, guarded score: cosine similarity minus the depth
    penalty. ``thresholds`` and ``embedder_name`` record the bar that score had
    to clear, so a decision can be audited long after the run.
    """

    candidate_title: str
    decision: Decision
    best_id: str | None
    score: float
    runners_up: list[tuple[str, float]] = field(default_factory=list)
    embedder_name: str = ""
    thresholds: tuple[float, float] = (REUSE_THRESHOLD, ADJUDICATE_THRESHOLD)
    lexical: bool = False

    @property
    def raised(self) -> bool:
        """True when the weak-evidence threshold raise was applied."""
        return self.lexical

    def explain(self) -> str:
        """One auditable line describing why this decision was made."""
        reuse, adjudicate = self.thresholds
        note = " (+raised: lexical fallback)" if self.lexical else ""
        return (
            f"{self.candidate_title!r}: {self.decision.value} "
            f"score={self.score:.3f} best={self.best_id} "
            f"bar={reuse:.2f}/{adjudicate:.2f}{note} via {self.embedder_name}"
        )


#: A judge is any callable that turns a middle-band match into a verdict. It
#: may return a bool or anything with ``same_skill`` and ``confidence``.
#: :func:`the_oracle.agents.critic.judge_fn` builds one from the Critic agent.
Judge = Callable[[Match], Any]


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def thresholds_for(embedder: Embedder | None) -> tuple[float, float]:
    """The active ``(reuse, adjudicate)`` bar for this embedder.

    The bar is a function of the evidence quality, not a global constant. A
    dense model keeps the base thresholds; the lexical fallback must clear a
    higher bar because its evidence is weaker.
    """
    if is_lexical(embedder):
        return (
            REUSE_THRESHOLD + LEXICAL_THRESHOLD_RAISE,
            ADJUDICATE_THRESHOLD + LEXICAL_THRESHOLD_RAISE,
        )
    return REUSE_THRESHOLD, ADJUDICATE_THRESHOLD


def objective_text(obj: "Objective | Any") -> str:
    """The text that represents one objective to an embedder."""
    tags = " ".join(getattr(obj, "tags", None) or [])
    parts = [obj.title, getattr(obj, "description", "") or "", tags]
    return "\n".join(p.strip() for p in parts if p and p.strip())


def depth_penalty(a: "Objective | Any", b: "Objective | Any") -> float:
    """How far apart two objectives sit in depth, as a score deduction.

    Same words, different depth, different audience: not the same skill.
    """
    try:
        bloom_gap = abs(
            BLOOM_ORDER.index(str(getattr(a, "bloom", "")))
            - BLOOM_ORDER.index(str(getattr(b, "bloom", "")))
        )
    except ValueError:
        bloom_gap = 0
    difficulty_gap = abs(int(getattr(a, "difficulty", 0)) - int(getattr(b, "difficulty", 0)))
    raw = (
        bloom_gap * DEPTH_PENALTY_PER_BLOOM_STEP
        + difficulty_gap * DEPTH_PENALTY_PER_DIFFICULTY_STEP
    )
    return min(MAX_DEPTH_PENALTY, raw)


def similarity(
    a: "Objective | Any",
    b: "Objective | Any",
    vector_a: Sequence[float],
    vector_b: Sequence[float],
) -> float:
    """Guarded similarity: cosine minus the depth penalty, floored at -1.

    Kept for callers that want a single blended number. Banding uses
    :func:`decide` with the raw cosine and the penalty passed separately.
    """
    return max(-1.0, cosine(vector_a, vector_b) - depth_penalty(a, b))


#: A depth gap this large or bigger blocks a silent reuse and asks the Critic.
DEPTH_DOWNGRADE_EPSILON = 0.02


def decide(
    score: float,
    thresholds: tuple[float, float],
    penalty: float = 0.0,
) -> Decision:
    """Map one similarity onto a band.

    ``score`` is the RAW cosine. The depth penalty is applied as a *downgrade*,
    not as a subtraction before banding.

    Why: subtracting the penalty first pushed genuine duplicates below the
    adjudicate threshold, so the Critic never saw them. Measured against the
    seed library, "Design a rolling-origin backtest" scored 0.858 raw against an
    existing objective that teaches exactly that, but 0.752 after the penalty —
    a silent false split, which is the failure the Critic exists to prevent.

    So: raw similarity chooses the band, and a depth gap can only make the
    decision more cautious, never less visible.
    """
    reuse, adjudicate = thresholds
    if score >= reuse:
        return Decision.REUSE if penalty < DEPTH_DOWNGRADE_EPSILON else Decision.ADJUDICATE
    if score >= adjudicate:
        return Decision.ADJUDICATE
    return Decision.MINT


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------


def _library() -> list["Objective"]:
    from the_oracle.domains import library

    return list(library.all_objectives())


def _resolve_embedder(embedder: Embedder | None) -> Embedder:
    return embedder if embedder is not None else embedding.get_embedder()


def match_one(
    candidate: "Objective",
    library_objs: Iterable["Objective"] | None = None,
    embedder: Embedder | None = None,
) -> Match:
    """Match one candidate against the library. Never merges on weak evidence."""
    return match_all([candidate], library_objs, embedder)[0]


def match_all(
    candidates: Sequence["Objective"],
    library_objs: Iterable["Objective"] | None = None,
    embedder: Embedder | None = None,
) -> list[Match]:
    """Match every candidate in one pass, embedding the library once."""
    active = _resolve_embedder(embedder)
    thresholds = thresholds_for(active)
    existing = list(library_objs) if library_objs is not None else _library()

    if is_lexical(active):
        log.warning(
            "dedupe running on the lexical fallback (%s): thresholds raised to "
            "%.2f/%.2f to avoid a false merge",
            active.name,
            *thresholds,
        )

    if not candidates:
        return []

    candidate_vectors = active.embed([objective_text(c) for c in candidates])
    existing_vectors = (
        active.embed([objective_text(o) for o in existing]) if existing else []
    )

    matches: list[Match] = []
    for candidate, vector in zip(candidates, candidate_vectors, strict=True):
        # Rank on RAW cosine, and carry the depth penalty alongside it so the
        # banding step can downgrade rather than hide a strong match.
        scored_full = sorted(
            (
                (obj.id, cosine(vector, other), depth_penalty(candidate, obj))
                for obj, other in zip(existing, existing_vectors, strict=True)
            ),
            key=lambda triple: (-triple[1], triple[0]),
        )
        scored = [(oid, raw) for oid, raw, _ in scored_full]
        if scored_full:
            best_id, best_score, best_penalty = scored_full[0]
        else:
            best_id, best_score, best_penalty = None, 0.0, 0.0
        decision = (
            decide(best_score, thresholds, best_penalty) if scored_full else Decision.MINT
        )
        match = Match(
            candidate_title=candidate.title,
            decision=decision,
            best_id=best_id if decision is not Decision.MINT else None,
            score=round(best_score, 6),
            runners_up=[(oid, round(s, 6)) for oid, s in scored[1 : 1 + RUNNERS_UP]],
            embedder_name=active.name,
            thresholds=thresholds,
            lexical=is_lexical(active),
        )
        log.info("%s", match.explain())
        matches.append(match)
    return matches


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------


def _accepts(verdict: Any) -> bool:
    """Read a judge result. Anything unclear means split."""
    if verdict is None:
        return False
    if isinstance(verdict, bool):
        return verdict
    same = bool(getattr(verdict, "same_skill", False))
    confidence = float(getattr(verdict, "confidence", 0.0))
    return same and confidence >= JUDGE_MIN_CONFIDENCE


def resolve(matches: Sequence[Match], judge: Judge | None = None) -> dict[str, str | None]:
    """Candidate title -> the library id to reuse, or ``None`` to mint.

    The judge is consulted for the middle band and nowhere else. With no judge
    the middle band splits, because an unadjudicated maybe is not evidence.
    """
    out: dict[str, str | None] = {}
    for match in matches:
        if match.decision is Decision.REUSE and match.best_id:
            out[match.candidate_title] = match.best_id
            continue
        if match.decision is not Decision.ADJUDICATE or match.best_id is None:
            out[match.candidate_title] = None
            continue
        if judge is None:
            log.info(
                "no judge for middle-band %r; splitting", match.candidate_title
            )
            out[match.candidate_title] = None
            continue
        try:
            verdict = judge(match)
        except Exception:  # noqa: BLE001 - a broken judge must never merge
            log.exception("judge failed on %r; splitting", match.candidate_title)
            out[match.candidate_title] = None
            continue
        accepted = _accepts(verdict)
        log.info(
            "judge on %r -> %s", match.candidate_title, "reuse" if accepted else "mint"
        )
        out[match.candidate_title] = match.best_id if accepted else None
    return out
