"""Retrieval-augmented drafting: show the Architect what already exists.

Post-hoc dedupe cannot fix a draft that was written blind. Live runs proved it:
the Architect invented an objective that scored 0.96 against an existing library
objective, and the Critic still refused to merge it, correctly, because the two
sat at different Bloom levels. That is bias-to-split working as designed, and it
is also a design gap. Left alone the library bloats with near-parallel
objectives and mastery stops transferring between domains, which is the whole
premise of a shared library.

So: **prevention over reconciliation.** Before the Architect drafts, embed what
the learner asked for, pull the nearest existing objectives out of the library,
and put them in the prompt with permission to reuse one verbatim.

There is one embedding path in this engine and this module reuses it
(:mod:`the_oracle.domains.embed` for vectors, :func:`dedupe.objective_text` for
what an objective looks like to an embedder). A second path would drift from the
first and quietly change what dedupe sees.

No subject matter lives in this file.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from the_oracle.domains import embed as embedding
from the_oracle.domains.dedupe import objective_text
from the_oracle.domains.embed import cosine

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from the_oracle.agents.architect import Interview
    from the_oracle.domains.embed import Embedder
    from the_oracle.domains.schema import Objective

log = logging.getLogger(__name__)

#: How many existing objectives the Architect gets to see. Enough to cover a
#: whole overlapping domain, small enough to leave the prompt readable.
RETRIEVAL_K = 20

#: The header the prompt block opens with. Tests assert on it, because a block
#: that is built and never interpolated is a bug this project has shipped twice.
CONTEXT_HEADER = "EXISTING OBJECTIVES IN THE SHARED LIBRARY"

REUSE_INSTRUCTION = """\
Some of these already teach what this learner needs. If an existing objective
covers a required skill, REUSE it: copy its title exactly and set `existing_id`
to its id. Do not restate the same skill at a different Bloom level just to
reword it. Draft a new objective only when nothing listed genuinely covers the
skill, or when the learner needs it at a materially different depth - and if you
do that, say why in the description."""


def query_text(interview: "Interview") -> str:
    """What this learner asked for, as one string to embed.

    Goal first, then the named must-cover terms, then the background. The
    background is last because it describes where the learner starts, not what
    they want, and it is the noisiest of the three.
    """
    parts = [interview.goal, *interview.must_cover, interview.background]
    return "\n".join(p.strip() for p in parts if p and p.strip())


def _library() -> list["Objective"]:
    from the_oracle.domains import library

    return list(library.all_objectives())


def retrieve_context(
    interview: "Interview",
    *,
    k: int = RETRIEVAL_K,
    embedder: "Embedder | None" = None,
    library_objs: "Sequence[Objective] | None" = None,
) -> list["Objective"]:
    """Nearest existing objectives to what this learner asked for.

    Best first, at most ``k``. An empty library returns ``[]`` and never raises:
    the first domain anyone builds has nothing to retrieve, and that is normal,
    not an error.
    """
    existing = list(library_objs) if library_objs is not None else _library()
    if not existing or k <= 0:
        return []
    query = query_text(interview)
    if not query:
        return []

    active = embedder if embedder is not None else embedding.get_embedder()
    vectors = active.embed([query, *(objective_text(o) for o in existing)])
    query_vector, existing_vectors = vectors[0], vectors[1:]
    scored = sorted(
        zip(existing, (cosine(query_vector, v) for v in existing_vectors), strict=True),
        key=lambda pair: (-pair[1], pair[0].id),
    )
    top = [obj for obj, _ in scored[:k]]
    log.info(
        "retrieval: %d of %d library objectives shown to the Architect via %s",
        len(top),
        len(existing),
        getattr(active, "name", "?"),
    )
    return top


def render_context(objectives: "Sequence[Objective] | Sequence[Any]") -> str:
    """The prompt block. Empty input renders an empty string, not a header."""
    if not objectives:
        return ""
    blocks: list[str] = [CONTEXT_HEADER]
    for obj in objectives:
        blocks.append(
            f"id: {obj.id}\n"
            f"title: {obj.title}\n"
            f"bloom: {obj.bloom} | difficulty: {obj.difficulty} | "
            f"minutes: {obj.est_minutes}\n"
            f"{obj.description}"
        )
    return "\n\n".join(blocks) + "\n\n" + REUSE_INSTRUCTION
