# Retrieval-Augmented Drafting (Phase 2.5)

Fixes the design gap found by live runs: the Architect drafts blind, invents
objectives that parallel existing ones at a different Bloom level, and post-hoc
dedupe cannot merge them without violating bias-to-split. Result: a bloated
library and mastery that stops transferring.

**Prevention over reconciliation.** Show the Architect what already exists.

## 1. Retrieval

New in `the_oracle/agents/architect.py` (or a small `retrieval.py` beside it):

```python
def retrieve_context(
    interview: Interview,
    *,
    k: int = 20,
    embedder: Embedder | None = None,
    library_objs: Sequence[Objective] | None = None,
) -> list[Objective]:
    """Nearest existing objectives to what this learner asked for."""
```
- Query text = `goal`, `must_cover`, and `background`, joined.
- Rank by cosine against `objective_text(...)` (reuse `dedupe`/`embed` helpers;
  do not write a second embedding path).
- Return the top `k`, best first. Empty library returns `[]` and must not raise.

## 2. Prompt

`draft_domain` calls `retrieve_context` and renders the results into the prompt:

```
EXISTING OBJECTIVES IN THE SHARED LIBRARY
id: ts_backtesting
title: Backtesting: Rolling Origin and Time Series Cross-Validation
bloom: evaluate | difficulty: 4 | minutes: 75
Evaluate forecasts with rolling-origin or expanding-window schemes...
```

Instruction to add, in the existing voice:

> Some of these already teach what this learner needs. If an existing objective
> covers a required skill, REUSE it: copy its title exactly and set
> `existing_id` to its id. Do not restate the same skill at a different Bloom
> level just to reword it. Draft a new objective only when nothing listed
> genuinely covers the skill, or when the learner needs it at a materially
> different depth — and if you do that, say why in the description.

## 3. Draft schema

`DraftObjective` gains:

```python
existing_id: str | None = Field(
    default=None,
    description="Id of a library objective this reuses verbatim. Null when new.",
)
```

## 4. Resolution in `plan_domain`

Before the dedupe pass:
- For each draft objective with `existing_id` that resolves in the library:
  reuse that id directly. Record a `Match` with `decision=REUSE`,
  `best_id=existing_id`, and a marker that it came from retrieval, not scoring,
  so the CLI report can distinguish the two.
- `existing_id` that does NOT resolve is not fatal: log it, drop the claim, and
  fall through to the normal dedupe path. A hallucinated id must never break a
  build, and must never silently reuse the wrong objective.
- Everything else: unchanged dedupe.

Reused objectives keep the library body. Their est_minutes, bloom, and stems come
from the library, never from the draft.

## 5. CLI

`domain add` report gains a reason column: `retrieved` (Architect chose it),
`matched` (dedupe score), `judged` (Critic agreed), or `new`. The summary line
becomes e.g. `12 objectives, 5 reused (4 retrieved, 1 judged), 7 new.`

## 6. Tests (`tests/test_retrieval.py`, plus additions to `test_architect.py`)

Offline, no API key, fake or lexical embedder:
1. `retrieve_context` returns nearest-first and respects `k`.
2. Empty library returns `[]`, no exception.
3. A draft objective with a valid `existing_id` reuses it, mints nothing, and
   keeps the LIBRARY body (assert est_minutes/bloom come from the library even
   when the draft disagrees).
4. A hallucinated `existing_id` falls through to dedupe and still builds.
5. An `existing_id` pointing at a real objective that is clearly a different
   skill is still honoured (the Architect saw it and chose it) — but assert it is
   reported as `retrieved`, so a human can audit the choice.
6. The prompt actually contains the retrieved ids (guard against the block being
   built and never interpolated — that bug has happened twice in this project).

## Rules

- No subject matter in `src/`. Tests must pass offline.
- Do not change `dedupe.decide` banding or the Critic wiring; both were just
  fixed against live evidence.
- `uv run pytest -q` must stay fully green (197 tests today).


---

# Outcome — verified live

Same overlapping topic that exposed the problem, before and after.

| | Before | After |
| --- | --- | --- |
| Objectives | 8 | 5 |
| Reused | **0** | **3** (all `retrieved`) |
| New | 8 | 2 |

The two new ones are genuinely new — combining calibration, sharpness and
scoring into a report, and diagnosing specific failure modes — not reworded
copies of library objectives.

## Mastery transfer, proven end to end

A learner masters three objectives while studying the ORIGINAL seed domain, then
opens the newly generated domain they have never touched:

```
ts_scoring_rules            p=1.000  MASTERED (transferred)
ts_calibration_sharpness    p=1.000  MASTERED (transferred)
ts_backtesting              p=1.000  MASTERED (transferred)
combining_...               p=0.200  to learn
diagnosing_...              p=0.200  to learn
```

`the-oracle plan` on the new domain then schedules **2.0 h instead of 4.6 h**,
reporting "already mastered 3". This is the shared-library premise working:
mastery is keyed `(learner_id, objective_id)` with no domain, so it transfers the
moment two domains share an objective.

## One real interaction found

Reused objectives carry LIBRARY durations. A practitioner-level objective is 60-75
minutes, so a 1 h/week, 1 week interview now fails the budget check:

> budget: the draft needs 165 minutes but the learner has about 48

That is correct behaviour — nothing was written, and the pack stayed clean — but
it means small budgets and deep reuse conflict. The Architect sees the minutes in
the retrieval block and should plan within them. Left as a known behaviour rather
than papered over.
