# Phase 4 Contract — "Study loop"

Frozen interface. Three workers build in parallel.

## File ownership (do not cross)

| Worker | Owns |
| --- | --- |
| session | `session/**`, `tests/test_session.py` |
| mastery2 | `mastery/incremental.py`, `mastery/gate.py`, `tests/test_gate.py` |
| commands4 | `commands/study.py`, `commands/review.py`, `tests/test_study_cli.py` |

Nobody edits `cli.py`, `store/models.py`, `agents/**`, `domains/**`, `corpus/**`,
`planning/**`, `mastery/{bkt,profile,reducers,scheduler,synthetic}.py`, or
`tests/cassettes/replay.py`. The orchestrator wires commands into `cli.py`.

## Reuse, do not reinvent

```python
from the_oracle.agents.assessor import (
    Item, Grade, ItemWriter, Grader, kind_for_bloom, feedback_is_immediate,
    next_difficulty, misconceptions_for,
)
from the_oracle.mastery import profile_for, MASTERY_THRESHOLD
from the_oracle.mastery.scheduler import Review, ReviewPlan, plan, next_review_at, is_due
from the_oracle.corpus.store import accepted_for
from the_oracle.planning.syllabus import build_syllabus
from the_oracle.store.events import EventLog, EventKind
```
`EventKind` already has `SESSION_STARTED`, `SESSION_ENDED`, `ITEM_PRESENTED`,
`RESPONSE_GRADED`, `MISCONCEPTION_DETECTED`, `REVIEW_SCHEDULED`.

## 1. `the_oracle.mastery.gate` — the strict teaching gate

**Phase 1 debt item 1.** The diagnostic's gate is deliberately loose: a
prerequisite counts as met if mastered OR just answered correctly, because
strict gating stalls a probe after one item. That relaxation must NOT reach
teaching.

```python
def prerequisites_met(domain, objective_id, profile, *, threshold=MASTERY_THRESHOLD) -> bool
def blocked_by(domain, objective_id, profile, *, threshold=MASTERY_THRESHOLD) -> list[str]
def teachable(domain, profile, *, threshold=MASTERY_THRESHOLD) -> list[str]
```
Strict: EVERY direct prerequisite must be at `p >= threshold`. No exceptions, no
"just answered correctly". A required test asserts the strict and loose gates
disagree on a constructed case, so the two can never silently converge.

## 2. `the_oracle.mastery.incremental` — apply without full replay

**Phase 1 debt item 2.** `run_diagnostic` calls `rebuild()` once per item. That
is fine at 12 items and hopeless in a session.

```python
def apply_event(state: MasteryState | None, event: Event) -> MasteryState
def apply_response(learner_id, payload: dict, *, engine=None) -> float   # returns new p
```
Must produce **exactly** what full replay produces. A required test drives N
random responses through both paths and asserts identical mastery, to 1e-9.
Full replay stays the source of truth; this is a fast path, not a second model.

## 3. `the_oracle.session` — the four-phase session

Per PLAN.md section 5.

```python
class Phase(StrEnum):
    WARMUP = "warmup"; NEW = "new"; PRACTICE = "practice"; CONSOLIDATE = "consolidate"

@dataclass(frozen=True)
class PhaseBudget:
    phase: Phase; minutes: float

DEFAULT_SHAPE: tuple[PhaseBudget, ...]   # 4 / 14 / 5 / 2 at 25 min, scaled for 10 and 50

@dataclass(frozen=True)
class SessionPlan:
    learner_id: str; domain_id: str; minutes: int
    warmup: list[str]        # objective ids, due review items
    focus: list[str]         # objective ids to learn, teachable only
    shape: tuple[PhaseBudget, ...]

def plan_session(domain_id, learner_id, *, minutes=25, now=None, engine=None) -> SessionPlan
```
- `warmup` = 2-3 due review objectives from the scheduler, oldest first.
  Review items are injected into a normal session, never a separate chore.
- `focus` = teachable objectives only, using `mastery.gate` (STRICT). Usually one.
- No due reviews yet -> warmup draws from the most recently studied objectives.

### Fatigue

```python
@dataclass(frozen=True)
class FatigueSignal:
    stop: bool; reason: str; observed: float; baseline: float

def fatigue(responses: Sequence[bool], *, baseline: float, min_items: int = 6) -> FatigueSignal
```
Compare recent accuracy against the learner's own baseline, not a constant.
Needs `min_items` before it may fire. When it fires the session ends early and
says why, in plain words. Never mid-attempt.

### Feedback timing

Use `assessor.feedback_is_immediate(kind)`. Recall and short items get instant
feedback; multi-step gets the full critique at the end of that attempt.
**Exception, required:** the same misconception twice in one session interrupts
and re-teaches immediately.

```python
def should_interrupt(seen_misconceptions: Sequence[str], new_id: str | None) -> bool
```

### Running a session

```python
@dataclass(frozen=True)
class SessionResult:
    session_id: str; asked: int; correct: int
    objectives_touched: list[str]; mastery_before: dict[str, float]
    mastery_after: dict[str, float]; ended_early: bool; reason: str

def run_session(plan: SessionPlan, *, answer_fn=None, on_event=None, engine=None) -> SessionResult
```
- Appends `SESSION_STARTED` / `SESSION_ENDED` and every item event.
- Updates mastery through `mastery.incremental`, never by direct write.
- `answer_fn(item) -> str` injectable, so a synthetic learner drives it in tests.
- Re-plan on drift: if a focus objective becomes non-teachable mid-session
  (a prerequisite decayed), drop it and say so rather than teaching blind.

## 4. Commands

`commands/study.py`: Typer `app`, `study <domain_id> [--minutes N]`. Runs a real
session with Rich. Shows the phase you are in, the resource for the objective
(`corpus.store.accepted_for`, first accepted), feedback per the timing rules, and
a close-out: what moved, what is due next, when to come back.
`commands/review.py`: `review` — everything due now, across all domains.

Both must state honestly when nothing is due, and never invent work.

## Rules

- Python 3.13, full type hints. No subject matter in `src/`.
- Tests offline, no key, no network. Cassettes for model calls.
- Do NOT make live calls. The orchestrator runs live verification.
- `uv run pytest -q` stays green (312 today).


---

# Phase 4 outcome — verified live

Whole loop exercised against real models: item writing, grading, mastery,
scheduling, and the CLI a learner actually uses.

## Debt cleared

**Strict gate (Phase 1 debt 1).** `mastery/gate.py` requires EVERY direct
prerequisite at `p >= 0.85`. The divergence is locked in by test: a learner who
answered one objective correctly once sits at `p = 0.63`; the diagnostic's loose
gate says met, the teaching gate says blocked. Asserted to disagree, so they
cannot silently converge.

**Incremental mastery (Phase 1 debt 2).** `mastery/incremental.py` matches full
replay to 1e-9 across 6 seeds x 40 mixed responses, on every field including the
review schedule. Speedup 2.1x at 50 responses, 4.2x at 150, 5.7x at 300 — the gap
widens as the log grows, which is the point. Full replay stays the source of truth.

## Live session

Real model calls throughout. Strict gate opened only after prerequisites were
genuinely mastered. Warm-up drew due reviews; focus took one teachable objective;
mastery moved 0.20 -> 0.28 on two items; the close-out gave real numbers and a
return date.

Fatigue on a realistic curve (baseline 85%): holds through two misses in the
6-item window (83%, then 67%), stops at 50% on item 13. It tolerates a bad
question and reacts to a bad run.

## Two defects the live run exposed, both fixed

**1. The practice item repeated the teaching question.** Both requests received
the whole stem list and the model picked the same stem twice. Practice that
repeats the teaching question is recognition, not retrieval.

First fix failed and is worth recording: filtering by the returned stem text does
nothing, because the model *rewrites* the stem it is given, so it never matches
the source. The rotation has to be decided before the call. `_Run.stems_for` now
hands over exactly ONE source stem, chosen by how many items that objective has
already produced this session. Deterministic, and it varies the idempotency key
as a side effect.

**2. "You worked 1 objectives".** Pluralisation. Small, but it is in the one
place the learner hears the Oracle speak.

## Honest notes

- `study` refuses to teach an objective with no accepted material and names the
  exact `resources` command: "Until that runs there is nothing to teach from,
  and I will not fake it." Warm-up retrieval is still allowed without material,
  which is correct - recall needs no reading.
- Minutes are budget minutes, not wall clock, so sessions stay deterministic and
  testable. A real timer can wrap this later.
- `SessionResult` has no dedicated drift or interrupt field; both land in
  `reason`. Fine for now, worth a field if Phase 5 reports on them.
