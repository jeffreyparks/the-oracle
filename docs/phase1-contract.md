# Phase 1 Contract — "Know me"

Frozen interface for Phase 1. Three workers build against it in parallel.
Do not change this file; if it is wrong, report it instead.

## File ownership (do not cross)

| Worker | Owns |
| --- | --- |
| mastery | `src/the_oracle/mastery/**`, `tests/test_mastery.py`, `tests/synthetic/**` |
| assessor | `src/the_oracle/agents/assessor.py`, `src/the_oracle/commands/assess.py`, `tests/test_assessor.py`, `tests/cassettes/**` |
| privacy | `src/the_oracle/store/privacy.py`, `src/the_oracle/commands/learner.py`, `tests/test_privacy.py` |

Nobody edits `cli.py`. Each worker exposes a Typer app in its own
`commands/<name>.py` module as a module-level object named `app`. The
orchestrator wires them into `cli.py` afterwards. Create
`src/the_oracle/commands/__init__.py` if missing; that one shared file may be an
empty stub, so keep it empty.

Nobody edits `store/models.py`. If a column is missing, report it.

## Existing interfaces you must use (already built, do not reimplement)

```python
from the_oracle.store.events import EventLog, EventKind, Event
log = EventLog(engine)                     # engine optional
log.append(kind, learner_id, payload)      # only write path
log.read(learner_id, kinds=[...])          # replay order

from the_oracle.store.rebuild import ReducerRegistry, ReplayState, REGISTRY
@REGISTRY.on(EventKind.RESPONSE_GRADED)
def _reduce(state: ReplayState, event: Event) -> None: ...

from the_oracle.agents.base import Agent, AgentResult, hash_input
from the_oracle.domains.registry import load_domain
from the_oracle.context import LearnerContext
from the_oracle.style import system_prompt
```

Relevant event kinds already defined: `ITEM_PRESENTED`, `RESPONSE_GRADED`,
`MISCONCEPTION_DETECTED`, `SESSION_STARTED`, `SESSION_ENDED`.

## `the_oracle.mastery` — owned by worker `mastery`

```python
@dataclass(frozen=True)
class BKTParams:
    p_init: float = 0.2      # prior knowledge
    p_transit: float = 0.15  # learning on an opportunity
    p_slip: float = 0.10     # knows it, answers wrong
    p_guess: float = 0.20    # does not know it, answers right

def posterior(p_prior: float, correct: bool, params: BKTParams) -> float: ...
    """Bayes update on one observation, then apply the learning rate."""

def update(p_prior: float, correct: bool, params: BKTParams | None = None) -> float: ...

MASTERY_THRESHOLD = 0.85

class MasteryProfile:
    def p(self, objective_id: str) -> float: ...
    def mastered(self) -> set[str]: ...
    def known_prerequisites_met(self, domain, objective_id: str) -> bool: ...
    def as_dict(self) -> dict[str, float]: ...

def profile_for(learner_id: str, engine=None) -> MasteryProfile: ...
```

Reducers live in `mastery/reducers.py` and register on `REGISTRY` at import
time, so `rebuild(learner_id)` recomputes `MasteryState` from the log alone.
`RESPONSE_GRADED` payload contract:

```python
{"objective_id": str, "item_id": str, "correct": bool, "difficulty": int,
 "bloom": str, "seconds": float, "misconception_id": str | None}
```

Replay must be pure: same log in, same mastery out, every time.

## `the_oracle.mastery.synthetic` — owned by worker `mastery`

```python
class SyntheticLearner:
    def __init__(self, true_mastery: dict[str, float], slip=0.1, guess=0.2, seed=0): ...
    def answer(self, objective_id: str, difficulty: int) -> bool: ...
```
Plus `recovery_error(true, estimated) -> float` (mean absolute error).

## `the_oracle.agents.assessor` — owned by worker `assessor`

```python
class Item(BaseModel):
    id: str; objective_id: str; stem: str; kind: Literal["recall","short","multi_step"]
    difficulty: int; bloom: str; expected: str; misconception_probes: list[str] = []

class ItemRequest(BaseModel):
    objective_id: str; bloom: str; difficulty: int; stems: list[str]

class Grade(BaseModel):
    correct: bool; confidence: float; misconception_id: str | None = None
    feedback: str; reasoning: str

class GradeRequest(BaseModel):
    item: Item; answer: str; misconceptions: list[dict] = []

class ItemWriter(Agent[ItemRequest, Item]): ...
class Grader(Agent[GradeRequest, Grade]): ...
```

Adaptive loop in the same module:

```python
def run_diagnostic(domain_id, learner_id, *, max_items=12, target_accuracy=0.7,
                   answer_fn=None, engine=None) -> MasteryProfile
```
- Picks the next objective by information gain: prefer `p` nearest 0.5, respect
  prerequisites, never repeat an objective.
- Stops early when every remaining ability interval is tight or `max_items` is hit.
- Appends `ITEM_PRESENTED` and `RESPONSE_GRADED` for every item. **Never writes
  `MasteryState` directly** — mastery comes only from reducers.
- `answer_fn(item) -> str` is injectable, so tests drive it with a synthetic learner.

## `the_oracle.store.privacy` — owned by worker `privacy`

```python
def export_learner(learner_id: str, engine=None) -> dict: ...   # JSON-safe, every learner-scoped table + full event log
def delete_learner(learner_id: str, engine=None) -> dict: ...   # hard delete, returns per-table counts
def learner_tables() -> list[str]: ...
```
Deleting one learner must never touch another learner's rows, and must never
touch shared tables (`Objective`, `Resource`, `Item` bank, etc.).

## Rules for everyone

- Python 3.13, full type hints, Pydantic v2 / SQLModel as already used.
- **No subject matter in `src/`** — a test greps for it.
- Use the project environment: `uv run pytest ...`. Never the kernel.
- Do not run `uv add` for a package another worker is likely to add; if you need
  a new dep, add it and tolerate one lock retry.
- Keep it simple and readable. Prefer plain functions over classes.


---

# Phase 1 outcome — gaps resolved and debt carried

Appended after the three workers landed. This section is the authority where it
disagrees with the frozen contract above.

## Gaps the contract left open

| Gap | Resolution |
| --- | --- |
| Confidence formula | `n / (n + 3)` where `n` is observations for that objective |
| `MISCONCEPTION_DETECTED` payload | `{"objective_id": str, "misconception_id": str}`; records the tag, does not move `p` |
| Unseen objective | `MasteryProfile.p()` returns `BKTParams.p_init` (0.2) |
| Feedback timing hook | `run_diagnostic(..., on_graded=...)` keyword-only callback |

## Measured behaviour (evidence, not claims)

- Replay purity proven twice: 3x rebuild gives identical rows; a DB-free fold repeats exactly.
- Recovery: polarised cohort, 40 items, worst MAE **0.075** over 50 seeds (asserted tolerance 0.10).
- Threshold classification exact from 20 items; at the **12-item cap, exact in 46/50 seeds**.
- Learners near `p = 0.5` saturate slowly. Inherent to BKT. Documented, not hidden.
- Cassette tests run offline with no API key. A cassette miss raises `CassetteMiss`;
  there is no network fallback.

## Debt carried into later phases

1. **Diagnostic prerequisite gate is deliberately loose.** During a diagnostic a
   prerequisite counts as cleared if it is mastered **or** was just answered
   correctly. Strict gating stalls after one item, because the pack has a single
   root. This relaxation is correct for *probing* and **must not leak into the
   study loop**, where mastery learning still requires `p >= 0.85` on every
   prerequisite before advancing. Phase 4 must implement the strict gate
   separately and test that the two differ.
2. **`run_diagnostic` calls `rebuild()` once per item** so the profile reflects
   the latest response. Fine at a 12-item cap; it is O(log) replay per item and
   will not survive a long study session. Phase 4 needs an incremental apply
   path alongside full replay, with a test that the two agree.
3. **Objective tags are weak** — derived from id prefixes during the Phase 0
   migration. Tags feed dedupe recall. Fix in Phase 2 with `domain add`.
4. **Scheduler is FSRS-*style*, not FSRS.** Stability, difficulty, and lapses are
   tracked; the retrievability curve and interval fuzz are not implemented. This
   is documented in `mastery/scheduler.py`. Revisit in Phase 4.
5. **Cassettes are hand-written**, not recorded from a live model. They prove the
   plumbing, not the prompt quality. Re-record against a real model in Phase 2
   with `ORACLE_CASSETTE_RECORD=1`.
