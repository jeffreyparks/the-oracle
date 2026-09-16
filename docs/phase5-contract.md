# Phase 5 Contract — "Push me"

Frozen interface. Three workers build in parallel. The last phase of the v1 loop.

## File ownership (do not cross)

| Worker | Owns |
| --- | --- |
| nudge | `nudge/**`, `tests/test_nudge.py` |
| coach | `agents/coach.py`, `commands/report.py`, `tests/test_report.py` |
| daemonless | `schedule/**`, `commands/cron.py`, `tests/test_schedule.py` |

Nobody edits `cli.py`, `store/models.py`, `agents/**` (except the coach worker's
own new file), `session/**`, `mastery/**`, `corpus/**`, `domains/**`, or another
worker's files. The orchestrator wires commands into `cli.py`.

## Reuse, do not reinvent

```python
from the_oracle.store.models import Nudge, NudgeState
from the_oracle.store.events import EventLog, EventKind   # NUDGE_SENT exists
from the_oracle.mastery.scheduler import ReviewPlan, is_due, next_review_at
from the_oracle.mastery import profile_for, MASTERY_THRESHOLD
from the_oracle.domains.registry import list_domains, load_domain
from the_oracle.style import system_prompt, VOICE
```
`Nudge(learner_id, rung, channel, state, body, scheduled_for, sent_at)` already
exists. Do NOT add columns.

## 1. `the_oracle.nudge` — the ladder (deterministic, no LLM)

```python
class Rung(IntEnum):
    QUIET = 0        # nothing owed, say nothing
    REMINDER = 1     # day 1 idle: a plain reminder
    SMALLER_ASK = 2  # day 3 idle: shrink the ask, "just 5 minutes"
    REPLAN = 3       # day 7 idle: offer to re-plan, "was the pace wrong?"
    PAUSE = 4        # day 14 idle: pause the plan and ask

@dataclass(frozen=True)
class NudgeDecision:
    rung: Rung; reason: str; body: str
    send: bool; suppressed_reason: str | None = None

IDLE_DAYS: dict[Rung, int] = {Rung.REMINDER: 1, Rung.SMALLER_ASK: 3,
                              Rung.REPLAN: 7, Rung.PAUSE: 14}

def decide(*, idle_days: float, due_count: int, last_nudge: Nudge | None,
           now: datetime, quiet_hours: tuple[int, int] | None = None) -> NudgeDecision
```
Hard rules, each with a test:
- **At most one nudge per calendar day.** A second call the same day returns
  `send=False`, `suppressed_reason="already nudged today"`.
- **Quiet hours are respected.** Inside them, `send=False`.
- **The ladder de-escalates.** Study activity resets the rung to QUIET.
- **Never nudge when nothing is owed** (`due_count == 0` and no idle plan).
- **One-word opt-out.** `learner.preferences["nudges"] == "off"` suppresses
  everything, permanently, no argument.

Bodies come from templates, not a model. Voice from `style.py`: plain, short,
never guilt-tripping. A nudge that shames the learner is a bug.

```python
# channels.py
class NudgeChannel(Protocol):
    name: str
    def send(self, nudge: Nudge) -> bool: ...

class TerminalChannel:  # writes to stdout via Rich
def get_channel(name: str = "terminal") -> NudgeChannel
```
Email/chat are later adapters. The protocol is the point.

## 2. `the_oracle.schedule` — due work without a daemon

PLAN.md section 9: **no daemon, no queue, nothing to restart.**

```python
@dataclass(frozen=True)
class DueSummary:
    learner_id: str; due_objectives: int; idle_days: float
    next_due_at: datetime | None; domains: dict[str, int]

def due_summary(learner_id: str, *, now=None, engine=None) -> DueSummary
def run_due_checks(learner_id: str, *, now=None, engine=None, send=True) -> NudgeDecision
```
`run_due_checks` is the one entry point a cron line calls. It computes the
summary, asks `nudge.decide`, records the `Nudge` row, appends `NUDGE_SENT`, and
delivers through the channel. Idempotent per day.

`commands/cron.py`: Typer `app` with
- `cron line` — print the exact crontab line, do not install anything.
- `cron install [--hour 9]` — add the line via `crontab`, refusing to duplicate.
- `cron uninstall` — remove only our line, identified by a marker comment.
- `cron status` — is it installed, when did it last run.

Never edit a user's crontab without showing the exact change first.

## 3. `the_oracle.agents.coach` + the weekly report

The ONLY place in Phase 5 that calls a model, and only once a week.

```python
class WeekFacts(BaseModel):
    learner_id: str; sessions: int; items: int; correct: int
    minutes: float; objectives_mastered: list[str]
    objectives_improved: list[tuple[str, float, float]]
    objectives_fading: list[tuple[str, float]]
    due_next_week: int; streak_days: int; idle_days: float

class WeeklyReport(BaseModel):
    headline: str                  # one sentence, honest
    what_moved: str
    what_is_fading: str
    what_is_next: str
    encouragement: str             # specific and earned, or omitted entirely

class Coach(Agent[WeekFacts, WeeklyReport]): ...

def gather_week(learner_id, *, now=None, engine=None) -> WeekFacts
def render(report: WeeklyReport, facts: WeekFacts) -> RenderableType
```
`gather_week` is pure and deterministic, computed from the event log. The model
only writes prose over facts it is given. **It must never invent a number.**
A test asserts every figure in the rendered output appears in `WeekFacts`.

A bad week gets an honest headline. If the learner did nothing, the report says
so without scolding, and `encouragement` is empty rather than hollow.

`commands/report.py`: `report [--weeks 1]`, Rich output, works with zero
activity.

## Rules

- Python 3.13, full type hints. No subject matter in `src/`.
- Tests offline, no key, no network, no real crontab writes (inject the runner).
- Freeze time in tests. Never assert against `datetime.now()`.
- `uv run pytest -q` stays green (371 today).


---

# Phase 5 outcome — verified live

## Live results

**Nudge ladder.** `cron run` for an idle learner sent rung 1:

> 1 item due, about 5 minutes.
> Run `the-oracle review` when you have a gap.

Second run the same day: `quiet: already nudged today`. One nudge per day holds
in the shipped path, not just in tests.

**Opt-out, end to end.** `learner pause` -> `cron run` -> `quiet: nudges off`.

**Weekly report**, real model, on a learner who had a bad week:

> You attended 1 session this week and got 0 out of 2 items right.
> what is fading: ...is fading at 0.17. You need to rebuild this before moving
> forward.

No encouragement section, because none was earned. The report told the truth
about a bad week without scolding.

**Cron** is one line and nothing else:

```
0 9 * * * .../the-oracle cron run >/dev/null 2>&1 # the-oracle:nudge
```

## Two defects found during integration

**1. The opt-out was unreachable.** The day-14 template told the learner to run
`the-oracle learner pause`. That command did not exist, and NOTHING in the
product could set `preferences["nudges"]`. The ladder read the preference
correctly; no human could ever write it. A documented opt-out you cannot reach is
not an opt-out, and it sits at the exact moment someone is disengaging.

Added `learner pause` / `learner resume`, which write the preference and append
`PREFERENCES_SET`.

Added `tests/test_optout.py::test_every_command_named_in_a_nudge_template_exists`:
it extracts every `the-oracle ...` command named in any template and asserts each
one runs. A template can never again promise a command the product lacks.

**2. Subject vocabulary leaked into the engine.** `agents/coach.py` used
"posterior" where it meant "mastery estimate". Caught by our own guard test, in a
file written by a worker that had never read the seed pack. Reworded.

## Notes

- The Coach enforces no-invented-numbers in three layers: pure `gather_week`,
  a post-generation scan that replaces any field containing an unsanctioned
  figure, and a renderer that prints only from facts. A lying cassette model is
  tested end to end.
- Zero activity never calls the model.
- No daemon anywhere: a test greps the `schedule` package for `threading`,
  `asyncio`, and `while True`.
- Tests can never touch a real crontab: the runner is injected, defaults to
  in-memory under pytest, and `SystemCrontab.write` raises if pytest is loaded.
