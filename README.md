# The Oracle

A tutor that learns you, plans the path, finds the material, and keeps you moving.

It assesses what you already know, builds a syllabus around the gaps, finds good
resources (or writes the lesson when nothing good exists), teaches in short
sessions, and reminds you to come back.

---

## Quick start

**1. Install**

```bash
git clone <this repo> && cd the-oracle
uv sync --extra embeddings          # the extra enables local semantic matching
uv tool install --editable .        # puts `oracle` on your PATH
```

Now just type `oracle` from anywhere. No `uv run`, no activated venv.

If `oracle` is not found, add uv's tool directory to your PATH:

```bash
uv tool update-shell    # then restart your shell
```

**2. Add your keys**

Put them in your data directory so `oracle` finds them from any folder:

```bash
mkdir -p ~/.the-oracle
cp .env.example ~/.the-oracle/.env
```

Set `ANTHROPIC_API_KEY` (required) and `SERPER_API_KEY` (required to find
resources on the web). Check they are visible:

```bash
oracle version
```

It reports which keys are set. It never prints their values.

A `.env` in the current directory, or any parent, also works — handy while
developing. Real environment variables always win.

**3. Install the starter subject**

```bash
oracle init
```

This copies the Bayesian forecasting pack into your data directory,
`~/.the-oracle`. That directory holds everything personal: your progress, your
generated subjects, and your database.

Keep it out of the repo. `ORACLE_HOME` overrides the location if you want it
somewhere else, but never point it at this checkout.

**4. Study**

```bash
oracle domain list                      # what you can learn
oracle assess bayesian_forecasting      # 12 questions, find your level
oracle plan bayesian_forecasting        # your syllabus, honest hours
oracle resources bayesian_forecasting --module m01_probability_foundations
oracle study bayesian_forecasting       # a 25 minute session
```

Run `resources` before `study`. The Oracle refuses to teach from an empty
corpus, and tells you the exact command to fix it.

**5. Keep going**

```bash
oracle review          # everything due now
oracle report          # what moved this week, what is fading
oracle cron install    # one crontab line, daily reminder
```


### Where your data lives

| Path | What | In git? |
| --- | --- | --- |
| `~/.the-oracle/objectives/` | Objective library, yours and shipped | no |
| `~/.the-oracle/domains/` | Your subjects, including generated ones | no |
| `~/.the-oracle/embeddings/` | Embedding cache, rebuildable | no |
| `~/.the-oracle/oracle.db` | Progress, evidence log, corpus | no |
| `data/packs/{objectives,domains}/` | **Seed** content shipped with the repo | yes |

The repo ships seed packs only. `oracle init` copies them out; it never
writes back. Nothing you do while studying touches the checkout.

---

## Learning something else

```bash
oracle domain add "Evaluate whether a forecast is well calibrated"
```

It interviews you about your goal, background, and time budget, then builds a
skill graph. Objectives you already mastered in another subject are **reused, not
re-taught** — so related topics start partly complete.

---

## Every command

| Command | What it does |
| --- | --- |
| `init` | Copy the starter subject into your data directory |
| `version` | Version, data directory, which keys are set |
| `domain list` / `show <id>` | What you can learn; modules and prerequisites |
| `domain add "<goal>"` | Build a new subject from an interview |
| `assess <domain>` | Adaptive diagnostic, capped at 12 items |
| `plan <domain>` | Syllabus with real hours and a finish date |
| `resources <domain> [--module M] [--dry-run]` | Find or write material |
| `study <domain> [--minutes N]` | A session. 10, 25, or 50 minutes |
| `review` | Everything due now, across subjects |
| `report` | The weekly report |
| `cron line \| install \| uninstall \| status` | The daily reminder |
| `learner pause` / `resume` | Stop or restart reminders |
| `learner export <id>` / `delete <id>` | Take your data, or erase it |
| `rebuild` | Recompute all progress from the event log |

---

## Useful things to know

**Nothing is taught until you are ready for it.** An objective is only taught
when every prerequisite is at 85% mastery. If that feels slow, it is the point.

**Sessions end early when you fade.** The Oracle watches your accuracy against
your own baseline, not a clock, and stops when the evidence says you are done.

**Reminders stop the moment you ask.** `oracle learner pause` is permanent
until you resume. There is no "are you sure".

**Your progress is rebuildable.** Everything is derived from an append-only
event log, so `oracle rebuild` can recompute your whole profile from scratch.

**It will tell you when a week went badly.** The weekly report does not spin a
bad week, and it leaves out praise it cannot justify.

---

## Telemetry (optional)

The Oracle can trace itself with [Logfire](https://logfire.pydantic.dev): one
span per command, one per agent run, plus every model call, HTTP request, and
SQL query.

```bash
uv sync --extra telemetry          # install it
export LOGFIRE_TOKEN=...           # then traces go to your project
oracle version                     # prints the telemetry state
```

Rules it follows:

- Off by the default. No token, nothing leaves your machine.
- Prompts and learner answers are **not** sent. Set
  `ORACLE_TELEMETRY_CAPTURE_CONTENT=1` if you want them.
- `ORACLE_TELEMETRY=0` turns instrumentation off completely.
- Telemetry never breaks a session. A missing or broken Logfire is reported
  and ignored.

---
---

# Architecture

Notes for anyone working on the code.

## The central idea: domains are data

Nothing in `src/the_oracle/` mentions any subject. Two tests enforce it: one
greps for subject vocabulary, and a stronger one asserts that no identifier from
a shipped pack appears anywhere in the source.

A **domain pack** is YAML in `~/.the-oracle`, not in the repo. A blank install is
fully capable with nothing loaded; `domain add` creates subjects at runtime.

## Objectives are shared atoms

Objectives live in one library. A domain is a **manifest** that references them
and adds its own prerequisite edges and modules. Two domains may reference the
same objective, and neither owns it.

Therefore `MasteryState` is keyed `(learner_id, objective_id)` with **no
domain**. Mastery transfers the moment two subjects share an objective. This is
the whole premise, and it is why dedupe biases toward splitting: a false split
costs one duplicate objective, a false merge silently corrupts mastery in both
directions.

Before drafting, the Architect is shown the nearest existing objectives and told
to reuse them. Prevention beats reconciliation — post-hoc dedupe cannot merge two
objectives that differ by Bloom level, and should not try.

## The agents

Seven specialists. **Six use a model.** Everything that does not need judgment is
plain Python: cheaper, faster, testable.

| Agent | Model | Job |
| --- | --- | --- |
| Orchestrator | no | Session loop, **sole writer of state**, budget ceiling |
| Assessor | yes | Item selection and grading; misconception detection |
| Architect | yes | Interview to skill graph to syllabus |
| Scout | yes | Ten candidate resources per objective |
| Reviewer | yes | Scores and rejects candidates |
| Author | yes | Writes the full lesson when the web has nothing |
| Coach | weekly | Ladder and scheduling are deterministic; only the report is written |

## Learner model

Append-only evidence, everything else derived. Bayesian Knowledge Tracing per
objective. `rebuild` replays the log and must reproduce current state exactly —
it is the recovery path, the model-swap path, and the main integration test.
`mastery/incremental.py` is a fast path that is tested to agree with full replay
to 1e-9.

## Study loop

Four phases: retrieval warm-up, new material, applied practice, consolidation.
Research gives no magic session length, so the **shape** is fixed and the
**length adapts** to measured fatigue. Spacing is the primary variable.

Feedback is immediate but never mid-attempt. The exception is mandatory: the same
misconception twice in one session interrupts and re-teaches.

## Stack

Python 3.13, uv, Pydantic AI 2.x (pinned), SQLModel + SQLite, Typer + Rich,
httpx + trafilatura, Serper for search, sentence-transformers for dedupe
embeddings (optional extra; falls back to lexical and says so), Logfire for
tracing (optional extra; no-op without it).

**Not here on purpose:** no daemon, no queue, no web UI, no Postgres, no Docker,
no vector database. One SQLite file, one directory of YAML, one CLI.

## Engineering rules

- Only the orchestrator commits state.
- Budget ceilings live in one place.
- Idempotency keys `(agent, objective_id, input_hash)` on every agent call.
- Cassettes for agent tests: recorded once, replayed offline, a miss raises.
- Domain packs validate fully on load or refuse to load.
- Synthetic learners test pedagogy in seconds.
- Respect `robots.txt`: fail open on an unreachable file, fail closed on an
  explicit `Disallow`.

## Testing

```bash
uv run pytest -q          # 475 tests, offline, no keys, no network
```

Development uses `uv run` as usual; `oracle` on your PATH is the editable
install, so code changes take effect immediately.

Every test runs without an API key. Tests cannot touch a real crontab. A live
model is never called from the suite.

Offline tests prove the plumbing. They have never once caught a product defect
that a live run then found — repeated items, empty misconception fields,
unreachable commands, and an unwired Critic all shipped past a green suite. Run
the thing for real before believing it works.

## Documents

`PLAN.md` is the design. `docs/*-contract.md` are the frozen interfaces each
phase was built against, each with an appended outcome section recording what
live runs actually revealed.
