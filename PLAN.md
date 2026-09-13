# The Oracle

A multi-agent tutor that learns the student, plans the path, finds or writes the
material, and keeps the student moving.

> Source of truth for the design. 19 decisions were settled in planning; this
> document supersedes all earlier drafts.

## 1. What it is

One engine, many domains, many learners.

| Promise | What the learner sees |
| --- | --- |
| Know me | A short adaptive diagnostic, then "here is what you already know" |
| Plan me | A syllabus with modules, hours, and checkpoints |
| Feed me | Curated readings and exercises, written from scratch when nothing good exists |
| Push me | Spaced review, honest nudges, a weekly report |

Surface: a CLI (`the-oracle`). No web UI in v1.

## 2. The central idea: domains are data

Nothing under `src/the_oracle/` may mention Bayes, statistics, or any subject
matter. If it does, the abstraction has leaked, and a test should fail.

- A **domain pack** is YAML in a user directory (`~/.the-oracle/`), not in the repo.
- `the-oracle domain add "..."` runs the Curriculum Architect to generate a new
  pack at runtime, after a short interview.
- A **blank Oracle** is the engine with an empty registry: every capability
  present, nothing to learn yet. That is the shippable artifact for a new user.
- Sharing a subject with someone is copying one file.

### Objectives are shared atoms

Objectives live in a **shared library**, not inside a domain pack. A domain is a
**manifest**: ordered references to objective ids, plus its own prerequisite
edges and modules. Two domains may reference the same objective; neither owns it.

Therefore **`MasteryState` is keyed by `(learner_id, objective_id)` with no
domain**. Mastery transfers across related domains automatically: add "Bayesian
A/B testing" beside "Bayesian forecasting" and the overlap is already known.

Consequences to respect:
- **Prerequisites belong to the manifest**, not the objective, because the same
  skill needs different scaffolding in different contexts.
- **Objectives are versioned and domains pin a version**, because an edit ripples
  into every domain that references it.
- **Dedupe biases toward split.** A false split is cheap; a false merge silently
  corrupts mastery in both directions. Reuse only on strong evidence.

### `domain add` flow

```
interview -> Architect drafts objectives -> embed each (API) and match against
the library -> high similarity: reuse id | middle band: Critic judges same skill
vs same words | low: mint new objective -> show the learner what was reused ->
validate DAG -> commit manifest
```

## 3. Agents

Seven specialists plus an orchestrator. **Six use an LLM.** Everything that does
not need judgment is plain Python: cheaper, faster, testable.

| Agent | LLM | Job |
| --- | --- | --- |
| Orchestrator | no | Session loop, routing, **sole writer of state**, budget enforcement |
| Assessor | yes | Diagnostic item selection **and** grading; ability + misconception detection |
| Curriculum Architect | yes | Interview -> skill graph -> syllabus, prerequisite ordering |
| Scout | yes | Finds 10 candidate resources per objective (Serper + fetch) |
| Critic | yes | Scores and rejects candidates; adjudicates dedupe middle band |
| Author | yes | Writes the full lesson when nothing good exists |
| Coach | weekly only | Nudge ladder, review scheduling, streaks — all deterministic; LLM writes the weekly report only |

The Evaluator is merged into the Assessor: scoring a diagnostic answer and
grading submitted work are the same operation at different times.

Loop: `Assessor -> Architect -> (Scout -> Critic -> Author?) -> study ->
Assessor -> Coach -> re-plan on drift`.

## 4. The learner model

- **Evidence log is append-only. All state is derived.** Never lose raw signal.
- **Mastery**: Bayesian Knowledge Tracing per objective — `p(mastery)`,
  confidence, last seen, next review. IRT/Elo is an upgrade path, not v1.
- **Misconception tags**: named wrong models the Assessor detects and the
  Author targets directly.
- **Preferences**: session length, modality, quiet hours, timezone, tone.
- **`the-oracle rebuild`** drops derived state and replays the log. This is the
  recovery path, the model-swap path, and the primary integration test. If a
  replay does not reproduce current state, something is broken.

## 5. The study session

Research gives **no magic session length**. Bradbury (2016) debunks the fixed
"attention span"; the 25-minute figure is Pomodoro folklore. What is well
supported is that **spacing dominates**: distributed practice d = 0.54 (Mawson,
2025 meta-analysis), with a further gain from spacing *retrieval* episodes
(Latimier, 2021). Optimal gap scales at roughly 10-20% of the retention interval.

So: **return gap is the primary variable, length is secondary.**

- Default **25 min**, learner-settable 10 / 25 / 50.
- **Adaptive early exit**: the Oracle tracks in-session accuracy against the
  learner's own baseline and ends the session when decay is detected, with a
  plain explanation. Measured fatigue beats an assumed constant.

Fixed structure, which matters more than duration:

| Phase | Time | Purpose |
| --- | --- | --- |
| Retrieval warm-up | ~4 min | 2-3 items from earlier modules |
| New material | ~14 min | one objective, rarely two |
| Applied practice | ~5 min | use it, do not just read it |
| Consolidation | ~2 min | summarise, preview, schedule the next session |

### Feedback timing

**Immediate, but never mid-attempt. Feedback lands at the smallest natural
boundary.**

- Recall and short items: instant.
- Multi-step or build tasks: full critique at the end of *that attempt*, not the
  end of the set. Interrupting a derivation destroys working memory.
- Exception: the **same misconception twice in one session** interrupts and
  re-teaches. A repeated wrong model strengthens with each repetition.

## 6. Pedagogy that maps to code

- **Bloom level** on each objective controls the item type the Assessor generates.
- **Adaptive diagnostic**: stop when the ability interval is tight, target ~70%
  correct, cap at 12 items.
- **Mastery learning**: no advance until prerequisites reach `p >= 0.85`.
- **Spaced repetition**: FSRS-style scheduling over the evidence log; review
  items are injected into normal sessions, never a separate chore.
- **Interleaving and retrieval practice**: built into the warm-up phase.
- **Desirable difficulty**: the Coach steers toward ~80% success.

## 7. Resources

**Lazy, per module.** Nothing is fetched for a module the learner has not reached.

```
objective -> Scout (10 candidates) -> Critic (score + reject) -> >=2 accepted?
   yes -> attach, cache in shared corpus
   no  -> Author writes the lesson -> Critic reviews -> attach
```

Critic rubric, 0-5 weighted: authority, accuracy, level fit, depth, recency,
accessibility/licence, effort-to-value. Hard reject on dead link, paywall-only,
or level mismatch beyond one band. Start at 10 candidates and tune from the
observed acceptance rate.

**Authored lessons are full lessons**: explainer prose, 2+ worked examples, 3-6
graded exercises with answer keys and wrong-answer feedback, runnable code where
the objective is computational, and a short summary for spaced review.

The corpus is keyed by `(objective_id)` and **shared across domains and
learners**. Learner N costs far less than learner 1; track that ratio.

## 8. Voice

One style spec, shared by every agent, so tone cannot drift.

**Friendly, encouraging, serious.**

- **Serious**: wrong is called wrong, plainly and at once. Difficulty is stated
  up front, never softened. It never claims you understand what the evidence
  does not support.
- **Encouraging**: praise is specific and earned. "You caught the divergence
  before checking R-hat" — not "Great job!". Progress is shown as evidence.
  After failure the frame is diagnostic, not consoling.
- **Friendly**: plain language, second person, contractions, short sentences. A
  knowledgeable colleague at a whiteboard.
- **Banned**: exclamation-mark enthusiasm, emoji, "Let's dive in!", apologising
  for hard material, false balance on a wrong answer.

## 9. Engineering rules

- **One writer.** Only the orchestrator commits state.
- **One cost chokepoint.** Budget ceilings checked in the orchestrator before
  every agent call. Nowhere else.
- **Idempotency keys** `(agent, objective_id, input_hash)` on every agent call,
  so a retry returns the cached result instead of authoring a duplicate lesson.
- **Cassettes.** Record real LLM responses once, replay in tests. Without this,
  agent testing stops within two weeks.
- **Fail loud.** A domain pack validates fully on load — acyclic, references
  resolve, versions pinned — or refuses to load.
- **Synthetic learners.** Simulated learners with known true mastery and a fixed
  error rate verify that the Assessor recovers the right profile and the
  Architect never violates a prerequisite. Pedagogy gets tested in seconds.
- **No daemon.** Due work is computed on demand at CLI start, plus one cron line
  for the daily nudge. No scheduler process, no queue, nothing to restart.
- **Privacy.** Per-learner export and delete from Phase 1.

## 10. Stack

Python 3.13, `uv`, `src/the_oracle` layout.
**Pydantic AI 2.x** (pinned; the 2.x line moves fast) — agents are stateless
objects, so the orchestrator keeps sole ownership of state and the learner model
stays the single source of truth. Rejected a thin custom loop (owning the
high-churn 20% with no gain in control) and LangGraph (it wants to own state,
creating a second source of truth). See `docs/framework-research.md`.

SQLModel + SQLite (no SQLite-only SQL, so Postgres is a connection string).
Typer + Rich. httpx + trafilatura. Serper via the `websearch` skill. Embeddings
via API for dedupe. pytest.

**Model routing** via the `compute-routing` skill: cheap models for fetch and
summarise, strong models for the Assessor, Architect, and Author.

**Not in v1**: web UI, Postgres, Docker, vector database, message queue.

## 11. Layout

```
src/the_oracle/
  cli.py              typer app: domain, assess, plan, study, review, report, rebuild
  config.py           settings, paths, model routing table
  context.py          LearnerContext - the auth seam (stubbed in Phase 0)
  style.py            the one voice spec, imported by every agent
  domains/
    registry.py       discover/load/validate packs from ~/.the-oracle
    schema.py         Domain, Manifest, Objective, Misconception, Module
    library.py        shared objective library; mint, version, lookup
    dedupe.py         embed + match + Critic adjudication
  store/
    db.py  models.py  events.py  rebuild.py
  agents/
    base.py           Agent[In, Out] on Pydantic AI; contract, usage, idempotency
    orchestrator.py   session loop; ONLY writer of state; budget gate
    assessor.py  architect.py  scout.py  critic.py  author.py  coach.py
  mastery/
    bkt.py            p(mastery) update
    scheduler.py      FSRS review dates, return-gap policy
  session/
    loop.py           four-phase structure, fatigue detection, feedback timing
  corpus/
    store.py  fetch.py
  nudge/
    ladder.py         deterministic escalation
    channels.py       NudgeChannel protocol; TerminalChannel first
tests/
  synthetic/          simulated learners
  cassettes/          recorded LLM responses
```

### Tables

**Shared** (the moat): `Objective`, `ObjectiveVersion`, `Misconception`,
`Domain`, `DomainEdge`, `Module`, `Resource`, `ResourceReview`, `Item`.

**Learner-scoped** (private, exportable, deletable): `Learner`, `Enrollment`,
`Goal`, `MasteryState`, `Response`, `StudySession`, `ReviewSchedule`, `Nudge`,
`EventLog`.

`Enrollment` joins learner to domain, so one learner can study several subjects
at once. `MasteryState` deliberately has **no** `domain_id`.

## 12. Phases

**Phase 0 - Skeleton.** CLI, config, store, `LearnerContext`, agent base, event
log, `rebuild`, style spec. Restructure the Bayesian pack into shared library +
manifest.
*Exit:* `the-oracle domain list` reads a validated pack; `rebuild` round-trips.

**Phase 1 - Know me.** Assessor, BKT, synthetic learners, export/delete.
*Exit:* a mastery profile in <=12 items; synthetic learner recovered correctly.

**Phase 2 - Plan me.** Architect, `domain add` with interview and dedupe.
*Exit:* a second domain is generated from scratch and reuses shared objectives.

**Phase 3 - Feed me.** Scout, Critic, Author, corpus, lazy per module.
*Exit:* a module's objectives all have >=2 accepted resources or a full lesson.

**Phase 4 - Study loop.** Four-phase session, feedback timing, fatigue
detection, review injection, re-plan on drift.
*Exit:* a real week of study with measured mastery change.

**Phase 5 - Push me.** Nudge ladder, terminal channel, cron, weekly report.
*Exit:* it pulls you back after two skipped days.

**Phase 6 - Surface.** FastAPI, web or chat client, more channels.

## 13. Risks

| Risk | Mitigation |
| --- | --- |
| Hallucinated facts in authored lessons | Critic review, cite-or-abstain, flag unverified claims |
| False merge in dedupe | Bias to split; middle band goes to the Critic; show reuse before commit |
| Shared objective edit breaks a domain | Versioned objectives, domains pin versions |
| Assessment feels like an exam | 12-item cap, conversational, value shown immediately |
| Nudge fatigue | One per day, ladder de-escalates, one-word opt-out |
| Cost creep | Shared corpus, lazy fetch, model routing, orchestrator budget gate |
| Over-engineering the learner model | BKT now; upgrade only when data justifies it |

## 14. Metrics

Diagnostic completion rate; objectives mastered per active week; 14- and 30-day
retention; Critic acceptance rate and learner thumbs-down rate; cost per
mastered objective; **cost ratio of learner N to learner 1** (corpus leverage);
objective reuse rate across domains.
