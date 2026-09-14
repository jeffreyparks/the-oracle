# Phase 2 Contract — "Plan me"

Frozen interface. Three workers build in parallel. Report problems, do not redesign.

## File ownership (do not cross)

| Worker | Owns |
| --- | --- |
| dedupe | `src/the_oracle/domains/embed.py`, `domains/dedupe.py`, `agents/critic.py`, `scripts/backfill_tags.py`, `tests/test_dedupe.py` |
| architect | `src/the_oracle/agents/architect.py`, `commands/domain_add.py`, `tests/test_architect.py` |
| planner | `src/the_oracle/planning/**`, `commands/plan.py`, `tests/test_planning.py` |

Nobody edits `cli.py` (the orchestrator wires commands), `store/models.py`,
`mastery/**`, `agents/assessor.py`, or another worker's files.
`tests/cassettes/replay.py` already exists and is shared — use it, do not modify it.

## Embeddings: no provider key is configured

The machine has `SERPER_API_KEY` only. There is no OpenAI/Voyage/Cohere key.
So embeddings must be **pluggable with an honest offline fallback**.

```python
class Embedder(Protocol):
    name: str
    dim: int
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

def get_embedder(settings=None) -> Embedder
```
- `LocalEmbedder` (**preferred**): sentence-transformers, default
  `BAAI/bge-small-en-v1.5` (384-dim), overridable via `ORACLE_EMBED_MODEL`.
  Free, offline, repeatable. Imported lazily; optional extra.
- `OpenAIEmbedder` and `VoyageEmbedder`: used when the matching key is present.
- `LexicalEmbedder`: deterministic last-resort fallback (character n-gram TF-IDF,
  L2-normalised). No network, no key, no cost, weaker signal.
- `get_embedder` order: explicit override -> Local -> API by key -> Lexical, and
  it **logs which one it chose**. Degraded mode must be visible, never silent.
- Embeddings are **cached** keyed `(embedder_name, text_sha256)` so `domain add`
  is cheap on a second run. Cache lives under `ORACLE_HOME/embeddings/`.

## `the_oracle.domains.dedupe`

```python
class Decision(StrEnum):
    REUSE = "reuse"; ADJUDICATE = "adjudicate"; MINT = "mint"

@dataclass(frozen=True)
class Match:
    candidate_title: str
    decision: Decision
    best_id: str | None
    score: float
    runners_up: list[tuple[str, float]]

REUSE_THRESHOLD = 0.92      # strong evidence only
ADJUDICATE_THRESHOLD = 0.78 # middle band -> Critic

def match_one(candidate: Objective, library_objs, embedder=None) -> Match
def match_all(candidates, library_objs=None, embedder=None) -> list[Match]
def resolve(matches, judge=None) -> dict[str, str | None]   # candidate title -> reused id or None
```

**Bias to split.** Thresholds are a function of the embedder, not a global
constant. A dense model (local or API) uses the base 0.92 / 0.78. `LexicalEmbedder`
raises both by 0.05, because lexical similarity is weaker evidence. A
false merge corrupts mastery in both directions; a false split only costs a
duplicate objective.

## `the_oracle.agents.critic`

```python
class SamenessRequest(BaseModel):
    candidate: Objective; existing: Objective
class SamenessVerdict(BaseModel):
    same_skill: bool; confidence: float; reasoning: str
class SamenessJudge(Agent[SamenessRequest, SamenessVerdict]): ...
```
Judges the middle band only. Its instruction must state plainly that the same
words at a different depth, or for a different audience, are NOT the same skill,
and that when unsure the answer is `same_skill=False`.

## `the_oracle.agents.architect`

```python
class Interview(BaseModel):
    goal: str; background: str; hours_per_week: float
    target_weeks: int; depth: Literal["survey","working","practitioner","expert"]
    must_cover: list[str] = []; exclude: list[str] = []

class DraftObjective(BaseModel):     # mirrors domains.schema.Objective, no id yet
    title: str; description: str; bloom: str; difficulty: int
    est_minutes: int; assessment_stems: list[str]; tags: list[str]

class DraftDomain(BaseModel):
    title: str; description: str
    objectives: list[DraftObjective]
    edges: list[tuple[str, str]]     # by TITLE at draft time, ids assigned later
    modules: list[dict]              # {id,title,goal,objectives:[title,...]}
    misconceptions: list[dict]

class Architect(Agent[Interview, DraftDomain]): ...

def build_domain(interview, *, embedder=None, judge=None, engine=None) -> tuple[Domain, list[Match]]
```
`build_domain` drafts, dedupes against the shared library, mints ids only for
genuinely new objectives, writes new objectives to the library, assembles the
manifest, and validates it before returning. It must NOT write a broken pack.

## `the_oracle.planning`

```python
@dataclass(frozen=True)
class PlannedObjective:
    objective_id: str; module_id: str; est_minutes: int
    p_mastery: float; status: Literal["mastered","review","learn"]

@dataclass(frozen=True)
class Syllabus:
    domain_id: str; learner_id: str
    items: list[PlannedObjective]
    total_minutes: int; weeks: float
    skipped_mastered: list[str]

def build_syllabus(domain_id, learner_id, *, hours_per_week=3.0, engine=None) -> Syllabus
```
Rules: respect `domain.teaching_order()`; drop objectives already at
`p >= MASTERY_THRESHOLD` into `skipped_mastered`; mark `0.5 <= p < 0.85` as
`review` with reduced minutes; everything else is `learn`. Pure and deterministic
— **no LLM in the planner**.

## Tag backfill (`scripts/backfill_tags.py`)

Phase 1 debt: the 70 migrated objectives have tags derived from id prefixes.
Regenerate them **deterministically** from title + description (stopworded,
stemmed-ish keyword extraction, max 6 tags). No LLM, no cost, reproducible.
Rewrite `data/packs/objectives/*.yaml` in place and bump nothing — tags are not
semantic content. Print a before/after diff summary.

## Rules for everyone

- Python 3.13, full type hints. No subject matter in `src/` — two tests check this.
- Tests must pass **offline with no API key**. Use a fake/lexical embedder and
  cassettes. A cassette miss raises; never fall back to the network.
- Use `uv run pytest ...`. Never the kernel interpreter.
- Commands live in `commands/<name>.py` exposing a module-level Typer `app`.


---

# Phase 2 outcome

## Embedder decision changed mid-phase

Planning chose an embeddings **API**. That was reversed: the machine has no
provider key, and a local model is free, offline, and repeatable. Dedupe runs on
every `domain add` against a growing library, so it is a repeated cost.

`LocalEmbedder` (`BAAI/bge-small-en-v1.5`, 384-dim) is now preferred.
Installed as an **optional extra**: `uv sync --extra embeddings`.
Cost: torch and friends take the venv to **933 MB**. The engine runs without it
and falls back to lexical, announcing the downgrade.

## Verified on the real 70-objective library

| Candidate | Decision | Score |
| --- | --- | --- |
| Near-identical to an existing objective | `reuse` | 1.000 |
| Same title, survey depth instead of practitioner | `mint` | 0.609 |
| Unrelated subject entirely | `mint` | 0.521 |

The middle row is the one that matters: the false merge that would corrupt
mastery across domains does not happen. `dedupe.py` adds Bloom and difficulty
depth penalties beyond the contract, which is what pushes that case down.

Note the dense-model floor: unrelated text still scores 0.52, so thresholds must
stay absolute, not relative.

## Fixed during integration

- **Tags were never backfilled** by the worker that wrote the script. Run now;
  Phase 1 debt item 3 is cleared. Example: `['ts']` became
  `['scores', 'proper', 'rules', 'score', 'crps', 'interval']`.
- **CLI option parsing bug.** Single-action commands were wired with
  `add_typer`, so click treated any option after the positional argument as a
  subcommand: `domain add "a goal" --yes` died with "No such command '--yes'".
  `cli.py` now registers those callbacks as real commands. The worker's strict
  `xfail` became a passing regression test against the shipped `cli` app.
- A stale `domain add` stub in `cli.py` was shadowing the real command.
- `sentence-transformers` 6.x renamed `get_sentence_embedding_dimension`;
  supported both without pinning.

## Debt carried

1. **Cassettes are still hand-written.** They prove plumbing, not prompt quality.
   No generated domain has been produced by a real model yet.
2. **`domain add` is untested against a live model.** The first real run is the
   true test of whether a generated curriculum is worth studying.
3. **Planner and syllabus are not wired to checkpoints in the `Syllabus` object**
   (the contract froze its fields); call `build_checkpoints` separately.


---

# Live `domain add` — first real model run

Run: survey depth, 1 h/week, 1 week, scratch `~/.the-oracle-live`.
Model: `anthropic:claude-sonnet-4-5`. ~40 s and a few cents per run.

Three bugs found that **every offline test had passed**. This is why hand-written
cassettes prove plumbing, not correctness.

## 1. `'RunUsage' object is not callable`

`agents/base.py` called `run.usage()`. pydantic-ai has shipped `usage` as both a
method and a property. Fixed with `_unwrap_usage` / `_run_usage`, which accept
either shape. A cassette can never catch this: it fabricates the run object.

## 2. Every misconception had an empty `wrong_model`

Root cause: `DraftDomain.modules` and `.misconceptions` were
`list[dict[str, Any]]`. An untyped dict puts **no required keys in the JSON
schema**, so the model simply omitted the field. Objective `tags` came back empty
for the same reason — declared but never described.

Fix: real `DraftModule` and `DraftMisconception` models with `min_length`
constraints and field descriptions. The lesson generalises: **the model fills
what the schema demands, and skips what it does not.** Never accept a free dict
from a structured-output agent.

Before: `wrong_model: ''` for all five.
After: `"A 30% chance of rain means it will rain for 30% of the time period."`
with a matching diagnostic question and objective refs.

## 3. Duplicate misconception ids

Ids were minted from the first six words of the belief. Two beliefs opened
identically ("a 30% chance of rain means...") and collided, silently collapsing
two distinct wrong models into one. Fixed by widening then numbering, and a new
**rule 6a** in `registry.py` rejects duplicate misconception ids at load time.
The validator should have caught this and did not.

## Result

Two domains now coexist in one library: the 70-objective seed pack and a
5-objective generated pack. Quality of the generated content is good — stems are
diagnose-and-explain, not recall, and the misconceptions are ones a real learner
holds.

## Still unproven

**Live reuse.** All five objectives scored 0.60-0.70 against the library and were
minted. That is almost certainly correct — consumer-level forecast reading is not
the same skill as practitioner-level calibration — but it means the live path
exercised `mint` only. Reuse is proven in unit tests (1.000 on a near-duplicate),
not yet against a real model. A second generated domain that deliberately
overlaps the seed pack would close this gap.


---

# Closing the reuse gap — what the live runs actually revealed

Four live runs against an overlapping topic ("evaluate whether a probabilistic
forecast is calibrated and sharp", practitioner depth, background stating
existing Bayesian/MCMC skill). Reuse never happened. Three causes, in order of
discovery.

## Cause 1 (bug, fixed): the depth penalty hid true duplicates from the Critic

`decide()` banded on *cosine minus depth penalty*. That pushed genuine
duplicates below the adjudicate bar, so the Critic never saw them.

Measured: "Design a rolling-origin backtest" scored **0.858 raw** against
`ts_backtesting`, but **0.752** after the penalty, landing in MINT.

Fixed: raw cosine chooses the band; a depth gap **downgrades** REUSE to
ADJUDICATE rather than suppressing the match. Effect on the same 7 candidates:
adjudications went from **0 to 5**.

## Cause 2 (bug, fixed): the Critic was never called

`commands/domain_add.py` called `plan_domain(draft, interview)` with no judge.
`resolve()` splits every middle-band match when `judge is None`, so the Critic
was built, unit-tested, and **completely unwired** — the same failure mode as the
stale `domain add` stub in Phase 0. `judge_fn` already existed as the adapter.

Wired in the command, not defaulted inside `plan_domain`, so library callers and
offline tests never make a live model call unless they ask for one.

## Cause 3 (not a bug): the Architect drafts blind

With both bugs fixed, a candidate scored **0.96** against `ts_backtesting` and
the Critic still refused, with a defensible argument:

> "The Bloom level differs (apply vs evaluate)... The candidate asks learners to
> construct a scheme, while the existing objective asks them to evaluate
> forecasts... A learner who can design a rolling-origin scheme may not yet judge
> reporting practices, and vice versa."

That is the bias-to-split policy working exactly as specified. It is also a
**design gap**: the Architect drafts with no knowledge of the existing library,
so it invents parallel objectives at slightly different Bloom levels, and
post-hoc dedupe is left to reconcile them. It cannot, and should not, merge
across a genuine Bloom gap.

Consequence if left alone: the library bloats with near-parallel objectives and
**mastery stops transferring**, which is the entire premise of the shared
library.

### Proposed fix (not yet built): retrieval-augmented drafting

Before drafting, embed the interview goal and `must_cover` terms, retrieve the
top ~20 nearest library objectives, and put them in the Architect's prompt with
an instruction to reuse an existing objective verbatim by id where it already
covers a needed skill, and to draft only what is genuinely missing.

Prevention beats reconciliation: dedupe stops being the primary mechanism and
becomes the safety net it was designed to be.
