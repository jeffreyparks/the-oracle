# Phase 3 Contract — "Feed me"

Frozen interface. Four workers build in parallel. Report problems, do not redesign.

## File ownership (do not cross)

| Worker | Owns |
| --- | --- |
| scout | `agents/scout.py`, `corpus/fetch.py`, `tests/test_scout.py` |
| reviewer | `agents/reviewer.py`, `tests/test_reviewer.py` |
| author | `agents/author.py`, `tests/test_author.py` |
| corpus | `corpus/store.py`, `corpus/pipeline.py`, `commands/resources.py`, `tests/test_corpus.py` |

Nobody edits `cli.py`, `store/models.py`, `agents/critic.py`, `agents/architect.py`,
`agents/retrieval.py`, `domains/**`, `mastery/**`, `planning/**`, or `tests/cassettes/replay.py`.
The orchestrator wires commands into `cli.py`.

**`agents/critic.py` is off limits.** It holds the dedupe `SamenessJudge` and was
fixed against live evidence. Resource review is a NEW module, `agents/reviewer.py`.

## Existing tables you must use (do not add columns)

`Resource(id, objective_id, kind: found|authored, url, title, body, meta: dict, created_at)`
`ResourceReview(id, resource_id, verdict: accept|reject, score, rubric: dict, notes, created_at)`
`Item(id, objective_id, bloom, difficulty, stem, answer_key: dict, misconception_ids: list, created_at)`

`Resource.meta` is the escape hatch for anything else (author, published date,
licence, word count, source rank). Use it instead of asking for a migration.

## `the_oracle.agents.scout`

```python
class Candidate(BaseModel):
    url: str; title: str; snippet: str = ""; source: str = ""; rank: int = 0

class ScoutRequest(BaseModel):
    objective_id: str; title: str; description: str
    bloom: str; difficulty: int; tags: list[str] = []

def search_candidates(req: ScoutRequest, *, limit: int = 10) -> list[Candidate]
```
- Uses Serper via `SERPER_API_KEY`. Build queries from title + tags + a level
  hint derived from bloom/difficulty. Two or three query variants, deduplicated
  by normalised URL, capped at `limit` (default **10**, per the planning decision).
- **No key -> raise `ScoutUnavailableError`.** Never silently return nothing.

`corpus/fetch.py`:
```python
@dataclass(frozen=True)
class Fetched:
    url: str; final_url: str; status: int; title: str; text: str
    word_count: int; error: str | None = None

def fetch(url: str, *, timeout: float = 15.0) -> Fetched
def fetch_many(urls: Sequence[str], *, max_workers: int = 6) -> list[Fetched]
```
httpx + trafilatura. Never raises on a bad URL: return `Fetched` with `error` set.
Respect a 2 MB cap and a text/html check. No JS rendering.

## `the_oracle.agents.reviewer`

```python
class ReviewRequest(BaseModel):
    objective_title: str; objective_description: str
    bloom: str; difficulty: int
    url: str; title: str; text: str          # text truncated by the caller

class Rubric(BaseModel):
    authority: int; accuracy: int; level_fit: int; depth: int
    recency: int; accessibility: int; effort_to_value: int   # each 0-5

class Review(BaseModel):
    verdict: Literal["accept","reject"]
    score: float                              # 0-5 weighted
    rubric: Rubric
    notes: str
    hard_reject_reason: str | None = None

class Reviewer(Agent[ReviewRequest, Review]): ...

HARD_REJECTS = ("dead_link", "paywall_only", "level_mismatch", "not_about_objective")
WEIGHTS: dict[str, float]        # must sum to 1.0
ACCEPT_SCORE = 3.2
```
Deterministic guards run BEFORE the model: empty text, fetch error, or under 200
words is a hard reject with no model call. Cheap rejects must not cost tokens.

## `the_oracle.agents.author`

```python
class LessonRequest(BaseModel):
    objective_id: str; title: str; description: str
    bloom: str; difficulty: int; est_minutes: int
    assessment_stems: list[str] = []
    misconceptions: list[dict] = []          # {id, wrong_model, diagnostic}
    accepted_sources: list[dict] = []        # {url, title, note} - may be empty

class Exercise(BaseModel):
    prompt: str; answer: str
    wrong_answer_feedback: dict[str, str] = {}   # wrong answer -> what it reveals
    bloom: str

class Lesson(BaseModel):
    explainer: str                 # markdown
    worked_examples: list[str]     # >= 2
    exercises: list[Exercise]      # 3-6
    code: str | None = None        # runnable snippet when computational
    review_summary: str            # <= 80 words, for spaced review
    citations: list[str] = []      # urls actually used

class Author(Agent[LessonRequest, Lesson]): ...
```
Per PLAN.md section 7: full lessons, not explainers. The instruction must carry
the cite-or-abstain rule — when no accepted source supports a specific claim, say
so plainly rather than inventing a reference. Voice comes from `style.py`.

## `the_oracle.corpus`

```python
# store.py
def put_resource(objective_id, *, kind, url=None, title=None, body=None, meta=None, engine=None) -> str
def record_review(resource_id, review, engine=None) -> None
def accepted_for(objective_id, engine=None) -> list[Resource]
def has_enough(objective_id, *, minimum: int = 2, engine=None) -> bool
def put_lesson(objective_id, lesson, engine=None) -> str       # kind=authored
def put_items(objective_id, exercises, engine=None) -> list[str]

# pipeline.py
@dataclass(frozen=True)
class ObjectiveResult:
    objective_id: str; searched: int; fetched: int
    accepted: int; rejected: int; authored: bool; skipped: bool

def ensure_objective(objective_id, domain, *, minimum=2, engine=None, ...) -> ObjectiveResult
def ensure_module(domain_id, module_id, *, engine=None, ...) -> list[ObjectiveResult]
```
Pipeline order, per PLAN.md section 7:
`already has enough -> skip` (this is what makes it LAZY) ->
`Scout 10 -> fetch -> Reviewer -> accepted >= minimum ? attach : Author writes a
lesson -> Reviewer checks the lesson -> attach`.

Every step appends `RESOURCE_ATTACHED` to the event log. Idempotent: running
`ensure_module` twice must not duplicate resources or re-pay for the same work.

`commands/resources.py`: Typer `app` exposing
`resources <domain_id> [--module M] [--dry-run] [--minimum N]`, showing per
objective what was searched, accepted, rejected, or authored, and the token cost.
`--dry-run` reports what WOULD be fetched without a network or model call.

## Rules for everyone

- Python 3.13, full type hints. No subject matter in `src/`.
- Tests pass **offline, no key, no network**: cassettes for model calls, fixtures
  for HTTP. A cassette miss raises. Never fall back to the network in a test.
- Do NOT make live model or Serper calls. The orchestrator runs live verification.
- `uv run pytest -q` must stay green (214 today).
- Commands live in `commands/<name>.py` exposing a module-level Typer `app`.


---

# Phase 3 outcome — verified live

Pipeline exercised end to end against the real web and a real model.

## Found path (Scout -> fetch -> Reviewer)

One objective, 3 candidates: **3 searched, 3 fetched, 2 accepted, 1 rejected**,
no authoring needed. Serper, robots.txt, fetch, and review all fired in order.

The Reviewer's judgement is good, and its notes are written for the learner:

> **REJECT** Wikipedia "Monte Carlo method" - `level_mismatch`
> "...written as an encyclopedia entry for someone looking up background, not as
> a tutorial for someone learning to apply these techniques."

Both accepted resources were university course notes with runnable code, scored
5.00, for an `apply`-level objective. That is the right call.

## Laziness, proven

Second run on the same objective: **0 HTTP requests, 0 model calls,
`skipped=True`, 0.58 s** against 33 s cold. The corpus is only paid for once.

## Authored path (web returns nothing)

Forced with empty seams. Produced a **1248-word lesson and 5 stored items**, with
`evaluate`-level exercises and wrong-answer feedback that names the
misconception:

> wrong answer "The posterior is underdispersed" -> "Wrong. U-shaped histograms
> mean overdispersion (too wide)..."

**Cite-or-abstain worked with zero sources.** The lesson opens with:

> "Nothing here is cited; the specifics are unverified."

That is the behaviour we specified: abstain plainly rather than invent a
reference.

## Known gaps

1. **The Author skipped code on a computational objective** and logged it:
   `computational objective came back without code`. Honest, but the prompt
   should insist harder. Fix before Phase 4 relies on runnable examples.
2. **Accuracy review is same-family.** The Reviewer cannot detect a confident,
   well-written falsehood better than the model behind it can. An accuracy charge
   now requires verbatim quotes that are checked against the source, and an
   unevidenced charge still rejects but is flagged. Cross-family escalation needs
   a second provider key.
3. **Cost is real at full scale.** A dry run of one 8-objective module reports 80
   searches. Lazy fetching and the shared corpus are what make this affordable;
   do not remove them.
