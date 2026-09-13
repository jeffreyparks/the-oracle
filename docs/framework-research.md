# Multi-Agent Python Framework Research — The Oracle

**Researched: 2026-09-13.**

> **Important correction to the brief.** The task asked for the "late 2025" landscape.
> The real date is **2026-09-13**. All data below is the *current* state, taken live from
> the PyPI JSON API, the GitHub API, and the vendors' own documentation sites on 2026-09-13.
> Several 2025-era assumptions are now wrong — most importantly **Microsoft AutoGen is in
> maintenance mode** and **Pydantic AI is at 2.x with five durable-execution backends**.
>
> The `websearch` skill was unavailable (no Serper API key configured), so recency comes from
> primary sources (package registries, repo metadata, official docs) rather than blog posts.
> That is arguably stronger evidence, but it means less "community sentiment" input.

## Project requirements being matched

From `PLAN.md`:

1. ~8 specialist agents (Orchestrator, Assessor, Architect, Scout, Critic, Author, Coach, Evaluator).
2. Typed Pydantic contracts in and out of every agent.
3. **Orchestrator owns all state commits.** No agent writes to the store.
4. SQLite persistence (SQLModel), append-only evidence, derived state.
5. Long-running scheduled background work (nudges, spaced review, weekly report).
6. Python 3.13, `uv`, Typer CLI first, FastAPI later.
7. Model routing across cheap/mid/expensive models per agent.

---

## 1. Recency data (hard numbers, 2026-09-13)

| Package | Latest version | Released | Releases in 2025 | GitHub stars | Last repo push |
| --- | --- | --- | --- | --- | --- |
| `pydantic-ai` | **2.43.0** | 2026-09-12 | 172 | 19.9k | 2026-09-12 |
| `pydantic-ai-harness` | 0.31.0 | 2026-09-12 | 0 (new in 2026) | — | — |
| `langgraph` | **1.2.11** | 2026-08-11 | 94 | 41.6k | 2026-09-13 |
| `openai-agents` | **0.22.2** | 2026-09-09 | 46 | 29.4k | 2026-09-12 |
| `crewai` | **1.15.21** | 2026-09-09 | 60 | 58.4k | 2026-09-11 |
| `autogen-agentchat` (MS) | 0.7.5 | **2025-09-30** | 34 | 61.0k | **2026-04-15** |
| `agent-framework` (MS, successor) | **1.18.0** | 2026-09-10 | 18 | — | — |
| `ag2` (community AutoGen fork) | **1.0.5** | 2026-09-11 | 49 | 4.9k | 2026-09-13 |
| `google-adk` | **2.9.0** | 2026-09-10 | 38 | 21.5k | 2026-09-13 |
| `atomic-agents` | **2.10.2** | 2026-08-24 | 35 | 6.2k | 2026-08-24 |
| `llama-index-workflows` | **2.23.3** | 2026-08-22 | 38 | 0.4k (own repo) | 2026-09-10 |
| `dspy` | **3.3.1** | 2026-08-21 | 44 | 38.0k | 2026-09-11 |
| `@mastra/core` | **1.66.0** | 2026-09-11 | — | 28.0k | 2026-09-13 |
| `agno` | **3.0.9** | 2026-09-08 | 166 | 42.2k | 2026-09-13 |
| `smolagents` | 1.26.0 | **2026-05-29** | 34 | 29.3k | 2026-08-25 |

**Headline changes since late 2025:**

- **Microsoft AutoGen is officially in maintenance mode.** Its README carries a `CAUTION`
  banner: "AutoGen is now in maintenance mode. It will not receive new features or
  enhancements and is community managed going forward." Last Python release was
  2025-09-30; last repo push 2026-04-15. The successor is **Microsoft Agent Framework
  (MAF)**, now GA at `agent-framework` 1.18.0.
- **Pydantic AI went 1.0 then 2.x** and grew a whole product surface: durable execution,
  a Harness SDK (`pydantic-ai-harness`), `SubAgents`, `StepPersistence`, guardrails,
  spend limits, and a realtime/voice stack. 172 releases in 2025 alone — extremely fast.
- **LangGraph is on 1.x** and is now the stable core under the LangChain 1.x rewrite.
- **CrewAI reached 1.x** (was 0.x through 2025).
- **Google ADK reached 2.x** and is multi-language (Python, TS, Go, Java, Kotlin).

---

## 2. Comparison table

Legend for **Lock-in**: L = low, M = medium, H = high.

| Framework | Maturity / last release | Typed output (Pydantic) | Multi-agent orchestration model | State persistence / durability | Human-in-the-loop | Observability | Lock-in | 1-line verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Pydantic AI** 2.43.0 | Very mature, 2.x, released 2026-09-12, MIT, Pydantic team | **Best in class.** `output_type=Model` is the native path; validation retries built in; deps are typed too | Plain Python by default; `SubAgents` capability (`delegate_task`), agent-as-tool delegation, programmatic hand-off, `pydantic-graph` for real state machines | `StepPersistence` (append-only step events, continuable snapshots, tool-effect ledger) + **5 durable-execution backends**: Temporal, DBOS, Prefect, Restate, AWS Lambda (plus Kitaru, Airflow) | Deferred tools / approval-required tools, durable pause-resume, guardrails | **Native OpenTelemetry**; first-class Logfire, but any OTel backend works | **L** — agents are stateless objects you call; no runtime owns your loop | Typed-contract orchestration with the least framework tax; the default answer for this project |
| **LangGraph** 1.2.11 | Very mature, 1.x, 41.6k stars, huge ecosystem | Good — structured output via the model layer and typed state schemas; state is a `TypedDict`/Pydantic reducer graph, not per-agent contracts | Explicit **graph of nodes and edges**; supervisor / swarm / hierarchical prebuilts; subgraphs | **Strongest OSS story.** Checkpointers (`langgraph-checkpoint-sqlite` 3.1.1 for SQLite), threads, stores, pending writes, time travel, durability modes | **Best in class.** `interrupt()` + resume is the canonical pattern, requires a checkpointer | OTel via LangSmith; LangSmith is the paved road | **M/H** — the graph *is* your app; your state schema and control flow become LangGraph objects. Migrating out is a rewrite | Unbeatable durability and HITL, but it wants to own the state that your orchestrator is supposed to own |
| **OpenAI Agents SDK** 0.22.2 | Mature-ish but still **0.x** after 18 months; 29.4k stars, very active | Good — `output_type=PydanticModel` on agents | `handoffs` (specialist takes over the turn) and `Agent.as_tool()` (manager keeps control). Loop is `Runner.run()` | `SQLiteSession` out of the box (also SQLAlchemy, encrypted sessions). **Conversation memory only — not workflow checkpointing.** No durable execution, no resume-after-crash | Tool approval hooks and guardrails; no first-class durable pause | Built-in tracing (OpenAI dashboard) + OTel exporters | **M** — provider-agnostic now, but tracing and Responses-API ergonomics pull toward OpenAI | Cleanest small API, but session memory is not the durable workflow state this project needs |
| **CrewAI** 1.15.21 | Mature by version, 58.4k stars, very active, commercial company | Decent — Pydantic `output_pydantic` on tasks; not as tight as Pydantic AI | Two layers: **Crews** (role-playing agents, autonomous collaboration) and **Flows** (event-driven, `@start`/`@listen`/`@router`, explicit control) | `@persist` decorator with **`SQLiteFlowPersistence` by default** — class- or method-level | Present but thinner; human input on tasks | CrewAI AMP / enterprise dashboard; OTel via integrations | **H** — role/task/crew abstractions are opinionated and pervasive; strong pull to the paid platform | Great demos and real SQLite flow persistence, but the role-playing abstraction fights an orchestrator that owns all commits |
| **AutoGen (Microsoft)** 0.7.5 | **Maintenance mode.** Last release 2025-09-30 | Yes (structured output in AgentChat) | GroupChat / team conversation patterns | Team state save/load; no durable engine | `UserProxyAgent` | OTel instrumentation | — | **Do not start here in 2026.** Officially superseded |
| **Microsoft Agent Framework (MAF)** 1.18.0 | New but GA, stable APIs, LTS commitment; successor to AutoGen **and** Semantic Kernel | Yes | Agents + **functional and graph-based workflows**; explicit execution paths | Workflow checkpointing; thread/context providers | Tool approval ("don't-ask-again"), planning, todo tracking | Built-in observability, OTel | **M/H** — Azure/Foundry gravity | The real AutoGen successor; credible, but Azure-shaped and young for a solo SQLite CLI |
| **AG2** 1.0.5 | Active community fork of AutoGen 0.2 lineage, Apache-2.0, only 4.9k stars | Yes | Conversable agents, group chat patterns | Limited; no durable engine | `UserProxyAgent`, human input modes | Basic | **M** | Alive and 1.0, but a small-community fork; no reason to pick it over MAF or the leaders |
| **Google ADK** 2.9.0 | Mature, 2.x, multi-language, 21.5k stars, Google-backed | **Caveat:** `output_schema` on an LLM agent disables tool use and transfers in ADK's design — awkward for typed agents that also need tools | Hierarchical agent tree with `sub_agents`; `SequentialAgent` / `ParallelAgent` / `LoopAgent` workflow agents; LLM-driven transfer | **`DatabaseSessionService`** with SQLAlchemy — `sqlite+aiosqlite:///./my_agent_data.db` works; sessions + events + state prefixes (`user:`, `app:`, `temp:`). Session state, not workflow checkpoints | Callbacks, plugins; long-running tool pattern | OTel; Cloud Trace / Agent Engine | **M/H** — strong Vertex AI / Agent Engine gravity | Solid, genuinely persistent sessions — but the typed-output-vs-tools restriction is a direct hit on this project's core rule |
| **Atomic Agents** 2.10.2 | Small but steady; 6.2k stars, last release 2026-08-24; requires **Python ≥3.12** | **Excellent by design** — every agent is `InputSchema -> OutputSchema`, built on Instructor | Deliberately none. You chain agents in plain Python; schema alignment is the contract | **None.** You own persistence entirely | You own it | You own it (Instructor + your own logging) | **L** — smallest surface of any framework here | Philosophically the exact match for this project's typed-contract rule, but it gives you nothing you could not write in a day |
| **LlamaIndex Workflows** 2.23.3 | Mature, split into its own package/repo (now "LlamaAgents"); document-centric positioning | Pydantic events; typed step signatures | **Event-driven steps**: async functions emit and consume events; branch, loop, parallelize. Agents are workflow steps | Context serialization, state persistence, failure recovery; durable orchestration is the pitch | `InputRequiredEvent` / `HumanResponseEvent` — clean HITL primitive | OTel + LlamaTrace/Arize | **M** — you write to the event/step model, but it is thin | Excellent event-driven engine; best fit if you were document-pipeline shaped, which this project is only partly |
| **DSPy** 3.3.1 | Very mature, 38.0k stars, active | **Signatures are typed I/O** — typed by construction | Modules compose; `ReAct` agents; **not** a multi-agent orchestrator | None — DSPy has no state store | None built in | MLflow autologging, callbacks | **M** — your prompts become DSPy programs | A prompt *optimizer*, not an orchestrator. Use it *inside* an agent (e.g. the Assessor's item generator), not as the framework |
| **Mastra** `@mastra/core` 1.66.0 | Very active, 28.0k stars | Zod schemas | Agents, workflows (suspend/resume), networks | Durable workflows, storage adapters incl. SQLite/LibSQL | Native `suspend()`/`resume()` | Built-in tracing UI | — | **TypeScript-only. Disqualified.** No Python SDK; this project is Python 3.13 + uv. Listed only to close the question |
| **Agno** 3.0.9 | Extremely active (166 releases in 2025), 42.2k stars | Pydantic `response_model` | Teams (route/coordinate/collaborate) + workflows | SQLite/Postgres storage for sessions and memory | Present (user-confirmation tools) | AgentOS UI, OTel | **M/H** — AgentOS platform pull | Fast and batteries-included; the batteries are the problem when your orchestrator must own commits |
| **Thin custom loop** (no framework) | n/a — you own it | Perfect: you call `instructor`/native structured output with your own models | Whatever you write; a `match` on an enum is a legitimate router | Perfect fit: SQLModel + SQLite, append-only evidence, derived state, exactly as `PLAN.md` specifies | You write it (and it is ~20 lines for a CLI) | You must wire OTel yourself | **None** | Maximum control, but you will re-implement retries, streaming, tool loops, usage limits, and OTel spans — and maintain them |

---

## 3. Additional notable frameworks found

- **Microsoft Agent Framework (MAF)** — the single most important 2026 change. GA, stable
  API, LTS. Absorbs AutoGen *and* Semantic Kernel.
- **Agno 3.x** — one of the fastest-moving Python agent frameworks (166 releases in 2025).
  Worth knowing, not worth adopting here.
- **Apache Burr 0.43.0** — `burr` moved to `apache-burr` under the ASF. A state-machine
  library for agents with pluggable persistence and a good local debug UI. Genuinely
  aligned with "orchestrator owns state", but a small community.
- **smolagents 1.26.0** — code-writing agents. Last release 2026-05-29, slowing. Wrong shape here.
- **Durable-execution engines as a separate layer**: `temporalio` 1.32.0, `dbos` 2.31.1
  (244 releases in 2025), `prefect` 3.8.5, Restate. Relevant because Pydantic AI now
  integrates all of them — you can add durability later without changing framework.

---

## 4. Ranked recommendation for The Oracle

### 🥇 1. Pydantic AI 2.x (+ `pydantic-ai-harness` optionally) — **recommended**

**Why it wins for this project specifically:**

- The project's central rule is *"each agent has one job and a typed contract (Pydantic in,
  Pydantic out)"*. Pydantic AI's `output_type` + typed `deps_type` is literally that model,
  from the team that maintains Pydantic itself. No adapter layer.
- The second rule is *"no agent writes to the store directly; the orchestrator commits
  state"*. Pydantic AI agents are **stateless, global objects you call**. They have no
  ambient state store and no runtime that owns your loop. Your orchestrator stays a plain
  Python function that calls agents and commits to SQLModel. Nothing fights you.
- 8 specialists map cleanly: each is an `Agent` with its own output model and model
  selector — which also makes your `compute-routing` per-agent model policy trivial
  (`Agent('anthropic:claude-sonnet-5')` vs Opus vs a cheap open model).
- Long-running scheduled work: start with APScheduler + a queue table as `PLAN.md` says.
  If reliability becomes a problem, Pydantic AI has **five durable-execution backends**
  (DBOS is the lightest — it is literally "ultra-lightweight durable execution in Python"
  and can sit on your existing SQLite/Postgres). You do not have to decide now.
- Observability is native OpenTelemetry, so `pytest` golden transcripts plus Logfire (or
  any OTel collector) give you per-agent traces for free.
- HITL ("skip this", "too easy") maps to deferred/approval-required tools.

**The tradeoff you accept:** Pydantic AI moves *fast* — 172 releases in 2025 and a major
version bump. You will do occasional small migrations, and `pydantic-ai-harness` is still
0.x with API churn between minors (their own docs say so). Mitigation: pin exactly, use
`pydantic-ai-slim`, and treat the Harness as optional sugar rather than a dependency.

### 🥈 2. Thin custom loop over `instructor` / native structured output

**Why it is a serious second, not a straw man:** `PLAN.md` already leans this way, and the
reasoning is sound — 8 agents, all contracts typed, one orchestrator, SQLite. That is a
few hundred lines. You get exactly the append-only-evidence / derived-state discipline you
designed, with zero framework opinions and zero upgrade treadmill.

**The tradeoff you accept:** you will re-implement and then *maintain* model-agnostic
calling, validation-retry loops, tool-call loops, streaming, usage and cost limits, and
OTel span conventions. That is the boring 20% of an agent framework, and it is exactly
what Pydantic AI gives you for free at the same level of control. Choose this only if you
want the learning, not the leverage.

### 🥉 3. LangGraph 1.x

**Why it is on the podium:** it has the strongest persistence and human-in-the-loop story
in open source, full stop. SQLite checkpointer, threads, stores, pending writes, time
travel, and `interrupt()`/resume. For an app with a long-lived learner session that pauses
for days and resumes, that is a real match. 41.6k stars and the largest ecosystem.

**The tradeoff you accept:** LangGraph wants to be the owner of your state. Your learner
model — the thing `PLAN.md` calls *"the core asset... everything else is replaceable; this
is not"* — would live in graph state and checkpoint tables alongside your SQLModel tables,
or you fight the framework to keep it out. That is duplicated state and a real migration
cost later.

---

### The single biggest reason to reject each of the other two

- **Reject the thin custom loop because:** it makes you the maintainer of the boring,
  high-churn 20% of an agent framework — retries, tool loops, streaming, usage limits,
  OTel — with no compensating control, since Pydantic AI already leaves the orchestrator
  and the data store entirely in your hands.

- **Reject LangGraph because:** it owns the state, and this project's whole design says the
  orchestrator owns the state. You would run two sources of truth (checkpointer + SQLModel)
  over the one asset you said must never be lost.

---

### Explicitly rejected, with the deciding reason

| Framework | Deciding reason to reject |
| --- | --- |
| Microsoft AutoGen | In maintenance mode since 2025; no new features. Dead end. |
| AG2 | Small-community fork; nothing it does better than the leaders. |
| Microsoft Agent Framework | Credible but Azure-shaped and young; overkill for a single-user SQLite CLI. |
| CrewAI | Role-playing crew abstraction conflicts with a single orchestrator owning all commits. |
| Google ADK | Setting `output_schema` disables tool use and agent transfer — a direct conflict with "typed contracts *and* tools". |
| Atomic Agents | Right philosophy, but it is a thin custom loop with a dependency; if you want thin, write thin. |
| LlamaIndex Workflows | Excellent event engine, but document-pipeline positioned; you would use ~30% of it. |
| DSPy | Not an orchestrator. **Do adopt it later inside the Assessor/Author** for prompt optimization against your evidence log. |
| Mastra | **TypeScript-only.** No Python SDK. Disqualified by stack. |
| Agno | Batteries-included storage and AgentOS pull state away from your orchestrator. |

---

## 5. Concrete next step

```toml
# pyproject.toml
dependencies = [
  "pydantic-ai-slim[anthropic,openai,logfire]==2.43.0",
  "sqlmodel>=0.0.42",
  "typer", "rich", "httpx", "trafilatura",
  "apscheduler>=3.11.3",
]
```

Build the orchestrator as a plain async function. Each of the 8 specialists is one
`Agent` with an explicit `output_type` and its own model selector from `compute-routing`.
Commit every transition through one `commit()` on the SQLModel layer. Add DBOS later
*only if* the scheduler proves unreliable.

**Second opinion worth having:** if you expect learner sessions to pause for days and
resume mid-workflow (not just mid-conversation), re-run the LangGraph-vs-Pydantic-AI
decision with that as the primary axis. It is the one requirement that could flip the ranking.

---

*Sources: PyPI JSON API and GitHub REST API queried live 2026-09-13; official docs at
ai.pydantic.dev / pydantic.dev/docs/ai, docs.langchain.com, openai.github.io/openai-agents-python,
google.github.io/adk-docs, docs.crewai.com, developers.llamaindex.ai,
learn.microsoft.com/agent-framework, github.com/microsoft/autogen README.*
