# Domain Pack Format & Module Contract

Frozen interface. Both the `domains/` subpackage and the `store/` + engine work
build against this. Do not change it without updating this file first.

## Filesystem

```
~/.the-oracle/
  objectives/            shared objective library, one YAML per objective
    <objective_id>.yaml
  domains/               domain manifests, one YAML per domain
    <domain_id>.yaml
  oracle.db              SQLite
```
Root is overridable by `ORACLE_HOME`. Default `~/.the-oracle`.

## objectives/<id>.yaml  (subject-neutral, shared, versioned)

```yaml
id: bayes_theorem_discrete          # stable snake_case, globally unique
version: 1                          # bump on any semantic edit
title: Bayes' Theorem for Discrete Events
description: One sentence, domain-neutral.
bloom: understand                   # remember|understand|apply|analyze|evaluate|create
difficulty: 2                       # 1-5 cognitive load
est_minutes: 40
assessment_stems:                   # 2-3, reveal mastery not recall
  - "..."
tags: [probability, bayes]          # free-form, aids dedupe recall
```

## domains/<id>.yaml  (a view over the library)

```yaml
id: bayesian_forecasting
version: 1
title: Bayesian Modeling and Forecasting
description: ...
objectives:                         # pinned references into the library
  - id: bayes_theorem_discrete
    version: 1
edges:                              # prerequisites belong to the DOMAIN
  - from: prob_sample_space_events  # prerequisite
    to: bayes_theorem_discrete      # dependent
modules:
  - id: m01_foundations
    title: Probability Foundations
    goal: ...
    objectives: [prob_sample_space_events, ...]   # ordered
misconceptions:
  - id: mc_base_rate_neglect
    wrong_model: Plain statement of the wrong mental model.
    objectives: [bayes_theorem_discrete]
    diagnostic: A question that forces it into the open.
```

Rules enforced on load (fail loud, never half-load):
1. Every `objectives[].id` resolves in the library at the pinned `version`.
2. Every edge endpoint appears in `objectives`.
3. The edge graph is acyclic.
4. Every objective appears in exactly one module.
5. Module order never violates an edge.
6. Every misconception objective ref resolves.

## Python contract

`the_oracle.domains.schema` exports Pydantic models:
`Objective`, `ObjectiveRef`, `Edge`, `Module`, `Misconception`, `Domain`.

`the_oracle.domains.registry`:
```python
def oracle_home() -> Path: ...
def list_domains() -> list[str]: ...
def load_domain(domain_id: str) -> Domain: ...        # raises PackValidationError
def validate_domain(domain_id: str) -> list[str]: ... # [] means valid
```

`the_oracle.domains.library`:
```python
def get(objective_id: str, version: int | None = None) -> Objective: ...
def put(obj: Objective) -> None: ...
def all_objectives() -> Iterator[Objective]: ...
def search(text: str, limit: int = 20) -> list[Objective]: ...
```

`Domain` exposes resolved helpers. `load_domain` calls `bind()` for you; call it
yourself only when constructing a `Domain` by hand:
```python
domain.bind(resolved: dict[str, Objective]) -> Domain   # attach library bodies
domain.objective(objective_id) -> Objective       # library-resolved, version-pinned
domain.prerequisites(objective_id) -> list[str]
domain.teaching_order() -> list[str]              # topologically valid
```

`PackValidationError` lives in `the_oracle.domains.errors`.

## Non-negotiables

- No subject matter in `src/the_oracle/`. A test greps for it.
- `MasteryState` is keyed `(learner_id, objective_id)`. No `domain_id`. Ever.
- Objectives are domain-neutral; prerequisites live in the domain manifest.
