"""Load-time contract tests for the domains subpackage (docs/pack-format.md)."""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from the_oracle.domains import library, registry
from the_oracle.domains.errors import PackValidationError
from the_oracle.domains.schema import Domain

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PACKS = REPO_ROOT / "data" / "packs"
DOMAIN_ID = "bayesian_forecasting"

SRC = REPO_ROOT / "src" / "the_oracle"
SUBJECT_WORDS = ("bayes", "bayesian", "posterior", "prior", "mcmc")


@pytest.fixture
def packs_home(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ORACLE_HOME at the migrated repo data."""
    monkeypatch.setenv("ORACLE_HOME", str(DATA_PACKS))
    return DATA_PACKS


@pytest.fixture
def broken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[[Callable[[dict[str, Any]], None]], list[str]]:
    """Copy the real pack into tmp, apply a mutation, return the problems found."""
    source = yaml.safe_load(
        (DATA_PACKS / "domains" / f"{DOMAIN_ID}.yaml").read_text(encoding="utf-8")
    )
    objectives_dir = tmp_path / "objectives"
    objectives_dir.mkdir()
    for path in (DATA_PACKS / "objectives").glob("*.yaml"):
        (objectives_dir / path.name).write_text(path.read_text(encoding="utf-8"))
    monkeypatch.setenv("ORACLE_HOME", str(tmp_path))

    def run(mutate: Callable[[dict[str, Any]], None]) -> list[str]:
        manifest = copy.deepcopy(source)
        mutate(manifest)
        target = tmp_path / "domains" / f"{DOMAIN_ID}.yaml"
        target.parent.mkdir(exist_ok=True)
        target.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
        return registry.validate_domain(DOMAIN_ID)

    return run


# --- round trip ------------------------------------------------------------


def test_list_domains(packs_home: Path) -> None:
    assert registry.list_domains() == [DOMAIN_ID]


def test_round_trip_load(packs_home: Path) -> None:
    domain = registry.load_domain(DOMAIN_ID)
    assert isinstance(domain, Domain)
    assert registry.validate_domain(DOMAIN_ID) == []
    assert len(domain.objectives) == 70
    assert len(domain.modules) == 10
    assert len(domain.misconceptions) == 15

    first = domain.objectives[0].id
    resolved = domain.objective(first)
    assert resolved.id == first
    assert resolved.version == domain.objectives[0].version
    assert resolved.est_minutes > 0


def test_library_get_put_search(packs_home: Path, tmp_path: Path,
                                monkeypatch: pytest.MonkeyPatch) -> None:
    obj = library.get("prob_sample_space_events", 1)
    assert obj.bloom == "understand"
    assert len(list(library.all_objectives())) == 70
    assert library.search(obj.title[:12], limit=5)

    monkeypatch.setenv("ORACLE_HOME", str(tmp_path))
    library.put(obj)
    assert library.get(obj.id).model_dump() == obj.model_dump()


def test_prerequisites_match_edges(packs_home: Path) -> None:
    domain = registry.load_domain(DOMAIN_ID)
    target = "bayes_theorem"
    expected = [e.from_id for e in domain.edges if e.to_id == target]
    assert domain.prerequisites(target) == expected
    assert expected


def test_teaching_order_respects_every_edge(packs_home: Path) -> None:
    domain = registry.load_domain(DOMAIN_ID)
    order = domain.teaching_order()
    assert len(order) == len(domain.objectives)
    assert set(order) == {ref.id for ref in domain.objectives}
    position = {oid: i for i, oid in enumerate(order)}
    for edge in domain.edges:
        assert position[edge.from_id] < position[edge.to_id], edge


# --- the six rules ---------------------------------------------------------


def test_rule1_unresolved_reference(broken: Any) -> None:
    def mutate(m: dict[str, Any]) -> None:
        m["objectives"].append({"id": "ghost_objective", "version": 1})
        m["modules"][0]["objectives"].append("ghost_objective")

    problems = broken(mutate)
    assert any(p.startswith("rule1:") and "ghost_objective" in p for p in problems)


def test_rule1_version_skew(broken: Any) -> None:
    def mutate(m: dict[str, Any]) -> None:
        m["objectives"][0]["version"] = 99

    problems = broken(mutate)
    assert any("rule1:" in p and "version" in p for p in problems)


def test_rule2_dangling_edge_endpoint(broken: Any) -> None:
    def mutate(m: dict[str, Any]) -> None:
        m["edges"].append({"from": "nowhere", "to": m["objectives"][0]["id"]})

    problems = broken(mutate)
    assert any(p.startswith("rule2:") and "nowhere" in p for p in problems)


def test_rule3_cycle(broken: Any) -> None:
    def mutate(m: dict[str, Any]) -> None:
        first = m["objectives"][0]["id"]
        second = m["objectives"][1]["id"]
        m["edges"].append({"from": first, "to": second})
        m["edges"].append({"from": second, "to": first})

    problems = broken(mutate)
    assert any(p.startswith("rule3:") and "cycle" in p for p in problems)


def test_rule4_objective_in_two_modules(broken: Any) -> None:
    def mutate(m: dict[str, Any]) -> None:
        duplicated = m["modules"][0]["objectives"][0]
        m["modules"][1]["objectives"].append(duplicated)

    problems = broken(mutate)
    assert any(p.startswith("rule4:") for p in problems)


def test_rule4_objective_in_no_module(broken: Any) -> None:
    def mutate(m: dict[str, Any]) -> None:
        m["modules"][0]["objectives"].pop()

    problems = broken(mutate)
    assert any("appears in no module" in p for p in problems)


def test_rule5_module_order_violates_edge(broken: Any) -> None:
    def mutate(m: dict[str, Any]) -> None:
        m["modules"].reverse()

    problems = broken(mutate)
    assert any(p.startswith("rule5:") for p in problems)


def test_rule6_misconception_ref(broken: Any) -> None:
    def mutate(m: dict[str, Any]) -> None:
        m["misconceptions"][0]["objectives"].append("no_such_objective")

    problems = broken(mutate)
    assert any(p.startswith("rule6:") and "no_such_objective" in p for p in problems)


def test_load_never_half_loads_and_lists_all_problems(broken: Any,
                                                      tmp_path: Path) -> None:
    def mutate(m: dict[str, Any]) -> None:
        m["edges"].append({"from": "nowhere", "to": m["objectives"][0]["id"]})
        m["misconceptions"][0]["objectives"].append("no_such_objective")

    problems = broken(mutate)
    assert len(problems) >= 2

    with pytest.raises(PackValidationError) as excinfo:
        registry.load_domain(DOMAIN_ID)
    assert excinfo.value.domain_id == DOMAIN_ID
    assert len(excinfo.value.problems) >= 2


def test_missing_domain_raises(tmp_path: Path,
                               monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORACLE_HOME", str(tmp_path))
    assert registry.list_domains() == []
    with pytest.raises(PackValidationError):
        registry.load_domain("absent")


def test_oracle_home_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ORACLE_HOME", raising=False)
    assert registry.oracle_home() == Path.home() / ".the-oracle"


# --- the abstraction must not leak ----------------------------------------


def _source_files() -> Iterator[Path]:
    yield from sorted(SRC.rglob("*.py"))


def test_no_subject_matter_in_src() -> None:
    pattern = re.compile(r"\b(" + "|".join(SUBJECT_WORDS) + r")\b", re.IGNORECASE)
    leaks: list[str] = []
    for path in _source_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                rel = path.relative_to(REPO_ROOT)
                leaks.append(f"{rel}:{lineno}: {line.strip()}")
    assert not leaks, "subject matter leaked into src/the_oracle:\n" + "\n".join(leaks)
