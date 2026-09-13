"""Build the assessor cassettes.

Run it through the project environment:

    uv run python -m tests.cassettes.record

With ``ORACLE_CASSETTE_RECORD=1`` and a real API key it calls the live model
once per case. Without one it falls back to the hand-written responder in
:mod:`tests.cassettes.offline`, which is how the committed cassettes were made.

The cases here are the single source of truth for the unit tests, so a prompt
change is re-recorded and re-asserted in one step.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]
UNIT_CASSETTE = "assessor_units"

#: The objective the unit cases are built from. Any objective would do.
OBJECTIVE_ID = "bayes_theorem"


def packs_home() -> Path:
    """The repo's domain packs. Set as ORACLE_HOME before building cases."""
    return REPO_ROOT / "data" / "packs"


def unit_cases() -> list[tuple[str, BaseModel]]:
    """Every agent call the unit tests replay, in order."""
    from the_oracle.agents.assessor import GradeRequest, Item, ItemRequest
    from the_oracle.domains.registry import load_domain

    os.environ.setdefault("ORACLE_HOME", str(packs_home()))
    domain = load_domain("bayesian_forecasting")
    objective = domain.objective(OBJECTIVE_ID)
    misconceptions: list[dict[str, Any]] = [
        {"id": m.id, "wrong_model": m.wrong_model, "diagnostic": m.diagnostic}
        for m in domain.misconceptions
        if OBJECTIVE_ID in m.objectives
    ]

    request = ItemRequest(
        objective_id=OBJECTIVE_ID,
        bloom=objective.bloom,
        difficulty=objective.difficulty,
        stems=list(objective.assessment_stems),
    )
    item = Item(
        id="item-fixed-0001",
        objective_id=OBJECTIVE_ID,
        stem=objective.assessment_stems[0],
        kind="short",
        difficulty=objective.difficulty,
        bloom=objective.bloom,
        expected=f"A correct, complete response to: {objective.assessment_stems[0]}",
    )
    right = GradeRequest(item=item, answer=item.expected, misconceptions=misconceptions)
    wrong = GradeRequest(
        item=item, answer=misconceptions[0]["wrong_model"], misconceptions=misconceptions
    )
    empty = GradeRequest(item=item, answer="no idea", misconceptions=misconceptions)
    return [
        ("item_writer", request),
        ("grader", right),
        ("grader", wrong),
        ("grader", empty),
    ]


def main() -> None:
    from the_oracle.agents.assessor import Grader, ItemWriter
    from the_oracle.store.db import build_engine, create_all

    from tests.cassettes.offline import synthesize
    from tests.cassettes.replay import Cassette, CASSETTE_DIR

    os.environ.setdefault("ORACLE_HOME", str(packs_home()))
    engine = build_engine("sqlite://")
    create_all(engine)
    agents = {"item_writer": ItemWriter(engine), "grader": Grader(engine)}

    tape = Cassette(CASSETTE_DIR / f"{UNIT_CASSETTE}.json", synthesize=synthesize)
    for name, payload in unit_cases():
        agent = agents[name]
        prompt = agent.prompt(payload)  # type: ignore[arg-type]
        import asyncio

        asyncio.run(tape.play(agent, prompt))
    tape.save()
    print(f"{len(tape.entries)} entries in {tape.path}")


if __name__ == "__main__":  # pragma: no cover
    main()
