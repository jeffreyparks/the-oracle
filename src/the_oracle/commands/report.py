"""``the-oracle report`` — the weekly report.

One model call a week, and only when there is something to write about. A week
with no activity is rendered from the facts alone: no key, no network, no
chance of a model inventing a week that did not happen.

The command never crashes on an empty log. If the model is unreachable, the
report still prints, computed.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Annotated

import typer
from rich.console import Console

from the_oracle.agents.coach import (
    Coach,
    WeekFacts,
    WeeklyReport,
    deterministic_report,
    gather_week,
    render,
)

console = Console()

app = typer.Typer(
    name="report",
    help="Print the weekly report.",
    invoke_without_command=True,
)
"""Wired into ``cli.py`` as a command::

    app.command("report", help="Print the weekly report.")(_report_cmd.report)
"""


def _write(facts: WeekFacts) -> tuple[WeeklyReport, str | None]:
    """The prose for ``facts``, plus a note when the model was not used.

    A silent downgrade would be a lie about where the words came from, so the
    reason is returned and printed.
    """
    if facts.is_empty_week:
        return deterministic_report(facts), None
    try:
        from the_oracle.context import LearnerContext

        ctx = LearnerContext.for_learner(facts.learner_id)
        coach = Coach()
        result = asyncio.run(coach.run(ctx, facts))
        return result.output, None
    except Exception as exc:  # noqa: BLE001 - the report must still print
        return deterministic_report(facts), f"{type(exc).__name__}: {exc}"


def report(
    weeks: Annotated[int, typer.Option("--weeks", help="How many weeks back to cover.")] = 1,
) -> None:
    """Print the weekly report: what moved, what is fading, what is next."""
    from the_oracle.context import LearnerContext
    from the_oracle.store.db import create_all, get_engine

    if weeks < 1:
        console.print("[red]--weeks must be at least 1.[/red]")
        raise typer.Exit(code=1)

    ctx = LearnerContext.resolve()
    engine = get_engine()
    create_all(engine)

    facts = gather_week(
        ctx.learner_id, now=datetime.now(timezone.utc), engine=engine, weeks=weeks
    )
    written, degraded = _write(facts)
    console.print(render(written, facts))
    if degraded is not None:
        console.print(
            f"[dim]Written from your event log without the model ({degraded}).[/dim]"
        )


app.callback(invoke_without_command=True)(report)

__all__ = ["app", "report"]
