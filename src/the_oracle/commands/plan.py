"""``oracle plan`` - show the syllabus for a domain, honestly costed.

The planner is deterministic, so this command is a renderer and nothing else.
It never rounds the total down to make the plan look friendlier. If the real
number is forty hours, it says forty hours.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:  # pragma: no cover
    from the_oracle.domains.schema import Domain
    from the_oracle.mastery.profile import MasteryProfile
    from the_oracle.planning import Checkpoint, Syllabus

console = Console()

app = typer.Typer(
    name="plan",
    help="Show the study plan for a domain.",
    invoke_without_command=True,
)
"""Wired into ``cli.py`` with ``app.add_typer``, which yields ``plan <domain_id>``.

Same shape as ``commands/assess.py``. One caveat the orchestrator should know:
a Typer group parses options before the positional argument, so with
``add_typer`` the working form is ``oracle plan --hours-per-week 2 demo``.
If you want options after the id, register the function directly instead:
``app.command("plan")(_plan_cmd.plan)``. Both entry points are exported.
"""

_STATUS_STYLE = {"learn": "cyan", "review": "yellow"}


def _hours(minutes: int) -> str:
    return f"{minutes / 60:.1f} h"


def _module_title(domain: "Domain", module_id: str) -> str:
    for module in domain.modules:
        if module.id == module_id:
            return module.title
    return module_id


def _render_module(
    domain: "Domain",
    syllabus: "Syllabus",
    module_id: str,
    checkpoint: "Checkpoint | None",
) -> None:
    """One module: its planned items, its cost, and the checkpoint that closes it."""
    items = syllabus.for_module(module_id)
    minutes = syllabus.module_minutes(module_id)
    table = Table(
        title=f"{_module_title(domain, module_id)} - {_hours(minutes)}",
        title_justify="left",
        box=None,
        pad_edge=False,
    )
    table.add_column("objective", style="bold")
    table.add_column("what", justify="left")
    table.add_column("p(mastery)", justify="right")
    table.add_column("minutes", justify="right")

    for item in items:
        label = "new" if item.status == "learn" else "review"
        style = _STATUS_STYLE.get(item.status, "white")
        table.add_row(
            item.objective_id,
            f"[{style}]{label}[/{style}]",
            f"{item.p_mastery:.2f}",
            str(item.est_minutes),
        )
    console.print(table)

    if checkpoint is not None:
        drawn = ", ".join(checkpoint.objective_ids)
        console.print(
            f"  [magenta]checkpoint[/magenta] {checkpoint.item_count} items, "
            f"{checkpoint.est_minutes} min, one session, drawn from this module: {drawn}"
        )
    console.print()


def _render_skipped(domain: "Domain", syllabus: "Syllabus") -> None:
    """What the learner does not have to do, and what that is worth."""
    if not syllabus.skipped_mastered:
        return
    saved = sum(domain.objective(oid).est_minutes for oid in syllabus.skipped_mastered)
    console.print(
        f"[green]Skipped as already mastered:[/green] {len(syllabus.skipped_mastered)} "
        f"objectives, {_hours(saved)} you do not have to sit through."
    )
    for objective_id in syllabus.skipped_mastered:
        console.print(f"  [dim]- {objective_id}[/dim]")
    console.print()


def _read_profile(learner_id: str) -> "MasteryProfile":
    """Read mastery, treating a store that does not exist yet as no evidence."""
    from sqlalchemy.exc import SQLAlchemyError

    from the_oracle.mastery.profile import MasteryProfile, profile_for

    try:
        return profile_for(learner_id, None)
    except SQLAlchemyError:
        return MasteryProfile({}, learner_id=learner_id)


@app.callback(invoke_without_command=True)
def plan(
    domain_id: Annotated[str, typer.Argument(help="Domain pack id to plan.")],
    hours_per_week: Annotated[
        float, typer.Option("--hours-per-week", "-h", help="Study time you will really give it.")
    ] = 3.0,
    session_minutes: Annotated[
        int, typer.Option("--session-minutes", help="Length of one session.")
    ] = 25,
) -> None:
    """Build and print the syllabus: what to learn, what to review, what to skip."""
    from the_oracle.context import LearnerContext
    from the_oracle.domains.errors import PackValidationError
    from the_oracle.domains.registry import load_domain
    from the_oracle.planning import build_checkpoints, build_syllabus, finish_date

    ctx = LearnerContext.resolve()
    try:
        domain = load_domain(domain_id)
    except PackValidationError as exc:
        console.print(f"[red]{domain_id} did not load.[/red]")
        for problem in exc.problems:
            console.print(f"  - {problem}")
        raise typer.Exit(code=1) from exc

    profile = _read_profile(ctx.learner_id)
    syllabus = build_syllabus(
        domain_id,
        ctx.learner_id,
        hours_per_week=hours_per_week,
        profile=profile,
        domain=domain,
    )

    if len(profile) == 0:
        console.print(
            Panel(
                "No mastery data for you yet, so this plan assumes you know none "
                f"of it. That is a guess, not a measurement. Run "
                f"[bold]the-oracle assess {domain_id}[/bold] first - it takes about "
                "12 items, and it usually removes real work from this plan.",
                title="unmeasured",
                border_style="yellow",
            )
        )
        console.print()

    checkpoints = {c.module_id: c for c in build_checkpoints(
        syllabus, domain=domain, session_minutes=session_minutes
    )}

    console.rule(domain.title)
    console.print()
    _render_skipped(domain, syllabus)

    for module_id in syllabus.module_ids():
        _render_module(domain, syllabus, module_id, checkpoints.get(module_id))

    if not syllabus.items:
        console.print(
            "Nothing left to plan. Every objective in this domain is above the "
            "mastery bar. Come back for review when the scheduler says so."
        )
        return

    counts = syllabus.counts()
    check_minutes = sum(c.est_minutes for c in checkpoints.values())
    grand_total = syllabus.total_minutes + check_minutes
    weeks = grand_total / 60.0 / hours_per_week
    end = finish_date(weeks, date.today())

    summary = Table(box=None, pad_edge=False, show_header=False)
    summary.add_column("", style="bold")
    summary.add_column("", justify="right")
    summary.add_row("new objectives", str(counts["learn"]))
    summary.add_row("review objectives", str(counts["review"]))
    summary.add_row("already mastered", str(counts["mastered"]))
    summary.add_row("study time", _hours(syllabus.total_minutes))
    summary.add_row(
        f"checkpoints ({len(checkpoints)})", _hours(check_minutes)
    )
    summary.add_row("total", _hours(grand_total))
    summary.add_row("at", f"{hours_per_week:g} h/week")
    summary.add_row("weeks", f"{weeks:.1f}")
    summary.add_row("finish", end.isoformat())
    console.print(summary)
    console.print()
    console.print(
        f"That is [bold]{_hours(grand_total)}[/bold] of real work, checkpoints "
        f"included, and about {weeks:.1f} weeks at {hours_per_week:g} hours a week. "
        f"Miss a week and the date moves. The plan does not shrink on its own."
    )


__all__ = ["app", "plan"]
