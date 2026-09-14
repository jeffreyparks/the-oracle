"""``the-oracle domain add`` — interview, draft, dedupe, commit.

The orchestrator wires :data:`app` into ``cli.py`` under the ``domain`` group.
This module owns the interactive surface only: ask the questions, show what the
Architect proposed to reuse, and write nothing until the learner says yes.

The reuse report is the point of the confirmation step. A false merge is the
one mistake in this pipeline that corrupts mastery silently in both directions
(PLAN.md section 2), so a human sees every reuse before it lands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:  # pragma: no cover
    from the_oracle.agents.architect import DomainPlan, Interview

console = Console()

app = typer.Typer(
    name="add",
    help="Generate a new domain pack from a short interview.",
    invoke_without_command=True,
)

DEPTHS = ("survey", "working", "practitioner", "expert")

DEPTH_HELP = {
    "survey": "you want the map, not the territory",
    "working": "you want to use it under supervision",
    "practitioner": "you want to do the work unaided",
    "expert": "you want to judge other people's work",
}


def _csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def run_interview() -> "Interview":
    """Ask the seven questions. Nothing here costs a token."""
    from the_oracle.agents.architect import Interview

    console.print(
        Panel(
            "Seven questions, then I draft the curriculum.\n"
            "Answer the background question honestly. If you overstate it, the "
            "plan starts above you and you will feel it in week one.",
            title="new domain",
            border_style="cyan",
        )
    )
    goal = typer.prompt("What do you want to be able to do")
    background = typer.prompt("What do you already know that is relevant")
    hours = typer.prompt("Hours per week you will really give this", type=float, default=3.0)
    weeks = typer.prompt("Weeks you want to take", type=int, default=8)

    console.print()
    for name in DEPTHS:
        console.print(f"  [bold]{name}[/bold] — {DEPTH_HELP[name]}")
    depth = typer.prompt("How deep", default="working")
    while depth not in DEPTHS:
        depth = typer.prompt(f"Pick one of {', '.join(DEPTHS)}", default="working")

    must = typer.prompt("Anything it must cover (comma separated)", default="")
    skip = typer.prompt("Anything to leave out (comma separated)", default="")
    return Interview(
        goal=goal.strip(),
        background=background.strip(),
        hours_per_week=hours,
        target_weeks=weeks,
        depth=depth,  # type: ignore[arg-type]
        must_cover=_csv(must),
        exclude=_csv(skip),
    )


#: What each reason word means, shown under the table so nobody has to guess.
REASON_HELP = {
    "retrieved": "the Architect saw it and chose it",
    "matched": "the similarity score was strong enough",
    "judged": "borderline, and the Critic agreed",
    "new": "nothing in your library covered it",
}


def dedupe_table(plan: "DomainPlan") -> Table:
    """Reused versus new, with the reason and score behind each decision.

    The reason column exists because reuse now has two different origins. A
    ``retrieved`` reuse is a judgement the model made from the library it was
    shown; a ``matched`` one is arithmetic. They deserve different amounts of
    trust, so a human gets to see which is which before saying yes.
    """
    table = Table(title="what this pack reuses", title_justify="left", box=None, pad_edge=False)
    table.add_column("objective", style="bold")
    table.add_column("decision")
    table.add_column("reason")
    table.add_column("id")
    table.add_column("score", justify="right")
    for draft_obj in plan.draft.objectives:
        title = draft_obj.title
        reason = plan.reason_for(title)
        # A retrieved reuse was never scored, so there is no number to show.
        score = "-" if reason == "retrieved" else f"{plan.score_for(title):.2f}"
        if title in plan.reused:
            table.add_row(title, "[green]reuse[/green]", reason, plan.reused[title], score)
        else:
            table.add_row(title, "[cyan]new[/cyan]", reason, plan.minted[title].id, score)
    return table


def reuse_summary(plan: "DomainPlan") -> str:
    """``12 objectives, 5 reused (4 retrieved, 1 judged), 7 new.``"""
    counts: dict[str, int] = {}
    for title in plan.reused:
        reason = plan.reason_for(title)
        counts[reason] = counts.get(reason, 0) + 1
    parts = [f"{counts[r]} {r}" for r in ("retrieved", "matched", "judged") if counts.get(r)]
    detail = f" ({', '.join(parts)})" if parts else ""
    return (
        f"{len(plan.domain.objectives)} objectives, "
        f"{len(plan.reused)} reused{detail}, {len(plan.minted)} new."
    )


def show_plan(plan: "DomainPlan") -> None:
    """Everything the learner needs to decide yes or no."""
    domain = plan.domain
    console.print()
    console.print(Panel(f"{domain.title}\n\n{domain.description}", title=domain.id, border_style="cyan"))

    for module in domain.modules:
        table = Table(title=module.title, title_justify="left", box=None, pad_edge=False)
        table.add_column("objective", style="bold")
        table.add_column("bloom")
        table.add_column("diff", justify="right")
        table.add_column("min", justify="right")
        for objective_id in module.objectives:
            body = plan.resolved[objective_id]
            table.add_row(body.title, body.bloom, str(body.difficulty), str(body.est_minutes))
        console.print(table)
        console.print()

    console.print(dedupe_table(plan))
    console.print(
        "[dim]reason: " + "; ".join(f"{k} = {v}" for k, v in REASON_HELP.items()) + "[/dim]"
    )
    console.print()

    hours = plan.total_minutes / 60
    weeks = plan.interview.target_weeks or 1
    console.print(
        f"{reuse_summary(plan)} {hours:.1f} hours of teaching time, "
        f"about {hours / weeks:.1f} hours a week over {weeks} weeks."
    )
    if plan.reused:
        console.print(
            "[dim]Reused objectives already carry your mastery. You will not be "
            "taught them twice.[/dim]"
        )


@app.callback(invoke_without_command=True)
def domain_add(
    goal: Annotated[str, typer.Argument(help="What you want to be able to do.")] = "",
    background: Annotated[str, typer.Option(help="What you already know.")] = "",
    hours_per_week: Annotated[float, typer.Option(help="Study hours per week.")] = 3.0,
    target_weeks: Annotated[int, typer.Option(help="Weeks to take.")] = 8,
    depth: Annotated[str, typer.Option(help=f"One of {', '.join(DEPTHS)}.")] = "working",
    must_cover: Annotated[str, typer.Option(help="Comma separated topics to include.")] = "",
    exclude: Annotated[str, typer.Option(help="Comma separated topics to leave out.")] = "",
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation.")] = False,
) -> None:
    """Interview, draft a curriculum, show what it reuses, then commit."""
    from the_oracle.agents.architect import (
        Interview,
        commit,
        draft_domain,
        plan_domain,
    )
    from the_oracle.domains.errors import PackValidationError

    if goal:
        if depth not in DEPTHS:
            console.print(f"[red]depth must be one of {', '.join(DEPTHS)}.[/red]")
            raise typer.Exit(code=2)
        interview = Interview(
            goal=goal,
            background=background,
            hours_per_week=hours_per_week,
            target_weeks=target_weeks,
            depth=depth,  # type: ignore[arg-type]
            must_cover=_csv(must_cover),
            exclude=_csv(exclude),
        )
    else:
        interview = run_interview()

    console.print()
    console.print(
        "Drafting the curriculum. This is one large model call, so it takes "
        "roughly a minute and it costs tokens. Nothing is written yet."
    )
    try:
        with console.status("thinking"):
            draft = draft_domain(interview)
    except Exception as exc:  # noqa: BLE001 - the CLI is the last line
        console.print(f"[red]The Architect failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print("Matching against the objectives you already have.")

    # Wire the Critic in. Without a judge every middle-band match splits, so the
    # Critic would exist, be tested, and never run: a live overlapping domain
    # scored 0.94 against an existing objective and still minted a duplicate.
    # Built here rather than defaulted inside plan_domain, so library callers
    # and offline tests stay free of live model calls unless they ask for one.
    from the_oracle.agents.architect import draft_objectives_as_objectives
    from the_oracle.agents.critic import judge_fn

    try:
        judge = judge_fn(draft_objectives_as_objectives(draft))
    except Exception as exc:  # noqa: BLE001 - never block a build on the judge
        console.print(f"[yellow]No sameness judge available ({exc}); borderline matches will split.[/yellow]")
        judge = None

    try:
        plan = plan_domain(draft, interview, judge=judge)
    except PackValidationError as exc:
        console.print("[red]The draft is not a valid curriculum, so nothing was written.[/red]")
        for problem in exc.problems:
            console.print(f"  - {problem}")
        console.print("[dim]Run the command again; the Architect gets a fresh attempt.[/dim]")
        raise typer.Exit(code=1) from exc

    show_plan(plan)
    if not yes and not typer.confirm("Write this pack", default=True):
        console.print("Nothing written.")
        raise typer.Exit(code=0)

    domain = commit(plan)
    console.print()
    console.print(
        Panel(
            f"Written. {len(domain.objectives)} objectives, "
            f"{len(plan.minted)} of them new.\n\n"
            f"Next: [bold]the-oracle assess {domain.id}[/bold] to find out where "
            "you actually stand.",
            title=domain.id,
            border_style="green",
        )
    )
