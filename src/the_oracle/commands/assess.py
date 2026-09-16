"""``oracle assess`` — run the adaptive diagnostic.

The orchestrator wires :data:`app` into ``cli.py``. This module owns the
interactive surface only: ask, show the verdict, print the profile.

Feedback timing follows PLAN.md section 5. Recall and short items get their
verdict the moment the answer is in. Multi-step work gets its critique at the
end of that attempt, never during it. The same wrong model twice in one
session interrupts.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

app = typer.Typer(
    name="assess",
    help="Run the adaptive diagnostic for a domain.",
    invoke_without_command=True,
)
"""Wired into ``cli.py`` with ``app.add_typer``, which yields ``assess <domain_id>``."""


def _ask(item: "object") -> str:
    """Put one item to the learner and read the answer back."""
    from the_oracle.agents.assessor import feedback_is_immediate

    stem = getattr(item, "stem", "")
    kind = getattr(item, "kind", "short")
    difficulty = getattr(item, "difficulty", 1)
    console.print()
    console.print(Panel(stem, title=f"{kind} · difficulty {difficulty}", border_style="cyan"))
    if not feedback_is_immediate(kind):
        console.print("[dim]Work it through. You get the full critique when you submit.[/dim]")
    return typer.prompt("your answer")


def _report(item: "object", answer: str, grade: "object", seen: dict[str, int]) -> None:
    """Show the verdict for one finished attempt."""
    correct = bool(getattr(grade, "correct", False))
    verdict = "[green]correct[/green]" if correct else "[red]wrong[/red]"
    console.print(f"{verdict} — {getattr(grade, 'feedback', '')}")

    misconception_id = getattr(grade, "misconception_id", None)
    if misconception_id:
        seen[misconception_id] = seen.get(misconception_id, 0) + 1
        if seen[misconception_id] >= 2:
            console.print(
                Panel(
                    f"That is the second time {misconception_id} has shown up this "
                    "session. Stopping here to fix the model before it sets.",
                    title="same wrong model twice",
                    border_style="yellow",
                )
            )


def _profile_table(domain: "object", profile: "object") -> None:
    """Print p(mastery) per objective, grouped by module."""
    from the_oracle.mastery import MASTERY_THRESHOLD

    for module in getattr(domain, "modules", []):
        table = Table(title=module.title, title_justify="left", box=None, pad_edge=False)
        table.add_column("objective", style="bold")
        table.add_column("p(mastery)", justify="right")
        table.add_column("state")
        shown = 0
        for objective_id in module.objectives:
            p = profile.p(objective_id)  # type: ignore[attr-defined]
            state = "mastered" if p >= MASTERY_THRESHOLD else ("unseen" if p == 0.2 else "learning")
            table.add_row(objective_id, f"{p:.2f}", state)
            shown += 1
        if shown:
            console.print(table)
            console.print()


@app.callback(invoke_without_command=True)
def assess(
    domain_id: Annotated[str, typer.Argument(help="Domain pack id to assess against.")],
    max_items: Annotated[int, typer.Option(help="Hard cap on items asked.")] = 12,
    target_accuracy: Annotated[float, typer.Option(help="Accuracy the loop steers toward.")] = 0.7,
) -> None:
    """Ask a short adaptive set of items and build the mastery profile."""
    from the_oracle.agents.assessor import run_diagnostic
    from the_oracle.context import LearnerContext
    from the_oracle.domains.errors import PackValidationError
    from the_oracle.domains.registry import load_domain

    ctx = LearnerContext.resolve()
    try:
        domain = load_domain(domain_id)
    except PackValidationError as exc:
        console.print(f"[red]{domain_id} did not load.[/red]")
        for problem in exc.problems:
            console.print(f"  - {problem}")
        raise typer.Exit(code=1) from exc

    console.print(
        Panel(
            f"{domain.title}\n\nUp to {max_items} items. This is a diagnostic, so "
            "expect to get some wrong. That is the point: wrong answers are where "
            "the information is.",
            title="diagnostic",
            border_style="cyan",
        )
    )

    seen: dict[str, int] = {}
    profile = run_diagnostic(
        domain_id,
        ctx.learner_id,
        max_items=max_items,
        target_accuracy=target_accuracy,
        answer_fn=_ask,
        on_graded=lambda item, answer, grade: _report(item, answer, grade, seen),
    )

    console.print()
    console.rule("where you stand")
    _profile_table(domain, profile)
