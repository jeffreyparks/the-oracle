"""``the-oracle review`` — everything that is due, right now, across domains.

Mastery is keyed by objective, not by domain, so due work crosses packs. This
command reads the derived schedule and reports it. It never generates work to
look useful: when nothing is due it says so and stops, and tells you when the
next thing lands so the silence is informative rather than empty.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

app = typer.Typer(
    name="review",
    help="Work through everything that is due.",
    invoke_without_command=True,
)
"""Wired into ``cli.py`` as a command::

    app.command("review", help="Work through everything that is due.")(_review_cmd.review)
"""


def _domains_by_objective() -> dict[str, list[str]]:
    """Objective id -> the domain packs that reference it. Best effort."""
    from the_oracle.domains import registry

    mapping: dict[str, list[str]] = {}
    try:
        domain_ids = registry.list_domains()
    except Exception:
        return mapping
    for domain_id in sorted(domain_ids):
        try:
            domain = registry.load_domain(domain_id)
        except Exception:
            continue  # a broken pack must not hide due work in the others
        for ref in getattr(domain, "objectives", []):
            mapping.setdefault(ref.id, []).append(domain_id)
    return mapping


def _rows(learner_id: str, engine: Any) -> list[Any]:
    from sqlmodel import Session, select

    from the_oracle.store import models

    with Session(engine) as db:
        rows = db.exec(
            select(models.MasteryState).where(models.MasteryState.learner_id == learner_id)
        ).all()
    return [row for row in rows if row.next_review_at is not None]


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _gap(hours: float) -> str:
    """Plain words for a span. Rounded, but never rounded to flatter."""
    whole = round(hours)
    if whole < 24:
        return f"{whole} hour" if whole == 1 else f"{whole} hours"
    days = hours / 24.0
    return f"{days:.1f} day" if round(days, 1) == 1.0 else f"{days:.1f} days"


def _overdue(due_at: datetime, now: datetime) -> str:
    return f"{_gap((now - _aware(due_at)).total_seconds() / 3600.0)} ago"


def _wait(due_at: datetime, now: datetime) -> str:
    return f"in {_gap((_aware(due_at) - now).total_seconds() / 3600.0)}"


@app.callback(invoke_without_command=True)
def review() -> None:
    """Show every objective due for review now, across every domain."""
    from the_oracle.context import LearnerContext
    from the_oracle.store.db import create_all, get_engine

    ctx = LearnerContext.resolve()
    engine = get_engine()
    create_all(engine)
    now = datetime.now(timezone.utc)

    scheduled = _rows(ctx.learner_id, engine)
    due = sorted(
        (row for row in scheduled if _aware(row.next_review_at) <= now),
        key=lambda row: row.next_review_at,
    )

    if not due:
        later = sorted(scheduled, key=lambda row: row.next_review_at)
        if not later:
            console.print(
                Panel(
                    "Nothing is due, and nothing is scheduled yet.\n"
                    "There is no review history to schedule from. Run "
                    "the-oracle assess <domain> or the-oracle study <domain> first.",
                    title="nothing due",
                    border_style="cyan",
                )
            )
            return
        nxt = later[0]
        console.print(
            Panel(
                "Nothing is due right now.\n"
                f"Next up is {nxt.objective_id}, {_wait(nxt.next_review_at, now)} "
                f"(on {_aware(nxt.next_review_at):%Y-%m-%d}).\n"
                "Reviewing early costs you most of the spacing benefit, so this is "
                "a real stop, not a nudge to do more.",
                title="nothing due",
                border_style="cyan",
            )
        )
        return

    by_objective = _domains_by_objective()
    table = Table(title="due now", title_justify="left", box=None, pad_edge=False)
    table.add_column("objective", style="bold")
    table.add_column("domain")
    table.add_column("p(mastery)", justify="right")
    table.add_column("seen", justify="right")
    table.add_column("due")
    for row in due:
        table.add_row(
            row.objective_id,
            ", ".join(by_objective.get(row.objective_id, [])) or "-",
            f"{row.p_mastery:.2f}",
            str(row.observations),
            _overdue(row.next_review_at, now),
        )
    console.print(table)
    console.print()

    domains = sorted({d for row in due for d in by_objective.get(row.objective_id, [])})
    noun = "objective" if len(due) == 1 else "objectives"
    console.print(f"{len(due)} {noun} due.")
    if domains:
        console.print(
            "These are injected into your next session, not done as a separate chore. "
            "Run: " + "  ".join(f"the-oracle study {d}" for d in domains)
        )
    else:
        console.print(
            "None of these appear in a domain pack you have installed, so there is "
            "nothing to run them from yet."
        )


__all__ = ["app", "review"]
