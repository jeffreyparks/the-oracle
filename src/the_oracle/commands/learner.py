"""``oracle learner`` - export and delete one learner's data.

Two commands, both plain. Export answers "what do you hold about me".
Delete answers "remove it". Delete cannot be undone, and says so once.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from the_oracle.store.privacy import delete_learner, export_learner

console = Console()

app = typer.Typer(
    name="learner",
    help="Export or delete one learner's data.",
    no_args_is_help=True,
)


@app.command("export")
def export(
    learner_id: Annotated[str, typer.Argument(help="The learner to export.")],
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Write JSON here instead of stdout."),
    ] = None,
) -> None:
    """Write everything held about a learner as JSON."""
    payload = export_learner(learner_id)
    text = json.dumps(payload, indent=2, sort_keys=True)

    if out is None:
        console.print_json(text)
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    total = sum(payload["counts"].values())
    console.print(f"Wrote {total} rows across {len(payload['counts'])} tables to {out}.")


@app.command("delete")
def delete(
    learner_id: Annotated[str, typer.Argument(help="The learner to erase.")],
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip the confirmation prompt."),
    ] = False,
) -> None:
    """Erase a learner's data. This cannot be undone."""
    if not yes:
        console.print(f"This deletes every row belonging to [bold]{learner_id}[/bold].")
        console.print("It cannot be undone. Export first if you want a copy.")
        if not typer.confirm(f"Delete {learner_id}?"):
            console.print("Nothing was deleted.")
            raise typer.Exit(code=1)

    counts = delete_learner(learner_id)

    table = Table(title=f"deleted: {learner_id}")
    table.add_column("table", style="bold")
    table.add_column("rows", justify="right")
    for name, count in sorted(counts.items()):
        table.add_row(name, str(count))
    console.print(table)

    total = sum(counts.values())
    if total == 0:
        console.print("No data was held for that learner. Nothing changed.")
    else:
        console.print(f"Removed {total} rows. The data is gone.")


__all__ = ["app"]


# ---------------------------------------------------------------------------
# Nudge consent
# ---------------------------------------------------------------------------
#
# The day-14 nudge tells the learner to run ``oracle learner pause``. That
# command did not exist, and nothing in the product could set
# ``preferences["nudges"]``, so the documented one-word opt-out was unreachable.
# An opt-out you cannot reach is not an opt-out.


def _set_nudges(learner_id: str, value: str) -> None:
    """Write the nudge preference and record it in the log."""
    from the_oracle.store.db import get_engine, session_scope
    from the_oracle.store.events import EventKind, EventLog
    from the_oracle.store.models import Learner

    engine = get_engine()
    with session_scope(engine) as session:
        learner = session.get(Learner, learner_id)
        if learner is None:
            learner = Learner(id=learner_id)
            session.add(learner)
        prefs = dict(learner.preferences or {})
        prefs["nudges"] = value
        learner.preferences = prefs
        session.add(learner)

    EventLog(engine).append(
        EventKind.PREFERENCES_SET, learner_id, {"nudges": value}
    )


@app.command("pause")
def pause(
    learner_id: Annotated[
        str | None, typer.Argument(help="Defaults to the current learner.")
    ] = None,
) -> None:
    """Stop all reminders. Progress is kept."""
    from the_oracle.context import LearnerContext

    target = learner_id or LearnerContext.resolve().learner_id
    _set_nudges(target, "off")
    console.print("Reminders are off. Your progress is untouched.")
    console.print("Start again whenever you want with [bold]the-oracle review[/bold].")


@app.command("resume")
def resume(
    learner_id: Annotated[
        str | None, typer.Argument(help="Defaults to the current learner.")
    ] = None,
) -> None:
    """Turn reminders back on."""
    from the_oracle.context import LearnerContext

    target = learner_id or LearnerContext.resolve().learner_id
    _set_nudges(target, "on")
    console.print("Reminders are back on.")
