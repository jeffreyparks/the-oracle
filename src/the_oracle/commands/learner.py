"""``the-oracle learner`` - export and delete one learner's data.

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
