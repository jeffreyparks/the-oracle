"""``oracle resources`` - fill the corpus for a module, lazily.

The command is a thin renderer over :mod:`the_oracle.corpus.pipeline`. It makes
one promise of its own: ``--dry-run`` touches no network and no model. That is
enforced twice - the pipeline returns before any seam is bound, and this command
also passes :meth:`Seams.dry_run`, whose four callables raise if anything ever
reaches them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:  # pragma: no cover
    from the_oracle.corpus.pipeline import ObjectiveResult

console = Console()

app = typer.Typer(
    name="resources",
    help="Find, review, and author the material for a module.",
    invoke_without_command=True,
)
"""Wired into ``cli.py``. Register the callback directly -

    app.command("resources", help="Fill the corpus for a module.")(_resources_cmd.resources)

- so options parse on either side of the domain id, exactly as ``plan`` does.
"""


def _outcome(result: "ObjectiveResult") -> str:
    if result.skipped:
        return "[dim]skipped[/dim]"
    if result.authored:
        return "[magenta]authored[/magenta]"
    if result.accepted:
        return "[green]attached[/green]"
    return "[yellow]nothing[/yellow]"


def _table(title: str, results: list["ObjectiveResult"]) -> Table:
    table = Table(title=title, title_justify="left", box=None, pad_edge=False)
    table.add_column("objective", style="bold")
    table.add_column("searched", justify="right")
    table.add_column("accepted", justify="right")
    table.add_column("rejected", justify="right")
    table.add_column("outcome")
    for result in results:
        table.add_row(
            result.objective_id,
            str(result.searched),
            str(result.accepted),
            str(result.rejected),
            _outcome(result),
        )
    return table


@app.callback(invoke_without_command=True)
def resources(
    domain_id: Annotated[str, typer.Argument(help="Domain pack id.")],
    module: Annotated[
        str | None, typer.Option("--module", "-m", help="One module id. Default: every module.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report what would happen. No network, no model.")
    ] = False,
    minimum: Annotated[
        int, typer.Option("--minimum", "-n", help="Accepted resources an objective needs.")
    ] = 2,
) -> None:
    """Fill the shared corpus for a module. Skips objectives that already have enough."""
    from the_oracle.context import LearnerContext
    from the_oracle.corpus.pipeline import Seams, ensure_module
    from the_oracle.corpus.store import tokens_spent
    from the_oracle.domains.errors import PackValidationError
    from the_oracle.domains.registry import load_domain
    from the_oracle.store.db import create_all, get_engine
    from the_oracle.store.models import utcnow

    ctx = LearnerContext.resolve()
    try:
        domain = load_domain(domain_id)
    except PackValidationError as exc:
        console.print(f"[red]{domain_id} did not load.[/red]")
        for problem in exc.problems:
            console.print(f"  - {problem}")
        raise typer.Exit(code=1) from exc

    module_ids = [m.id for m in domain.modules]
    if module is not None:
        if module not in module_ids:
            console.print(f"[red]{module!r} is not a module of {domain_id!r}.[/red]")
            console.print("modules: " + (", ".join(module_ids) or "none"))
            raise typer.Exit(code=1)
        module_ids = [module]
    if not module_ids:
        console.print(f"[yellow]{domain_id} has no modules.[/yellow]")
        raise typer.Exit(code=1)

    engine = get_engine()
    create_all(engine)
    started = utcnow()
    # Belt and braces: in a dry run every seam raises if it is ever called.
    seams = Seams.dry_run() if dry_run else None

    console.rule(domain.title)
    if dry_run:
        console.print(
            Panel(
                "Dry run. Nothing is searched, fetched, or written, and no model "
                "is called. The numbers below are the work this command would do.",
                title="dry run",
                border_style="cyan",
            )
        )

    totals = {"skipped": 0, "authored": 0, "accepted": 0, "rejected": 0, "searched": 0}
    for module_id in module_ids:
        results = ensure_module(
            domain_id,
            module_id,
            minimum=minimum,
            engine=engine,
            ctx=ctx,
            seams=seams,
            dry_run=dry_run,
            domain=domain,
        )
        console.print(_table(module_id, results))
        console.print()
        for result in results:
            totals["skipped"] += int(result.skipped)
            totals["authored"] += int(result.authored)
            totals["accepted"] += result.accepted
            totals["rejected"] += result.rejected
            totals["searched"] += result.searched

    tokens = 0 if dry_run else tokens_spent(engine, since=started)
    console.print(
        f"searched {totals['searched']}  accepted {totals['accepted']}  "
        f"rejected {totals['rejected']}  authored {totals['authored']}  "
        f"skipped {totals['skipped']}"
    )
    console.print(f"tokens: [bold]{tokens}[/bold]" + (" [dim](dry run)[/dim]" if dry_run else ""))


__all__ = ["app", "resources"]
