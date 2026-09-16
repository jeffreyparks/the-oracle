"""The ``the-oracle`` command line.

Phase 0 implements ``domain list``, ``domain show``, ``rebuild``, and
``version``. Every other command is a loud, honest stub.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from the_oracle import __version__
from the_oracle.config import get_settings
from the_oracle.context import LearnerContext

console = Console()

app = typer.Typer(
    name="the-oracle",
    help="A tutor that learns you, plans the path, finds the material, and keeps you moving.",
    no_args_is_help=True,
    add_completion=False,
)
domain_app = typer.Typer(name="domain", help="Work with domain packs.", no_args_is_help=True)
app.add_typer(domain_app)

# Feature commands own their own modules; the CLI only wires them in.
#
# Single-action commands are registered as COMMANDS, not sub-apps. Wiring a
# one-callback Typer group with add_typer makes click treat any option after the
# positional argument as a subcommand name, so `domain add "goal" --yes` fails
# with "No such command '--yes'". Registering the function directly parses
# options in either position.
from the_oracle.commands import assess as _assess_cmd  # noqa: E402
from the_oracle.commands import domain_add as _domain_add_cmd  # noqa: E402
from the_oracle.commands import learner as _learner_cmd  # noqa: E402
from the_oracle.commands import plan as _plan_cmd  # noqa: E402
from the_oracle.commands import cron as _cron_cmd  # noqa: E402
from the_oracle.commands import init as _init_cmd  # noqa: E402
from the_oracle.commands import report as _report_cmd  # noqa: E402
from the_oracle.commands import resources as _resources_cmd  # noqa: E402
from the_oracle.commands import review as _review_cmd  # noqa: E402
from the_oracle.commands import study as _study_cmd  # noqa: E402

app.command("init", help="Copy the seed packs into your data directory.")(_init_cmd.init)
app.command("assess", help="Run the adaptive diagnostic for a domain.")(_assess_cmd.assess)
app.command("resources", help="Fill the corpus for a module.")(_resources_cmd.resources)
app.command("study", help="Run a study session.")(_study_cmd.study)
app.command("review", help="Work through everything that is due.")(_review_cmd.review)
app.command("report", help="Print the weekly report.")(_report_cmd.report)
app.add_typer(_cron_cmd.app, name="cron")
app.command("plan", help="Build or refresh the syllabus.")(_plan_cmd.plan)
domain_app.command("add", help="Create a new domain pack from an interview.")(
    _domain_add_cmd.domain_add
)

# `learner` is a genuine group with several commands, so it stays a sub-app.
app.add_typer(_learner_cmd.app, name="learner")


@app.callback()
def _bootstrap(ctx: typer.Context) -> None:
    """Run before every command. Sets up telemetry and one span per command."""
    from contextlib import ExitStack

    from the_oracle import telemetry

    telemetry.configure()
    stack = ExitStack()
    stack.enter_context(
        telemetry.span("cli {command}", command=ctx.invoked_subcommand or "root")
    )
    ctx.call_on_close(stack.close)

def _not_built(command: str, phase: str) -> None:
    console.print(
        Panel(
            f"[bold]{command}[/bold] is not built yet.\nIt lands in {phase}.",
            title="not built yet",
            border_style="yellow",
        )
    )
    raise typer.Exit(code=1)


def _registry():  # type: ignore[no-untyped-def]
    """Import the domains registry lazily, so the engine loads without it."""
    from the_oracle.domains import registry

    return registry


@app.command()
def version() -> None:
    """Print the version and the resolved data directory."""
    settings = get_settings()
    console.print(f"oracle [bold]{__version__}[/bold]")
    console.print(f"home: {settings.home}")
    console.print(f"database: {settings.sqlalchemy_url}")

    # Credentials: report presence only, never a value.
    import os

    from the_oracle.config import DOTENV_FILES

    if DOTENV_FILES:
        console.print("env files: " + ", ".join(str(p) for p in DOTENV_FILES))
    else:
        console.print("env files: [dim]none found[/dim]")

    from the_oracle import telemetry

    console.print(f"telemetry: {telemetry.configure().describe()}")

    watched = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "SERPER_API_KEY", "VOYAGE_API_KEY")
    found = [name for name in watched if os.environ.get(name)]
    missing = [name for name in watched if not os.environ.get(name)]
    console.print("keys set: " + (", ".join(found) if found else "[dim]none[/dim]"))
    if missing:
        console.print(f"keys missing: [dim]{', '.join(missing)}[/dim]")


@domain_app.command("list")
def domain_list() -> None:
    """List the domain packs found in the data directory."""
    settings = get_settings()
    try:
        ids = _registry().list_domains()
    except ModuleNotFoundError:
        console.print("[yellow]The domains package is not installed yet.[/yellow]")
        raise typer.Exit(code=1) from None

    if not ids:
        console.print(
            Panel(
                f"No domain packs in [bold]{settings.domains_dir}[/bold].\n"
                "Run [bold]the-oracle domain add[/bold] to generate one.",
                title="blank oracle",
                border_style="cyan",
            )
        )
        return

    table = Table(title="domains", show_lines=False)
    table.add_column("id", style="bold")
    table.add_column("title")
    table.add_column("objectives", justify="right")
    table.add_column("modules", justify="right")

    registry = _registry()
    for domain_id in sorted(ids):
        try:
            domain = registry.load_domain(domain_id)
        except Exception as exc:  # fail loud, keep listing the rest
            table.add_row(domain_id, f"[red]invalid: {exc}[/red]", "-", "-")
            continue
        table.add_row(
            domain.id,
            getattr(domain, "title", ""),
            str(len(getattr(domain, "objectives", []))),
            str(len(getattr(domain, "modules", []))),
        )
    console.print(table)


@domain_app.command("show")
def domain_show(
    domain_id: Annotated[str, typer.Argument(help="Domain pack id.")],
) -> None:
    """Show one pack: modules, objectives in teaching order, prerequisites."""
    registry = _registry()
    try:
        domain = registry.load_domain(domain_id)
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    console.print(Panel(getattr(domain, "description", "") or "", title=getattr(domain, "title", domain_id)))

    table = Table(title="modules")
    table.add_column("#", justify="right")
    table.add_column("module", style="bold")
    table.add_column("objectives")
    for position, module in enumerate(getattr(domain, "modules", []), start=1):
        table.add_row(str(position), module.title, ", ".join(module.objectives))
    console.print(table)

    try:
        order = domain.teaching_order()
    except Exception:
        order = []
    if order:
        console.print("teaching order: " + " -> ".join(order))


@app.command()
def rebuild(
    learner: Annotated[str | None, typer.Option(help="Learner id. Defaults to the current one.")] = None,
) -> None:
    """Drop derived state and replay the event log."""
    from the_oracle.store.db import create_all, get_engine
    from the_oracle.store.rebuild import rebuild as run_rebuild

    settings = get_settings()
    settings.ensure_home()
    engine = get_engine(settings)
    create_all(engine)

    ctx = (
        LearnerContext.for_learner(learner, settings)
        if learner
        else LearnerContext.resolve(settings)
    )
    report = run_rebuild(ctx.learner_id, engine)

    table = Table(title=f"rebuild: {report.learner_id}")
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("events read", str(report.events_read))
    table.add_row("events handled", str(report.events_handled))
    table.add_row("events unhandled", str(report.events_skipped))
    table.add_row("derived rows", str(report.rows_written))
    table.add_row("tables cleared", ", ".join(report.tables_cleared) or "-")
    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    app()
