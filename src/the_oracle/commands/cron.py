"""``oracle cron`` — the whole scheduler, in one crontab line.

There is no daemon. ``cron run`` is the single entry point the line calls; it
computes due work, decides whether a nudge is owed, and delivers it. The other
commands exist only to put that one line in place, safely:

* ``cron line`` prints it and installs nothing.
* ``cron install`` shows the exact change first, then applies it, and refuses
  to leave a duplicate behind.
* ``cron uninstall`` removes only the marked line.
* ``cron status`` says whether it is installed and when it last ran.

A crontab belongs to the user, not to us. Nothing here writes one without
printing the exact change first, and the ``crontab`` runner is injected so a
test can never touch the real one.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from the_oracle.schedule import cron as cron_lib

console = Console()

app = typer.Typer(
    name="cron",
    help="Install the one crontab line that runs the daily check.",
    no_args_is_help=True,
)
"""Wired into ``cli.py`` as a sub-app::

    from the_oracle.commands import cron as _cron_cmd
    app.add_typer(_cron_cmd.app, name="cron")
"""


def _runner() -> cron_lib.CrontabRunner:
    """The injected seam. Tests call ``cron_lib.set_runner(fake)``."""
    return cron_lib.get_runner()


def _show(change: cron_lib.CrontabChange) -> None:
    """Print the exact change. Always called before anything is written."""
    console.print(
        Panel(
            change.diff(),
            title=f"exact change: {change.action}",
            border_style="yellow" if change.changed else "cyan",
        )
    )


@app.command("line")
def line(
    hour: Annotated[int, typer.Option(help="Hour of the day, 0-23.")] = cron_lib.DEFAULT_HOUR,
    minute: Annotated[int, typer.Option(help="Minute of the hour, 0-59.")] = cron_lib.DEFAULT_MINUTE,
) -> None:
    """Print the exact crontab line. Installs nothing."""
    try:
        text = cron_lib.cron_line(hour=hour, minute=minute)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from None
    console.print(text)
    console.print()
    console.print(
        "[dim]That is the whole scheduler. Add it yourself, or run "
        "oracle cron install.[/dim]"
    )


@app.command("install")
def install(
    hour: Annotated[int, typer.Option(help="Hour of the day, 0-23.")] = cron_lib.DEFAULT_HOUR,
    minute: Annotated[int, typer.Option(help="Minute of the hour, 0-59.")] = cron_lib.DEFAULT_MINUTE,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation.")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show the change and stop.")] = False,
) -> None:
    """Add the line to your crontab, after showing the exact change."""
    try:
        change = cron_lib.plan_install(_runner(), hour=hour, minute=minute)
    except (ValueError, cron_lib.CrontabError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from None

    _show(change)
    if not change.changed:
        console.print(f"Nothing to do: {change.reason}.")
        return
    if dry_run:
        console.print("[dim]--dry-run: your crontab was not touched.[/dim]")
        return
    if not yes and sys.stdin.isatty() and not typer.confirm("Apply this change?", default=True):
        console.print("Left your crontab alone.")
        raise typer.Exit(code=1)

    try:
        cron_lib.apply(change, _runner())
    except cron_lib.CrontabError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print(f"Installed. {change.reason.capitalize()}.")


@app.command("uninstall")
def uninstall(
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation.")] = False,
) -> None:
    """Remove only our line. Every other crontab entry is left alone."""
    try:
        change = cron_lib.plan_uninstall(_runner())
    except cron_lib.CrontabError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from None

    _show(change)
    if not change.changed:
        console.print(f"Nothing to do: {change.reason}.")
        return
    if not yes and sys.stdin.isatty() and not typer.confirm("Apply this change?", default=True):
        console.print("Left your crontab alone.")
        raise typer.Exit(code=1)

    try:
        cron_lib.apply(change, _runner())
    except cron_lib.CrontabError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print("Removed. Your other crontab entries were not touched.")


@app.command("status")
def status() -> None:
    """Is the line installed, and when did it last run?"""
    try:
        report = cron_lib.status(_runner())
    except cron_lib.CrontabError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    table = Table(title="cron", title_justify="left", box=None, pad_edge=False)
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("installed", "yes" if report.installed else "no")
    table.add_row("line", report.lines[0] if report.lines else "-")
    table.add_row(
        "last run",
        f"{report.last_run_at:%Y-%m-%d %H:%M %Z}" if report.last_run_at else "never",
    )
    table.add_row("last result", report.last_result or "-")
    console.print(table)

    if not report.installed:
        console.print()
        console.print(
            "Not installed. Run [bold]the-oracle cron install[/bold], or "
            "[bold]the-oracle cron line[/bold] to see the line first."
        )


@app.command("run")
def run(
    learner: Annotated[str | None, typer.Option(help="Learner id. Defaults to the current one.")] = None,
    send: Annotated[bool, typer.Option("--send/--no-send", help="Deliver, or only decide.")] = True,
) -> None:
    """The daily check. This is what the crontab line calls."""
    from the_oracle.context import LearnerContext
    from the_oracle.schedule import due_summary, run_due_checks
    from the_oracle.store.db import create_all, get_engine

    ctx = (
        LearnerContext.for_learner(learner) if learner else LearnerContext.resolve()
    )
    engine = create_all(get_engine())
    now = datetime.now(UTC)

    summary = due_summary(ctx.learner_id, now=now, engine=engine)
    try:
        decision = run_due_checks(ctx.learner_id, now=now, engine=engine, send=send)
    except ModuleNotFoundError as exc:  # the ladder is not installed
        cron_lib.record_run(at=now, result=f"error: {exc}")
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    result = (
        f"sent rung {int(decision.rung)}"
        if decision.send
        else f"quiet: {decision.suppressed_reason or decision.reason}"
    )
    cron_lib.record_run(at=now, result=result)

    console.print(
        f"{summary.due_objectives} due, idle {summary.idle_days:.1f} days. {result}."
    )


__all__ = ["app", "install", "line", "run", "status", "uninstall"]
