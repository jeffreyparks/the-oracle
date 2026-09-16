"""``the-oracle init`` - copy the seed packs into your data directory.

The repo ships SEED content under ``data/packs``: objective bodies and domain
manifests, and nothing else. Your progress, your generated domains, your
embedding cache and your database belong in ``ORACLE_HOME`` (``~/.the-oracle``
by default).

Before this command existed the README told people to point ``ORACLE_HOME`` at
the checkout, which wrote runtime data into a git repository. Copy once, then
leave the repo alone.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

console = Console()

app = typer.Typer(name="init", help="Copy the seed packs into your data directory.")

#: Subdirectories that are shipped content. Anything else is runtime data.
SEED_DIRS: tuple[str, ...] = ("objectives", "domains")


def seed_root() -> Path | None:
    """Locate ``data/packs`` in the checkout, if this is a source install."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "data" / "packs"
        if (candidate / "objectives").is_dir():
            return candidate
    return None


def copy_seed(source: Path, target: Path, *, force: bool = False) -> tuple[int, int]:
    """Copy seed files. Returns ``(copied, skipped)``. Never overwrites silently."""
    copied = skipped = 0
    for name in SEED_DIRS:
        src_dir = source / name
        if not src_dir.is_dir():
            continue
        dst_dir = target / name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(src_dir.glob("*.yaml")):
            dst = dst_dir / src.name
            if dst.exists() and not force:
                skipped += 1
                continue
            shutil.copy2(src, dst)
            copied += 1
    return copied, skipped


@app.callback(invoke_without_command=True)
def init(
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite packs that already exist.")
    ] = False,
) -> None:
    """Copy the packs shipped with this repo into your data directory."""
    from the_oracle.config import get_settings

    settings = get_settings()
    target = Path(settings.home).expanduser()

    source = seed_root()
    if source is None:
        console.print("No seed packs found. This looks like a non-source install.")
        console.print("Create your own subject with [bold]the-oracle domain add[/bold].")
        raise typer.Exit(code=1)

    if source.resolve() == target.resolve():
        console.print(
            "[yellow]ORACLE_HOME points at the seed packs inside the repo.[/yellow]"
        )
        console.print(
            "Unset it, or set it somewhere outside the checkout, so your progress "
            "and generated subjects are not written into git."
        )
        raise typer.Exit(code=1)

    copied, skipped = copy_seed(source, target, force=force)
    console.print(f"Copied {copied} pack files into {target}.")
    if skipped:
        console.print(
            f"Left {skipped} existing file(s) alone. Use --force to overwrite."
        )
    console.print("\nNext: [bold]the-oracle domain list[/bold]")
