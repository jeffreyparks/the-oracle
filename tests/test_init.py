"""Seed packs are shipped content; runtime data must never land in the repo."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from the_oracle.cli import app as cli_app
from the_oracle.commands.init import SEED_DIRS, copy_seed, seed_root

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("ORACLE_HOME", str(tmp_path))
    from the_oracle.config import get_settings

    get_settings.cache_clear()
    return tmp_path


def test_init_copies_the_seed_packs(home) -> None:
    result = CliRunner().invoke(cli_app, ["init"])
    assert result.exit_code == 0, result.output
    assert len(list((home / "objectives").glob("*.yaml"))) == 70
    assert (home / "domains" / "bayesian_forecasting.yaml").exists()


def test_init_does_not_overwrite_without_force(home) -> None:
    runner = CliRunner()
    runner.invoke(cli_app, ["init"])
    target = home / "domains" / "bayesian_forecasting.yaml"
    target.write_text("id: edited\n", encoding="utf-8")

    runner.invoke(cli_app, ["init"])
    assert target.read_text(encoding="utf-8") == "id: edited\n"

    runner.invoke(cli_app, ["init", "--force"])
    assert "edited" not in target.read_text(encoding="utf-8")


def test_init_refuses_when_home_is_the_repo(monkeypatch) -> None:
    """Pointing ORACLE_HOME at the checkout is what put runtime data in git."""
    monkeypatch.setenv("ORACLE_HOME", str(REPO_ROOT / "data" / "packs"))
    from the_oracle.config import get_settings

    get_settings.cache_clear()
    result = CliRunner().invoke(cli_app, ["init"])
    assert result.exit_code == 1
    assert "inside the repo" in result.output


def test_copy_seed_only_copies_shipped_directories(home) -> None:
    """Only objectives and domains ship. Caches and databases never do."""
    source = seed_root()
    assert source is not None
    copy_seed(source, home)
    assert sorted(p.name for p in home.iterdir() if p.is_dir()) == sorted(SEED_DIRS)


def test_no_runtime_data_is_tracked_by_git() -> None:
    """A database or embedding cache must never be committed again."""
    out = subprocess.run(
        ["git", "ls-files", "data/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert out, "expected the seed packs to be tracked"
    for path in out:
        assert not path.endswith((".db", ".sqlite3")), path
        assert "/embeddings/" not in path, path
        assert path.startswith(tuple(f"data/packs/{d}/" for d in SEED_DIRS)), path
