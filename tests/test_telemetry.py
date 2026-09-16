"""Telemetry is optional, silent by default, and never fatal."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from the_oracle import telemetry
from the_oracle.cli import app
from the_oracle.config import Settings

runner = CliRunner()


@pytest.fixture(autouse=True)
def _clean():
    telemetry.reset()
    yield
    telemetry.reset()


def test_disabled_by_settings_is_a_noop():
    status = telemetry.configure(Settings(telemetry=False))
    assert status.requested is False
    assert status.active is False
    assert "disabled" in status.describe()


def test_span_without_configure_yields_none():
    with telemetry.span("anything", a=1) as active:
        assert active is None
    telemetry.set_attributes(None, b=2)  # must not raise


def test_configure_is_cached():
    first = telemetry.configure(Settings(telemetry=False))
    second = telemetry.configure(Settings(telemetry=True))
    assert second is first


def test_no_token_means_nothing_is_sent(monkeypatch):
    monkeypatch.delenv("LOGFIRE_TOKEN", raising=False)
    status = telemetry.configure(Settings(telemetry=True))
    assert status.sending is False


def test_missing_logfire_is_reported_not_raised(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _no_logfire(name, *args, **kwargs):
        if name == "logfire":
            raise ImportError("no logfire")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_logfire)
    status = telemetry.configure(Settings(telemetry=True))
    assert status.installed is False
    assert status.active is False


def test_version_command_reports_telemetry():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "telemetry:" in result.stdout
