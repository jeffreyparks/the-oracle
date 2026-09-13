"""Privacy tests: export is complete and JSON-safe, delete is surgical.

The rule under test is simple. One learner's erasure removes every row that
belongs to that learner, and nothing else -- not another learner's rows, not
the shared corpus.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import inspect as sa_inspect
from sqlmodel import Session, select
from typer.testing import CliRunner

from the_oracle.commands.learner import app as learner_app
from the_oracle.store import models, privacy
from the_oracle.store.db import build_engine, create_all
from the_oracle.store.events import EventKind, EventLog

MINE = "learner_mine"
THEIRS = "learner_theirs"

SHARED_TABLES = (
    models.Objective,
    models.Resource,
    models.Item,
)


@pytest.fixture
def engine():
    eng = build_engine("sqlite:///:memory:")
    create_all(eng)
    yield eng
    eng.dispose()


def _seed_learner(engine, learner_id: str) -> None:
    """Write one row into every learner-scoped table we can populate."""
    log = EventLog(engine)
    log.append(EventKind.LEARNER_CREATED, learner_id, {"id": learner_id})
    log.append(EventKind.RESPONSE_GRADED, learner_id, {"objective_id": "obj_a", "correct": True})
    log.append(EventKind.SESSION_ENDED, learner_id, {"minutes": 25})

    with Session(engine) as session:
        session.add(models.Learner(id=learner_id, display_name=learner_id, preferences={"tone": "plain"}))
        session.add(models.Enrollment(learner_id=learner_id, domain_id="dom_a"))
        session.add(models.Goal(learner_id=learner_id, statement="a stated goal"))
        session.add(models.MasteryState(learner_id=learner_id, objective_id="obj_a", p_mastery=0.7))
        session.add(models.StudySession(id=f"sess_{learner_id}", learner_id=learner_id))
        session.add(models.Response(learner_id=learner_id, objective_id="obj_a", correct=True))
        session.add(models.ReviewSchedule(learner_id=learner_id, objective_id="obj_a"))
        session.add(models.Nudge(learner_id=learner_id, rung=1))
        session.commit()


def _seed_shared(engine) -> None:
    with Session(engine) as session:
        session.add(models.Objective(id="obj_a", title="t", description="d"))
        session.add(models.Resource(id="res_a", objective_id="obj_a"))
        session.add(models.Item(id="item_a", objective_id="obj_a", stem="s"))
        session.commit()


@pytest.fixture
def seeded(engine):
    _seed_shared(engine)
    _seed_learner(engine, MINE)
    _seed_learner(engine, THEIRS)
    return engine


def _count(engine, model) -> int:
    with Session(engine) as session:
        return len(session.exec(select(model)).all())


# -- the table list -------------------------------------------------------


def test_table_list_is_derived_from_metadata(engine):
    """Not a hand-written list. Every mapped learner column shows up."""
    names = privacy.learner_tables()

    # Named via the model, never as a literal, so a rename cannot pass silently.
    assert models.MasteryState.__tablename__ in names
    assert models.EventLog.__tablename__ in names
    assert models.Learner.__tablename__ in names

    expected = {
        table.name
        for table in models.SQLModel.metadata.tables.values()
        if "learner_id" in table.columns
    }
    assert expected <= set(names)


def test_shared_tables_are_not_learner_scoped(engine):
    names = set(privacy.learner_tables())
    for model in SHARED_TABLES:
        assert model.__tablename__ not in names


def test_every_listed_table_really_exists(engine):
    live = set(sa_inspect(engine).get_table_names())
    assert set(privacy.learner_tables()) <= live


# -- export ---------------------------------------------------------------


def test_export_covers_every_learner_table(seeded):
    payload = privacy.export_learner(MINE, seeded)
    assert set(payload["tables"]) == set(privacy.learner_tables())
    for name, rows in payload["tables"].items():
        assert rows, f"{name} exported empty; the seed or the query is wrong"


def test_export_includes_the_full_event_log(seeded):
    payload = privacy.export_learner(MINE, seeded)
    logged = EventLog(seeded).read(MINE)

    assert len(payload["events"]) == len(logged) == 3
    assert [row["kind"] for row in payload["events"]] == [event.kind for event in logged]
    assert payload["events"] == payload["tables"][models.EventLog.__tablename__]


def test_export_is_json_serialisable(seeded):
    payload = privacy.export_learner(MINE, seeded)
    text = json.dumps(payload)
    assert json.loads(text) == payload


def test_export_renders_datetimes_as_iso_strings(seeded):
    payload = privacy.export_learner(MINE, seeded)
    row = payload["tables"][models.EventLog.__tablename__][0]
    assert isinstance(row["occurred_at"], str)
    assert row["occurred_at"].startswith("20")


def test_export_holds_only_that_learner(seeded):
    payload = privacy.export_learner(MINE, seeded)
    text = json.dumps(payload)
    assert THEIRS not in text


def test_export_of_unknown_learner_is_empty_not_an_error(seeded):
    payload = privacy.export_learner("nobody", seeded)
    assert payload["learner_id"] == "nobody"
    assert set(payload["tables"]) == set(privacy.learner_tables())
    assert sum(payload["counts"].values()) == 0


# -- delete ---------------------------------------------------------------


def test_delete_removes_that_learner(seeded):
    counts = privacy.delete_learner(MINE, seeded)

    assert set(counts) == set(privacy.learner_tables())
    assert sum(counts.values()) > 0
    after = privacy.export_learner(MINE, seeded)
    assert sum(after["counts"].values()) == 0


def test_delete_removes_the_event_log_rows(seeded):
    privacy.delete_learner(MINE, seeded)
    assert EventLog(seeded).count(MINE) == 0


def test_delete_leaves_the_other_learner_untouched(seeded):
    before = privacy.export_learner(THEIRS, seeded)
    privacy.delete_learner(MINE, seeded)
    after = privacy.export_learner(THEIRS, seeded)
    assert after["tables"] == before["tables"]
    assert sum(after["counts"].values()) > 0


def test_delete_leaves_shared_tables_untouched(seeded):
    before = {model.__tablename__: _count(seeded, model) for model in SHARED_TABLES}
    privacy.delete_learner(MINE, seeded)
    after = {model.__tablename__: _count(seeded, model) for model in SHARED_TABLES}
    assert after == before
    assert all(value == 1 for value in after.values())


def test_delete_of_unknown_learner_is_a_clean_no_op(seeded):
    counts = privacy.delete_learner("nobody", seeded)
    assert set(counts) == set(privacy.learner_tables())
    assert sum(counts.values()) == 0
    assert EventLog(seeded).count(MINE) == 3


def test_delete_is_idempotent(seeded):
    first = privacy.delete_learner(MINE, seeded)
    second = privacy.delete_learner(MINE, seeded)
    assert sum(first.values()) > 0
    assert sum(second.values()) == 0


def test_event_log_api_still_refuses_deletion(seeded):
    """Erasure is a SQL-layer exception. The API rule is unchanged."""
    from the_oracle.store.events import ImmutableEventLogError

    with pytest.raises(ImmutableEventLogError):
        EventLog(seeded).delete(1)


# -- the cli --------------------------------------------------------------


@pytest.fixture
def cli(seeded, monkeypatch):
    monkeypatch.setattr(privacy, "get_engine", lambda *a, **k: seeded)
    return CliRunner()


def test_cli_export_writes_a_file(cli, tmp_path):
    out = tmp_path / "nested" / "export.json"
    result = cli.invoke(learner_app, ["export", MINE, "--out", str(out)])
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text())
    assert payload["learner_id"] == MINE
    assert payload["events"]


def test_cli_export_to_stdout_is_json(cli):
    result = cli.invoke(learner_app, ["export", MINE])
    assert result.exit_code == 0, result.output
    assert MINE in result.output


def test_cli_delete_needs_confirmation(cli):
    result = cli.invoke(learner_app, ["delete", MINE], input="n\n")
    assert result.exit_code == 1
    assert "cannot be undone" in result.output
    assert EventLog(privacy.get_engine()).count(MINE) == 3


def test_cli_delete_yes_skips_the_prompt_and_reports_counts(cli):
    result = cli.invoke(learner_app, ["delete", MINE, "--yes"])
    assert result.exit_code == 0, result.output
    assert models.MasteryState.__tablename__ in result.output
    assert EventLog(privacy.get_engine()).count(MINE) == 0


def test_cli_delete_of_unknown_learner_says_nothing_changed(cli):
    result = cli.invoke(learner_app, ["delete", "nobody", "--yes"])
    assert result.exit_code == 0, result.output
    assert "Nothing changed." in result.output
