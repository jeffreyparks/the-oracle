"""Dedupe tests. Offline, free, deterministic, no API key.

The question these tests answer is not "does dedupe work". It is "can dedupe
merge two objectives that are not the same skill". That failure is silent and
permanent: mastery is keyed by ``(learner_id, objective_id)`` with no domain,
so a bad merge pollutes every domain that shares the id.

Every objective here is invented filler with no subject matter in it.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from the_oracle.agents.critic import SamenessJudge, SamenessRequest, SamenessVerdict
from the_oracle.domains import dedupe
from the_oracle.domains import embed as embedding
from the_oracle.domains.dedupe import (
    ADJUDICATE_THRESHOLD,
    LEXICAL_THRESHOLD_RAISE,
    REUSE_THRESHOLD,
    Decision,
    Match,
    match_all,
    match_one,
    resolve,
    thresholds_for,
)
from the_oracle.domains.embed import (
    CachedEmbedder,
    LexicalEmbedder,
    cosine,
    is_lexical,
)
from the_oracle.domains.schema import Objective


# --- fixtures and fakes ----------------------------------------------------


def make_objective(
    oid: str,
    title: str,
    description: str,
    *,
    bloom: str = "understand",
    difficulty: int = 3,
    tags: Sequence[str] = (),
) -> Objective:
    """A syntactically valid objective with no subject matter in it."""
    return Objective(
        id=oid,
        version=1,
        title=title,
        description=description,
        bloom=bloom,  # type: ignore[arg-type]
        difficulty=difficulty,
        est_minutes=30,
        assessment_stems=["State the rule.", "Apply the rule to one case."],
        tags=list(tags),
    )


class AngleEmbedder:
    """A fake embedder with a dial: every text after the first sits at a fixed
    cosine from the first. Lets a test put a score exactly where it wants it."""

    def __init__(self, score: float, name: str = "fake-dense", dim: int = 2) -> None:
        self.name = name
        self.dim = dim
        self.score = score
        self._angle = math.acos(max(-1.0, min(1.0, score)))
        self._seen: dict[str, list[float]] = {}
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        out: list[list[float]] = []
        for text in texts:
            if text not in self._seen:
                if not self._seen:
                    self._seen[text] = [1.0, 0.0]
                else:
                    self._seen[text] = [math.cos(self._angle), math.sin(self._angle)]
            out.append(self._seen[text])
        return out


class CountingEmbedder:
    """Wraps a real lexical embedder and counts how many texts it embedded."""

    def __init__(self, name: str = "counting-dense") -> None:
        self.inner = LexicalEmbedder()
        self.name = name
        self.dim = self.inner.dim
        self.texts_embedded = 0
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        self.texts_embedded += len(texts)
        return self.inner.embed(texts)


class CountingJudge:
    """Records every match handed to it and answers with a scripted verdict."""

    def __init__(self, verdict: SamenessVerdict | bool) -> None:
        self.verdict = verdict
        self.seen: list[Match] = []

    def __call__(self, match: Match) -> SamenessVerdict | bool:
        self.seen.append(match)
        return self.verdict


@pytest.fixture
def lexical() -> LexicalEmbedder:
    return LexicalEmbedder()


@pytest.fixture(autouse=True)
def _no_provider_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No key, no override, and a throwaway cache directory."""
    for key in ("OPENAI_API_KEY", "VOYAGE_API_KEY", "ORACLE_EMBEDDER", "ORACLE_EMBED_MODEL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ORACLE_HOME", str(tmp_path / "home"))


# --- similarity floor and ceiling -----------------------------------------


def test_identical_text_scores_one(lexical: LexicalEmbedder) -> None:
    """The same words at the same depth must score ~1.0 and reuse."""
    existing = make_objective(
        "obj_alpha",
        "Reading a Summary Table",
        "Read one summary table and state what each column reports.",
        tags=["tables", "reading"],
    )
    candidate = make_objective("draft_1", existing.title, existing.description,
                               tags=list(existing.tags))
    match = match_one(candidate, [existing], lexical)
    assert match.score == pytest.approx(1.0, abs=1e-6)
    assert match.decision is Decision.REUSE
    assert match.best_id == "obj_alpha"


def test_unrelated_text_scores_low(lexical: LexicalEmbedder) -> None:
    existing = make_objective(
        "obj_alpha",
        "Reading a Summary Table",
        "Read one summary table and state what each column reports.",
    )
    candidate = make_objective(
        "draft_1",
        "Negotiating a Delivery Schedule",
        "Agree a delivery date with a supplier and record the commitment.",
    )
    match = match_one(candidate, [existing], lexical)
    assert match.score < ADJUDICATE_THRESHOLD
    assert match.decision is Decision.MINT
    assert match.best_id is None


def test_vectors_are_unit_length_and_deterministic(lexical: LexicalEmbedder) -> None:
    text = "Read one summary table and state what each column reports."
    first = lexical.embed([text])[0]
    second = LexicalEmbedder().embed([text])[0]
    assert first == second
    assert math.sqrt(sum(x * x for x in first)) == pytest.approx(1.0)
    assert cosine(first, second) == pytest.approx(1.0)


# --- the three bands -------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.999, Decision.REUSE),
        (REUSE_THRESHOLD, Decision.REUSE),
        (REUSE_THRESHOLD - 0.001, Decision.ADJUDICATE),
        (ADJUDICATE_THRESHOLD, Decision.ADJUDICATE),
        (ADJUDICATE_THRESHOLD - 0.001, Decision.MINT),
        (0.0, Decision.MINT),
    ],
)
def test_three_decision_bands(score: float, expected: Decision) -> None:
    base = (REUSE_THRESHOLD, ADJUDICATE_THRESHOLD)
    assert dedupe.decide(score, base) is expected


def test_thresholds_are_a_function_of_the_embedder(lexical: LexicalEmbedder) -> None:
    """A dense model keeps the base bar; the lexical fallback must clear more."""
    dense = AngleEmbedder(0.5, name="local:some-dense-model")
    assert thresholds_for(dense) == (REUSE_THRESHOLD, ADJUDICATE_THRESHOLD)
    assert thresholds_for(lexical) == (
        REUSE_THRESHOLD + LEXICAL_THRESHOLD_RAISE,
        ADJUDICATE_THRESHOLD + LEXICAL_THRESHOLD_RAISE,
    )
    assert not is_lexical(dense)
    assert is_lexical(lexical)
    assert is_lexical(None)


def test_lexical_raise_changes_a_borderline_decision() -> None:
    """One score, two embedders, two different answers. 0.80 adjudicates on a
    dense model and mints on the fallback, because 0.80 < 0.78 + 0.05."""
    existing = make_objective("obj_alpha", "Alpha Task", "Do the alpha task.")
    candidate = make_objective("draft_1", "Alpha Duty", "Do the alpha duty.")
    borderline = 0.80
    assert ADJUDICATE_THRESHOLD < borderline < ADJUDICATE_THRESHOLD + LEXICAL_THRESHOLD_RAISE

    dense = match_one(candidate, [existing], AngleEmbedder(borderline, "local:dense"))
    fallback = match_one(
        candidate, [existing], AngleEmbedder(borderline, "lexical-fake")
    )

    assert dense.score == pytest.approx(borderline, abs=1e-6)
    assert fallback.score == pytest.approx(borderline, abs=1e-6)
    assert dense.decision is Decision.ADJUDICATE
    assert fallback.decision is Decision.MINT
    assert not dense.raised
    assert fallback.raised
    assert "raised" in fallback.explain()
    assert "lexical" in fallback.explain()


def test_match_all_reports_runners_up_in_order(lexical: LexicalEmbedder) -> None:
    library = [
        make_objective("obj_a", "Reading a Summary Table", "Read one summary table."),
        make_objective("obj_b", "Reading a Summary Chart", "Read one summary chart."),
        make_objective("obj_c", "Packing a Shipping Crate", "Pack one shipping crate."),
    ]
    candidate = make_objective("draft_1", "Reading a Summary Table", "Read one summary table.")
    match = match_all([candidate], library, lexical)[0]
    assert match.best_id == "obj_a"
    scores = [score for _, score in match.runners_up]
    assert scores == sorted(scores, reverse=True)
    assert match.runners_up[0][0] == "obj_b"


def test_match_all_on_an_empty_library_mints(lexical: LexicalEmbedder) -> None:
    candidate = make_objective("draft_1", "Alpha Task", "Do the alpha task.")
    match = match_all([candidate], [], lexical)[0]
    assert match.decision is Decision.MINT
    assert match.best_id is None
    assert match_all([], [], lexical) == []


# --- the failure mode that corrupts mastery --------------------------------


def test_same_words_at_different_depth_are_not_merged() -> None:
    """The adversarial case. Two objectives name the same technique with nearly
    the same words, but one is a survey and the other is expert work. Merging
    them would credit a beginner with expert mastery and vice versa.

    The embedder here is deliberately *fooled*: it reports 0.99 similarity. The
    depth guard must still refuse to reuse."""
    survey = make_objective(
        "obj_survey",
        "The Standard Transform: An Overview",
        "Recognise the standard transform and say when it is used.",
        bloom="remember",
        difficulty=1,
        tags=["standard", "transform"],
    )
    expert = make_objective(
        "draft_expert",
        "The Standard Transform: An Overview of Advanced Practice",
        "Derive the standard transform from first principles and judge when its "
        "guarantees fail.",
        bloom="evaluate",
        difficulty=5,
        tags=["standard", "transform"],
    )

    fooled = AngleEmbedder(0.99, name="local:dense-but-fooled")
    match = match_one(expert, [survey], fooled)
    assert match.decision is not Decision.REUSE, match.explain()
    assert resolve([match]) == {expert.title: None}

    # And the same verdict on the offline fallback, which sees the shared words.
    fallback = match_one(expert, [survey], LexicalEmbedder())
    assert fallback.decision is not Decision.REUSE, fallback.explain()

    # A judge that says "same" without confidence still does not merge.
    unsure = CountingJudge(
        SamenessVerdict(same_skill=True, confidence=0.3, reasoning="not sure")
    )
    assert resolve([match], unsure) == {expert.title: None}


def test_depth_penalty_grows_with_the_gap() -> None:
    shallow = make_objective("a", "T", "t", bloom="remember", difficulty=1)
    middle = make_objective("b", "T", "t", bloom="apply", difficulty=3)
    deep = make_objective("c", "T", "t", bloom="create", difficulty=5)
    assert dedupe.depth_penalty(shallow, shallow) == 0.0
    assert dedupe.depth_penalty(shallow, middle) < dedupe.depth_penalty(shallow, deep)
    assert dedupe.depth_penalty(shallow, deep) <= dedupe.MAX_DEPTH_PENALTY


# --- resolve and the judge -------------------------------------------------


def _match(decision: Decision, score: float) -> Match:
    return Match(
        candidate_title=f"candidate {decision.value}",
        decision=decision,
        best_id="obj_alpha" if decision is not Decision.MINT else None,
        score=score,
        runners_up=[],
        embedder_name="local:dense",
        thresholds=(REUSE_THRESHOLD, ADJUDICATE_THRESHOLD),
    )


def test_resolve_consults_the_judge_only_in_the_middle_band() -> None:
    matches = [
        _match(Decision.REUSE, 0.97),
        _match(Decision.ADJUDICATE, 0.84),
        _match(Decision.MINT, 0.10),
    ]
    judge = CountingJudge(SamenessVerdict(same_skill=True, confidence=0.9, reasoning="same"))
    out = resolve(matches, judge)

    assert [m.decision for m in judge.seen] == [Decision.ADJUDICATE]
    assert out["candidate reuse"] == "obj_alpha"
    assert out["candidate adjudicate"] == "obj_alpha"
    assert out["candidate mint"] is None


def test_an_unsure_judge_mints_rather_than_reuses() -> None:
    match = _match(Decision.ADJUDICATE, 0.84)
    for verdict in (
        SamenessVerdict(same_skill=False, confidence=0.9, reasoning="different depth"),
        SamenessVerdict(same_skill=False, confidence=0.2, reasoning="unsure"),
        SamenessVerdict(same_skill=True, confidence=0.2, reasoning="maybe"),
    ):
        assert resolve([match], CountingJudge(verdict)) == {match.candidate_title: None}

    confident = SamenessVerdict(same_skill=True, confidence=0.95, reasoning="same skill")
    assert resolve([match], CountingJudge(confident)) == {match.candidate_title: "obj_alpha"}


def test_no_judge_means_split() -> None:
    assert resolve([_match(Decision.ADJUDICATE, 0.84)]) == {"candidate adjudicate": None}


def test_a_broken_judge_never_merges() -> None:
    def explode(match: Match) -> SamenessVerdict:
        raise RuntimeError("model unavailable")

    assert resolve([_match(Decision.ADJUDICATE, 0.84)], explode) == {
        "candidate adjudicate": None
    }


# --- the Critic ------------------------------------------------------------


def test_the_judge_instruction_states_the_split_rule() -> None:
    role = SamenessJudge.role.casefold()
    assert "depth" in role
    assert "audience" in role
    assert "same_skill=false" in role
    assert "unsure" in role


def test_sameness_prompt_carries_depth_signals() -> None:
    survey = make_objective("a", "T", "t", bloom="remember", difficulty=1)
    expert = make_objective("b", "T", "t", bloom="evaluate", difficulty=5)
    prompt = SamenessJudge.prompt_for(SamenessRequest(candidate=expert, existing=survey))
    assert "bloom: remember" in prompt
    assert "bloom: evaluate" in prompt
    assert "difficulty: 1/5" in prompt
    assert "difficulty: 5/5" in prompt


# --- the embedding cache ---------------------------------------------------


def test_cache_avoids_a_second_call(tmp_path: Path) -> None:
    inner = CountingEmbedder()
    cache_dir = tmp_path / "embeddings"
    cached = CachedEmbedder(inner, cache_dir)
    texts = ["first text", "second text"]

    first = cached.embed(texts)
    assert inner.texts_embedded == 2

    second = cached.embed(texts)
    assert second == first
    assert inner.texts_embedded == 2, "memory cache did not hold"

    # A brand new wrapper with an empty memory still pays nothing: the vectors
    # are on disk under (embedder name, sha256(text)).
    fresh_inner = CountingEmbedder()
    fresh = CachedEmbedder(fresh_inner, cache_dir)
    assert fresh.embed(texts) == first
    assert fresh_inner.texts_embedded == 0

    # Only the new text is embedded on a partial hit.
    assert fresh.embed([*texts, "third text"])
    assert fresh_inner.texts_embedded == 1


def test_cache_is_keyed_by_embedder_name(tmp_path: Path) -> None:
    cache_dir = tmp_path / "embeddings"
    one = CountingEmbedder(name="embedder-one")
    two = CountingEmbedder(name="embedder-two")
    CachedEmbedder(one, cache_dir).embed(["shared text"])
    CachedEmbedder(two, cache_dir).embed(["shared text"])
    assert one.texts_embedded == 1
    assert two.texts_embedded == 1, "a different embedder must not read that cache"


def test_cache_survives_a_corrupt_file(tmp_path: Path) -> None:
    cache_dir = tmp_path / "embeddings"
    inner = CountingEmbedder()
    cached = CachedEmbedder(inner, cache_dir)
    cached.embed(["first text"])
    for path in cache_dir.rglob("*.json"):
        path.write_text("not json", encoding="utf-8")
    fresh = CachedEmbedder(CountingEmbedder(), cache_dir)
    assert len(fresh.embed(["first text"])[0]) == inner.dim


# --- provider selection ----------------------------------------------------


def test_fallback_is_chosen_and_logged_loudly(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(embedding, "local_available", lambda: False)
    with caplog.at_level(logging.WARNING, logger="the_oracle.domains.embed"):
        chosen = embedding.get_embedder()
    assert is_lexical(chosen)
    assert "falling back" in caplog.text
    assert "degraded" in embedding.describe(chosen)


def test_a_provider_key_beats_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embedding, "local_available", lambda: False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
    chosen = embedding.get_embedder()
    assert not is_lexical(chosen)
    assert chosen.name.startswith("openai:")


def test_the_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORACLE_EMBEDDER", "lexical")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
    assert is_lexical(embedding.get_embedder())


def test_local_embedder_when_installed() -> None:
    """Exercised only on a machine that has sentence-transformers."""
    pytest.importorskip(
        "sentence_transformers", reason="optional local embedding extra is not installed"
    )
    local = embedding.LocalEmbedder()
    vectors = local.embed(["first text", "first text", "a completely different line"])
    assert local.dim > 0
    assert not is_lexical(local)
    assert thresholds_for(local) == (REUSE_THRESHOLD, ADJUDICATE_THRESHOLD)
    assert cosine(vectors[0], vectors[1]) == pytest.approx(1.0, abs=1e-4)
    assert cosine(vectors[0], vectors[2]) < 0.99


# --- tag backfill ----------------------------------------------------------


def test_backfill_tags_is_deterministic_and_bounded() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import backfill_tags

    title = "Reading a Summary Table Under Time Pressure"
    description = "Read one summary table quickly and state what each column reports."
    first = backfill_tags.tags_for(title, description)
    assert first == backfill_tags.tags_for(title, description)
    assert 0 < len(first) <= backfill_tags.MAX_TAGS
    assert "the" not in first and "one" not in first
    assert all(tag == tag.casefold() for tag in first)
    assert "summary" in first
