"""Scout and fetch, offline. No network, no key, no model call."""

from __future__ import annotations

import httpx
import pytest

from the_oracle.agents import scout as scout_mod
from the_oracle.agents.scout import (
    Candidate,
    ScoutRequest,
    ScoutUnavailableError,
    build_queries,
    level_band,
    normalise_url,
    search_candidates,
)
from the_oracle.corpus import fetch as fetch_mod
from the_oracle.corpus.fetch import MAX_BYTES, Fetched, fetch, fetch_many

HTML = """<html><head><title>Bayes for beginners</title></head><body>
<article><h1>Bayes for beginners</h1>
<p>Bayes' theorem updates a prior belief with new evidence. You start with a
prior, you observe data, and you land on a posterior that reflects both. The
denominator is the marginal likelihood, which normalises the result so the
posterior is a genuine probability distribution over the hypotheses.</p>
<p>Work an example. A test is ninety-nine percent accurate, the disease is rare,
and the posterior probability of disease given a positive test is still low.
That surprises people, and it is the whole point of the theorem.</p>
</article></body></html>"""


@pytest.fixture(autouse=True)
def _no_politeness_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the host throttle logic, drop the wall-clock cost, in tests only."""
    monkeypatch.setattr(fetch_mod, "MIN_HOST_INTERVAL", 0.0)
    fetch_mod._host_next_free.clear()


@pytest.fixture(autouse=True)
def _clear_robots_cache() -> None:
    """robots.txt is cached per process; never let it leak between tests."""
    fetch_mod._ROBOTS.clear()
    yield
    fetch_mod._ROBOTS.clear()


def test_robots_disallow_blocks_the_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit Disallow is consent withheld, so we do not fetch."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private/")
        return httpx.Response(200, text=HTML, headers={"content-type": "text/html"})

    install_transport(monkeypatch, handler)
    blocked = fetch_mod.fetch("https://example.com/private/page")
    assert blocked.error == "blocked by robots.txt"
    assert blocked.text == ""

    allowed = fetch_mod.fetch("https://example.com/public/page")
    assert allowed.error is None


def test_missing_robots_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing robots.txt is not consent withheld."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404, text="")
        return httpx.Response(200, text=HTML, headers={"content-type": "text/html"})

    install_transport(monkeypatch, handler)
    assert fetch_mod.fetch("https://example.com/page").error is None


def test_robots_is_fetched_once_per_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """The parsed file is cached, so a module of pages costs one robots GET."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /")
        return httpx.Response(200, text=HTML, headers={"content-type": "text/html"})

    install_transport(monkeypatch, handler)
    for n in range(4):
        fetch_mod.fetch(f"https://example.com/page-{n}")
    assert calls.count("/robots.txt") == 1


def test_robots_can_be_bypassed_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    """The override exists, is off by default, and must be opt-in."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /")
        return httpx.Response(200, text=HTML, headers={"content-type": "text/html"})

    install_transport(monkeypatch, handler)
    assert fetch_mod.fetch("https://example.com/page").error == "blocked by robots.txt"
    monkeypatch.setenv(fetch_mod.IGNORE_ROBOTS_ENV, "1")
    assert fetch_mod.fetch("https://example.com/page").error is None


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERPER_API_KEY", "test-key-not-real")


def make_request(**overrides: object) -> ScoutRequest:
    payload: dict[str, object] = {
        "objective_id": "obj-1",
        "title": "Bayes theorem",
        "description": "State and use Bayes' theorem.",
        "bloom": "understand",
        "difficulty": 2,
        "tags": ["probability", "statistics"],
    }
    payload.update(overrides)
    return ScoutRequest(**payload)  # type: ignore[arg-type]


def serper_payload(*links: str) -> dict[str, object]:
    return {
        "organic": [
            {"link": link, "title": f"Page {i}", "snippet": "snippet"}
            for i, link in enumerate(links, start=1)
        ]
    }


def stub_search(monkeypatch: pytest.MonkeyPatch, responses: object) -> list[str]:
    """Replace the Serper call. ``responses`` is a dict, list, or callable."""
    seen: list[str] = []

    def fake(query: str, *, api_key: str, num: int = 10, timeout: float = 15.0):
        seen.append(query)
        if callable(responses):
            return responses(query)
        if isinstance(responses, dict):
            return responses.get(query, {"organic": []}).get("organic", [])
        index = min(len(seen) - 1, len(responses) - 1)
        return responses[index].get("organic", [])

    monkeypatch.setattr(scout_mod, "serper_search", fake)
    return seen


# --------------------------------------------------------------------------
# queries and level hints


def test_level_band_spans_intro_to_deep() -> None:
    assert level_band("remember", 1) == "intro"
    assert level_band("apply", 3) == "working"
    assert level_band("evaluate", 5) == "deep"


def test_query_variants_differ_by_level() -> None:
    easy = build_queries(make_request(bloom="remember", difficulty=1))
    hard = build_queries(make_request(bloom="evaluate", difficulty=5))

    assert 2 <= len(easy) <= 3
    assert 2 <= len(hard) <= 3
    assert easy != hard
    assert set(easy).isdisjoint(hard)
    assert any("introduction" in q for q in easy)
    assert any("beginner" in q for q in easy)
    assert any("in depth" in q or "advanced" in q for q in hard)
    assert not any("beginner" in q for q in hard)
    assert all(q.startswith("Bayes theorem") for q in easy + hard)


def test_queries_use_tags_and_are_unique() -> None:
    queries = build_queries(make_request())
    assert any("probability" in q for q in queries)
    assert len(set(queries)) == len(queries)


def test_queries_survive_missing_tags() -> None:
    queries = build_queries(make_request(tags=[]))
    assert queries
    assert all("Bayes theorem" in q for q in queries)


def test_unknown_bloom_verb_lands_mid_band() -> None:
    assert level_band("wondering", 2) == "working"


# --------------------------------------------------------------------------
# url normalisation and dedupe


@pytest.mark.parametrize(
    "variant",
    [
        "http://example.com/a/b",
        "https://example.com/a/b",
        "https://www.example.com/a/b",
        "https://example.com/a/b/",
        "https://example.com/a/b?utm_source=twitter&utm_campaign=x",
        "https://EXAMPLE.com/a/b#section",
        "https://www.example.com/a/b/?utm_medium=email",
    ],
)
def test_normalise_url_collapses_same_page(variant: str) -> None:
    assert normalise_url(variant) == normalise_url("https://example.com/a/b")


def test_normalise_url_keeps_meaningful_query() -> None:
    a = normalise_url("https://example.com/watch?v=abc&utm_source=x")
    b = normalise_url("https://example.com/watch?v=def")
    assert a != b
    assert a == normalise_url("http://www.example.com/watch?v=abc")


def test_search_dedupes_across_query_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    same_page = [
        "http://example.com/a/b",
        "https://www.example.com/a/b/",
        "https://example.com/a/b?utm_source=news",
    ]
    stub_search(
        monkeypatch,
        lambda q: serper_payload(*same_page, "https://other.org/x")["organic"],
    )
    results = search_candidates(make_request())
    assert [c.url for c in results] == ["http://example.com/a/b", "https://other.org/x"]
    assert results[0].source == "example.com"


def test_search_respects_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    links = [f"https://site{i}.com/page" for i in range(30)]
    stub_search(monkeypatch, lambda q: serper_payload(*links)["organic"])
    assert len(search_candidates(make_request())) == 10
    assert len(search_candidates(make_request(), limit=3)) == 3
    assert search_candidates(make_request(), limit=0) == []


def test_search_runs_every_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = stub_search(monkeypatch, lambda q: [])
    search_candidates(make_request())
    assert seen == build_queries(make_request())


def test_search_skips_junk_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_search(
        monkeypatch,
        lambda q: [
            {"title": "no link"},
            {"link": "ftp://example.com/file"},
            {"link": "https://good.example/page", "title": "Good"},
        ],
    )
    results = search_candidates(make_request())
    assert [c.url for c in results] == ["https://good.example/page"]


def test_missing_key_raises_rather_than_returning_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    called: list[str] = []
    monkeypatch.setattr(
        scout_mod,
        "serper_search",
        lambda *a, **k: called.append("nope") or [],  # type: ignore[func-returns-value]
    )
    with pytest.raises(ScoutUnavailableError):
        search_candidates(make_request())
    assert called == []


def test_blank_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERPER_API_KEY", "   ")
    with pytest.raises(ScoutUnavailableError):
        search_candidates(make_request())


def test_total_search_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(query: str, **kwargs: object) -> list[dict[str, object]]:
        raise ScoutUnavailableError("quota exhausted")

    monkeypatch.setattr(scout_mod, "serper_search", boom)
    with pytest.raises(ScoutUnavailableError):
        search_candidates(make_request())


def test_partial_search_failure_still_returns_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def flaky(query: str, **kwargs: object) -> list[dict[str, object]]:
        calls.append(query)
        if len(calls) == 1:
            raise ScoutUnavailableError("transient")
        return serper_payload("https://ok.example/p")["organic"]

    monkeypatch.setattr(scout_mod, "serper_search", flaky)
    results = search_candidates(make_request())
    assert [c.url for c in results] == ["https://ok.example/p"]


def test_candidate_is_a_plain_model() -> None:
    c = Candidate(url="https://a.example/", title="A")
    assert c.rank == 0 and c.snippet == "" and c.source == ""


# --------------------------------------------------------------------------
# fetch


def install_transport(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    """Route every httpx.Client through a MockTransport. No sockets."""
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def client(*args: object, **kwargs: object) -> httpx.Client:
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx, "Client", client)


def html_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, html=HTML)


def test_fetch_extracts_text(monkeypatch: pytest.MonkeyPatch) -> None:
    install_transport(monkeypatch, html_response)
    result = fetch("https://example.com/bayes")
    assert result.error is None
    assert result.status == 200
    assert result.word_count > 30
    assert "Bayes" in result.text
    assert result.final_url == "https://example.com/bayes"


def test_fetch_records_final_url_after_redirect(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(301, headers={"Location": "https://example.com/new"})
        return httpx.Response(200, html=HTML)

    install_transport(monkeypatch, handler)
    result = fetch("https://example.com/old")
    assert result.error is None
    assert result.url == "https://example.com/old"
    assert result.final_url == "https://example.com/new"


def test_fetch_404_returns_error_not_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    install_transport(monkeypatch, lambda r: httpx.Response(404, text="gone"))
    result = fetch("https://example.com/missing")
    assert isinstance(result, Fetched)
    assert result.status == 404
    assert result.error == "http 404"
    assert result.text == ""


def test_fetch_timeout_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("too slow", request=request)

    install_transport(monkeypatch, handler)
    result = fetch("https://slow.example/page", timeout=0.01)
    assert result.error is not None
    assert "ConnectTimeout" in result.error
    assert result.status == 0


def test_fetch_rejects_non_html(monkeypatch: pytest.MonkeyPatch) -> None:
    install_transport(
        monkeypatch,
        lambda r: httpx.Response(200, content=b"%PDF-1.7", headers={"content-type": "application/pdf"}),
    )
    result = fetch("https://example.com/paper.pdf")
    assert result.error is not None
    assert "content-type" in result.error


def test_fetch_rejects_declared_oversize(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"<html></html>",
            headers={
                "content-type": "text/html",
                "content-length": str(MAX_BYTES + 1),
            },
        )

    install_transport(monkeypatch, handler)
    result = fetch("https://example.com/huge")
    assert result.error is not None
    assert "too large" in result.error


def test_fetch_rejects_streamed_oversize(monkeypatch: pytest.MonkeyPatch) -> None:
    big = b"<html><body>" + (b"padding " * (MAX_BYTES // 4)) + b"</body></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=big, headers={"content-type": "text/html"})

    install_transport(monkeypatch, handler)
    result = fetch("https://example.com/huge2")
    assert result.error is not None
    assert "too large" in result.error


def test_fetch_reports_empty_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    install_transport(monkeypatch, lambda r: httpx.Response(200, html="<html><body></body></html>"))
    result = fetch("https://example.com/blank")
    assert result.error == "no extractable text"


def test_fetch_rejects_bad_scheme_without_network() -> None:
    assert fetch("ftp://example.com/x").error is not None
    assert fetch("").error == "empty url"
    assert fetch("   ").error == "empty url"


def test_fetch_many_preserves_order_and_isolates_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/bad":
            return httpx.Response(500, text="boom")
        if path == "/boom":
            raise httpx.ReadError("socket died")
        return httpx.Response(200, html=HTML)

    install_transport(monkeypatch, handler)
    urls = [
        "https://a.example/one",
        "https://b.example/bad",
        "https://c.example/boom",
        "https://d.example/two",
    ]
    results = fetch_many(urls, max_workers=4)

    assert [r.url for r in results] == urls
    assert results[0].error is None
    assert results[1].error == "http 500"
    assert results[2].error is not None
    assert results[3].error is None
    assert results[3].word_count > 30


def test_fetch_many_empty_input() -> None:
    assert fetch_many([]) == []


def test_fetched_is_frozen() -> None:
    f = Fetched(url="u", final_url="u", status=200, title="t", text="x", word_count=1)
    assert f.ok
    with pytest.raises(Exception):
        f.status = 404  # type: ignore[misc]
