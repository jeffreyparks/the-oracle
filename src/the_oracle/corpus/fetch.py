"""Fetch a candidate page and extract its readable text.

httpx for transport, trafilatura for extraction. No JS rendering, no retries,
no crawling: one GET per URL, politely, and never an exception for the caller.
A bad URL comes back as a :class:`Fetched` with ``error`` set.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Final, Sequence
from urllib.parse import urlsplit

MAX_BYTES: Final[int] = 2 * 1024 * 1024
"""2 MB cap. A page bigger than this is not a lesson, it is a download."""

ALLOWED_CONTENT_TYPES: Final[tuple[str, ...]] = (
    "text/html",
    "application/xhtml+xml",
    "text/plain",
)

USER_AGENT: Final[str] = (
    "the-oracle/0.1 (learning-resource fetcher; +https://github.com/jeffreyparks/the-oracle)"
)

DEFAULT_HEADERS: Final[dict[str, str]] = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml;q=0.9,text/plain;q=0.8,*/*;q=0.1",
    "Accept-Language": "en",
}

MIN_HOST_INTERVAL: Final[float] = 0.5
"""Seconds between two requests to the same host. Cheap politeness."""

_host_lock = threading.Lock()
_host_next_free: dict[str, float] = {}


@dataclass(frozen=True)
class Fetched:
    """One fetch attempt. ``error`` is None only when text was extracted."""

    url: str
    final_url: str
    status: int
    title: str
    text: str
    word_count: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _failure(url: str, error: str, *, final_url: str = "", status: int = 0) -> Fetched:
    return Fetched(
        url=url,
        final_url=final_url or url,
        status=status,
        title="",
        text="",
        word_count=0,
        error=error,
    )


def _throttle(url: str) -> None:
    """Space out requests to one host. Never blocks a different host."""
    import time

    host = (urlsplit(url).hostname or "").lower()
    if not host:
        return
    with _host_lock:
        now = time.monotonic()
        wait = max(0.0, _host_next_free.get(host, 0.0) - now)
        _host_next_free[host] = max(now, _host_next_free.get(host, 0.0)) + MIN_HOST_INTERVAL
    if wait > 0:
        time.sleep(min(wait, MIN_HOST_INTERVAL * 4))


def _extract(html: str, url: str) -> tuple[str, str]:
    """Return ``(title, text)`` from HTML. Empty strings when nothing useful."""
    import trafilatura

    text = ""
    title = ""
    try:
        text = trafilatura.extract(html, url=url, include_comments=False, include_tables=True) or ""
    except Exception:
        text = ""
    try:
        meta = trafilatura.extract_metadata(html)
        title = (getattr(meta, "title", "") or "") if meta is not None else ""
    except Exception:
        title = ""
    return title.strip(), text.strip()


#: Per-host robots.txt, parsed once and cached for the process.
_ROBOTS: dict[str, Any] = {}
_ROBOTS_LOCK = threading.Lock()

#: Set ``ORACLE_IGNORE_ROBOTS=1`` to bypass. Off by default, and it stays off in
#: anything a learner runs.
IGNORE_ROBOTS_ENV: Final[str] = "ORACLE_IGNORE_ROBOTS"


def robots_allows(url: str, *, timeout: float = 5.0) -> bool:
    """Does this host's robots.txt permit our user agent to fetch ``url``?

    Decision: The Oracle respects robots.txt. It fetches pages a learner could
    open themselves, one GET at a time, with no crawling — so the risk is low —
    but this is a product, and a tool that quietly ignores robots.txt is not one
    we want to ship.

    **Fail open on an unreachable robots.txt, fail closed on an explicit
    Disallow.** A missing or broken robots file is not consent withheld; a rule
    that names us is.
    """
    import urllib.robotparser
    from urllib.parse import urlsplit

    if os.environ.get(IGNORE_ROBOTS_ENV) == "1":
        return True

    parts = urlsplit(url)
    origin = f"{parts.scheme}://{parts.netloc}"

    with _ROBOTS_LOCK:
        parser = _ROBOTS.get(origin)
        if parser is None:
            import httpx

            parser = urllib.robotparser.RobotFileParser()
            try:
                with httpx.Client(
                    follow_redirects=True, timeout=timeout, headers=DEFAULT_HEADERS
                ) as client:
                    response = client.get(f"{origin}/robots.txt")
                if response.status_code >= 400:
                    parser.parse([])  # no rules: allow
                else:
                    parser.parse(response.text.splitlines())
            except Exception:  # noqa: BLE001 - unreachable robots means allow
                parser.parse([])
            _ROBOTS[origin] = parser

    try:
        return bool(parser.can_fetch(USER_AGENT, url))
    except Exception:  # noqa: BLE001 - a malformed rule must not block a fetch
        return True


def fetch(url: str, *, timeout: float = 15.0) -> Fetched:
    """GET one URL and extract its text. Never raises.

    Follows redirects and records the final URL. Refuses a non-HTML content
    type, any body over :data:`MAX_BYTES`, and anything robots.txt disallows.
    """
    import httpx

    clean = (url or "").strip()
    if not clean:
        return _failure(url or "", "empty url")
    if not clean.lower().startswith(("http://", "https://")):
        return _failure(clean, f"unsupported scheme: {clean.split(':', 1)[0]!r}")

    if not robots_allows(clean):
        return _failure(clean, "blocked by robots.txt")

    _throttle(clean)

    try:
        with httpx.Client(
            follow_redirects=True, timeout=timeout, headers=DEFAULT_HEADERS
        ) as client:
            with client.stream("GET", clean) as response:
                final_url = str(response.url)
                status = int(response.status_code)
                if status >= 400:
                    return _failure(
                        clean, f"http {status}", final_url=final_url, status=status
                    )

                content_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
                if content_type and not content_type.startswith(ALLOWED_CONTENT_TYPES):
                    return _failure(
                        clean,
                        f"unsupported content-type: {content_type}",
                        final_url=final_url,
                        status=status,
                    )

                declared = response.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > MAX_BYTES:
                    return _failure(
                        clean,
                        f"too large: {int(declared)} bytes > {MAX_BYTES}",
                        final_url=final_url,
                        status=status,
                    )

                body = bytearray()
                for chunk in response.iter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_BYTES:
                        return _failure(
                            clean,
                            f"too large: body exceeds {MAX_BYTES} bytes",
                            final_url=final_url,
                            status=status,
                        )
                encoding = response.encoding or "utf-8"
    except Exception as exc:
        return _failure(clean, f"{type(exc).__name__}: {exc}")

    html = bytes(body).decode(encoding, errors="replace")
    title, text = _extract(html, final_url)
    if not text:
        return _failure(clean, "no extractable text", final_url=final_url, status=status)

    return Fetched(
        url=clean,
        final_url=final_url,
        status=status,
        title=title,
        text=text,
        word_count=len(text.split()),
        error=None,
    )


def fetch_many(urls: Sequence[str], *, max_workers: int = 6) -> list[Fetched]:
    """Fetch several URLs concurrently. Output order matches input order.

    One failure never affects another URL: every slot gets a :class:`Fetched`.
    """
    items = list(urls)
    if not items:
        return []
    workers = max(1, min(int(max_workers), len(items)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="oracle-fetch") as pool:
        futures = [pool.submit(fetch, url) for url in items]
        results: list[Fetched] = []
        for url, future in zip(items, futures, strict=True):
            try:
                results.append(future.result())
            except Exception as exc:  # defensive: fetch() should not raise
                results.append(_failure(url, f"{type(exc).__name__}: {exc}"))
    return results


__all__ = ("Fetched", "MAX_BYTES", "fetch", "fetch_many")
