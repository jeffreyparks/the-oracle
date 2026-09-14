"""A hand-written stand-in for the Architect, and for dedupe until it lands.

No API key was available when this suite was written, so the Architect
cassette is hand-written rather than recorded. This module holds the draft
that produced it, so the cassette can be regenerated from source at any time.

It also carries a minimal :mod:`the_oracle.domains.dedupe` stand-in. The real
module is another worker's file and lands in parallel. The stand-in implements
the frozen Phase 2 contract exactly, so the tests here exercise
``build_domain`` against that contract either way, and start exercising the
real implementation the moment it exists.
"""

from __future__ import annotations

import re
import sys
import types
from enum import StrEnum
from typing import Any

#: The interview the cassette was written for. Change it and re-record.
INTERVIEW: dict[str, Any] = {
    "goal": "Design, build and ship a small production web service on my own",
    "background": "Two years of scripting. I have never run anything in production.",
    "hours_per_week": 3.0,
    "target_weeks": 4,
    "depth": "working",
    "must_cover": ["authentication"],
    "exclude": ["front-end frameworks"],
}

#: Titles are the draft's only identifiers. Ids are minted after dedupe.
DRAFT: dict[str, Any] = {
    "title": "Building and Shipping a Small Web Service",
    "description": (
        "Take a service from a first route to something running, guarded and "
        "observable, without a team behind you."
    ),
    "objectives": [
        {
            "title": "Request and Response Model",
            "description": "Explain how one request becomes one response and what carries meaning in each.",
            "bloom": "understand",
            "difficulty": 1,
            "est_minutes": 30,
            "assessment_stems": [
                "Given a request that returns the wrong body, decide which part of the exchange is at fault.",
                "Predict what the caller sees when the handler raises before it writes anything.",
            ],
            "tags": ["protocol", "request"],
        },
        {
            "title": "Routing and Path Parameters",
            "description": "Map an incoming path onto the one handler that should own it.",
            "bloom": "apply",
            "difficulty": 2,
            "est_minutes": 40,
            "assessment_stems": [
                "Two routes match the same path. Decide which one wins and change the table so the other does.",
                "Given this set of paths, write the route table with no ambiguity left.",
            ],
            "tags": ["routing"],
        },
        {
            "title": "Request Validation with Schemas",
            "description": "Reject bad input at the edge instead of inside the handler.",
            "bloom": "apply",
            "difficulty": 3,
            "est_minutes": 50,
            "assessment_stems": [
                "This payload passed validation and still broke the handler. Find the gap in the schema.",
                "Given three failing inputs, write the schema that rejects all three and accepts the valid one.",
            ],
            "tags": ["validation", "schema"],
        },
        {
            "title": "Persistence with a Relational Store",
            "description": "Store and query the service's state without losing it or corrupting it.",
            "bloom": "apply",
            "difficulty": 3,
            "est_minutes": 60,
            "assessment_stems": [
                "Two writes race on the same row. Predict the final state and fix the code so it cannot happen.",
                "Given this query plan, decide whether the index earns its cost.",
            ],
            "tags": ["persistence", "database"],
        },
        {
            "title": "Authentication and Session Handling",
            "description": "Prove who is calling and keep that proof from leaking.",
            "bloom": "analyze",
            "difficulty": 4,
            "est_minutes": 60,
            "assessment_stems": [
                "This login flow is exploitable. Find the step that lets an attacker in.",
                "Decide whether this token belongs in a cookie or a header here, and defend the choice.",
            ],
            "tags": ["authentication", "security"],
        },
        {
            "title": "Error Handling and Status Codes",
            "description": "Tell the caller what went wrong in a way they can act on.",
            "bloom": "apply",
            "difficulty": 2,
            "est_minutes": 35,
            "assessment_stems": [
                "A handler returns success while the write failed. Diagnose what the caller will do next.",
                "Given five failures, choose the status code each one deserves and say why.",
            ],
            "tags": ["errors"],
        },
        {
            "title": "Deploying and Observing a Service",
            "description": "Run the service somewhere real and know when it is unhealthy.",
            "bloom": "evaluate",
            "difficulty": 4,
            "est_minutes": 55,
            "assessment_stems": [
                "The service is up and users report failures. Decide which signal you trust first.",
                "Judge this deployment plan: name the failure it cannot recover from.",
            ],
            "tags": ["deployment", "observability"],
        },
    ],
    "edges": [
        ["Request and Response Model", "Routing and Path Parameters"],
        ["Routing and Path Parameters", "Request Validation with Schemas"],
        ["Routing and Path Parameters", "Error Handling and Status Codes"],
        ["Request Validation with Schemas", "Persistence with a Relational Store"],
        ["Persistence with a Relational Store", "Authentication and Session Handling"],
        ["Authentication and Session Handling", "Deploying and Observing a Service"],
        ["Error Handling and Status Codes", "Deploying and Observing a Service"],
    ],
    "modules": [
        {
            "id": "foundations",
            "title": "The Exchange",
            "goal": "Handle one request correctly, end to end.",
            "objectives": [
                "Request and Response Model",
                "Routing and Path Parameters",
                "Error Handling and Status Codes",
            ],
        },
        {
            "id": "state",
            "title": "State and Identity",
            "goal": "Keep data and decide who may touch it.",
            "objectives": [
                "Request Validation with Schemas",
                "Persistence with a Relational Store",
                "Authentication and Session Handling",
            ],
        },
        {
            "id": "ship",
            "title": "Shipping",
            "goal": "Run it in front of real users.",
            "objectives": ["Deploying and Observing a Service"],
        },
    ],
    "misconceptions": [
        {
            "id": "validation_is_the_handler",
            "wrong_model": "Checking input inside the handler is the same as validating it at the edge.",
            "objectives": ["Request Validation with Schemas", "Error Handling and Status Codes"],
            "diagnostic": "Where does a malformed payload stop, and what does the caller see when it does?",
        },
        {
            "id": "auth_is_a_login_page",
            "wrong_model": "Authentication is a login screen, so a service without a UI does not need it.",
            "objectives": ["Authentication and Session Handling"],
            "diagnostic": "Your service has no UI. Who is calling it right now, and how do you know?",
        },
    ],
}


def synthesize(agent_name: str, prompt: str) -> dict[str, Any]:
    """Cassette responder: the Architect answers with the draft above."""
    if agent_name == "architect":
        return DRAFT
    raise KeyError(f"no offline responder for agent {agent_name!r}")


# ---------------------------------------------------------------------------
# A stand-in for the dedupe module, used only while the real one is missing
# ---------------------------------------------------------------------------


def _tokens(obj: Any) -> set[str]:
    text = f"{obj.title} {obj.description} {' '.join(obj.tags)}".casefold()
    return {t for t in re.split(r"[^a-z0-9]+", text) if t}


def _similarity(a: Any, b: Any) -> float:
    left, right = _tokens(a), _tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def build_stub_dedupe() -> types.ModuleType:
    """A contract-faithful, lexical stand-in for ``domains.dedupe``."""
    module = types.ModuleType("the_oracle.domains.dedupe")

    class Decision(StrEnum):
        REUSE = "reuse"
        ADJUDICATE = "adjudicate"
        MINT = "mint"

    from dataclasses import dataclass, field

    @dataclass(frozen=True)
    class Match:
        candidate_title: str
        decision: Decision
        best_id: str | None
        score: float
        runners_up: list[tuple[str, float]] = field(default_factory=list)

    reuse_threshold = 0.97  # lexical evidence is weak, so the bar is raised
    adjudicate_threshold = 0.83

    def match_one(candidate: Any, library_objs: Any, embedder: Any = None) -> Match:
        scored = sorted(
            ((obj.id, _similarity(candidate, obj)) for obj in library_objs),
            key=lambda pair: (-pair[1], pair[0]),
        )
        if not scored:
            return Match(candidate.title, Decision.MINT, None, 0.0, [])
        best_id, score = scored[0]
        if score >= reuse_threshold:
            decision = Decision.REUSE
        elif score >= adjudicate_threshold:
            decision = Decision.ADJUDICATE
        else:
            decision = Decision.MINT
        return Match(candidate.title, decision, best_id, score, scored[1:4])

    def match_all(candidates: Any, library_objs: Any = None, embedder: Any = None) -> list[Match]:
        pool = list(library_objs or [])
        return [match_one(c, pool, embedder) for c in candidates]

    def resolve(matches: Any, judge: Any = None) -> dict[str, str | None]:
        out: dict[str, str | None] = {}
        for match in matches:
            if match.decision == Decision.REUSE:
                out[match.candidate_title] = match.best_id
            elif match.decision == Decision.ADJUDICATE and judge is not None:
                out[match.candidate_title] = match.best_id if judge(match) else None
            else:
                out[match.candidate_title] = None
        return out

    module.Decision = Decision  # type: ignore[attr-defined]
    module.Match = Match  # type: ignore[attr-defined]
    module.REUSE_THRESHOLD = reuse_threshold  # type: ignore[attr-defined]
    module.ADJUDICATE_THRESHOLD = adjudicate_threshold  # type: ignore[attr-defined]
    module.match_one = match_one  # type: ignore[attr-defined]
    module.match_all = match_all  # type: ignore[attr-defined]
    module.resolve = resolve  # type: ignore[attr-defined]
    return module


def dedupe_is_real() -> bool:
    """True once the dedupe worker's module is importable."""
    try:
        import the_oracle.domains.dedupe  # noqa: F401
    except ImportError:
        return False
    return True


def install_stub_dedupe() -> bool:
    """Put the stand-in in ``sys.modules`` unless the real module exists.

    Returns True when the stand-in was installed. The caller is responsible
    for removing it again.
    """
    if dedupe_is_real():
        return False
    sys.modules["the_oracle.domains.dedupe"] = build_stub_dedupe()
    return True
