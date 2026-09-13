"""A hand-written stand-in for the model.

No API key was available when this suite was written, so the cassettes are
hand-written rather than recorded. This module holds the logic that produced
them, and it doubles as the responder for the one test whose prompts depend on
another worker's component.

It reads the same prompt the real model would read, and answers in the same
schema. It is deliberately dumb: exact-match grading plus misconception
matching by wrong-model text.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any


def _field(prompt: str, label: str) -> str:
    match = re.search(rf"^{re.escape(label)}: (.*)$", prompt, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _block(prompt: str, label: str) -> list[str]:
    """The ``- `` bullet lines that follow ``label:``."""
    lines = prompt.splitlines()
    try:
        start = lines.index(f"{label}:")
    except ValueError:
        return []
    out: list[str] = []
    for line in lines[start + 1 :]:
        if not line.startswith("- "):
            break
        out.append(line[2:].strip())
    return out


def _item_id(objective_id: str, stem: str) -> str:
    digest = hashlib.sha256(f"{objective_id}|{stem}".encode()).hexdigest()[:12]
    return f"item-{digest}"


def write_item(prompt: str) -> dict[str, Any]:
    """Answer an ItemWriter prompt."""
    objective_id = _field(prompt, "objective_id")
    bloom = _field(prompt, "bloom")
    kind = _field(prompt, "item kind")
    difficulty = int((_field(prompt, "difficulty") or "1").split()[0])
    stems = [s for s in _block(prompt, "assessment stems to draw on") if "(none" not in s]
    stem = stems[0] if stems else f"Explain {objective_id.replace('_', ' ')} in your own words."
    return {
        "id": _item_id(objective_id, stem),
        "objective_id": objective_id,
        "stem": stem,
        "kind": kind or "short",
        "difficulty": difficulty,
        "bloom": bloom,
        "expected": f"A correct, complete response to: {stem}",
        "misconception_probes": [],
    }


def grade_answer(prompt: str) -> dict[str, Any]:
    """Answer a Grader prompt."""
    expected = _field(prompt, "expected answer")
    answer = prompt.split("learner answer:", 1)[-1].strip()
    known = [line.split(": ", 1) for line in _block(prompt, "known misconceptions")]
    known = [pair for pair in known if len(pair) == 2]

    if answer and expected and answer == expected:
        return {
            "correct": True,
            "confidence": 0.92,
            "misconception_id": None,
            "feedback": "That matches the expected answer. You named the rule and applied it.",
            "reasoning": "Answer is identical in substance to the expected response.",
        }

    for mis_id, wrong_model in known:
        if answer and (answer in wrong_model or wrong_model[:40] in answer):
            return {
                "correct": False,
                "confidence": 0.8,
                "misconception_id": mis_id,
                "feedback": (
                    "That is wrong, and it is wrong in a specific way: you are "
                    f"using the model behind {mis_id}. Fix that model first."
                ),
                "reasoning": f"Answer restates the wrong model recorded as {mis_id}.",
            }

    return {
        "correct": False,
        "confidence": 0.7,
        "misconception_id": None,
        "feedback": "That is wrong. It does not reach the point the item asks for.",
        "reasoning": "Answer does not match the expected response or any known wrong model.",
    }


def synthesize(agent_name: str, prompt: str) -> dict[str, Any]:
    """Cassette responder: dispatch on the agent name."""
    if agent_name == "item_writer":
        return write_item(prompt)
    if agent_name == "grader":
        return grade_answer(prompt)
    raise KeyError(f"no offline responder for agent {agent_name!r}")
