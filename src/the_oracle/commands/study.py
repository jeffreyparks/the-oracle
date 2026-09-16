"""``oracle study`` — run one real session.

This is the only place a learner hears the Oracle speak, so the voice in
``style.py`` is the specification, not a suggestion. Praise names the exact
move the learner made. A wrong answer is called wrong, at once. Numbers at the
close-out are the measured ones, even when they are unflattering.

Structure follows PLAN.md section 5: retrieval warm-up, new material, applied
practice, consolidation. The learner is told which phase they are in and why,
because "recall last week's work before you start something new" is a
pedagogical decision they deserve an explanation for.

Two seams keep this module testable offline. :func:`_session` and
:func:`_gate` import their modules lazily and are the only door to
``the_oracle.session`` and ``the_oracle.mastery.gate``; tests replace them with
fakes, so no model, no key and no network is ever needed here.
"""

from __future__ import annotations

import inspect
from dataclasses import replace
from datetime import datetime, timezone
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

app = typer.Typer(
    name="study",
    help="Run a study session.",
    invoke_without_command=True,
)
"""Wired into ``cli.py`` as a command, not a sub-app::

    app.command("study", help="Run a study session.")(_study_cmd.study)

Registering the callback directly lets ``--minutes`` parse on either side of
the domain id, exactly as ``plan`` and ``resources`` do.
"""

#: What each phase is for, in the learner's words. Keyed by the phase value.
PHASE_PURPOSE: dict[str, str] = {
    "warmup": (
        "Pull back what you did earlier. Retrieving something is what makes it "
        "stick, so old work comes before new work, every time."
    ),
    "new": "One new objective, sometimes two. This is the only new material today.",
    "practice": "Use it. Reading it again is not practice.",
    "consolidate": "Say what you learned, then set the date you come back.",
}


# --- seams ------------------------------------------------------------------


def _session() -> Any:
    """The session engine. Imported lazily; replaced wholesale in tests."""
    from the_oracle import session

    return session


def _gate() -> Any:
    """The strict teaching gate. Imported lazily; replaced wholesale in tests."""
    from the_oracle.mastery import gate

    return gate


# --- rendering --------------------------------------------------------------


def _plan_panel(domain_title: str, plan: Any) -> None:
    """Show the shape of the session and what each phase is for."""
    table = Table(box=None, pad_edge=False, show_header=True)
    table.add_column("phase", style="bold")
    table.add_column("min", justify="right")
    table.add_column("what it is for")
    for budget in getattr(plan, "shape", ()):  # PhaseBudget
        name = str(getattr(budget.phase, "value", budget.phase))
        table.add_row(name, f"{budget.minutes:g}", PHASE_PURPOSE.get(name, ""))
    console.print(Panel(domain_title, title=f"study · {plan.minutes} min", border_style="cyan"))
    console.print(table)
    console.print()


def _warmup_note(plan: Any) -> None:
    """Explain the review items before they arrive, not after."""
    warmup = list(getattr(plan, "warmup", ()))
    if not warmup:
        console.print(
            "[dim]Nothing is due for review, so you start straight on new material.[/dim]"
        )
        return
    count = len(warmup)
    noun = "item" if count == 1 else "items"
    console.print(
        f"You start with {count} {noun} from earlier work: " + ", ".join(warmup) + "."
    )
    console.print(
        "[dim]" + PHASE_PURPOSE["warmup"] + " These are due today, so they are part "
        "of the session rather than a separate chore.[/dim]"
    )
    console.print()


def _module_for(domain: Any, objective_id: str) -> str | None:
    for module in getattr(domain, "modules", []):
        if objective_id in module.objectives:
            return str(module.id)
    return None


def _resource_line(objective_id: str, engine: Any) -> str | None:
    """First accepted resource for an objective, rendered. ``None`` when bare."""
    from the_oracle.corpus.store import accepted_for

    accepted = accepted_for(objective_id, engine)
    if not accepted:
        return None
    first = accepted[0]
    title = getattr(first, "title", "") or objective_id
    url = getattr(first, "url", "") or ""
    return f"{title} — {url}" if url else title


def _missing_material(
    domain: Any, domain_id: str, missing: list[str]
) -> None:
    """Say plainly that there is nothing to teach from, and what to run."""
    lines = ["There is no accepted material for:"]
    for objective_id in missing:
        lines.append(f"  - {objective_id}")
    lines.append("")
    modules = sorted({m for m in (_module_for(domain, o) for o in missing) if m})
    if modules:
        for module_id in modules:
            lines.append(f"  the-oracle resources {domain_id} --module {module_id}")
    else:
        lines.append(f"  the-oracle resources {domain_id}")
    lines.append("")
    lines.append("Until that runs there is nothing to teach from, and I will not fake it.")
    console.print(Panel("\n".join(lines), title="empty corpus", border_style="yellow"))


# --- the interactive surface ------------------------------------------------


class _Surface:
    """Holds the per-session display state: phase, misconceptions, timing."""

    def __init__(self, domain: Any, domain_id: str, engine: Any) -> None:
        self.domain = domain
        self.domain_id = domain_id
        self.engine = engine
        self.phase: str = ""
        self.seen_misconceptions: list[str] = []
        self.shown_resources: set[str] = set()
        #: True when the session engine reports grades through ``on_feedback``.
        self.uses_feedback: bool = False

    # -- phase ---------------------------------------------------------------

    def enter(self, phase: str) -> None:
        if not phase or phase == self.phase:
            return
        self.phase = phase
        console.rule(f"{phase} — {PHASE_PURPOSE.get(phase, '')}")

    def resource(self, objective_id: str) -> None:
        """Show the material for an objective once, before the first item."""
        if not objective_id or objective_id in self.shown_resources:
            return
        self.shown_resources.add(objective_id)
        line = _resource_line(objective_id, self.engine)
        if line is None:
            console.print(
                f"[yellow]No accepted material for {objective_id}. "
                f"Run: the-oracle resources {self.domain_id}"
                + (f" --module {m}" if (m := _module_for(self.domain, objective_id)) else "")
                + "[/yellow]"
            )
            return
        console.print(Panel(line, title=f"read first · {objective_id}", border_style="green"))

    # -- items ---------------------------------------------------------------

    def ask(self, item: Any) -> str:
        """Put one item to the learner. Never interrupts a multi-step attempt."""
        from the_oracle.agents.assessor import feedback_is_immediate

        self.enter(str(getattr(item, "phase", "") or self.phase))
        self.resource(str(getattr(item, "objective_id", "") or ""))
        kind = str(getattr(item, "kind", "short"))
        difficulty = getattr(item, "difficulty", 1)
        console.print()
        console.print(
            Panel(
                str(getattr(item, "stem", "")),
                title=f"{kind} · difficulty {difficulty}",
                border_style="cyan",
            )
        )
        if not feedback_is_immediate(kind):
            console.print(
                "[dim]Multi-step. Work the whole thing through. I hold the critique "
                "until you submit, because breaking into a derivation costs you the "
                "working memory you are using to do it.[/dim]"
            )
        return typer.prompt("your answer")

    def graded(self, payload: dict[str, Any]) -> None:
        """Report one finished attempt. Wrong is called wrong, at once.

        Two paths land here and only one runs per session. ``on_feedback``
        carries the grader's prose and is preferred; ``RESPONSE_GRADED``
        carries the verdict and is the fallback. Either way the attempt is
        already over when this fires, which is the timing rule: a multi-step
        attempt is never interrupted, because nothing can arrive mid-way.
        """
        correct = bool(payload.get("correct", False))
        verdict = "[green]correct[/green]" if correct else "[red]wrong[/red]"
        feedback = str(payload.get("feedback", "") or "")
        console.print(f"{verdict} — {feedback}" if feedback else verdict)

        misconception_id = payload.get("misconception_id")
        if not misconception_id:
            return
        session = _session()
        should_interrupt = getattr(session, "should_interrupt", None)
        if callable(should_interrupt):
            interrupt = bool(should_interrupt(tuple(self.seen_misconceptions), misconception_id))
        else:  # the rule, if the seam is absent: twice in one session interrupts
            interrupt = misconception_id in self.seen_misconceptions
        self.seen_misconceptions.append(str(misconception_id))
        if interrupt:
            console.print(
                Panel(
                    f"That is {misconception_id} for the second time this session. "
                    "Stopping the flow to fix it now. A wrong model gets stronger "
                    "every time you use it, so it does not wait until the end.",
                    title="same wrong model twice",
                    border_style="yellow",
                )
            )

    def feedback(self, item: Any, answer: str, grade: Any, phase: Any = None) -> None:
        """``session.run_session(on_feedback=...)``: prose, once per attempt."""
        self.enter(str(getattr(phase, "value", phase) or self.phase))
        self.graded(
            {
                "correct": bool(getattr(grade, "correct", False)),
                "feedback": getattr(grade, "feedback", ""),
                "misconception_id": getattr(grade, "misconception_id", None),
            }
        )

    # -- events --------------------------------------------------------------

    def event(self, event: Any) -> None:
        """Follow the session's own events for phase changes and drift."""
        kind = str(getattr(event, "kind", "") or "")
        payload = getattr(event, "payload", None) or {}
        if not isinstance(payload, dict):
            return
        phase = payload.get("phase")
        if isinstance(phase, str):
            self.enter(phase)
        if kind.endswith("item_presented"):
            objective_id = payload.get("objective_id")
            if isinstance(objective_id, str):
                self.resource(objective_id)
        if kind.endswith("response_graded") and not self.uses_feedback:
            self.graded(payload)


# --- close-out --------------------------------------------------------------


def _due_rows(learner_id: str, engine: Any) -> list[Any]:
    from sqlmodel import Session, select

    from the_oracle.store import models

    with Session(engine) as db:
        rows = db.exec(
            select(models.MasteryState).where(models.MasteryState.learner_id == learner_id)
        ).all()
    return [r for r in rows if r.next_review_at is not None]


def _gap(hours: float) -> str:
    """Plain words for a span. Rounded, but never rounded to flatter."""
    whole = round(hours)
    if whole < 24:
        return f"{whole} hour" if whole == 1 else f"{whole} hours"
    days = hours / 24.0
    return f"{days:.1f} day" if round(days, 1) == 1.0 else f"{days:.1f} days"


def _when(due_at: datetime, now: datetime) -> str:
    """Plain words for a date."""
    if due_at.tzinfo is None:
        due_at = due_at.replace(tzinfo=timezone.utc)
    hours = (due_at - now).total_seconds() / 3600.0
    if hours <= 0:
        return "now"
    return f"in {_gap(hours)}, on {due_at:%Y-%m-%d}"


def _closeout(result: Any, learner_id: str, engine: Any, now: datetime) -> None:
    """What moved, what is due next, when to come back. Measured numbers only."""
    asked = int(getattr(result, "asked", 0))
    correct = int(getattr(result, "correct", 0))
    before: dict[str, float] = dict(getattr(result, "mastery_before", {}) or {})
    after: dict[str, float] = dict(getattr(result, "mastery_after", {}) or {})

    console.print()
    console.rule("what moved")
    if asked == 0:
        console.print("You answered nothing this session, so nothing moved.")
    else:
        accuracy = correct / asked
        console.print(f"You answered {correct} of {asked} right. That is {accuracy:.0%}.")

    table = Table(box=None, pad_edge=False)
    table.add_column("objective", style="bold")
    table.add_column("before", justify="right")
    table.add_column("after", justify="right")
    table.add_column("change", justify="right")
    moved = False
    for objective_id in sorted(set(before) | set(after)):
        start = float(before.get(objective_id, 0.0))
        end = float(after.get(objective_id, start))
        change = end - start
        if abs(change) >= 0.005:
            moved = True
        arrow = f"{change:+.2f}"
        table.add_row(objective_id, f"{start:.2f}", f"{end:.2f}", arrow)
    if before or after:
        console.print(table)
    if not moved and asked:
        console.print("[dim]No objective moved enough to report. One session often does not.[/dim]")

    reason = str(getattr(result, "reason", "") or "")
    if getattr(result, "ended_early", False):
        console.print(
            Panel(
                f"Session ended early. {reason}".strip(),
                title="stopped early",
                border_style="yellow",
            )
        )
    elif reason:
        console.print(reason)

    console.rule("due next")
    rows = _due_rows(learner_id, engine)
    upcoming = sorted(rows, key=lambda r: r.next_review_at)
    if not upcoming:
        console.print("Nothing is scheduled yet. Answer a few more items and a schedule appears.")
        return
    table = Table(box=None, pad_edge=False)
    table.add_column("objective", style="bold")
    table.add_column("p(mastery)", justify="right")
    table.add_column("due")
    for row in upcoming[:5]:
        table.add_row(row.objective_id, f"{row.p_mastery:.2f}", _when(row.next_review_at, now))
    console.print(table)
    soonest = upcoming[0]
    console.print(f"Come back {_when(soonest.next_review_at, now)}, starting with {soonest.objective_id}.")


# --- the command ------------------------------------------------------------


@app.callback(invoke_without_command=True)
def study(
    domain_id: Annotated[str, typer.Argument(help="Domain pack id to study.")],
    minutes: Annotated[
        int, typer.Option("--minutes", "-m", help="Session length. 10, 25, or 50.")
    ] = 25,
) -> None:
    """Run one session: warm-up, new material, practice, consolidation."""
    from the_oracle.context import LearnerContext
    from the_oracle.domains.errors import PackValidationError
    from the_oracle.domains.registry import load_domain
    from the_oracle.store.db import create_all, get_engine

    ctx = LearnerContext.resolve()
    try:
        domain = load_domain(domain_id)
    except PackValidationError as exc:
        console.print(f"[red]{domain_id} did not load.[/red]")
        for problem in exc.problems:
            console.print(f"  - {problem}")
        raise typer.Exit(code=1) from exc

    engine = get_engine()
    create_all(engine)
    now = datetime.now(timezone.utc)

    session = _session()
    plan = session.plan_session(domain_id, ctx.learner_id, minutes=minutes, now=now, engine=engine)

    _plan_panel(getattr(domain, "title", domain_id), plan)
    _warmup_note(plan)

    focus = list(getattr(plan, "focus", ()))
    if not focus:
        blocked = []
        gate = _gate()
        try:
            from the_oracle.mastery import profile_for

            profile = profile_for(ctx.learner_id, engine)
            for objective_id in domain.teaching_order():
                blocked = gate.blocked_by(domain, objective_id, profile)
                if blocked:
                    break
        except Exception:  # the gate is advisory here, never fatal
            blocked = []
        console.print(
            Panel(
                "Nothing here is teachable right now.\n"
                + (
                    "The next objective is waiting on: " + ", ".join(blocked) + ".\n"
                    if blocked
                    else ""
                )
                + f"Run: the-oracle assess {domain_id} to place yourself, "
                f"or the-oracle plan {domain_id}.",
                title="nothing to teach",
                border_style="yellow",
            )
        )
        raise typer.Exit(code=0)

    missing = [oid for oid in focus if _resource_line(oid, engine) is None]
    if missing:
        _missing_material(domain, domain_id, missing)
        focus = [oid for oid in focus if oid not in missing]
        plan = replace(plan, focus=focus)

    if not focus and not list(getattr(plan, "warmup", ())):
        console.print("Nothing to run this session. Fill the corpus, then come back.")
        raise typer.Exit(code=1)

    surface = _Surface(domain, domain_id, engine)
    calls: dict[str, Any] = {
        "answer_fn": surface.ask,
        "on_event": surface.event,
        "engine": engine,
    }
    # The session engine reports grades through ``on_feedback`` when it has one,
    # because that call carries the grader's prose. The event stream is the
    # fallback and carries only the verdict.
    try:
        parameters = inspect.signature(session.run_session).parameters
    except (TypeError, ValueError):  # pragma: no cover - a callable without a signature
        parameters = {}
    if "on_feedback" in parameters:
        calls["on_feedback"] = surface.feedback
        surface.uses_feedback = True
    result = session.run_session(plan, **calls)
    _closeout(result, ctx.learner_id, engine, now)


__all__ = ["PHASE_PURPOSE", "app", "study"]
