"""Interactive user prompts (v1.4.8): the AskUserQuestion tool.

A :class:`QuestionSpec` (1..N questions, each with 2..M options) is the currency
for the ``AskUserQuestion`` tool (as of v1.5.3 permission no longer uses it — it
is denied directly inside the Executor). An async ``ask_user`` port (held by the
:class:`~neutrix.context_manager.ContextManager`, the shape of CC's ``canUseTool``)
takes a ``QuestionSpec`` and returns an
:class:`Answer`. This module is pure data + parse/render/format; the event-loop
plumbing (the Future the input loop resolves) lives in ``terminal_chat`` and the
dispatch interception lives in ``executor`` — see
``docs/PRDs/v1.4.8-ask-user-question.md``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

ASK_NOT_AVAILABLE = (
    "[AskUserQuestion needs an interactive terminal — not available here]"
)
HEADER_MAX = 12
MAX_QUESTIONS = 4
MIN_OPTIONS = 2
MAX_OPTIONS = 4


@dataclass(frozen=True)
class Option:
    """One selectable choice. ``value`` is the machine token (defaults to the
    label); the permission prompt uses ``yes``/``always``/``no`` values behind
    human labels."""

    label: str
    description: str = ""
    value: str = ""

    def effective_value(self) -> str:
        return self.value or self.label


@dataclass(frozen=True)
class Question:
    question: str
    header: str
    options: tuple[Option, ...]
    multi_select: bool = False


@dataclass(frozen=True)
class QuestionSpec:
    questions: tuple[Question, ...]


@dataclass(frozen=True)
class QuestionAnswer:
    """The answer to one question: human ``display`` text plus the machine
    ``values`` of the chosen options (empty when the user typed free text)."""

    question: str
    display: str
    values: tuple[str, ...] = ()
    is_other: bool = False
    other_text: str = ""


@dataclass(frozen=True)
class Answer:
    per_question: tuple[QuestionAnswer, ...]


if TYPE_CHECKING:
    # The async port the harness (TerminalChat) injects into the ContextManager
    # so it can ask the human a question mid-turn. The Executor never holds this
    # — it yields a ``needs_user_input`` event and the CM drives the prompt,
    # keeping UI→CM→Executor layering intact (v1.4.8). ``None`` everywhere
    # non-interactive (tests, piped stdin, inside a subagent).
    AskUserPort = Callable[[QuestionSpec], Awaitable[Answer]]


def parse_question_spec(args_json: str) -> QuestionSpec:
    """Validate ``AskUserQuestion`` tool args into a :class:`QuestionSpec`.

    Enforces the CC bounds (1..4 questions, 2..4 options each, unique question
    text, unique option labels per question, non-empty text, header ≤12 chars).
    Raises :class:`ValueError` with a model-readable message on any violation.
    """
    try:
        args = json.loads(args_json) if args_json else {}
    except ValueError as exc:
        raise ValueError(f"invalid JSON args: {exc}") from exc
    if not isinstance(args, dict):
        raise ValueError("args must be a JSON object")
    raw_qs = args.get("questions")
    if not isinstance(raw_qs, list) or not (1 <= len(raw_qs) <= MAX_QUESTIONS):
        raise ValueError(f"'questions' must be a list of 1..{MAX_QUESTIONS} items")
    seen_q: set[str] = set()
    questions: list[Question] = []
    for q in raw_qs:
        if not isinstance(q, dict):
            raise ValueError("each question must be an object")
        text = str(q.get("question", "")).strip()
        header = str(q.get("header", "")).strip()
        if not text:
            raise ValueError("question text is required")
        if not header:
            raise ValueError("question header is required")
        if len(header) > HEADER_MAX:
            raise ValueError(f"header must be ≤{HEADER_MAX} chars: {header!r}")
        if text in seen_q:
            raise ValueError(f"duplicate question: {text!r}")
        seen_q.add(text)
        raw_opts = q.get("options")
        if not isinstance(raw_opts, list) or not (
            MIN_OPTIONS <= len(raw_opts) <= MAX_OPTIONS
        ):
            raise ValueError(
                f"each question needs {MIN_OPTIONS}..{MAX_OPTIONS} options"
            )
        seen_l: set[str] = set()
        options: list[Option] = []
        for o in raw_opts:
            if not isinstance(o, dict):
                raise ValueError("each option must be an object")
            label = str(o.get("label", "")).strip()
            if not label:
                raise ValueError("option label is required")
            if label in seen_l:
                raise ValueError(f"duplicate option label: {label!r}")
            seen_l.add(label)
            options.append(
                Option(label=label, description=str(o.get("description", "")).strip())
            )
        questions.append(
            Question(
                question=text,
                header=header,
                options=tuple(options),
                multi_select=bool(q.get("multiSelect", False)),
            )
        )
    return QuestionSpec(questions=tuple(questions))


def parse_answer_line(line: str, question: Question) -> QuestionAnswer | None:
    """Map a typed answer line to a :class:`QuestionAnswer` (split #9).

    All-comma-separated integers in ``[1..N]`` → those option labels (first only
    for single-select); anything else → Other (verbatim free text). Empty input
    → ``None`` (the caller re-prompts).
    """
    text = line.strip()
    if not text:
        return None
    n = len(question.options)
    tokens = [t.strip() for t in text.split(",") if t.strip()]
    nums: list[int] = []
    all_ints = bool(tokens)
    for tok in tokens:
        if tok.isdigit() and 1 <= int(tok) <= n:
            nums.append(int(tok))
        else:
            all_ints = False
            break
    if all_ints and nums:
        if not question.multi_select:
            nums = nums[:1]
        ordered: list[int] = []
        for i in nums:
            if i not in ordered:
                ordered.append(i)
        chosen = [question.options[i - 1] for i in ordered]
        return QuestionAnswer(
            question=question.question,
            display=", ".join(o.label for o in chosen),
            values=tuple(o.effective_value() for o in chosen),
        )
    return QuestionAnswer(
        question=question.question,
        display=text,
        values=(),
        is_other=True,
        other_text=text,
    )


def render_question(question: Question, idx: int = 0, total: int = 1) -> str:
    """Numbered scrollback block for one question (split #2)."""
    prefix = f"[{idx + 1}/{total}] " if total > 1 else ""
    lines = [f"{prefix}{question.question}"]
    for i, o in enumerate(question.options, 1):
        lines.append(f"  {i}. {o.label} — {o.description}" if o.description
                     else f"  {i}. {o.label}")
    multi = ", comma-separated for multiple" if question.multi_select else ""
    lines.append(f"(type a number{multi}, or your own answer)")
    return "\n".join(lines)


def format_answers_result(answer: Answer) -> str:
    """CC-keyed tool result: ``{"answers": {question_text: display, …}}``."""
    return json.dumps(
        {"answers": {qa.question: qa.display for qa in answer.per_question}},
        ensure_ascii=False,
    )
