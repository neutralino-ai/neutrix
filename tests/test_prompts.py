"""v1.4.8 — pure parse/validate/render for interactive prompts."""
from __future__ import annotations

import json

import pytest

from neutrix.prompts import (
    Answer,
    Option,
    Question,
    QuestionAnswer,
    format_answers_result,
    parse_answer_line,
    parse_question_spec,
    render_question,
)


def _q(args: dict) -> str:
    return json.dumps(args)


# ---- schema validation ----------------------------------------------------


def test_parse_question_spec_happy():
    spec = parse_question_spec(_q({
        "questions": [
            {
                "question": "Which DB?",
                "header": "Database",
                "multiSelect": False,
                "options": [
                    {"label": "Postgres", "description": "relational"},
                    {"label": "SQLite", "description": "embedded"},
                ],
            }
        ]
    }))
    assert len(spec.questions) == 1
    q = spec.questions[0]
    assert q.header == "Database"
    assert q.multi_select is False
    assert [o.label for o in q.options] == ["Postgres", "SQLite"]


@pytest.mark.parametrize(
    "args,msg",
    [
        ({"questions": []}, "1..4"),
        ({"questions": [{"question": "q?", "header": "h", "options": [{"label": "a"}]}]}, "2..4"),
        ({"questions": [{"question": "", "header": "h", "options": [{"label": "a"}, {"label": "b"}]}]}, "question text"),
        ({"questions": [{"question": "q?", "header": "", "options": [{"label": "a"}, {"label": "b"}]}]}, "header"),
        ({"questions": [{"question": "q?", "header": "0123456789abcd", "options": [{"label": "a"}, {"label": "b"}]}]}, "≤12"),
        ({"questions": [{"question": "q?", "header": "h", "options": [{"label": "a"}, {"label": "a"}]}]}, "duplicate option"),
    ],
)
def test_parse_question_spec_rejects(args, msg):
    with pytest.raises(ValueError, match=msg):
        parse_question_spec(_q(args))


def test_parse_question_spec_rejects_duplicate_questions():
    with pytest.raises(ValueError, match="duplicate question"):
        parse_question_spec(_q({
            "questions": [
                {"question": "same?", "header": "a", "options": [{"label": "1"}, {"label": "2"}]},
                {"question": "same?", "header": "b", "options": [{"label": "1"}, {"label": "2"}]},
            ]
        }))


def test_parse_question_spec_bad_json():
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_question_spec("{not json")


# ---- answer parsing (split #9) --------------------------------------------


def _question(multi: bool = False) -> Question:
    return Question(
        question="Pick one?",
        header="Pick",
        options=(Option("Alpha"), Option("Beta"), Option("Gamma")),
        multi_select=multi,
    )


def test_parse_answer_single_number():
    qa = parse_answer_line("2", _question())
    assert qa is not None
    assert qa.display == "Beta"
    assert qa.values == ("Beta",)
    assert qa.is_other is False


def test_parse_answer_single_select_takes_first_of_many():
    qa = parse_answer_line("3,1", _question(multi=False))
    assert qa.display == "Gamma"
    assert qa.values == ("Gamma",)


def test_parse_answer_multi_select_joins():
    qa = parse_answer_line("1, 3", _question(multi=True))
    assert qa.display == "Alpha, Gamma"
    assert qa.values == ("Alpha", "Gamma")


def test_parse_answer_multi_dedupes_preserving_order():
    qa = parse_answer_line("3,1,3", _question(multi=True))
    assert qa.values == ("Gamma", "Alpha")


def test_parse_answer_out_of_range_is_other():
    qa = parse_answer_line("9", _question())
    assert qa.is_other is True
    assert qa.display == "9"
    assert qa.values == ()


def test_parse_answer_free_text_is_other():
    qa = parse_answer_line("use mongo instead", _question())
    assert qa.is_other is True
    assert qa.other_text == "use mongo instead"


def test_parse_answer_mixed_is_other():
    qa = parse_answer_line("1, banana", _question(multi=True))
    assert qa.is_other is True
    assert qa.display == "1, banana"


def test_parse_answer_empty_is_none():
    assert parse_answer_line("", _question()) is None
    assert parse_answer_line("   ", _question()) is None


# ---- render + result ------------------------------------------------------


def test_render_question_numbers_options():
    q = Question("Which?", "Which", (Option("A", "first"), Option("B", "second")), False)
    text = render_question(q)
    assert "1. A — first" in text
    assert "2. B — second" in text
    assert "your own answer" in text
    assert "comma-separated" not in text  # single-select


def test_render_question_multi_hint_and_prefix():
    q = Question("Which?", "Which", (Option("A"), Option("B")), True)
    text = render_question(q, idx=1, total=3)
    assert text.startswith("[2/3] ")
    assert "comma-separated for multiple" in text


def test_format_answers_result_cc_keyed():
    answer = Answer(
        per_question=(
            QuestionAnswer("Which DB?", "Postgres", ("Postgres",)),
            QuestionAnswer("Region?", "us-east, eu-west", ("us-east", "eu-west")),
        )
    )
    out = json.loads(format_answers_result(answer))
    assert out == {"answers": {"Which DB?": "Postgres", "Region?": "us-east, eu-west"}}
