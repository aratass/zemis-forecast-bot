"""End to end, with the network replaced by fakes.

The point of this file is the payload validator: whatever the models say, what
goes on the wire has to be something the Metaculus API accepts. A rejected
submission scores zero, and zeros are what the squared prize rule punishes.
"""

from datetime import datetime, timedelta, timezone

import pytest


from bot import forecast as fc
from bot import research as research_mod
from bot.cdf import validate_cdf

NOW = datetime.now(timezone.utc)
SOON = (NOW + timedelta(days=30)).isoformat()


def q_binary():
    return {
        "id": 101,
        "type": "binary",
        "title": "Will the policy rate be cut before the end of the year?",
        "description": "Background.",
        "resolution_criteria": "Resolves YES if the rate is cut.",
        "scheduled_close_time": SOON,
        "scheduled_resolve_time": SOON,
    }


def q_numeric(open_lower=False, open_upper=True, zero_point=None, count=200):
    return {
        "id": 102,
        "type": "numeric",
        "title": "How many units will ship?",
        "resolution_criteria": "The reported figure.",
        "scheduled_close_time": SOON,
        "scheduled_resolve_time": SOON,
        "unit": "units",
        "open_lower_bound": open_lower,
        "open_upper_bound": open_upper,
        "inbound_outcome_count": count,
        "scaling": {"range_min": 0, "range_max": 1000, "zero_point": zero_point},
    }


def q_discrete():
    q = q_numeric(count=11)
    q["id"] = 103
    q["type"] = "discrete"
    q["scaling"] = {"range_min": 0, "range_max": 10, "zero_point": None}
    q["open_lower_bound"] = False
    q["open_upper_bound"] = False
    return q


def q_date():
    return {
        "id": 104,
        "type": "date",
        "title": "When will the report be published?",
        "resolution_criteria": "Publication date.",
        "scheduled_close_time": SOON,
        "scheduled_resolve_time": SOON,
        "open_lower_bound": False,
        "open_upper_bound": True,
        "inbound_outcome_count": 200,
        "scaling": {
            "range_min": datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp(),
            "range_max": datetime(2028, 1, 1, tzinfo=timezone.utc).timestamp(),
            "zero_point": None,
        },
    }


def q_mc():
    return {
        "id": 105,
        "type": "multiple_choice",
        "title": "Which outcome occurs?",
        "resolution_criteria": "As announced.",
        "scheduled_close_time": SOON,
        "scheduled_resolve_time": SOON,
        "options": ["Alpha", "Beta", "Gamma", "None of these"],
    }


POST = {"id": 9001, "title": "post title"}


REPLIES = {
    "binary": "Reasoning about the status quo.\nPROBABILITY: 31%",
    "numeric": (
        "Reasoning.\nPERCENTILES:\n5: 120\n10: 160\n20: 210\n40: 280\n"
        "60: 340\n80: 430\n90: 520\n95: 610\n"
    ),
    "discrete": (
        "Reasoning.\nPERCENTILES:\n5: 1\n10: 2\n20: 3\n40: 4\n60: 5\n80: 7\n90: 8\n95: 9\n"
    ),
    "date": (
        "Reasoning.\nPERCENTILES:\n5: 2026-04-01\n10: 2026-06-01\n20: 2026-09-01\n"
        "40: 2027-01-15\n60: 2027-04-01\n80: 2027-08-01\n90: 2027-10-01\n95: 2027-11-15\n"
    ),
    "multiple_choice": (
        "Reasoning.\nPROBABILITIES:\nAlpha: 45%\nBeta: 30%\nGamma: 20%\nNone of these: 5%\n"
    ),
}


@pytest.fixture
def fake_llm(monkeypatch):
    """One reply per question type, plus a JSON array for the query step."""
    calls = {"n": 0}
    state = {"kind": "binary"}

    def fake_chat(messages, models, **kwargs):
        calls["n"] += 1
        text = " ".join(m["content"] for m in messages)
        if "search queries" in text.lower() or "JSON array" in text:
            return '["query one", "query two", "query three"]', models[0]
        return REPLIES[state["kind"]], models[0]

    monkeypatch.setattr(fc, "chat_with_fallback", fake_chat)
    monkeypatch.setattr(
        research_mod, "gather", lambda queries, include_markets=True: research_mod.ResearchReport()
    )
    return state, calls


def run(question, state, models=("a/one", "b/two", "c/three"), runs=5):
    state["kind"] = question["type"]
    return fc.forecast_question(
        post=POST,
        question=question,
        research_text="no evidence",
        research_sources=["Fake"],
        models=list(models),
        runs=runs,
    )


def assert_payload_ok(forecast, question):
    p = forecast.payload
    assert p["question"] == question["id"], "keyed by question id, never post id"
    if question["type"] == "binary":
        assert set(p) == {"question", "probability_yes"}
        assert 0.001 <= p["probability_yes"] <= 0.999
    elif question["type"] == "multiple_choice":
        assert set(p) == {"question", "probability_yes_per_category"}
        cat = p["probability_yes_per_category"]
        assert set(cat) == set(question["options"])
        assert all(0.001 <= v <= 0.999 for v in cat.values())
        assert abs(sum(cat.values()) - 1.0) < 1e-6
    else:
        assert set(p) == {"question", "continuous_cdf"}
        errors = validate_cdf(
            p["continuous_cdf"],
            question["inbound_outcome_count"],
            question["open_lower_bound"],
            question["open_upper_bound"],
        )
        assert not errors, errors


@pytest.mark.parametrize(
    "factory",
    [q_binary, q_numeric, q_discrete, q_date, q_mc],
    ids=["binary", "numeric", "discrete", "date", "multiple_choice"],
)
def test_each_question_type_produces_a_submittable_payload(factory, fake_llm):
    state, _ = fake_llm
    question = factory()
    forecast = run(question, state)
    assert_payload_ok(forecast, question)
    assert forecast.comment.strip()
    assert forecast.post_id == POST["id"]
    assert forecast.runs_used > 0


@pytest.mark.parametrize(
    "zero_point,open_lower,open_upper",
    [(None, False, False), (None, True, True), (-1.0, False, True), (-1.0, True, False)],
)
def test_numeric_variants_all_validate(zero_point, open_lower, open_upper, fake_llm):
    state, _ = fake_llm
    question = q_numeric(open_lower=open_lower, open_upper=open_upper, zero_point=zero_point)
    forecast = run(question, state)
    assert_payload_ok(forecast, question)


def test_date_median_lands_inside_the_range(fake_llm):
    state, _ = fake_llm
    question = q_date()
    forecast = run(question, state)
    assert_payload_ok(forecast, question)
    assert "2027" in forecast.headline, forecast.headline


def test_comment_records_models_sources_and_calibration(fake_llm):
    state, _ = fake_llm
    forecast = run(q_binary(), state)
    assert "Fake" in forecast.comment
    assert "ensemble" in forecast.comment
    assert "Forecast:" in forecast.comment


def test_a_model_that_says_nothing_useful_leaves_the_question_open(monkeypatch, fake_llm):
    """No uniform placeholder: it would score worse than no forecast at all."""
    state, _ = fake_llm

    def useless(messages, models, **kwargs):
        text = " ".join(m["content"] for m in messages)
        if "JSON array" in text:
            return "[]", models[0]
        return "I am not able to give a number.", models[0]

    monkeypatch.setattr(fc, "chat_with_fallback", useless)
    question = q_numeric()
    state["kind"] = "numeric"
    with pytest.raises(fc.LLMError):
        fc.forecast_question(
            post=POST,
            question=question,
            research_text="",
            research_sources=[],
            models=["a/one"],
            runs=3,
        )


def test_binary_with_no_usable_output_raises_rather_than_guessing(monkeypatch, fake_llm):
    """A silent 50% would be a fabricated forecast; failing lets the retry work."""
    state, _ = fake_llm

    def useless(messages, models, **kwargs):
        text = " ".join(m["content"] for m in messages)
        if "JSON array" in text:
            return "[]", models[0]
        return "no number", models[0]

    monkeypatch.setattr(fc, "chat_with_fallback", useless)
    with pytest.raises(Exception):
        fc.forecast_question(
            post=POST,
            question=q_binary(),
            research_text="",
            research_sources=[],
            models=["a/one"],
            runs=2,
        )


def test_ensemble_rotates_across_model_families(monkeypatch, fake_llm):
    state, _ = fake_llm
    seen = []

    def record(messages, models, **kwargs):
        text = " ".join(m["content"] for m in messages)
        if "JSON array" in text:
            return '["q"]', models[0]
        seen.append(models[0])
        return REPLIES["binary"], models[0]

    monkeypatch.setattr(fc, "chat_with_fallback", record)
    run(q_binary(), state, models=("a/one", "b/two", "c/three"), runs=6)
    assert len(set(seen)) == 3, seen


def test_unsupported_type_is_reported_not_guessed(fake_llm):
    state, _ = fake_llm
    question = {"id": 999, "type": "conditional", "title": "x"}
    with pytest.raises(ValueError):
        fc.forecast_question(POST, question, "", [], ["a/one"], runs=1)
