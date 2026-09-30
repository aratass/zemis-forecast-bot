"""Regression tests for the bugs the first live run exposed.

The expensive one: with every model unavailable, the bot submitted a flat
uniform distribution to a real MiniBench question and then recorded that
question as done, so it would never have gone back once the outage cleared. A
placeholder forecast is worse than no forecast, because it burns the question.
"""

import pytest

from bot import forecast as fc
from bot import llm
from bot.llm import DEAD_MODELS, ModelUnavailable, NoModelsAvailable, _adapt_model_name
from tests.test_pipeline import POST, REPLIES, q_binary, q_mc, q_numeric


@pytest.fixture(autouse=True)
def clean_dead_models():
    DEAD_MODELS.clear()
    yield
    DEAD_MODELS.clear()


def _all_models_dead(messages, models, **kwargs):
    raise NoModelsAvailable("every model is unavailable")


def _responds_with_junk(messages, models, **kwargs):
    return "I cannot answer that.", models[0]


def _run(question, chat_fn, monkeypatch, runs=3):
    monkeypatch.setattr(fc, "chat_with_fallback", chat_fn)
    return fc.forecast_question(
        post=POST,
        question=question,
        research_text="",
        research_sources=[],
        models=["a/one", "b/two"],
        runs=runs,
    )


# -- the bug ---------------------------------------------------------------
@pytest.mark.parametrize("factory", [q_numeric, q_mc], ids=["numeric", "multiple_choice"])
def test_no_placeholder_is_submitted_when_nothing_responds(factory, monkeypatch):
    with pytest.raises(NoModelsAvailable):
        _run(factory(), _all_models_dead, monkeypatch)


def test_binary_also_refuses(monkeypatch):
    with pytest.raises(NoModelsAvailable):
        _run(q_binary(), _all_models_dead, monkeypatch)


# -- and the behaviour it must not have broken -----------------------------
@pytest.mark.parametrize("factory", [q_numeric, q_mc], ids=["numeric", "multiple_choice"])
def test_no_placeholder_either_when_models_answer_badly(factory, monkeypatch):
    """Unusable prose used to become a uniform. A uniform scores far below the
    zero of a question left alone, and it uses the question up, so the question
    now stays open for the next poll, as a binary one always did."""
    with pytest.raises(llm.LLMError) as err:
        _run(factory(), _responds_with_junk, monkeypatch)
    assert "nothing submitted" in str(err.value)


def test_a_working_ensemble_is_unaffected(monkeypatch):
    def good(messages, models, **kwargs):
        text = " ".join(m["content"] for m in messages)
        if "JSON array" in text:
            return '["q"]', models[0]
        return REPLIES["binary"], models[0]

    forecast = _run(q_binary(), good, monkeypatch)
    assert 0.001 <= forecast.payload["probability_yes"] <= 0.999


# -- model naming, which is why the proxy rejected everything ---------------
@pytest.mark.parametrize(
    "provider,model,expected",
    [
        ("openrouter", "openai/gpt-5", "openai/gpt-5"),
        ("metaculus", "openai/gpt-5", "gpt-5"),
        ("metaculus", "anthropic/claude-sonnet-4", "claude-sonnet-4"),
        ("metaculus", "gpt-4o", "gpt-4o"),
        # Groq ids contain a slash of their own; stripping it gave a 404.
        ("groq", "meta-llama/llama-4-maverick-17b-128e-instruct",
         "meta-llama/llama-4-maverick-17b-128e-instruct"),
        ("groq", "openai/gpt-oss-120b", "openai/gpt-oss-120b"),
        ("groq", "llama-3.3-70b-versatile", "llama-3.3-70b-versatile"),
        ("gemini", "google/gemini-2.5-pro", "gemini-2.5-pro"),
    ],
)
def test_each_provider_gets_the_model_name_it_actually_serves(provider, model, expected):
    assert _adapt_model_name(llm.PROVIDERS[provider], model) == expected


# -- dead model bookkeeping, which is why one outage became hundreds of calls
def test_an_unavailable_model_is_not_retried(monkeypatch):
    calls = []

    def fake_chat(messages, model, **kwargs):
        calls.append(model)
        DEAD_MODELS.add(model)
        raise ModelUnavailable("no allowance")

    monkeypatch.setattr(llm, "chat", fake_chat)

    with pytest.raises(NoModelsAvailable):
        llm.chat_with_fallback([{"role": "user", "content": "x"}], ["a/one", "b/two"])
    assert calls == ["a/one", "b/two"]

    # Second question: both are known dead, so nothing is attempted at all.
    with pytest.raises(NoModelsAvailable):
        llm.chat_with_fallback([{"role": "user", "content": "x"}], ["a/one", "b/two"])
    assert calls == ["a/one", "b/two"], "a dead model must not be called again"


def test_a_transient_failure_does_not_kill_a_model(monkeypatch):
    calls = []

    def flaky(messages, model, **kwargs):
        calls.append(model)
        if model == "a/one":
            raise llm.LLMError("timeout")
        return "ok"

    monkeypatch.setattr(llm, "chat", flaky)
    text, used = llm.chat_with_fallback([{"role": "user", "content": "x"}], ["a/one", "b/two"])
    assert (text, used) == ("ok", "b/two")
    assert "a/one" not in DEAD_MODELS, "a timeout is not an allowance error"
