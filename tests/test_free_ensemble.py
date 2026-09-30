"""The free-tier ensemble: strong models first, stand-ins only when they must.

What the watcher logs showed from 21 to 30 September, and what each test here
guards against:

* 63 of 68 answers behind tournament forecasts came from gpt-oss-120b, the
  weakest model in the ensemble by Metaculus's own model leaderboard, because
  the Gemini Flash member answered 429 or 503 and the run moved on to Groq
  within a minute. Now every run tries all the Flash versions first, and a
  question with too few strong answers waits for the next poll instead of
  being submitted on the stand-in.
* Only the newest two Flash versions were ever asked. Each version has its own
  free allowance and its own load; 3.6 Flash answered 17 times in the one run
  that asked it.
* "HTTP 429" was all the log said, which cannot tell a per-minute limit from a
  used-up daily allowance. The two need opposite handling.
* The comment said "(not captured)" where the reasoning belongs, while the
  rules ask for comments "so everyone can see their reasoning".
"""

from datetime import datetime, timedelta, timezone

import pytest

from bot import forecast as fc
from bot import llm
from bot.forecast import EnsembleTooThin
from tests.test_pipeline import POST, REPLIES, q_binary, q_mc, q_numeric

FLASH = [
    "gemini/models/gemini-3.8-flash",
    "gemini/models/gemini-3.7-flash",
    "gemini/models/gemini-3.6-flash",
    "gemini/models/gemini-3.5-flash",
]
STAND_IN = "groq/openai/gpt-oss-120b"


def _serve(monkeypatch, catalogues, openrouter=None):
    monkeypatch.setattr(llm, "_CATALOGUE_CACHE", openrouter, raising=False)
    monkeypatch.setattr(
        llm, "provider_catalogue", lambda prov: list(catalogues.get(prov.name, []))
    )


def _keys(monkeypatch, **present):
    for name in (
        "BOT_MODELS",
        "OPENROUTER_API_KEY",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "GITHUB_MODELS_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in present.items():
        monkeypatch.setenv(name, value)


# -- which models --------------------------------------------------------------
def test_up_to_four_flash_versions_newest_first_one_id_each(monkeypatch):
    _keys(monkeypatch, GEMINI_API_KEY="g", GROQ_API_KEY="q")
    _serve(
        monkeypatch,
        {
            "gemini": [
                "models/gemini-2.5-flash",
                "models/gemini-3.5-flash-lite",
                "models/gemini-3.6-flash",
                "models/gemini-flash-latest",
                "models/gemini-3.8-flash-001",
                "models/gemini-3.8-flash",
                "models/gemini-3.5-flash",
                "models/gemini-2.5-pro",
                "models/gemini-3.7-flash",
                "models/gemini-3.9-flash-exp",
            ],
            "groq": ["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
        },
    )
    picked = llm.resolve_models(3)
    assert picked == FLASH + [STAND_IN], picked
    assert llm.FALLBACK_ONLY == {STAND_IN}
    assert llm.PRIMARY_PROVIDER == "gemini"


@pytest.mark.parametrize(
    "mid,expected",
    [
        ("models/gemini-3.8-flash", 3.8),
        ("gemini-3-flash", 3.0),
        # The largest number in the id is a date, not a version.
        ("models/gemini-2.5-flash-preview-09-2025", 2.5),
        ("models/gemini-flash-latest", None),
    ],
)
def test_the_gemini_version_is_the_number_after_gemini(mid, expected):
    assert llm.gemini_version(mid) == expected


def test_old_and_lite_flash_models_are_not_primaries():
    # Leaderboard: 2.5 Flash -8.07, 3 Flash +6.39, 3.5 Flash-Lite +0.70.
    for mid in ("models/gemini-2.5-flash", "models/gemini-3-flash", "models/gemini-3.5-flash-lite"):
        assert not llm.is_primary_flash(mid), mid
    assert llm.is_primary_flash("models/gemini-3.5-flash")


def test_with_too_few_flash_models_the_old_spread_is_kept(monkeypatch):
    _keys(monkeypatch, GEMINI_API_KEY="g", GROQ_API_KEY="q")
    _serve(
        monkeypatch,
        {"gemini": ["models/gemini-3.8-flash", "models/gemini-2.5-flash"], "groq": ["openai/gpt-oss-120b"]},
    )
    picked = llm.resolve_models(3)
    assert picked[:2] == ["gemini/models/gemini-3.8-flash", STAND_IN], picked
    assert not llm.FALLBACK_ONLY


def test_free_stand_ins_are_ordered_strongest_first(monkeypatch):
    """Nemotron 3 Ultra +5.83, GPT-OSS 120B -0.26. GitHub Models is retired."""
    _keys(
        monkeypatch,
        GEMINI_API_KEY="g",
        GROQ_API_KEY="q",
        OPENROUTER_API_KEY="o",
        GITHUB_MODELS_TOKEN="h",
    )
    _serve(
        monkeypatch,
        {"gemini": ["models/gemini-3.8-flash", "models/gemini-3.7-flash"], "groq": ["openai/gpt-oss-120b"]},
        openrouter=[
            {"id": "openai/gpt-6.1-sol", "created": 99},
            {"id": "nvidia/nemotron-3-ultra-550b-a55b:free", "created": 50},
            {"id": "nvidia/nemotron-3.5-lightning:free", "created": 60},
        ],
    )
    picked = llm.resolve_models(3)
    assert picked[2:] == [
        "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
        STAND_IN,
    ], picked
    assert not any(m.startswith("github/") for m in picked), "GitHub Models was retired on 30 July 2026"
    assert "openai/gpt-6.1-sol" not in " ".join(picked), "a free key must never be sent to a paid model"


def test_the_run_pace_follows_the_strong_provider(monkeypatch):
    _keys(monkeypatch, GEMINI_API_KEY="g", GROQ_API_KEY="q", OPENROUTER_API_KEY="o")
    _serve(
        monkeypatch,
        {"gemini": ["models/gemini-3.8-flash", "models/gemini-3.7-flash"], "groq": ["openai/gpt-oss-120b"]},
        openrouter=[],
    )
    llm.resolve_models(3)
    assert llm.active_provider().name == "gemini"
    assert llm.provider_is_metered()


# -- how the calls fail ---------------------------------------------------------
class Resp:
    def __init__(self, status=200, text="", headers=None, content="ok"):
        self.status_code = status
        self.text = text
        self.headers = headers or {}
        self._content = content

    @property
    def ok(self):
        return self.status_code < 400

    def json(self):
        return {"choices": [{"message": {"content": self._content}, "finish_reason": "stop"}], "usage": {}}


@pytest.fixture
def wire(monkeypatch):
    """Replace the network and the clock's sleeps; record what was sent."""
    sent, sleeps, queue = [], [], []

    def post(url, headers=None, json=None, timeout=None):
        sent.append({"url": url, "body": dict(json)})
        return queue.pop(0) if queue else Resp()

    monkeypatch.setattr(llm.requests, "post", post)
    monkeypatch.setattr(llm.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(llm.LIMITER, "wait", lambda key: None)
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("GROQ_API_KEY", "q")
    llm.DEAD_MODELS.clear()
    llm.NO_REASONING.clear()
    yield sent, sleeps, queue
    llm.DEAD_MODELS.clear()
    llm.NO_REASONING.clear()


GOOGLE_DAILY = (
    '[{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "You exceeded your '
    'current quota", "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", '
    '"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}, '
    '{"retryDelay": "54s"}]}}]'
)
GOOGLE_MINUTE = GOOGLE_DAILY.replace("PerDay", "PerMinute")


def test_a_used_up_daily_allowance_is_not_retried_until_the_reset(wire):
    sent, sleeps, queue = wire
    queue.append(Resp(429, GOOGLE_DAILY))
    with pytest.raises(llm.ModelUnavailable):
        llm.chat([{"role": "user", "content": "x"}], FLASH[0])
    assert len(sent) == 1 and not sleeps, "no retries and no sleeping against a daily limit"
    until = llm.EXHAUSTED_UNTIL[FLASH[0]]
    assert 0 < until - datetime.now(timezone.utc).timestamp() <= 25 * 3600
    assert not llm.model_available(FLASH[0])

    # The next question skips it without a request.
    text, used = llm.chat_with_fallback([{"role": "user", "content": "y"}], FLASH[:2])
    assert used == FLASH[1]
    assert len(sent) == 2


def test_a_per_minute_limit_is_retried_briefly_then_left_to_the_next_version(wire):
    sent, sleeps, queue = wire
    queue.extend([Resp(429, GOOGLE_MINUTE), Resp(429, GOOGLE_MINUTE)])
    with pytest.raises(llm.LLMError) as err:
        llm.chat([{"role": "user", "content": "x"}], FLASH[0], attempts=2)
    assert not isinstance(err.value, llm.ModelUnavailable)
    assert "PerMinute" in str(err.value), "the log must say which limit was hit"
    assert sleeps == [llm.RATE_LIMIT_MAX_SLEEP], sleeps
    assert llm.model_available(FLASH[0])


def test_groq_out_of_daily_tokens_is_out_for_the_stated_time(wire):
    sent, sleeps, queue = wire
    body = '{"error":{"message":"Rate limit reached on tokens per day (TPD): Limit 200000","type":"tokens"}}'
    queue.append(Resp(429, body, headers={"Retry-After": "4957"}))
    with pytest.raises(llm.ModelUnavailable):
        llm.chat([{"role": "user", "content": "x"}], STAND_IN)
    assert llm.EXHAUSTED_UNTIL[STAND_IN] > datetime.now(timezone.utc).timestamp() + 4000


def test_an_overloaded_model_is_not_written_off(wire):
    sent, sleeps, queue = wire
    overloaded = '{"error": {"code": 503, "status": "UNAVAILABLE", "message": "The model is overloaded."}}'
    queue.extend([Resp(503, overloaded), Resp(503, overloaded)])
    with pytest.raises(llm.LLMError) as err:
        llm.chat([{"role": "user", "content": "x"}], FLASH[0], attempts=2)
    assert "overloaded" in str(err.value)
    assert llm.model_available(FLASH[0])


def test_a_model_that_just_refused_moves_behind_its_own_tier_not_behind_stand_ins(wire):
    sent, sleeps, queue = wire
    llm.FALLBACK_ONLY.add(STAND_IN)
    llm._COOLDOWN_UNTIL[FLASH[0]] = llm.time.monotonic() + 60
    text, used = llm.chat_with_fallback([{"role": "user", "content": "x"}], [FLASH[0], FLASH[1], STAND_IN])
    assert used == FLASH[1]

    llm._COOLDOWN_UNTIL[FLASH[1]] = llm.time.monotonic() + 60
    text, used = llm.chat_with_fallback([{"role": "user", "content": "x"}], [FLASH[0], FLASH[1], STAND_IN])
    assert used in (FLASH[0], FLASH[1]), "an overloaded Flash model still beats the stand-in"


def test_a_prompt_too_long_does_not_kill_the_model(wire):
    sent, sleeps, queue = wire
    queue.append(Resp(413, '{"error":{"code":"tokens_limit_reached","message":"Request body too large"}}'))
    with pytest.raises(llm.LLMError) as err:
        llm.chat([{"role": "user", "content": "x"}], STAND_IN)
    assert not isinstance(err.value, llm.ModelUnavailable)
    assert STAND_IN not in llm.DEAD_MODELS


# -- the ensemble ---------------------------------------------------------------
def _question(factory, minutes_to_close):
    q = factory()
    q["scheduled_close_time"] = (
        datetime.now(timezone.utc) + timedelta(minutes=minutes_to_close)
    ).isoformat()
    return q


def _forecast(question, monkeypatch, answer, runs=4):
    monkeypatch.setattr(fc, "chat_with_fallback", answer)
    return fc.forecast_question(
        post=POST,
        question=question,
        research_text="",
        research_sources=[],
        models=FLASH + [STAND_IN],
        runs=runs,
    )


def test_each_run_is_led_by_a_different_flash_version_and_stand_ins_come_last(monkeypatch):
    llm.FALLBACK_ONLY.add(STAND_IN)
    orders = []

    def record(messages, models, **kwargs):
        orders.append(list(models))
        return REPLIES["binary"], models[0]

    _forecast(_question(q_binary, 10), monkeypatch, record)
    assert sorted(o[0] for o in orders) == sorted(FLASH)
    assert all(o[-1] == STAND_IN for o in orders), orders


def test_while_there_is_time_stand_ins_are_not_even_asked(monkeypatch):
    llm.FALLBACK_ONLY.add(STAND_IN)
    orders = []

    def record(messages, models, **kwargs):
        orders.append(list(models))
        return REPLIES["binary"], models[0]

    _forecast(_question(q_binary, 80), monkeypatch, record)
    assert orders and all(STAND_IN not in o for o in orders), orders


def test_too_few_strong_answers_waits_and_keeps_what_it_has(monkeypatch):
    llm.FALLBACK_ONLY.add(STAND_IN)
    calls = []

    def one_answer(messages, models, **kwargs):
        calls.append(models[0])
        if len(calls) == 1:
            return REPLIES["binary"], models[0]
        raise llm.LLMError("HTTP 503 overloaded")

    question = _question(q_binary, 80)
    with pytest.raises(EnsembleTooThin):
        _forecast(question, monkeypatch, one_answer)
    assert len(calls) == 4

    # Next poll: the Flash models answer, and only the three missing runs are asked.
    calls.clear()

    def all_answer(messages, models, **kwargs):
        calls.append(models[0])
        return REPLIES["binary"], models[0]

    forecast = _forecast(question, monkeypatch, all_answer)
    assert len(calls) == 3, calls
    assert forecast.runs_used == 4


def test_at_the_deadline_the_stand_in_is_used_rather_than_nothing(monkeypatch):
    llm.FALLBACK_ONLY.add(STAND_IN)

    def only_stand_in(messages, models, **kwargs):
        assert models[-1] == STAND_IN
        return REPLIES["binary"], STAND_IN

    forecast = _forecast(_question(q_binary, 10), monkeypatch, only_stand_in)
    assert forecast.models_used == [STAND_IN]
    assert any("stand-ins used" in n for n in forecast.notes), forecast.notes


def test_at_the_deadline_one_strong_answer_is_not_outvoted_by_stand_ins(monkeypatch):
    llm.FALLBACK_ONLY.add(STAND_IN)
    replies = iter(
        [
            ("PROBABILITY: 12%", FLASH[0]),
            ("PROBABILITY: 70%", STAND_IN),
            ("PROBABILITY: 72%", STAND_IN),
            ("PROBABILITY: 75%", STAND_IN),
        ]
    )
    forecast = _forecast(_question(q_binary, 10), monkeypatch, lambda m, models, **k: next(replies))
    assert forecast.models_used == [FLASH[0]]
    assert forecast.payload["probability_yes"] < 0.2
    assert any("used alone" in n for n in forecast.notes), forecast.notes


def test_with_enough_strong_answers_the_stand_ins_are_left_out(monkeypatch):
    llm.FALLBACK_ONLY.add(STAND_IN)
    replies = iter(
        [
            ("PROBABILITY: 20%", FLASH[0]),
            ("PROBABILITY: 24%", FLASH[1]),
            ("PROBABILITY: 90%", STAND_IN),
            ("PROBABILITY: 88%", STAND_IN),
        ]
    )
    forecast = _forecast(_question(q_binary, 10), monkeypatch, lambda m, models, **k: next(replies))
    assert set(forecast.models_used) == {FLASH[0], FLASH[1]}
    assert forecast.payload["probability_yes"] < 0.3


def test_when_no_strong_model_can_answer_the_question_waits(monkeypatch):
    llm.FALLBACK_ONLY.add(STAND_IN)
    for m in FLASH:
        llm.EXHAUSTED_UNTIL[m] = datetime.now(timezone.utc).timestamp() + 3600

    def nobody(messages, models, **kwargs):
        raise llm.NoModelsAvailable("all out for the day")

    with pytest.raises(EnsembleTooThin):
        _forecast(_question(q_numeric, 80), monkeypatch, nobody)


@pytest.mark.parametrize("factory", [q_binary, q_numeric, q_mc], ids=["binary", "numeric", "mc"])
def test_the_comment_carries_real_reasoning(factory, monkeypatch):
    kind = factory()["type"]

    def answer(messages, models, **kwargs):
        return "Base rate reasoning, step by step.\n" + REPLIES[kind], models[0]

    forecast = _forecast(_question(factory, 10), monkeypatch, answer)
    assert "Base rate reasoning, step by step." in forecast.comment
    assert "(not captured)" not in forecast.comment


def test_the_binary_comment_quotes_the_run_nearest_the_median(monkeypatch):
    replies = iter(
        [
            ("far low\nPROBABILITY: 5%", FLASH[0]),
            ("near\nPROBABILITY: 30%", FLASH[1]),
            ("far high\nPROBABILITY: 70%", FLASH[2]),
        ]
    )
    forecast = _forecast(_question(q_binary, 10), monkeypatch, lambda m, models, **k: next(replies), runs=3)
    assert "near" in forecast.comment and "far high" not in forecast.comment


def test_a_long_reasoning_is_trimmed_not_dropped():
    text = "start " + "x" * 10000 + " PROBABILITY: 12%"
    out = fc._reasoning_excerpt(text)
    assert len(out) <= fc.COMMENT_REASONING_CHARS + 20
    assert out.startswith("start") and out.endswith("PROBABILITY: 12%")


def test_without_tiers_nothing_waits(monkeypatch):
    """Sponsored credits or a single provider: the old behaviour, unchanged."""
    assert not llm.FALLBACK_ONLY

    def one_answer(messages, models, **kwargs):
        return REPLIES["binary"], models[0]

    forecast = _forecast(_question(q_binary, 80), monkeypatch, one_answer, runs=1)
    assert forecast.runs_used == 1


def test_an_unusable_answer_is_not_kept_for_the_next_poll(monkeypatch):
    llm.FALLBACK_ONLY.add(STAND_IN)
    replies = iter([("I cannot say.", FLASH[0])] + [(REPLIES["binary"], FLASH[1])] * 0)

    def first_poll(messages, models, **kwargs):
        try:
            return next(replies)
        except StopIteration:
            raise llm.LLMError("HTTP 503 overloaded")

    question = _question(q_binary, 80)
    with pytest.raises(EnsembleTooThin):
        _forecast(question, monkeypatch, first_poll)
    assert fc._STRONG_ANSWERS[(question["id"], "binary")] == []

    calls = []

    def second_poll(messages, models, **kwargs):
        calls.append(models[0])
        return REPLIES["binary"], models[0]

    forecast = _forecast(question, monkeypatch, second_poll)
    assert len(calls) == 4, "all four runs are asked again, the junk answer holds no slot"
    assert forecast.runs_used == 4


def test_the_probe_reports_each_model_without_thinking(wire):
    sent, sleeps, queue = wire
    queue.extend([Resp(content="OK"), Resp(503, '{"error":{"status":"UNAVAILABLE","message":"overloaded"}}')])
    out = llm.probe([FLASH[0], FLASH[1]])
    assert out[FLASH[0]].startswith("answered")
    assert out[FLASH[1]].startswith("failed") and "overloaded" in out[FLASH[1]]
    assert all("reasoning_effort" not in s["body"] for s in sent)
    assert len(sent) == 2, "one request per model, no retries"


class BadBody(Resp):
    def json(self):
        raise ValueError("Expecting value: line 1 column 1 (char 0)")


def test_an_ok_answer_with_an_unreadable_body_falls_through_to_the_next_model(wire):
    """The first dry run ended here: 3.7 Flash answered 200 with a body that was not JSON."""
    sent, sleeps, queue = wire
    queue.extend([BadBody(200, ""), BadBody(200, ""), Resp(content="PROBABILITY: 20%")])
    text, used = llm.chat_with_fallback(
        [{"role": "user", "content": "x"}], [FLASH[1], FLASH[2]], attempts=2
    )
    assert used == FLASH[2] and text == "PROBABILITY: 20%"
    assert llm.model_available(FLASH[1]), "an unreadable answer is not a reason to write the model off"


def test_the_probe_survives_anything(wire, monkeypatch):
    def explode(*a, **k):
        raise KeyError("surprise")

    monkeypatch.setattr(llm, "chat", explode)
    out = llm.probe([FLASH[0]])
    assert out[FLASH[0]].startswith("crashed")
