"""Model output is prose with JSON somewhere inside it. Parsing must not be brittle."""


import pytest


from bot import llm
from bot.llm import MODEL_PREFERENCES, STATIC_FALLBACK, extract_json, resolve_models


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"a": 1}', {"a": 1}),
        ("Here you go:\n```json\n{\"a\": 2}\n```\nHope that helps", {"a": 2}),
        ('Sure!\n{"a": 3, "b": [1,2,3,],}\n', {"a": 3, "b": [1, 2, 3]}),
        ("prose [1, 2, 3] more prose", [1, 2, 3]),
        ('```\n[{"p":0.1,"v":5}]\n```', [{"p": 0.1, "v": 5}]),
        ('[{"p":0.1},{"p":0.9}]', [{"p": 0.1}, {"p": 0.9}]),
        ('Reasoning follows.\n\n```json\n{"probability": 0.42}\n```', {"probability": 0.42}),
        ('{"nested": {"a": [1, {"b": 2}]}}', {"nested": {"a": [1, {"b": 2}]}}),
        ("```json\n{\"x\": 1}\n```\n```json\n{\"y\": 2}\n```", {"x": 1}),
        ('{"s": "a string with } and ] inside"}', {"s": "a string with } and ] inside"}),
    ],
)
def test_extract_json(text, expected):
    assert extract_json(text) == expected


@pytest.mark.parametrize("text", ["", "no json at all", "{unclosed", None])
def test_extract_json_rejects_garbage(text):
    with pytest.raises(ValueError):
        extract_json(text)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_resolve_models_falls_back_when_the_catalogue_is_unreachable(monkeypatch):
    """An unreachable catalogue must degrade to the static list, not crash."""
    monkeypatch.delenv("BOT_MODELS", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    import bot.llm as llm

    def boom(*args, **kwargs):
        raise OSError("no route to host")

    monkeypatch.setattr(llm.requests, "get", boom)
    llm._CATALOGUE_CACHE = None
    try:
        assert resolve_models(3) == STATIC_FALLBACK[:3]
    finally:
        llm._CATALOGUE_CACHE = None


def test_the_catalogue_is_read_without_a_key(monkeypatch):
    """The endpoint is public, and gating it on a key was the original bug.

    Before the fix, a run with no OpenRouter key skipped the catalogue entirely
    and used hardcoded model names from training data, which the Metaculus proxy
    then rejected. The resolver must use the live catalogue either way.
    """
    monkeypatch.delenv("BOT_MODELS", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    import bot.llm as llm

    seen = {}

    def fake_get(url, headers=None, timeout=None):
        seen["url"] = url
        seen["headers"] = headers
        return _FakeResponse(
            {
                "data": [
                    {"id": "openai/gpt-9-turbo", "created": 30},
                    {"id": "anthropic/claude-opus-9", "created": 20},
                    {"id": "google/gemini-9-pro", "created": 25},
                ]
            }
        )

    monkeypatch.setattr(llm.requests, "get", fake_get)
    llm._CATALOGUE_CACHE = None
    try:
        picked = resolve_models(3)
    finally:
        llm._CATALOGUE_CACHE = None

    assert seen["headers"] == {}, "no credentials should be sent"
    assert picked == ["openai/gpt-9-turbo", "anthropic/claude-opus-9", "google/gemini-9-pro"]
    assert picked != STATIC_FALLBACK[:3]


def test_resolve_models_honours_an_explicit_override(monkeypatch):
    monkeypatch.setenv("BOT_MODELS", " a/one , b/two ")
    assert resolve_models(5) == ["a/one", "b/two"]


def test_resolve_models_prefers_one_model_per_vendor(monkeypatch):
    monkeypatch.delenv("BOT_MODELS", raising=False)
    import bot.llm as llm

    llm._CATALOGUE_CACHE = [
        {"id": "openai/gpt-6-turbo", "created": 30},
        {"id": "openai/gpt-6-mini", "created": 31},
        {"id": "anthropic/claude-opus-5", "created": 20},
        {"id": "google/gemini-4-pro", "created": 25},
        {"id": "google/gemini-4-flash", "created": 26},
        {"id": "meta/llama-9", "created": 40},
        {"id": "openai/gpt-6-turbo:free", "created": 50},
    ]
    picked = resolve_models(3)
    llm._CATALOGUE_CACHE = None
    vendors = [m.split("/")[0] for m in picked]
    assert len(set(vendors)) == 3, picked
    assert not any(":free" in m for m in picked)
    assert "meta/llama-9" not in picked


def test_preference_patterns_compile():
    import re

    for pattern, score in MODEL_PREFERENCES:
        re.compile(pattern)
        assert 0 < score <= 100


# --------------------------------------------------------------------------
# Live-run findings: the catalogue handed back routing variants, and with no
# OpenRouter key the resolver must follow whichever provider does have one.
# --------------------------------------------------------------------------
def test_routing_variants_are_rejected(monkeypatch):
    """":batch" can take hours to return, against a three hour question window."""
    monkeypatch.delenv("BOT_MODELS", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    import bot.llm as llm

    llm._CATALOGUE_CACHE = [
        {"id": "openai/gpt-6-astra:batch", "created": 99},
        {"id": "openai/gpt-6-astra", "created": 98},
        {"id": "anthropic/claude-fable-5.1:batch", "created": 97},
        {"id": "anthropic/claude-fable-5.1", "created": 96},
        {"id": "google/gemini-2.5-pro:batch", "created": 95},
        {"id": "google/gemini-2.5-pro", "created": 94},
    ]
    try:
        picked = resolve_models(3)
    finally:
        llm._CATALOGUE_CACHE = None
    assert not any(":" in m for m in picked), picked
    assert picked == [
        "openai/gpt-6-astra",
        "anthropic/claude-fable-5.1",
        "google/gemini-2.5-pro",
    ]


def test_resolution_follows_the_provider_that_has_a_key(monkeypatch):
    """With only a Gemini key, the ensemble must be Gemini models, prefixed."""
    import bot.llm as llm

    monkeypatch.delenv("BOT_MODELS", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "x")

    def fake_get(url, headers=None, timeout=None):
        assert "generativelanguage" in url, url
        return _FakeResponse(
            {
                "data": [
                    {"id": "models/gemini-2.5-flash-lite"},
                    {"id": "models/gemini-2.5-pro"},
                    {"id": "models/gemini-3-flash"},
                ]
            }
        )

    monkeypatch.setattr(llm.requests, "get", fake_get)
    llm._CATALOGUE_CACHE = None
    try:
        picked = resolve_models(2)
    finally:
        llm._CATALOGUE_CACHE = None

    assert all(m.startswith("gemini/") for m in picked), picked
    assert picked[0] == "gemini/models/gemini-3-flash", picked
    assert "lite" not in picked[0]


def test_model_ranking_matches_what_the_live_run_taught_us(monkeypatch):
    """The first live Gemini run picked three models and all three failed.

    models/aqa is not a chat endpoint and returned 404. gemini-2.5-pro is
    closed to new keys and returned 404. gemini-pro-latest returned 429 on
    every attempt because the pro class per-minute limit on a free key cannot
    feed an ensemble. Flash answers; capability it cannot deliver is worth
    nothing.
    """
    from bot.llm import _rank_bare

    live = [
        "models/gemini-2.5-pro",
        "models/gemini-pro-latest",
        "models/aqa",
        "models/embedding-001",
        "models/imagen-4",
        "models/veo-3",
        "models/gemini-3-flash",
        "models/gemini-3-flash-lite",
        "models/gemini-2.5-flash",
    ]
    ranked = _rank_bare(live)

    for junk in ("models/aqa", "models/embedding-001", "models/imagen-4", "models/veo-3"):
        assert junk not in ranked, junk
    assert ranked[0] == "models/gemini-3-flash"
    assert ranked.index("models/gemini-3-flash") < ranked.index("models/gemini-3-pro") if "models/gemini-3-pro" in ranked else True
    assert ranked.index("models/gemini-2.5-flash") < ranked.index("models/gemini-2.5-pro")
    assert ranked.index("models/gemini-3-flash") < ranked.index("models/gemini-2.5-flash"), "newest first"


def test_the_rate_limiter_spaces_requests(monkeypatch):
    """Fifteen simultaneous requests at a free key produced four minutes of 429s."""
    import time as _time

    from bot.llm import PROVIDER_LIMITS, _RateLimiter

    limiter = _RateLimiter()
    PROVIDER_LIMITS.setdefault("fake", (60.0, 1))
    started = _time.monotonic()
    for _ in range(3):
        limiter.wait("fake")
    elapsed = _time.monotonic() - started
    assert elapsed >= 1.8, f"three calls at 60/min should take about 2s, took {elapsed:.2f}s"


def test_a_gemini_prefixed_model_routes_to_gemini_with_the_bare_name(monkeypatch):
    """The prefix picks the provider; the provider gets the name it understands.

    Google's catalogue returns ids as "models/gemini-2.5-pro" but its
    OpenAI-compatible endpoint is documented with the bare name, so both the
    routing prefix and the catalogue namespace come off.
    """
    import bot.llm as llm

    monkeypatch.setenv("GEMINI_API_KEY", "x")
    prov, bare = llm._provider_for("gemini/models/gemini-2.5-pro")
    assert prov.name == "gemini"
    assert bare == "gemini-2.5-pro"


def test_a_bare_provider_name_is_passed_through_untouched(monkeypatch):
    """Groq ids have no namespace at all; nothing may be stripped off them."""
    import bot.llm as llm

    monkeypatch.setenv("GROQ_API_KEY", "x")
    prov, bare = llm._provider_for("groq/llama-3.3-70b-versatile")
    assert prov.name == "groq"
    assert bare == "llama-3.3-70b-versatile"


def test_openrouter_keeps_its_vendor_prefix_in_the_call(monkeypatch):
    import bot.llm as llm

    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    prov, bare = llm._provider_for("openrouter/openai/gpt-6-astra")
    assert prov.name == "openrouter"
    assert bare == "openai/gpt-6-astra"


# --------------------------------------------------------------------------
# Throughput. Run #7 spent 83 rate-limit sleeps and timed out with a third of
# the questions unforecast, which under this scoring rule is the same as
# getting them wrong.
# --------------------------------------------------------------------------
def test_each_model_gets_its_own_allowance():
    """Free tiers meter per model, so three models are three budgets, not one."""
    import time as _time

    from bot.llm import PROVIDER_LIMITS, _RateLimiter

    PROVIDER_LIMITS.setdefault("slowfake", (60.0, 2))
    limiter = _RateLimiter()

    started = _time.monotonic()
    limiter.wait("slowfake|model-a")
    limiter.wait("slowfake|model-b")
    limiter.wait("slowfake|model-c")
    spread = _time.monotonic() - started
    assert spread < 0.3, f"different models must not queue behind each other ({spread:.2f}s)"

    started = _time.monotonic()
    limiter.wait("slowfake|model-a")
    limiter.wait("slowfake|model-a")
    same = _time.monotonic() - started
    assert same >= 0.9, f"the same model must be spaced ({same:.2f}s)"


class _FakeRateLimited:
    def __init__(self, headers=None, text=""):
        self.headers = headers or {}
        self.text = text
        self.status_code = 429


def test_the_stated_retry_delay_is_used():
    """Google puts the wait in the error body; guessing ignored it."""
    from bot.llm import _retry_delay

    assert _retry_delay(_FakeRateLimited({"Retry-After": "17"})) == 17.0
    body = '{"error":{"code":429,"details":[{"retryDelay":"23s"}]}}'
    assert _retry_delay(_FakeRateLimited(text=body)) == 23.0
    assert _retry_delay(_FakeRateLimited(text="no idea")) is None
    # A header takes precedence over the body.
    assert _retry_delay(_FakeRateLimited({"Retry-After": "5"}, body)) == 5.0


# -- ensembles across providers -------------------------------------------
# The first design spread the ensemble across providers, one model each. The
# watcher logs for 21 to 30 September show what that did on free keys: 63 of the
# 68 answers that reached a tournament forecast came from Groq's gpt-oss-120b
# (Metaculus FutureEval model leaderboard: -0.26), because the Gemini member
# (Gemini 3.5 Flash +12.17, 3.6 Flash +13.22 on the same board) answered 429 or
# 503 and the fallback moved on. Family diversity is worth having between
# models of similar strength; between these two it hands the forecast to the
# weaker one. So the Flash versions form the ensemble and other providers
# stand in behind them.


def _serve(monkeypatch, catalogues):
    monkeypatch.setattr(llm, "_CATALOGUE_CACHE", None, raising=False)
    monkeypatch.setattr(
        llm, "provider_catalogue", lambda prov: list(catalogues.get(prov.name, []))
    )


def test_flash_versions_lead_and_other_providers_stand_in(monkeypatch):
    monkeypatch.setattr(llm, "STAND_INS_ENABLED", True)
    monkeypatch.delenv("BOT_MODELS", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_MODELS_TOKEN", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("GROQ_API_KEY", "q")
    monkeypatch.setenv("METACULUS_TOKEN", "t")
    _serve(
        monkeypatch,
        {
            "gemini": ["models/gemini-3.8-flash", "models/gemini-3.7-flash"],
            "groq": ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"],
        },
    )
    picked = llm.resolve_models(3)
    assert picked == [
        "gemini/models/gemini-3.8-flash",
        "gemini/models/gemini-3.7-flash",
        "groq/openai/gpt-oss-120b",
    ], picked
    assert llm.FALLBACK_ONLY == {"groq/openai/gpt-oss-120b"}
    # The Metaculus token is always set, so it must not crowd out a real key.
    assert not any(m.startswith("metaculus/") for m in picked), picked


def test_one_keyed_provider_still_fills_the_ensemble(monkeypatch):
    monkeypatch.delenv("BOT_MODELS", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("METACULUS_TOKEN", "t")
    _serve(
        monkeypatch,
        {
            "gemini": [
                "models/gemini-3.8-flash",
                "models/gemini-3.7-flash",
                "models/gemini-3.6-flash",
            ]
        },
    )
    picked = llm.resolve_models(3)
    assert len(picked) == 3
    assert all(m.startswith("gemini/") for m in picked), picked


def test_the_proxy_is_used_only_when_nothing_else_is_keyed(monkeypatch):
    monkeypatch.delenv("BOT_MODELS", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("METACULUS_TOKEN", "t")
    _serve(monkeypatch, {"metaculus": ["gpt-5", "claude-sonnet-4"]})
    picked = llm.resolve_models(2)
    assert all(m.startswith("metaculus/") for m in picked), picked
