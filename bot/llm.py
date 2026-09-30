"""LLM access with a fallback chain, speaking one dialect to four providers.

Every provider here exposes an OpenAI-compatible /chat/completions endpoint, so
there is a single code path and no litellm (which is 133 MB and pulls in boto3).

Model ids are resolved at runtime from the provider catalogue rather than
pinned, because the season runs four months and model names churn inside that
window. BOT_MODELS overrides the resolver when a specific set is wanted.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import threading

import requests

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    key_env: str
    auth_scheme: str = "Bearer"

    @property
    def key(self) -> str | None:
        return os.environ.get(self.key_env) or None


PROVIDERS: dict[str, Provider] = {
    # Where the sponsored tournament credits live.
    "openrouter": Provider("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    # Metaculus's own proxy, authenticated with the bot token. Documented as a
    # fallback: it does not support every model.
    "metaculus": Provider(
        "metaculus",
        "https://llm-proxy.metaculus.com/proxy/openai/v1",
        "METACULUS_TOKEN",
        auth_scheme="Token",
    ),
    "gemini": Provider(
        "gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"
    ),
    "groq": Provider("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    # GitHub Models, reached with the workflow's own GITHUB_TOKEN when the
    # workflow grants "models: read". Free, but small: 50 requests a day on the
    # "high" tier, at most 8,000 tokens in and 4,000 out per request.
    "github": Provider("github", "https://models.github.ai/inference", "GITHUB_MODELS_TOKEN"),
}

# A free OpenRouter key buys the ":free" variants and nothing else; asking it
# for a paid model returns 402. This bot runs at zero cost, so unless this is
# switched off an OpenRouter key is treated as a free-tier key.
OPENROUTER_FREE_ONLY = (os.environ.get("OPENROUTER_FREE_ONLY") or "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)

# Ordered preference patterns. Higher score wins; ties break on how recently the
# model was published. Only vendors whose credits Metaculus sponsors are listed,
# because those are the ones that cost nothing.
MODEL_PREFERENCES: list[tuple[str, int]] = [
    (r"^openai/(gpt-[6-9]|o[5-9])(?!.*mini)", 100),
    (r"^anthropic/claude.*(opus|fable)", 100),
    (r"^google/gemini.*pro", 95),
    (r"^openai/gpt-[5-9]", 85),
    (r"^anthropic/claude.*sonnet", 85),
    (r"^google/gemini.*flash(?!.*lite)", 80),
    (r"^openai/", 50),
    (r"^anthropic/", 50),
    (r"^google/gemini", 50),
]

# Any ":suffix" on an OpenRouter id is a routing variant, not a different
# model. The live catalogue handed back "openai/gpt-6-astra:batch", and batch
# routing can take hours to return, against a question window of three. Reject
# every variant rather than blocklisting them one at a time.
VARIANT = re.compile(r":")
EXCLUDE = re.compile(r"(-preview|audio|image|tts|embed|moderation|search)", re.I)

# Fallback if the catalogue cannot be read. Deliberately family-diverse.
STATIC_FALLBACK = ["openai/gpt-5", "anthropic/claude-sonnet-4", "google/gemini-2.5-pro"]


# Requests per minute and maximum concurrency per provider. The first live run
# fired fifteen requests at once at a free Gemini key and got nothing but 429s
# for four minutes. Sending fewer requests is what makes them succeed.
PROVIDER_LIMITS: dict[str, tuple[float, int]] = {
    "openrouter": (120.0, 6),
    "gemini": (10.0, 2),
    "groq": (25.0, 3),
    "metaculus": (20.0, 2),
    "github": (10.0, 2),
}

# Free OpenRouter models are limited to 20 requests a minute.
OPENROUTER_FREE_LIMITS: tuple[float, int] = (20.0, 1)


def provider_limits(name: str) -> tuple[float, int]:
    if name == "openrouter" and OPENROUTER_FREE_ONLY:
        return OPENROUTER_FREE_LIMITS
    return PROVIDER_LIMITS.get(name, (60.0, 4))


# -- who forecasts, and who only stands in ---------------------------------
# From 21 to 30 September the bot was, in practice, a gpt-oss-120b bot. The
# watcher logs show 68 successful model calls in tournament mode, and 63 of
# them came from Groq's gpt-oss-120b: the Gemini models answered 429 or 503
# nearly every time, and the fallback order moved on to Groq after a minute.
# Metaculus's own FutureEval model leaderboard (read 30 Sep 2026, head-to-head
# peer score with GPT-4o at 0) puts GPT-OSS 120B at -0.26, Gemini 3.5 Flash at
# +12.17 and Gemini 3.6 Flash at +13.22. The free Flash models are among the
# best forecasters on that board; the free Groq model is not.
#
# So the ensemble is built from the Gemini Flash versions the key can see, and
# every other model is a stand-in, asked only when no Flash model has answered
# and the question is close to closing. Each Flash version has its own free
# allowance and its own load, so spreading runs over several of them is also
# what gets them answered.
PRIMARY_MODELS = int(os.environ.get("PRIMARY_MODELS") or 4)
PRIMARY_GEMINI_MIN_VERSION = float(os.environ.get("PRIMARY_GEMINI_MIN_VERSION") or 3.5)

# Stand-ins in the order they are asked, strongest first by the same board:
# Nemotron 3 Ultra +5.83 (OpenRouter free), GPT-4.1 +2.77 (GitHub Models),
# GPT-OSS 120B -0.26 (Groq).
FALLBACK_PROVIDER_ORDER = ("openrouter", "github", "groq")
OPENROUTER_FREE_PREFERENCES = (r"^nvidia/nemotron-3-ultra[^:]*:free$",)
GITHUB_MODELS_PREFERENCES = ("openai/gpt-4.1",)

# Filled in by resolve_models. A model in this set is only ever asked after
# every primary model in the same run has failed.
FALLBACK_ONLY: set[str] = set()


# The strongest controlled result Metaculus has published: eight pairs of bots
# differing only in reasoning effort, and the higher-effort one won all eight
# (one-sided sign test p = 0.004). Their example pair scored 11.3 against 4.56
# peer points per question. Nothing else in their Spring 2026 analysis reached
# significance, so this is the first thing to turn on and the last to turn off.
#
# BOT_REASONING=off disables it.
REASONING_EFFORT = (os.environ.get("BOT_REASONING") or "high").strip().lower()

# Thinking tokens are billed against the output budget, so a model that thinks
# hard inside a 3000 token ceiling can spend the whole allowance reasoning and
# return an empty answer. That is worse than not thinking at all: it turns a
# good forecast into a parse failure.
REASONING_MAX_TOKENS = 8000

# Models that answered "I do not know that parameter", or that truncated with
# reasoning on. Remembered for the process so one probe does not become one per
# call.
NO_REASONING: set[str] = set()

_REASONING_REJECTED = re.compile(
    r"(reasoning|thinking|effort|unknown (field|parameter)|unrecognized|not supported|invalid.*param)",
    re.I,
)


def _reasoning_params(prov: "Provider", limit_key: str) -> dict:
    """The provider's spelling of "think harder", or nothing."""
    if REASONING_EFFORT in ("", "off", "none", "0", "false"):
        return {}
    if limit_key in NO_REASONING:
        return {}
    if prov.name == "github":
        # GPT-4.1 is not a reasoning model, and the free tier caps output at
        # 4,000 tokens, so there is no room for thinking anyway.
        return {}
    if prov.name == "openrouter":
        return {"reasoning": {"effort": REASONING_EFFORT}}
    return {"reasoning_effort": REASONING_EFFORT}


# GitHub Models' free tier refuses more than 4,000 output tokens per request.
GITHUB_MAX_OUTPUT_TOKENS = 4000


class _RateLimiter:
    """A minimum spacing plus a concurrency cap, keyed by provider and model.

    Free tiers meter each model separately, so throttling the provider as a
    whole throws away most of the budget: an ensemble of three models has three
    separate allowances, not one shared one.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_free: dict[str, float] = {}
        self._slots: dict[str, threading.Semaphore] = {}

    def slot(self, key: str) -> threading.Semaphore:
        provider = key.split("|", 1)[0]
        with self._lock:
            if key not in self._slots:
                _, concurrency = provider_limits(provider)
                self._slots[key] = threading.Semaphore(concurrency)
            return self._slots[key]

    def wait(self, key: str) -> None:
        provider = key.split("|", 1)[0]
        rpm, _ = provider_limits(provider)
        spacing = 60.0 / max(rpm, 1.0)
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_free.get(key, 0.0))
            self._next_free[key] = start + spacing
        delay = start - time.monotonic()
        if delay > 0:
            time.sleep(delay)


LIMITER = _RateLimiter()


class LLMError(RuntimeError):
    pass


class ModelUnavailable(LLMError):
    """The model will never work with these credentials: wrong name, or no allowance.

    Distinct from a transient failure. Retrying it on the next question wastes a
    request and, when the whole ensemble is unavailable, turns one dead provider
    into hundreds of pointless calls a minute against a shared proxy.
    """


class NoModelsAvailable(LLMError):
    """Every model in the ensemble is unavailable. The run cannot forecast at all."""


# Models that answered with an allowance or authentication error this process.
DEAD_MODELS: set[str] = set()

# Free-tier refusals come in two kinds that need opposite handling. A per-minute
# limit clears in seconds, so wait briefly and try again. A daily allowance does
# not clear until the provider's day rolls over, and every retry before then
# wastes a request and a minute of a question that is open for about ninety.
# Google names the quota in the error body; Groq names the limit ("tokens per
# day (TPD)") and hands back a Retry-After of an hour or more.
_DAILY_LIMIT = re.compile(r"(PerDay|per day|\(RPD\)|\(TPD\))", re.I)
LONG_RETRY_SECONDS = 600.0
RATE_LIMIT_MAX_SLEEP = 20.0

# Models whose daily allowance is used up, and the wall-clock time it returns.
EXHAUSTED_UNTIL: dict[str, float] = {}

# A model that just answered 429 or 503 goes to the back of its tier for a
# minute, so parallel runs try another Flash version first instead of queueing
# behind the one that is overloaded. The watcher logs show why: the same Flash
# model was retried three times per question, a minute apart, and lost.
COOLDOWN_SECONDS = 60.0
_COOLDOWN_UNTIL: dict[str, float] = {}

# GitHub Models answers an over-long prompt with a 400 or 413. That is about the
# prompt, not the model, so it must not write the model off.
_PROMPT_TOO_LONG = re.compile(
    r"(tokens_limit_reached|too large|too long|maximum context|context length)", re.I
)


def model_available(model: str, now: float | None = None) -> bool:
    """False for a model that is dead, or out of its daily allowance until later."""
    if model in DEAD_MODELS:
        return False
    until = EXHAUSTED_UNTIL.get(model)
    return until is None or (now if now is not None else time.time()) >= until


def _next_pacific_midnight(now: float | None = None) -> float:
    """When Google's free daily allowances reset: midnight Pacific time."""
    now = time.time() if now is None else now
    try:
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/Los_Angeles")
        local = datetime.fromtimestamp(now, tz)
        nxt = (local + timedelta(days=1)).replace(hour=0, minute=5, second=0, microsecond=0)
        return nxt.timestamp()
    except Exception:  # noqa: BLE001 - no tz database: assume an hour
        return now + 3600.0


def _error_brief(resp) -> str:
    """A short, secret-free summary of a provider's error body, for the log.

    The old log said only "HTTP 429", which could not tell a per-minute limit
    from a used-up daily allowance, and those need opposite handling.
    """
    text = getattr(resp, "text", "") or ""
    try:
        data = json.loads(text)
    except ValueError:
        return re.sub(r"\s+", " ", text)[:160]
    if isinstance(data, list) and data:
        data = data[0]
    err = data.get("error") if isinstance(data, dict) else None
    if not isinstance(err, dict):
        return re.sub(r"\s+", " ", text)[:160]
    parts = [str(err.get("status") or err.get("type") or err.get("code") or "")]
    quotas = sorted(set(re.findall(r'"quotaId"\s*:\s*"([^"]+)"', text)))
    if quotas:
        parts.append("quota " + ",".join(quotas))
    # Google names the size of the allowance too, which is the only public
    # place a free key's daily limit can be read.
    values = sorted(set(re.findall(r'"quotaValue"\s*:\s*"?(\d+)', text)))
    if values:
        parts.append("limit " + ",".join(values))
    message = err.get("message")
    if message:
        parts.append(re.sub(r"\s+", " ", str(message))[:160])
    return " | ".join(p for p in parts if p)


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    by_model: dict[str, int] = field(default_factory=dict)

    def record(self, model: str, usage: dict | None, cost: float = 0.0) -> None:
        self.calls += 1
        self.by_model[model] = self.by_model.get(model, 0) + 1
        self.cost_usd += cost
        if usage:
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.completion_tokens += int(usage.get("completion_tokens") or 0)

    def summary(self) -> str:
        return (
            f"{self.calls} calls, {self.prompt_tokens}+{self.completion_tokens} tokens, "
            f"${self.cost_usd:.3f}, models={self.by_model}"
        )


USAGE = Usage()


def _provider_for(model: str) -> tuple[Provider, str]:
    """Route by an explicit prefix, else to whichever provider has a key."""
    for name, prov in PROVIDERS.items():
        prefix = f"{name}/"
        if model.startswith(prefix):
            return prov, _adapt_model_name(prov, model[len(prefix):])
    order = ["openrouter", "metaculus", "gemini", "groq"]
    for name in order:
        if PROVIDERS[name].key:
            return PROVIDERS[name], _adapt_model_name(PROVIDERS[name], model)
    raise LLMError(
        "No LLM credentials found. Set OPENROUTER_API_KEY (tournament credits), "
        "or METACULUS_TOKEN to use the Metaculus proxy, or GEMINI_API_KEY / GROQ_API_KEY."
    )


def _retry_delay(resp) -> float | None:
    """How long the provider says to wait, from the header or the error body."""
    header = resp.headers.get("Retry-After")
    if header:
        try:
            return max(float(header), 1.0)
        except ValueError:
            pass
    match = re.search(r'"retryDelay"\s*:\s*"?(\d+(?:\.\d+)?)s', resp.text or "")
    if match:
        try:
            return max(float(match.group(1)), 1.0)
        except ValueError:
            pass
    return None


def _adapt_model_name(prov: Provider, model: str) -> str:
    """OpenRouter uses ``vendor/model``; the other providers want the bare name.

    Sending "openai/gpt-5" to the Metaculus proxy produced
    "You don\'t have an allowance for model <openai/gpt-5> on <Openai>", because
    the vendor prefix is part of the name it looks up. The same single-segment
    strip also turns Google's catalogue form "models/gemini-2.5-pro" into the
    "gemini-2.5-pro" its OpenAI-compatible endpoint is documented with, and
    leaves a bare Groq id such as "llama-3.3-70b-versatile" untouched.

    Groq is the exception a blanket strip gets wrong: it serves ids like
    "meta-llama/llama-4-maverick-17b-128e-instruct" and "openai/gpt-oss-120b",
    where the slash is part of the name and removing it produces a 404. Groq
    and OpenRouter ids go out exactly as their catalogues gave them.
    """
    if prov.name in ("openrouter", "groq", "github"):
        return model
    return model.split("/", 1)[1] if "/" in model else model


def chat(
    messages: Sequence[dict],
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 3000,
    timeout: float = 240.0,
    attempts: int = 3,
    reasoning: bool = True,
) -> str:
    prov, bare_model = _provider_for(model)
    key = prov.key
    if not key:
        raise LLMError(f"{prov.key_env} is not set, needed for model {model}")

    headers = {
        "Authorization": f"{prov.auth_scheme} {key}",
        "Content-Type": "application/json",
    }
    if prov.name == "openrouter":
        headers["X-Title"] = "metaculus-forecast-bot"

    limit_key = f"{prov.name}|{bare_model}"
    body = {
        "model": bare_model,
        "messages": list(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    thinking = _reasoning_params(prov, limit_key) if reasoning else {}
    if thinking:
        body.update(thinking)
        body["max_tokens"] = max(max_tokens, REASONING_MAX_TOKENS)
    if prov.name == "github":
        body["max_tokens"] = min(body["max_tokens"], GITHUB_MAX_OUTPUT_TOKENS)

    last = None
    for attempt in range(1, attempts + 1):
        with LIMITER.slot(limit_key):
            LIMITER.wait(limit_key)
            try:
                resp = requests.post(
                    f"{prov.base_url}/chat/completions",
                    headers=headers,
                    json=body,
                    timeout=timeout,
                )
            except requests.RequestException as exc:
                last = f"{type(exc).__name__}: {exc}"
                time.sleep(min(2 ** attempt, 30))
                continue

        if resp.status_code == 429:
            detail = _error_brief(resp)
            delay = _retry_delay(resp)
            if _DAILY_LIMIT.search(resp.text or "") or (
                delay is not None and delay >= LONG_RETRY_SECONDS
            ):
                # Out for the day. Retrying before the reset only burns time.
                if prov.name == "gemini":
                    until = _next_pacific_midnight()
                else:
                    until = time.time() + (delay or 3600.0)
                EXHAUSTED_UNTIL[model] = until
                log.warning(
                    "%s %s is out of its daily allowance until %s UTC (%s)",
                    prov.name,
                    bare_model,
                    time.strftime("%Y-%m-%d %H:%M", time.gmtime(until)),
                    detail,
                )
                raise ModelUnavailable(f"{prov.name} {bare_model} daily allowance used up: {detail}")
            # A per-minute limit. Both the header and Google's error body say
            # how long to wait; a run that ignored them spent 83 sleeps and
            # still timed out. But the other Flash versions have their own
            # allowances, so wait briefly, then let the caller move on.
            wait = min(delay or 10.0 * attempt, RATE_LIMIT_MAX_SLEEP)
            last = f"HTTP 429 ({detail})"
            _COOLDOWN_UNTIL[model] = time.monotonic() + COOLDOWN_SECONDS
            log.info("%s %s rate limited (%s)", prov.name, bare_model, detail)
            if attempt < attempts:
                time.sleep(wait)
            continue

        if resp.status_code in (500, 502, 503, 504):
            detail = _error_brief(resp)
            last = f"HTTP {resp.status_code} ({detail})"
            _COOLDOWN_UNTIL[model] = time.monotonic() + COOLDOWN_SECONDS
            log.info("%s %s -> HTTP %s (%s)", prov.name, bare_model, resp.status_code, detail)
            if attempt < attempts:
                time.sleep(min(5.0 * attempt, 20.0))
            continue
        if resp.status_code in (400, 413) and _PROMPT_TOO_LONG.search(resp.text or ""):
            raise LLMError(
                f"{prov.name} {bare_model}: prompt too long for this model ({_error_brief(resp)})"
            )
        if resp.status_code in (400, 401, 403, 404):
            # Do not bury a working model because it does not know one optional
            # parameter. Drop the parameter, remember that, and try again.
            if body.get("reasoning") or body.get("reasoning_effort"):
                if resp.status_code == 400 and _REASONING_REJECTED.search(resp.text or ""):
                    log.info("%s does not take a reasoning setting; dropping it", bare_model)
                    NO_REASONING.add(limit_key)
                    body.pop("reasoning", None)
                    body.pop("reasoning_effort", None)
                    body["max_tokens"] = max_tokens
                    continue
            DEAD_MODELS.add(model)
            raise ModelUnavailable(
                f"{prov.name} {bare_model} -> HTTP {resp.status_code}: {resp.text[:300]}"
            )
        if not resp.ok:
            raise LLMError(f"{prov.name} {bare_model} -> HTTP {resp.status_code}: {resp.text[:400]}")

        try:
            data = resp.json()
        except ValueError:
            # Seen in the first dry run: gemini-3.7-flash answered 200 with a
            # body that was not JSON, and the exception escaped every handler
            # and ended the run. Treat it like an overload.
            last = f"HTTP {resp.status_code} with an unreadable body {(resp.text or '')[:60]!r}"
            _COOLDOWN_UNTIL[model] = time.monotonic() + COOLDOWN_SECONDS
            log.info("%s %s: %s", prov.name, bare_model, last)
            if attempt < attempts:
                time.sleep(min(5.0 * attempt, 20.0))
            continue
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"{prov.name} returned no content: {json.dumps(data)[:400]}")
        if (
            (body.get("reasoning") or body.get("reasoning_effort"))
            and choice.get("finish_reason") == "length"
            and not (text or "").strip()
        ):
            # It spent the whole output budget thinking. An empty answer scores
            # nothing, so a shallower answer is strictly better.
            log.info("%s ran out of output while thinking; retrying without it", bare_model)
            NO_REASONING.add(limit_key)
            body.pop("reasoning", None)
            body.pop("reasoning_effort", None)
            body["max_tokens"] = max_tokens
            last = "truncated while reasoning"
            continue
        cost = 0.0
        usage = data.get("usage") or {}
        if isinstance(usage.get("cost"), (int, float)):
            cost = float(usage["cost"])
        USAGE.record(model, usage, cost)
        if not (text or "").strip():
            # An empty answer parses to nothing, and returning it would end the
            # fallback chain on a run that produced no forecast. Let the next
            # model try instead.
            raise LLMError(
                f"{prov.name} {bare_model} returned an empty answer "
                f"(finish_reason {choice.get('finish_reason')!r})"
            )
        return text

    raise LLMError(f"{prov.name} {bare_model} failed after {attempts} attempts: {last}")


def chat_with_fallback(
    messages: Sequence[dict],
    models: Sequence[str],
    **kwargs,
) -> tuple[str, str]:
    """Try each model in turn. Returns (text, model_that_answered).

    Order is kept, with two adjustments. Dead models and models out of their
    daily allowance are skipped. And a model that refused in the last minute
    moves to the back of its own tier, never behind a stand-in: an overloaded
    Flash model is still a better forecaster than the fallback.
    """
    errors = []
    now = time.time()
    live = [m for m in models if model_available(m, now)]
    if not live:
        raise NoModelsAvailable(
            "every model is unavailable with the current credentials: "
            + ", ".join(sorted(set(DEAD_MODELS) | set(EXHAUSTED_UNTIL)))
        )
    mono = time.monotonic()
    live.sort(key=lambda m: (m in FALLBACK_ONLY, _COOLDOWN_UNTIL.get(m, 0.0) > mono))
    for model in live:
        try:
            return chat(messages, model, **kwargs), model
        except ModelUnavailable as exc:
            errors.append(f"{model}: {exc}")
            log.warning("model %s is unavailable, will not retry it: %s", model, str(exc)[:200])
        except LLMError as exc:
            errors.append(f"{model}: {exc}")
            log.warning("model %s failed, trying next: %s", model, str(exc)[:200])
    if not any(model_available(m) for m in models):
        raise NoModelsAvailable("all models failed permanently:\n" + "\n".join(errors))
    raise LLMError("all models failed:\n" + "\n".join(errors))


def probe(models: Sequence[str]) -> dict[str, str]:
    """One tiny request per model: which ones answer right now, and how fast.

    Run at the start of a dry run, so its log says model by model whether the
    free allowances are answering before any forecast depends on them.
    """
    out: dict[str, str] = {}
    for model in models:
        started = time.monotonic()
        try:
            text = chat(
                [{"role": "user", "content": "Reply with the word OK and nothing else."}],
                model,
                max_tokens=64,
                attempts=1,
                reasoning=False,
            )
            out[model] = f"answered in {time.monotonic() - started:.1f}s: {text.strip()[:20]!r}"
        except ModelUnavailable as exc:
            out[model] = f"unavailable: {str(exc)[:200]}"
        except LLMError as exc:
            out[model] = f"failed: {str(exc)[:200]}"
        except Exception as exc:  # noqa: BLE001 - a probe must never end the run
            out[model] = f"crashed: {type(exc).__name__}: {str(exc)[:160]}"
    return out


# -- model resolution ------------------------------------------------------
_CATALOGUE_CACHE: list[dict] | None = None

# Not every id a provider lists is a chat model. The first live run picked
# "models/aqa", which is an attributed-question-answering endpoint and returns
# 404 for generateContent, and "gemini-2.5-pro", which Google has closed to new
# keys. Both cost a whole run.
# Groq adds two more shapes of the same mistake: "whisper-large-v3" is speech
# to text and "llama-guard-4" is a safety classifier that answers "safe", not a
# forecast. Both survive the OpenRouter-shaped EXCLUDE list.
_NOT_A_CHAT_MODEL = re.compile(
    r"(aqa|embed|imagen|veo|image-gen|^models/text-|tts|speech|audio|live"
    r"|vision-only|learnlm|whisper|guard|prompt-?guard|moderation|rerank)",
    re.I,
)

# On a free tier the flash class is what actually answers: the pro class has a
# per-minute limit low enough that a five member ensemble exhausts it on the
# first question. Capability is worth less than a reply.
_FLASH = re.compile(r"flash", re.I)
_LITE = re.compile(r"(lite|mini|nano|tiny|8b|instant)", re.I)
_VERSION = re.compile(r"(\d+(?:\.\d+)?)")


# Set by resolve_models when the ensemble is built from one provider's models
# with stand-ins behind them; that provider's limits decide the run's pace.
PRIMARY_PROVIDER: str | None = None


def active_provider() -> Provider | None:
    """Whichever provider this run can actually reach, in preference order."""
    if PRIMARY_PROVIDER and PROVIDERS[PRIMARY_PROVIDER].key:
        return PROVIDERS[PRIMARY_PROVIDER]
    for name in ("openrouter", "gemini", "groq", "metaculus"):
        if PROVIDERS[name].key:
            return PROVIDERS[name]
    return None


def keyed_providers() -> list[Provider]:
    """Every provider this run holds a usable key for, in preference order.

    METACULUS_TOKEN is always set, because it is also how the bot posts its
    forecasts, so the proxy would otherwise always look available. Until the
    sponsored credits land it answers "you don\'t have an allowance" to every
    model, so it only joins the ensemble when nothing else is keyed.
    """
    named = [
        PROVIDERS[n] for n in ("openrouter", "gemini", "groq", "github") if PROVIDERS[n].key
    ]
    if named:
        return named
    return [PROVIDERS["metaculus"]] if PROVIDERS["metaculus"].key else []


def _round_robin(served: dict[str, list[str]], count: int) -> list[str]:
    """One model from each provider before a second from any of them."""
    picked: list[str] = []
    depth = 0
    while len(picked) < count and any(len(v) > depth for v in served.values()):
        for name, models in served.items():
            if depth < len(models):
                picked.append(f"{name}/{models[depth]}")
                if len(picked) >= count:
                    break
        depth += 1
    return picked


def provider_catalogue(prov: Provider) -> list[str]:
    """Ask a provider which models it serves. Empty list if it will not say."""
    headers = {"Authorization": f"{prov.auth_scheme} {prov.key}"} if prov.key else {}
    try:
        resp = requests.get(f"{prov.base_url}/models", headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        log.warning("could not list %s models: %s", prov.name, str(exc)[:200])
        return []
    entries = data.get("data") if isinstance(data, dict) else data
    out = []
    for entry in entries or []:
        mid = entry.get("id") if isinstance(entry, dict) else entry
        if mid and not EXCLUDE.search(str(mid)):
            out.append(str(mid))
    return out


def catalogue_report() -> list[str]:
    """What each keyed free provider offers, one line each, for a dry run.

    Chooses nothing. It exists so that a dry run's annotation shows which
    strong models were on offer, not only the ones the resolver picked.
    """
    lines: list[str] = []
    gemini = PROVIDERS["gemini"]
    if gemini.key:
        flash = sorted(
            (m for m in provider_catalogue(gemini) if _FLASH.search(m)),
            key=lambda m: (-(gemini_version(m) or 0.0), m),
        )
        lines.append("gemini flash ids: " + (", ".join(m.split("/", 1)[-1] for m in flash[:12]) or "none listed"))
    github = PROVIDERS["github"]
    if github.key:
        try:
            resp = requests.get(
                "https://models.github.ai/catalog/models",
                headers={"Authorization": f"Bearer {github.key}", "Accept": "application/vnd.github+json"},
                timeout=30,
            )
            resp.raise_for_status()
            entries = resp.json()
            rows = [
                f"{e.get('id')} ({e.get('rate_limit_tier') or '?'})"
                for e in (entries if isinstance(entries, list) else [])
                if isinstance(e, dict) and re.match(r"^(openai|xai|deepseek|meta|mistral-ai|microsoft|cohere|ai21-labs)/", str(e.get("id") or ""))
            ]
            lines.append("github models: " + (", ".join(rows[:40]) or "none listed"))
        except Exception as exc:  # noqa: BLE001 - a report must not stop a run
            lines.append(f"github models: catalogue unreadable ({str(exc)[:120]})")
    if PROVIDERS["openrouter"].key:
        free = [str(e.get("id")) for e in _catalogue() if str(e.get("id") or "").endswith(":free")]
        lines.append("openrouter free ids: " + (", ".join(free[:25]) or "none listed"))
    groq = PROVIDERS["groq"]
    if groq.key:
        lines.append("groq ids: " + (", ".join(provider_catalogue(groq)[:20]) or "none listed"))
    return lines


def _rank_bare(ids: Sequence[str]) -> list[str]:
    """Order a provider's own model ids: answerable first, newest first."""

    def score(mid: str) -> tuple[float, float, str]:
        tier = 0.0 if _FLASH.search(mid) else 1.0
        if _LITE.search(mid):
            tier += 0.5
        # For Gemini the version is the number after "gemini-". Taking the
        # largest number anywhere in the id would rank a dated snapshot such
        # as "-09-2025" above every real version.
        version = gemini_version(mid)
        if version is None:
            version = max([float(v) for v in _VERSION.findall(mid)] or [0.0])
        return (tier, -version, mid)

    usable = [m for m in ids if not _NOT_A_CHAT_MODEL.search(m)]
    return sorted(usable, key=score)


def _catalogue() -> list[dict]:
    global _CATALOGUE_CACHE
    if _CATALOGUE_CACHE is not None:
        return _CATALOGUE_CACHE
    prov = PROVIDERS["openrouter"]
    # This endpoint needs no credentials, so the resolver still works before the
    # tournament credits arrive. Gating it on a key was why a run with no key
    # fell back to hardcoded model names that no longer exist.
    headers = {"Authorization": f"Bearer {prov.key}"} if prov.key else {}
    try:
        resp = requests.get(
            f"{prov.base_url}/models",
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        _CATALOGUE_CACHE = resp.json().get("data") or []
    except Exception as exc:  # catalogue is an optimisation, never fatal
        log.warning("could not read the model catalogue: %s", exc)
        _CATALOGUE_CACHE = []
    return _CATALOGUE_CACHE


_GEMINI_VERSION = re.compile(r"gemini-(\d+(?:\.\d+)?)", re.I)
_NOT_PRIMARY = re.compile(r"(-exp|live|tts|image|audio|thinking|latest)", re.I)


def gemini_version(mid: str) -> float | None:
    """3.8 for "models/gemini-3.8-flash"; None when the id carries no version."""
    match = _GEMINI_VERSION.search(mid or "")
    return float(match.group(1)) if match else None


def is_primary_flash(mid: str) -> bool:
    """A full Flash model at or above the version floor.

    The floor is where the leaderboard evidence is: Gemini 3.5 Flash +12.17 and
    3.6 Flash +13.22, while Gemini 3 Flash scored +6.39 and 2.5 Flash -8.07.
    Lite variants score near zero (3.5 Flash-Lite +0.70). Aliases without a
    version ("-latest") are skipped because they duplicate a pinned version.
    """
    # Not _LITE here: "gemini" itself contains "mini".
    if not _FLASH.search(mid) or re.search(r"lite", mid, re.I) or _NOT_PRIMARY.search(mid):
        return False
    if _NOT_A_CHAT_MODEL.search(mid):
        return False
    version = gemini_version(mid)
    return version is not None and version >= PRIMARY_GEMINI_MIN_VERSION


def primary_flash_models(ids: Sequence[str], limit: int = PRIMARY_MODELS) -> list[str]:
    """Newest Flash versions first, one id per version."""
    candidates = sorted(
        (m for m in ids if is_primary_flash(m)),
        key=lambda m: (-(gemini_version(m) or 0.0), len(m), m),
    )
    picked: list[str] = []
    seen: set[float] = set()
    for mid in candidates:
        version = gemini_version(mid)
        if version in seen:
            continue
        seen.add(version)
        picked.append(mid)
        if len(picked) >= limit:
            break
    return picked


def fallback_model(prov: Provider) -> str | None:
    """The one stand-in model worth asking on this provider, or None."""
    if prov.name == "openrouter":
        ids = [str(e.get("id") or "") for e in _catalogue()]
        for pattern in OPENROUTER_FREE_PREFERENCES:
            for mid in ids:
                if re.search(pattern, mid):
                    return f"openrouter/{mid}"
        return None
    if prov.name == "github":
        return f"github/{GITHUB_MODELS_PREFERENCES[0]}"
    ranked = _rank_bare(provider_catalogue(prov))
    return f"{prov.name}/{ranked[0]}" if ranked else None


def resolve_models(count: int = 3) -> list[str]:
    """Pick the ensemble.

    On free keys, the strong models are the Gemini Flash versions and every
    other provider contributes one stand-in, listed after them and recorded in
    FALLBACK_ONLY (see the comment above PRIMARY_MODELS for the evidence).
    Otherwise, with sponsored credits, ``count`` capable models from distinct
    vendors: the bot-maker surveys found ensembling across comparably strong
    model families worth far more than which single model is best.
    """
    global PRIMARY_PROVIDER
    FALLBACK_ONLY.clear()
    PRIMARY_PROVIDER = None
    override = os.environ.get("BOT_MODELS", "").strip()
    if override:
        return [m.strip() for m in override.split(",") if m.strip()]

    keyed = keyed_providers()
    free_keys = bool(keyed) and (
        OPENROUTER_FREE_ONLY or not any(p.name == "openrouter" for p in keyed)
    )
    gemini = PROVIDERS["gemini"]
    if free_keys and gemini.key:
        primaries = [f"gemini/{m}" for m in primary_flash_models(provider_catalogue(gemini))]
        if len(primaries) >= 2:
            fallbacks = []
            for name in FALLBACK_PROVIDER_ORDER:
                prov = PROVIDERS[name]
                if prov.key:
                    mid = fallback_model(prov)
                    if mid:
                        fallbacks.append(mid)
            FALLBACK_ONLY.update(fallbacks)
            PRIMARY_PROVIDER = "gemini"
            return primaries + fallbacks
        log.warning("fewer than two Gemini Flash models at %.1f or newer; mixing providers",
                    PRIMARY_GEMINI_MIN_VERSION)

    if free_keys:
        # No OpenRouter key, so the ensemble comes from whichever providers we
        # do have. Ask each what it serves; hardcoding names is what produced
        # "You don\'t have an allowance for model <openai/gpt-5>".
        #
        # Spread across providers rather than taking three models from one. Two
        # reasons, and the second is the one that showed up in production: three
        # models from one family make correlated mistakes, and three models on
        # one free tier share one allowance. A live watcher on a Gemini-only
        # ensemble spent most of its wall clock asleep on 429s while a perfectly
        # good Groq key sat unused.
        served = {}
        for prov in keyed:
            if prov.name in ("openrouter", "github"):
                # Free keys here buy one specific model, not the catalogue.
                mid = fallback_model(prov)
                if mid:
                    served[prov.name] = [mid.split("/", 1)[1]]
                continue
            ranked = _rank_bare(provider_catalogue(prov))
            if ranked:
                served[prov.name] = ranked
            else:
                log.warning("%s served no model list", prov.name)
        picked = _round_robin(served, count)
        if picked:
            return picked
        log.warning("no keyed provider served a model list; falling back")

    catalogue = _catalogue()
    if not catalogue:
        return STATIC_FALLBACK[:count] if count <= len(STATIC_FALLBACK) else STATIC_FALLBACK

    scored: list[tuple[int, int, str, str]] = []
    for entry in catalogue:
        mid = entry.get("id") or ""
        if VARIANT.search(mid) or EXCLUDE.search(mid):
            continue
        for pattern, score in MODEL_PREFERENCES:
            if re.search(pattern, mid):
                vendor = mid.split("/", 1)[0]
                scored.append((score, int(entry.get("created") or 0), vendor, mid))
                break

    scored.sort(reverse=True)
    picked: list[str] = []
    seen_vendors: set[str] = set()
    for _, _, vendor, mid in scored:
        if vendor in seen_vendors:
            continue
        picked.append(mid)
        seen_vendors.add(vendor)
        if len(picked) >= count:
            break
    # Top up from the remainder if fewer vendors were available than requested.
    if len(picked) < count:
        for _, _, _, mid in scored:
            if mid not in picked:
                picked.append(mid)
            if len(picked) >= count:
                break
    return picked or STATIC_FALLBACK[:count]


# -- structured output -----------------------------------------------------
_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> Any:
    """Pull a JSON value out of model prose.

    Models wrap JSON in fences, prefix it with commentary, and occasionally emit
    trailing commas. All of that is recoverable and none of it is worth a retry.
    """
    if text is None:
        raise ValueError("no text to parse")
    candidates: list[str] = []
    for match in _JSON_BLOCK.finditer(text):
        candidates.append(match.group(1))
    candidates.append(text)

    for chunk in candidates:
        chunk = chunk.strip()
        # Try whichever bracket opens first, so a list of objects is read as a
        # list rather than as its first element.
        pairs = [("{", "}"), ("[", "]")]
        pairs.sort(key=lambda pair: chunk.find(pair[0]) if chunk.find(pair[0]) != -1 else len(chunk) + 1)
        for opener, closer in pairs:
            start = chunk.find(opener)
            end = chunk.rfind(closer)
            if start == -1 or end <= start:
                continue
            body = chunk[start : end + 1]
            for attempt in (body, re.sub(r",\s*([}\]])", r"\1", body)):
                try:
                    return json.loads(attempt)
                except json.JSONDecodeError:
                    continue
    raise ValueError(f"no JSON found in model output: {text[:300]!r}")


def run_parallel(tasks: Sequence[Callable[[], Any]], workers: int = 6) -> list[Any]:
    """Run independent LLM calls concurrently; failures come back as exceptions."""
    results: list[Any] = [None] * len(tasks)
    if not tasks:
        return results
    with cf.ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
        futures = {pool.submit(task): i for i, task in enumerate(tasks)}
        for fut in cf.as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:  # noqa: BLE001 - surfaced to the caller
                results[i] = exc
    return results


def metaculus_proxy_models() -> list[str]:
    """Ask the Metaculus LLM proxy which models the bot token is allowed to use.

    The proxy rejects a name it does not recognise with an "allowance" error, so
    guessing is expensive. This is reported by --check-sources.
    """
    prov = PROVIDERS["metaculus"]
    if not prov.key:
        return []
    try:
        resp = requests.get(
            f"{prov.base_url}/models",
            headers={"Authorization": f"Token {prov.key}"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        log.warning("could not list the Metaculus proxy models: %s", str(exc)[:200])
        return []
    entries = data.get("data") if isinstance(data, dict) else data
    out = []
    for entry in entries or []:
        mid = entry.get("id") if isinstance(entry, dict) else entry
        if mid:
            out.append(str(mid))
    return sorted(out)


def provider_is_metered(threshold: float = 30.0) -> bool:
    """True when the run is on a low rate limit and should spend calls sparingly."""
    prov = active_provider()
    if prov is None:
        return True
    rpm, _ = provider_limits(prov.name)
    return rpm < threshold
