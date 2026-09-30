"""Turning one Metaculus question into one submittable forecast.

Every question type ends in the same place: a payload the API will accept and a
comment explaining it, because prize eligibility requires a comment under every
question the bot forecasts.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

from . import aggregate, config, parsing, prompts
from .cdf import DEFAULT_INBOUND_OUTCOME_COUNT, build_cdf, percentiles_from_cdf, safe_cdf
from .llm import (
    FALLBACK_ONLY,
    cooling_down,
    LLMError,
    NoModelsAvailable,
    chat_with_fallback,
    extract_json,
    model_available,
    provider_is_metered,
    run_parallel,
)
from .scaling import Scaling

log = logging.getLogger(__name__)

CONTINUOUS_TYPES = {"numeric", "discrete", "date"}


class EnsembleTooThin(LLMError):
    """Too few strong-model answers, and the question is open long enough to wait.

    Nothing is submitted. The answers already in are kept for this process, so
    the next poll only asks for the missing ones.
    """


# Strong-model answers per (question id, question type), kept for the life of
# the watcher so a question that waits does not pay for its answers twice.
_STRONG_ANSWERS: dict[tuple, list[tuple[str, str]]] = {}

def forget(question_id: int) -> None:
    """Drop the kept answers for a question once its forecast has gone out."""
    for key in [k for k in _STRONG_ANSWERS if k and k[0] == question_id]:
        _STRONG_ANSWERS.pop(key, None)


# How much of one run's reasoning goes into the comment.
COMMENT_REASONING_CHARS = 3500


@dataclass
class Forecast:
    question_id: int
    post_id: int
    question_type: str
    payload: dict
    comment: str
    headline: str
    notes: list[str] = field(default_factory=list)
    runs_used: int = 0
    models_used: list[str] = field(default_factory=list)


def _iso(value: Any) -> str:
    if not value:
        return ""
    return str(value)[:19].replace("T", " ")


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def date_to_epoch(raw: str) -> float | None:
    """Read a date a model wrote, as unix seconds."""
    if raw is None:
        return None
    text = str(raw).strip()
    match = re.search(r"(\d{4})[-/](\d{1,2})(?:[-/](\d{1,2}))?", text)
    if match:
        year, month, day = match.group(1), match.group(2), match.group(3) or "1"
        try:
            return datetime(int(year), int(month), int(day), tzinfo=timezone.utc).timestamp()
        except ValueError:
            return None
    match = re.search(r"\b(19|20)\d{2}\b", text)
    if match:
        try:
            return datetime(int(match.group(0)), 7, 1, tzinfo=timezone.utc).timestamp()
        except ValueError:
            return None
    return None


def build_context(post: dict, question: dict) -> dict:
    qtype = question.get("type") or ""
    close = _parse_dt(question.get("scheduled_close_time"))
    resolve = _parse_dt(question.get("scheduled_resolve_time"))
    now = datetime.now(timezone.utc)
    days_left = None
    if resolve:
        days_left = max(0, int((resolve - now).total_seconds() // 86400))

    ctx: dict[str, Any] = {
        "post_id": post.get("id"),
        "question_id": question.get("id"),
        "close_dt": close,
        "type": qtype,
        "title": question.get("title") or post.get("title") or "",
        "description": (question.get("description") or "")[:6000],
        "resolution_criteria": (question.get("resolution_criteria") or "")[:4000],
        "fine_print": (question.get("fine_print") or "")[:2000],
        "close_time": _iso(question.get("scheduled_close_time")),
        "resolve_time": _iso(question.get("scheduled_resolve_time")),
        "days_left": days_left,
        "options": list(question.get("options") or []),
        "unit": question.get("unit") or "",
    }

    if qtype in CONTINUOUS_TYPES:
        scaling = Scaling.from_question(question)
        ctx["scaling"] = scaling
        ctx["open_lower_bound"] = bool(question.get("open_lower_bound"))
        ctx["open_upper_bound"] = bool(question.get("open_upper_bound"))
        ctx["inbound_outcome_count"] = int(
            question.get("inbound_outcome_count") or DEFAULT_INBOUND_OUTCOME_COUNT
        )
        if qtype == "date":
            ctx["range_min_label"] = _epoch_label(scaling.range_min)
            ctx["range_max_label"] = _epoch_label(scaling.range_max)
            ctx["unit"] = "date (YYYY-MM-DD)"
        else:
            ctx["range_min_label"] = _num_label(scaling.range_min)
            ctx["range_max_label"] = _num_label(scaling.range_max)
    return ctx


def _epoch_label(ts: float) -> str:
    try:
        return datetime.fromtimestamp(float(ts), timezone.utc).strftime("%Y-%m-%d")
    except (OverflowError, OSError, ValueError):
        return str(ts)


def _num_label(v: float) -> str:
    if abs(v - round(v)) < 1e-9 and abs(v) < 1e15:
        return f"{int(round(v)):,}"
    return f"{v:,.4g}"


# -- research --------------------------------------------------------------
def search_queries(ctx: dict, models: Sequence[str]) -> list[str]:
    fallback = [ctx["title"][:200]]
    if provider_is_metered():
        # One model call per question spent on phrasing search terms is a
        # luxury when the budget is measured in calls per minute. The title is
        # a decent query on its own.
        return fallback
    try:
        text, _ = chat_with_fallback(
            prompts.search_queries_prompt(ctx), models, temperature=0.3, max_tokens=300
        )
        parsed = extract_json(text)
        queries = [str(q).strip() for q in parsed if str(q).strip()] if isinstance(parsed, list) else []
        return (queries or fallback)[:3]
    except (LLMError, ValueError) as exc:
        log.warning("query generation failed, using the title: %s", str(exc)[:200])
        return fallback


# -- per-type pipelines ----------------------------------------------------
# Set by the runner for the last poll of a dry run, so that questions still
# waiting are carried through the deadline path and every type gets checked.
FORCE_DEADLINE = False


def can_defer(ctx: dict, models: Sequence[str] = (), now: datetime | None = None) -> bool:
    """May this question wait for the next poll?

    Only when the ensemble has stand-ins behind its strong models (otherwise
    there is nothing better to wait for), only while the question stays open
    for longer than the safety margin, and only if some strong model can
    answer before that margin. When every strong model is out of its daily
    allowance until after the question closes, waiting wins nothing and
    spends the stand-in's time: on 30 September all four Flash allowances
    were used up by 22:40 UTC and came back at 07:00.
    """
    if not FALLBACK_ONLY or FORCE_DEADLINE:
        return False
    close = ctx.get("close_dt")
    if close is None:
        return False
    now = now or datetime.now(timezone.utc)
    margin = config.DEFER_MARGIN_MINUTES * 60
    if (close - now).total_seconds() <= margin:
        return False
    primaries = [m for m in models if m not in FALLBACK_ONLY]
    if primaries:
        last_useful = close.timestamp() - margin
        if not any(model_available(m, last_useful) for m in primaries):
            return False
    return True


def _run_ensemble(
    messages,
    models: Sequence[str],
    runs: int,
    temperature: float,
    cache_key: tuple | None = None,
    allow_fallback: bool = True,
    usable: Callable[[str], bool] | None = None,
) -> list[tuple[str, str]]:
    """Collect ``runs`` answers, each run led by a different strong model.

    Every run tries the strong models first, starting from a different one each
    time, and only then the stand-ins, and those only when ``allow_fallback``.
    Usable strong answers from an earlier poll of the same question count
    towards ``runs``; an unusable one is not kept, or it would fill a slot
    forever without ever counting as an answer.
    """
    primaries = [m for m in models if m not in FALLBACK_ONLY]
    fallbacks = [m for m in models if m in FALLBACK_ONLY]
    if not primaries:
        primaries, fallbacks = list(models), []
    tail = fallbacks if allow_fallback else []

    strong_attempts, strong_timeout = config.ENSEMBLE_ATTEMPTS, 240.0
    wanted = runs
    if tail:
        # At the deadline the stand-in must get its turn in time. A strong
        # model that refused within the last minute or is out for the day is
        # skipped, and the rest get one short try each, so four overloaded
        # Flash models cannot use up the last half hour. The stand-in keeps
        # its normal retries: Groq's free gpt-oss-120b allows about 8,000
        # tokens a minute, so a burst of calls meets per-minute refusals.
        cool = cooling_down()
        primaries = [m for m in primaries if m not in cool and model_available(m)]
        strong_attempts, strong_timeout = 1, 120.0
        if not primaries:
            # Only stand-ins can answer. Fewer runs, so that a burst of
            # questions at the deadline stays inside the stand-in's limits.
            wanted = min(runs, config.STAND_IN_RUNS)

    def make_task(k: int):
        strong = primaries[k:] + primaries[:k] if primaries else []

        def task():
            if strong:
                try:
                    return chat_with_fallback(
                        messages,
                        strong,
                        temperature=temperature,
                        attempts=strong_attempts,
                        timeout=strong_timeout,
                    )
                except LLMError:
                    if not tail:
                        raise
            return chat_with_fallback(
                messages, tail, temperature=temperature, attempts=config.ENSEMBLE_ATTEMPTS
            )

        return task

    cached = list(_STRONG_ANSWERS.get(cache_key, [])) if cache_key else []
    tasks = [make_task(i % len(primaries) if primaries else 0) for i in range(len(cached), wanted)]
    out: list[tuple[str, str]] = []
    dead = 0
    for result in run_parallel(tasks, workers=min(max(len(tasks), 1), 5)):
        if isinstance(result, Exception):
            if isinstance(result, NoModelsAvailable):
                dead += 1
            log.warning("ensemble member failed: %s", str(result)[:200])
            continue
        out.append(result)
    if cache_key is not None:
        _STRONG_ANSWERS.setdefault(cache_key, []).extend(
            r for r in out if r[1] not in FALLBACK_ONLY and (usable is None or usable(r[0]))
        )
    combined = cached + out
    if not combined and not allow_fallback and fallbacks:
        # Nothing strong answered, overloaded or out for the day, and the
        # question is open long enough to ask again. Stand-ins stay behind for
        # the deadline; nothing is submitted now.
        raise EnsembleTooThin("no strong model answered this poll; waiting for the next one")
    if not combined and dead:
        # Nothing answered at all. That is the LLM layer being down, not the
        # models being unsure, and the two must not be confused: see the callers.
        raise NoModelsAvailable(
            f"all {runs} ensemble members failed because every model is unavailable"
        )
    return combined


def _choose(ctx: dict, parsed: list, runs: int, deferrable: bool, notes: list[str]) -> list:
    """Keep the answers worth aggregating: strong ones, or all of them at the deadline.

    ``parsed`` holds (value, text, model) triples. With enough strong answers the
    stand-ins are left out entirely, because a median of two strong answers and
    two much weaker ones is half the weaker model.
    """
    if not FALLBACK_ONLY:
        return parsed
    strong = [p for p in parsed if p[2] not in FALLBACK_ONLY]
    needed = max(1, min(config.MIN_STRONG_ANSWERS, runs))
    if len(strong) >= needed:
        dropped = len(parsed) - len(strong)
        if dropped:
            notes.append(f"{dropped} stand-in answer(s) left out; {len(strong)} strong answers")
        return strong
    if deferrable:
        raise EnsembleTooThin(
            f"{len(strong)} of {needed} strong answers so far; waiting for the next poll"
        )
    if strong:
        # At the deadline with fewer strong answers than wanted, the strong
        # ones still speak alone. The gap is about thirteen points on the
        # FutureEval model board (Gemini 3.6 Flash +13.22, GPT-OSS 120B -0.26),
        # so a median that lets three stand-ins outvote one Flash answer is
        # mostly the weaker model.
        dropped = len(parsed) - len(strong)
        notes.append(
            f"only {len(strong)} strong answer(s) before the deadline; used alone"
            + (f", {dropped} stand-in answer(s) left out" if dropped else "")
        )
        return strong
    if parsed:
        notes.append("no strong answer before the deadline; stand-ins used")
    return parsed


def _with_thinking(text: str) -> str:
    """The answer plus the provider's separate reasoning, when the answer is bare.

    A reasoning model served through some providers returns its thinking in a
    separate field and only the final block as content. The comment should
    carry the reasoning, since the rules ask for comments that show it.
    """
    thought = getattr(text, "reasoning", "") or ""
    if thought and len((text or "").strip()) < 600:
        return thought.strip() + "\n\n" + str(text).strip()
    return str(text or "")


def _reasoning_excerpt(text: str, limit: int = COMMENT_REASONING_CHARS) -> str:
    text = _with_thinking(text).strip()
    if len(text) <= limit:
        return text
    head = limit - 800
    return text[:head].rstrip() + "\n[...]\n" + text[-700:].lstrip()


def forecast_binary(ctx: dict, research: str, models: Sequence[str], runs: int):
    deferrable = can_defer(ctx, models)
    results = _run_ensemble(
        prompts.binary_prompt(ctx, research),
        models,
        runs,
        0.4,
        cache_key=(ctx["question_id"], "binary"),
        allow_fallback=not deferrable,
        usable=lambda text: parsing.parse_probability(text) is not None,
    )
    notes: list[str] = []
    parsed = []
    for text, model in results:
        p = parsing.parse_probability(text)
        if p is None:
            notes.append(f"{model}: no probability found")
            continue
        parsed.append((p, text, model))
    parsed = _choose(ctx, parsed, runs, deferrable, notes)

    if not parsed:
        raise LLMError("no ensemble member produced a usable probability")

    probs = [p for p, _, _ in parsed]
    used = [m for _, _, m in parsed]
    raw = aggregate.aggregate_binary(probs)
    disagreement = aggregate.spread(probs)
    final = aggregate.calibrate_binary(raw)
    notes.append(
        f"ensemble {[round(p, 3) for p in probs]} -> median {raw:.3f} -> calibrated {final:.3f} "
        f"(spread {disagreement:.3f})"
    )
    # The run nearest the median speaks for the ensemble in the comment.
    reasoning = min(parsed, key=lambda item: abs(item[0] - raw))[1]
    headline = f"{final:.1%}"
    return (
        {"question": ctx["question_id"], "probability_yes": round(final, 6)},
        headline,
        notes,
        len(probs),
        used,
        reasoning,
    )


def forecast_numeric(ctx: dict, research: str, models: Sequence[str], runs: int):
    is_date = ctx["type"] == "date"
    deferrable = can_defer(ctx, models)
    results = _run_ensemble(
        prompts.numeric_prompt(ctx, research),
        models,
        runs,
        0.4,
        cache_key=(ctx["question_id"], ctx["type"]),
        allow_fallback=not deferrable,
        usable=lambda text: len(parsing.parse_percentiles(text)) >= 2,
    )
    notes: list[str] = []
    parsed = []
    for text, model in results:
        points = parsing.parse_percentiles(text)
        if is_date:
            converted = []
            for p, raw in points:
                epoch = date_to_epoch(str(raw)) if not _looks_like_epoch(raw) else float(raw)
                if epoch is not None:
                    converted.append((p, epoch))
            points = converted
        if len(points) < 2:
            notes.append(f"{model}: no usable percentiles")
            continue
        parsed.append((points, text, model))

    scaling: Scaling = ctx["scaling"]
    count = ctx["inbound_outcome_count"]
    if not parsed and not results:
        # Submitting a uniform here would be the worst of both worlds: it scores
        # badly AND marks the question as forecast, so the bot never returns to
        # it once the outage clears. Fail instead, and let the next poll retry.
        raise NoModelsAvailable(
            "no ensemble member responded; refusing to submit a placeholder distribution"
        )
    parsed = _choose(ctx, parsed, runs, deferrable, notes)
    parsed_runs = [points for points, _, _ in parsed]
    used = [m for _, _, m in parsed]
    if not parsed_runs:
        notes.append("models responded but no usable percentiles; submitting a uniform distribution")
        cdf = safe_cdf(count, ctx["open_lower_bound"], ctx["open_upper_bound"])
        return (
            {"question": ctx["question_id"], "continuous_cdf": cdf},
            "uniform (no usable model output)",
            notes,
            0,
            used,
            "",
        )

    merged = aggregate.aggregate_percentiles(parsed_runs)
    widened = aggregate.extend_tails(aggregate.widen_percentiles(merged))
    cdf, build_notes = build_cdf(
        widened,
        scaling,
        ctx["open_lower_bound"],
        ctx["open_upper_bound"],
        count,
    )
    notes.extend(build_notes)

    def fmt(v: float) -> str:
        return _epoch_label(v) if is_date else _num_label(v)

    # Report what was actually submitted, not the raw anchors. The anchors
    # include tail extensions that can sit outside the question's range and get
    # conditioned away when the CDF is built, which made an early run look far
    # wider than the distribution really was.
    read = percentiles_from_cdf(cdf, scaling, (0.05, 0.5, 0.95))
    median = read[0.5]
    notes.append(f"{len(parsed_runs)} usable runs; submitted median {fmt(median)}")
    headline = f"median {fmt(median)} (90% between {fmt(read[0.05])} and {fmt(read[0.95])})"
    return (
        {"question": ctx["question_id"], "continuous_cdf": cdf},
        headline,
        notes,
        len(parsed_runs),
        used,
        parsed[0][1],
    )


def _looks_like_epoch(v: Any) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return 1e8 < f < 4e9


def forecast_multiple_choice(ctx: dict, research: str, models: Sequence[str], runs: int):
    options = ctx["options"]
    deferrable = can_defer(ctx, models)
    results = _run_ensemble(
        prompts.multiple_choice_prompt(ctx, research),
        models,
        runs,
        0.4,
        cache_key=(ctx["question_id"], "multiple_choice"),
        allow_fallback=not deferrable,
        usable=lambda text: bool(parsing.parse_multiple_choice(text, options)),
    )
    notes: list[str] = []
    parsed = []
    for text, model in results:
        got = parsing.parse_multiple_choice(text, options)
        if not got:
            notes.append(f"{model}: no usable option probabilities")
            continue
        parsed.append((got, text, model))

    if not parsed and not results:
        raise NoModelsAvailable(
            "no ensemble member responded; refusing to submit a placeholder distribution"
        )
    parsed = _choose(ctx, parsed, runs, deferrable, notes)
    parsed_runs = [got for got, _, _ in parsed]
    used = [m for _, _, m in parsed]
    merged = aggregate.aggregate_multiple_choice(parsed_runs, options)
    if merged is None:
        notes.append("models responded but no usable option probabilities; using the uniform")
        merged = {o: 1.0 / len(options) for o in options}
    final = aggregate.calibrate_multiple_choice(merged, options)
    top = max(final, key=lambda o: final[o])
    notes.append(f"{len(parsed_runs)} usable runs; top option {top} at {final[top]:.1%}")
    headline = ", ".join(f"{o} {final[o]:.0%}" for o in options[:6])
    return (
        {
            "question": ctx["question_id"],
            "probability_yes_per_category": {o: round(v, 6) for o, v in final.items()},
        },
        headline,
        notes,
        len(parsed_runs),
        used,
        parsed[0][1] if parsed else "",
    )


# -- entry point -----------------------------------------------------------
def forecast_question(
    post: dict,
    question: dict,
    research_text: str,
    research_sources: Sequence[str],
    models: Sequence[str],
    runs: int = 5,
    sample_reasoning: str = "",
) -> Forecast:
    ctx = build_context(post, question)
    qtype = ctx["type"]

    if qtype == "binary":
        payload, headline, notes, n, used, reasoning = forecast_binary(ctx, research_text, models, runs)
        method = "median of ensemble, log-odds shrink then 5% cap"
    elif qtype in CONTINUOUS_TYPES:
        payload, headline, notes, n, used, reasoning = forecast_numeric(ctx, research_text, models, runs)
        method = "median percentile by percentile, widened, 5% uniform mix"
    elif qtype == "multiple_choice":
        payload, headline, notes, n, used, reasoning = forecast_multiple_choice(
            ctx, research_text, models, runs
        )
        method = "median per option, mixed toward uniform"
    else:
        raise ValueError(f"unsupported question type {qtype!r} on question {ctx['question_id']}")

    # The rules ask for comments "so everyone can see their reasoning", and they
    # are made public when the question closes. Until this change the comment
    # carried the numbers and "(not captured)" where the reasoning belonged.
    comment = prompts.SUMMARY_TEMPLATE.format(
        models=", ".join(sorted(set(used))) or ", ".join(models),
        sources=", ".join(research_sources) or "none reached",
        n_runs=n,
        method=method,
        calibration="\n".join(f"- {note}" for note in notes),
        prediction_line=f"Forecast: {headline}",
        reasoning=_reasoning_excerpt(sample_reasoning or reasoning) or "(not captured)",
    )

    return Forecast(
        question_id=ctx["question_id"],
        post_id=ctx["post_id"],
        question_type=qtype,
        payload=payload,
        comment=comment,
        headline=headline,
        notes=notes,
        runs_used=n,
        models_used=sorted(set(used)),
    )
