"""The loop that actually earns the score.

Questions open at random hours and close to bots about three hours later, and a
question the bot never sees scores zero. Under a prize rule proportional to the
square of the summed score, missed questions are the most expensive failure
available, well ahead of any forecast being slightly off. So this module is
built around not missing things: every question is attempted independently,
every failure is caught and logged rather than allowed to end the tick, and the
bot's own already-submitted forecasts are the source of truth for what to skip.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone

from . import config, research as research_mod
from .client import (
    MetaculusClient,
    MetaculusError,
    MissingForecastHistory,
    already_forecast,
    sub_questions,
)
from .forecast import (
    CONTINUOUS_TYPES,
    EnsembleTooThin,
    build_context,
    forecast_question,
    search_queries,
)
from .llm import (
    DEAD_MODELS,
    EXHAUSTED_UNTIL,
    FALLBACK_ONLY,
    USAGE,
    LLMError,
    NoModelsAvailable,
    metaculus_proxy_models,
    probe,
    provider_is_metered,
    resolve_models,
)

log = logging.getLogger("bot")

SUPPORTED = {"binary", "multiple_choice"} | CONTINUOUS_TYPES


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _seen_before(client: MetaculusClient, post: dict, question: dict) -> bool:
    """Has the bot already forecast this question, according to the server?

    The list endpoint only carries my_forecasts when it was asked for it. If it
    is missing anyway, fall back to the post detail, which always carries it.

    If neither can answer, forecast it. The tournament asks for one forecast
    per question, but spot peer scoring counts only the last forecast, so a
    second one costs nothing in points, while a question never forecast scores
    zero under a prize rule proportional to the square of the total. The error
    is logged loudly because a run full of these means the with_cp parameter
    has regressed again.
    """
    try:
        return already_forecast(question)
    except MissingForecastHistory as exc:
        log.error("%s", exc)
    try:
        detail = client.get_post(post.get("id"))
    except MetaculusError as exc:
        log.error("could not re-check post %s: %s", post.get("id"), exc)
        return False
    for candidate in sub_questions(detail or {}):
        if candidate.get("id") == question.get("id"):
            try:
                return already_forecast(candidate)
            except MissingForecastHistory:
                break
    log.error("post %s detail could not say whether q%s was forecast; forecasting it",
              post.get("id"), question.get("id"))
    return False


def collect_targets(
    client: MetaculusClient, tournaments: list[str], include_forecast: bool = False
) -> list[tuple[dict, dict]]:
    """Open questions in these tournaments that the bot has not forecast yet.

    ``include_forecast`` also returns questions already forecast. It exists for
    dry runs in the bot testing area, where nothing is submitted and Metaculus
    encourages resubmission, so the whole pipeline can be exercised on the same
    handful of test questions. It is never used against a tournament.
    """
    targets: list[tuple[dict, dict]] = []
    seen: set[int] = set()
    for tournament in tournaments:
        try:
            posts = list(client.iter_posts(tournament))
        except MetaculusError as exc:
            log.error("could not list %s: %s", tournament, exc)
            continue
        log.info("%s: %d open post(s)", tournament, len(posts))
        for post in posts:
            for question in sub_questions(post):
                qid = question.get("id")
                qtype = question.get("type")
                if not qid or qid in seen:
                    continue
                seen.add(qid)
                if qtype not in SUPPORTED:
                    log.info("skipping question %s: unsupported type %r", qid, qtype)
                    continue
                if not include_forecast and _seen_before(client, post, question):
                    continue
                targets.append((post, question))
    return targets


# Research per question for the life of the watcher. A question that waits
# for more strong answers is polled again every few minutes, and repeating the
# search each time would spend the finite AskNews allowance for nothing.
_RESEARCH: dict[int, tuple[str, list[str]]] = {}


def _research(ctx: dict, models: list[str]) -> tuple[str, list[str]]:
    qid = ctx["question_id"]
    if qid in _RESEARCH:
        return _RESEARCH[qid]
    queries = search_queries(ctx, models)
    report = research_mod.gather(queries, ctx=ctx)
    log.info("q%s evidence: %s", qid, report.source_mix)
    if report.errors:
        log.warning("q%s research issues: %s", qid, "; ".join(report.errors)[:300])
    _RESEARCH[qid] = (report.render(), report.source_names)
    return _RESEARCH[qid]


def handle_one(client: MetaculusClient, post: dict, question: dict, models: list[str], runs: int) -> str:
    ctx = build_context(post, question)
    qid = ctx["question_id"]
    title = ctx["title"][:90]

    research_text, research_sources = _research(ctx, models)

    forecast = forecast_question(
        post=post,
        question=question,
        research_text=research_text,
        research_sources=research_sources,
        models=models,
        runs=runs,
    )

    client.submit_forecasts([forecast.payload])
    # The comment is a prize-eligibility requirement, so a failure here is
    # logged loudly rather than swallowed, but it must not undo the forecast.
    try:
        client.post_comment(forecast.post_id, forecast.comment)
    except MetaculusError as exc:
        log.error("q%s forecast submitted but comment FAILED: %s", qid, exc)

    log.info("q%s %-14s %s | %s", qid, forecast.question_type, forecast.headline, title)
    log.info("q%s answered by %s", qid, ", ".join(forecast.models_used) or "nobody")
    for note in forecast.notes:
        log.info("  q%s: %s", qid, note)
    if client.dry_run:
        log.info("q%s dry run payload: %s", qid, _payload_summary(forecast.payload))
        log.info("q%s dry run comment (%d chars):\n%s", qid, len(forecast.comment), forecast.comment[:1500])
    _RESEARCH.pop(qid, None)
    return forecast.headline


def _payload_summary(payload: dict) -> str:
    """What would go on the wire, short enough for one log line."""
    if "probability_yes" in payload:
        return f"probability_yes={payload['probability_yes']}"
    if "probability_yes_per_category" in payload:
        cats = payload["probability_yes_per_category"]
        return f"categories={cats} sum={sum(cats.values()):.6f}"
    cdf = payload.get("continuous_cdf") or []
    if cdf:
        return f"cdf {len(cdf)} points, first {cdf[0]:.5f}, last {cdf[-1]:.5f}"
    return str(payload)[:200]


def run_tick(
    client: MetaculusClient,
    tournaments: list[str],
    models: list[str],
    runs: int,
    limit: int,
    include_forecast: bool = False,
) -> int:
    targets = collect_targets(client, tournaments, include_forecast=include_forecast)
    if not targets:
        log.info("nothing new to forecast")
        return 0
    if len(targets) > limit:
        # Oldest first: those are closest to closing.
        log.warning("%d questions pending, taking the %d nearest to closing", len(targets), limit)
        targets = targets[:limit]

    log.info("forecasting %d question(s) with %s", len(targets), ", ".join(models))
    done = 0
    outage = 0
    waiting = 0
    with cf.ThreadPoolExecutor(max_workers=config.QUESTION_WORKERS) as pool:
        futures = {
            pool.submit(handle_one, client, post, question, models, runs): question.get("id")
            for post, question in targets
        }
        for fut in cf.as_completed(futures):
            qid = futures[fut]
            try:
                fut.result()
                done += 1
            except EnsembleTooThin as exc:
                # Not a failure: the question waits for more strong answers.
                waiting += 1
                log.info("q%s waiting: %s", qid, str(exc)[:200])
            except NoModelsAvailable as exc:
                outage += 1
                log.error("q%s skipped, no model available: %s", qid, str(exc)[:300])
            except (LLMError, MetaculusError, ValueError) as exc:
                log.error("q%s failed: %s", qid, str(exc)[:400])
            except Exception:  # noqa: BLE001 - one bad question must not end the tick
                log.error("q%s crashed:\n%s", qid, traceback.format_exc()[:1500])
    if waiting:
        log.info("%d question(s) waiting for more strong-model answers", waiting)
    if outage:
        # Grinding through the rest of the list would be hundreds of doomed
        # requests against a shared proxy. The questions are untouched and the
        # next poll picks them up.
        log.error(
            "%d question(s) skipped because no model was reachable. Dead models: %s. "
            "Out for the day: %s. "
            "Nothing was submitted for them, so they will be retried next run.",
            outage,
            ", ".join(sorted(DEAD_MODELS)) or "none recorded",
            ", ".join(sorted(EXHAUSTED_UNTIL)) or "none",
        )
    return done


def check_sources() -> int:
    """Run every research source once and report which actually returned data.

    Worth running from the environment the bot runs in, not a developer machine:
    a source can be reachable in one place and blocked in another.
    """
    setup_logging()
    report = research_mod.gather(["european central bank interest rate decision"])
    by_source: dict[str, int] = {}
    for item in report.items:
        by_source[item.source] = by_source.get(item.source, 0) + 1
    print(json.dumps({"items_by_source": by_source, "errors": report.errors}, indent=2))
    try:
        models = resolve_models(config.ENSEMBLE_MODELS)
        print(json.dumps({"models_resolved": models}, indent=2))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"models_error": str(exc)}, indent=2))

    # Guessing a name the proxy does not serve costs a whole run, so ask it.
    proxy = metaculus_proxy_models()
    print(json.dumps({"metaculus_proxy_models": proxy or "none listed"}, indent=2))
    return 0 if by_source else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Forecast Metaculus tournament questions.")
    parser.add_argument("--mode", choices=sorted(config.MODES), default="tournament")
    parser.add_argument("--tournament", action="append", help="override the tournament slug or id")
    parser.add_argument("--runs", type=int, default=config.RUNS_PER_QUESTION)
    parser.add_argument("--limit", type=int, default=config.MAX_QUESTIONS_PER_TICK)
    parser.add_argument(
        "--watch",
        type=int,
        default=0,
        metavar="SECONDS",
        help="keep polling for this long inside one process, for continuous coverage",
    )
    parser.add_argument("--interval", type=int, default=300, help="seconds between polls in watch mode")
    parser.add_argument("--dry-run", action="store_true", help="do everything except submit")
    parser.add_argument("--check-sources", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.check_sources:
        return check_sources()

    setup_logging(args.verbose)
    tournaments = args.tournament or config.MODES[args.mode]
    log.info("mode=%s tournaments=%s dry_run=%s", args.mode, tournaments, args.dry_run)

    try:
        client = MetaculusClient(dry_run=args.dry_run)
    except MetaculusError as exc:
        log.error("%s", exc)
        return 2

    models = resolve_models(config.ENSEMBLE_MODELS)
    runs = args.runs
    if provider_is_metered() and runs == config.RUNS_PER_QUESTION:
        runs = config.RUNS_PER_QUESTION_METERED if FALLBACK_ONLY else config.RUNS_PER_QUESTION_UNTIERED
        log.info("provider is rate limited, using %d runs per question", runs)
    strong = [m for m in models if m not in FALLBACK_ONLY]
    log.info("ensemble: %s", ", ".join(strong))
    if FALLBACK_ONLY:
        log.info("stand-ins, used only near the deadline: %s", ", ".join(m for m in models if m in FALLBACK_ONLY))

    # A dry run in the bot testing area re-forecasts questions the bot has
    # already answered there, since nothing is submitted; anywhere else a dry
    # run still skips them. And it looks at a handful, because it spends the
    # same free allowances the live watcher needs.
    include_forecast = bool(args.dry_run and args.mode == "test" and not args.tournament)
    limit = args.limit
    if args.dry_run and limit == config.MAX_QUESTIONS_PER_TICK:
        limit = config.DRY_RUN_LIMIT
    if args.dry_run:
        for model, outcome in probe(models).items():
            log.info("probe %s: %s", model, outcome)

    deadline = time.monotonic() + args.watch if args.watch else None
    total = 0
    while True:
        started = time.monotonic()
        try:
            total += run_tick(client, tournaments, models, runs, limit, include_forecast)
        except Exception:  # noqa: BLE001 - a watch loop must outlive one bad tick
            log.error("tick crashed:\n%s", traceback.format_exc()[:2000])

        if deadline is None:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        nap = max(30.0, min(args.interval - (time.monotonic() - started), remaining))
        log.info("sleeping %.0fs (%.0fs left in this watch window)", nap, remaining)
        time.sleep(nap)

    log.info("forecast %d question(s) this run. LLM usage: %s", total, USAGE.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
