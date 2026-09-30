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

from . import config, report, research as research_mod
from . import forecast as forecast_mod
from . import llm as llm_mod
from .audit import latest_comments, run_audit
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
    forget,
    search_queries,
)
from .llm import (
    DEAD_MODELS,
    EXHAUSTED_UNTIL,
    FALLBACK_ONLY,
    REASONING_EFFORT,
    USAGE,
    LLMError,
    NoModelsAvailable,
    catalogue_report,
    failure_summary,
    model_available,
    metaculus_proxy_models,
    probe,
    provider_is_metered,
    resolve_models,
)

from .monitor import ForecastRecord, RunTally, check_comment, check_payload, is_rejectable

log = logging.getLogger("bot")

# What this process has done, for the end-of-run tally and the annotations.
TALLY = RunTally()

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
    client: MetaculusClient,
    tournaments: list[str],
    include_forecast: bool = False,
    skip: set | None = None,
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
                if not qid or qid in seen or (skip and qid in skip):
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
_RESEARCH: dict[int, tuple[str, list[str], str]] = {}


def _research(ctx: dict, models: list[str]) -> tuple[str, list[str], str]:
    qid = ctx["question_id"]
    if qid in _RESEARCH:
        return _RESEARCH[qid]
    queries = search_queries(ctx, models)
    report_ = research_mod.gather(queries, ctx=ctx)
    log.info("q%s evidence: %s", qid, report_.source_mix)
    if report_.errors:
        log.warning("q%s research issues: %s", qid, "; ".join(report_.errors)[:300])
    _RESEARCH[qid] = (report_.render(), report_.source_names, report_.source_mix)
    return _RESEARCH[qid]


def handle_one(client: MetaculusClient, post: dict, question: dict, models: list[str], runs: int) -> str:
    ctx = build_context(post, question)
    qid = ctx["question_id"]
    title = ctx["title"][:90]

    research_text, research_sources, source_mix = _research(ctx, models)

    forecast = forecast_question(
        post=post,
        question=question,
        research_text=research_text,
        research_sources=research_sources,
        models=models,
        runs=runs,
    )

    problems = check_payload(forecast.payload, question) + check_comment(forecast.comment)
    if is_rejectable(problems):
        # The server would refuse it, and a refused forecast still costs the
        # attempt. Keep the question open for the next poll instead.
        TALLY.hold_back(qid)
        _RESEARCH.pop(qid, None)
        forget(qid)
        raise ValueError(f"held back, would be rejected: {'; '.join(problems)[:300]}")

    client.submit_forecasts([forecast.payload])
    forget(qid)
    # The comment is a prize-eligibility requirement, so a failure here is
    # logged loudly rather than swallowed, but it must not undo the forecast.
    try:
        client.post_comment(forecast.post_id, forecast.comment)
    except MetaculusError as exc:
        log.error("q%s forecast submitted but comment FAILED: %s", qid, exc)
        TALLY.comment_failed(qid)
        report.annotate("error", f"comment failed on q{qid}", f"{title}\n{str(exc)[:600]}")

    log.info("q%s %-14s %s | %s", qid, forecast.question_type, forecast.headline, title)
    log.info("q%s answered by %s", qid, ", ".join(forecast.models_used) or "nobody")
    for note in forecast.notes:
        log.info("  q%s: %s", qid, note)
    if problems:
        log.warning("q%s checks: %s", qid, "; ".join(problems))
    if client.dry_run:
        log.info("q%s dry run payload: %s", qid, _payload_summary(forecast.payload))
        log.info("q%s dry run comment (%d chars):\n%s", qid, len(forecast.comment), forecast.comment[:1500])

    TALLY.forecast(
        ForecastRecord(
            question_id=qid,
            question_type=forecast.question_type,
            headline=forecast.headline,
            models=list(forecast.models_used),
            sources=source_mix,
            problems=problems,
            comment_chars=len(forecast.comment),
        )
    )
    lines = [
        f"{title}",
        f"{'would submit' if client.dry_run else 'submitted'}: {_payload_summary(forecast.payload)}",
        f"headline: {forecast.headline}",
        f"answered by: {', '.join(forecast.models_used) or 'nobody'}",
        f"evidence: {source_mix}",
        f"checks: {'; '.join(problems) if problems else 'all passed'}",
        f"comment: {len(forecast.comment)} characters",
    ]
    if client.dry_run:
        lines.append("")
        lines.append(forecast.comment[:1600])
    report.annotate(
        "warning" if problems else "notice",
        f"{'dry run ' if client.dry_run else ''}q{qid} {forecast.question_type}",
        "\n".join(lines),
        reserve=1,
    )
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


def one_of_each_type(targets: list[tuple[dict, dict]], limit: int) -> list[tuple[dict, dict]]:
    """Take questions type by type, so a small dry run still covers every type.

    The bot testing area holds one open question of each type plus two groups;
    the first eight in server order were mostly group members of one type.
    """
    by_type: dict[str, list[tuple[dict, dict]]] = {}
    for pair in targets:
        by_type.setdefault(pair[1].get("type") or "", []).append(pair)
    picked: list[tuple[dict, dict]] = []
    depth = 0
    while len(picked) < limit and any(len(v) > depth for v in by_type.values()):
        for pairs in by_type.values():
            if depth < len(pairs) and len(picked) < limit:
                picked.append(pairs[depth])
        depth += 1
    return picked


def run_tick(
    client: MetaculusClient,
    tournaments: list[str],
    models: list[str],
    runs: int,
    limit: int,
    include_forecast: bool = False,
    diverse: bool = False,
) -> int:
    # A dry run submits nothing, so the server cannot tell it what this
    # process already did; the tally can.
    done_here = TALLY.forecast_ids() if client.dry_run else None
    targets = collect_targets(client, tournaments, include_forecast=include_forecast, skip=done_here)
    if not targets:
        log.info("nothing new to forecast")
        return 0
    if diverse:
        targets = one_of_each_type(targets, limit)
    elif len(targets) > limit:
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
                TALLY.wait(qid)
                log.info("q%s waiting: %s", qid, str(exc)[:200])
            except NoModelsAvailable as exc:
                outage += 1
                TALLY.fail(qid, f"no model available: {exc}")
                log.error("q%s skipped, no model available: %s", qid, str(exc)[:300])
            except (LLMError, MetaculusError, ValueError) as exc:
                TALLY.fail(qid, str(exc))
                log.error("q%s failed: %s", qid, str(exc)[:400])
                report.annotate("warning", f"q{qid} failed", str(exc)[:1500], reserve=1)
            except Exception:  # noqa: BLE001 - one bad question must not end the tick
                trace = traceback.format_exc()
                TALLY.fail(qid, trace.strip().splitlines()[-1])
                log.error("q%s crashed:\n%s", qid, trace[:1500])
                report.annotate("error", f"q{qid} crashed", trace[-1500:], reserve=1)
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
    found = research_mod.gather(["european central bank interest rate decision"])
    by_source: dict[str, int] = {}
    for item in found.items:
        by_source[item.source] = by_source.get(item.source, 0) + 1
    print(json.dumps({"items_by_source": by_source, "errors": found.errors}, indent=2))
    try:
        models = resolve_models(config.ENSEMBLE_MODELS)
        print(json.dumps({"models_resolved": models}, indent=2))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"models_error": str(exc)}, indent=2))

    # Guessing a name the proxy does not serve costs a whole run, so ask it.
    proxy = metaculus_proxy_models()
    print(json.dumps({"metaculus_proxy_models": proxy or "none listed"}, indent=2))

    # The same run can read the bot's own record at no model cost.
    source_lines = [f"{name}: {n} item(s)" for name, n in sorted(by_source.items())] or ["no source answered"]
    source_lines += [f"error: {e[:160]}" for e in found.errors[:6]]
    report.annotate("notice", "research sources", "\n".join(source_lines))
    try:
        _self_audit(MetaculusClient(dry_run=True))
    except MetaculusError as exc:
        log.error("%s", exc)
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
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="write the self-audit and the resolved ensemble, forecast nothing",
    )
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

    _self_audit(client)

    models = resolve_models(config.ENSEMBLE_MODELS)
    if args.audit_only:
        # Its own workflow step, so that a failure here never stops a watcher.
        strong = [m for m in models if m not in FALLBACK_ONLY]
        lines = [
            f"strong models: {', '.join(strong) or 'none'}",
            f"stand-ins (near the deadline only): {', '.join(m for m in models if m in FALLBACK_ONLY) or 'none'}",
            f"keys present: {', '.join(_keys_present()) or 'none'}",
        ]
        report.annotate("notice", "ensemble at start", "\n".join(lines))
        return 0
    runs = args.runs
    if provider_is_metered() and runs == config.RUNS_PER_QUESTION:
        runs = config.RUNS_PER_QUESTION_METERED if llm_mod.TIERED or FALLBACK_ONLY else config.RUNS_PER_QUESTION_UNTIERED
        log.info("provider is rate limited, using %d runs per question", runs)
    strong = [m for m in models if m not in FALLBACK_ONLY]
    stand_ins = [m for m in models if m in FALLBACK_ONLY]
    log.info("ensemble: %s", ", ".join(strong))
    if FALLBACK_ONLY:
        log.info("stand-ins, used only near the deadline: %s", ", ".join(stand_ins))

    # A dry run in the bot testing area re-forecasts questions the bot has
    # already answered there, since nothing is submitted; anywhere else a dry
    # run still skips them. And it looks at a handful, because it spends the
    # same free allowances the live watcher needs.
    include_forecast = bool(args.dry_run and args.mode == "test" and not args.tournament)
    limit = args.limit
    if args.dry_run and limit == config.MAX_QUESTIONS_PER_TICK:
        limit = config.DRY_RUN_LIMIT
    ensemble_lines = [
        f"mode {args.mode}, tournaments {', '.join(map(str, tournaments))}, "
        f"{'dry run, nothing submitted' if args.dry_run else 'live'}",
        f"strong models: {', '.join(strong) or 'none'}",
        f"stand-ins (near the deadline only): {', '.join(stand_ins) or 'none, by design (BOT_STAND_INS=off)'}",
        f"runs per question: {runs}; strong answers needed: {config.MIN_STRONG_ANSWERS}; "
        f"stand-ins allowed within {config.DEFER_MARGIN_MINUTES} min of close",
        f"reasoning effort: {REASONING_EFFORT}",
        f"keys present: {', '.join(_keys_present()) or 'none'}",
    ]
    if args.dry_run:
        for model, outcome in probe(models).items():
            log.info("probe %s: %s", model, outcome)
            ensemble_lines.append(f"probe {model}: {outcome}")
        try:
            ensemble_lines.extend(catalogue_report())
        except Exception as exc:  # noqa: BLE001 - a report must not stop a run
            ensemble_lines.append(f"catalogues unreadable: {str(exc)[:120]}")
    report.annotate("notice", "ensemble", "\n".join(ensemble_lines))
    report.summary("### Ensemble\n\n" + "\n".join(f"- {line}" for line in ensemble_lines))

    deadline = time.monotonic() + args.watch if args.watch else None
    interval = args.interval
    if args.dry_run and not args.watch and config.DRY_RUN_PATIENCE_SECONDS > 0:
        # A single pass sees a question wait and then stops, which tests the
        # waiting but never the forecast. So a dry run polls like a watcher
        # for a while, until nothing waits, and never more than the patience.
        deadline = time.monotonic() + config.DRY_RUN_PATIENCE_SECONDS
        interval = config.DRY_RUN_INTERVAL_SECONDS
    total = 0
    final_dry_poll = False
    while True:
        started = time.monotonic()
        if final_dry_poll:
            # The last poll of a dry run takes every question still waiting
            # through the deadline path, so each type is carried to a checked
            # payload and comment even when the strong models are out.
            forecast_mod.FORCE_DEADLINE = True
            log.info("last dry-run poll: questions still waiting go through the deadline path")
        try:
            total += run_tick(
                client,
                tournaments,
                models,
                runs,
                limit,
                include_forecast,
                diverse=bool(args.dry_run and include_forecast),
            )
        except Exception:  # noqa: BLE001 - a watch loop must outlive one bad tick
            trace = traceback.format_exc()
            log.error("tick crashed:\n%s", trace[:2000])
            report.annotate("error", "tick crashed", trace[-1500:], reserve=1)

        if deadline is None or final_dry_poll:
            break
        if args.dry_run and not args.watch and not TALLY.waiting:
            break
        remaining = deadline - time.monotonic()
        strong_back = any(model_available(m, time.time() + remaining) for m in strong)
        if args.dry_run and not args.watch and (remaining <= interval or not strong_back):
            # Out of patience, or every strong model is out for the day
            # beyond it: waiting longer cannot change what the dry run sees.
            final_dry_poll = True
            nap = max(30.0, min(interval - (time.monotonic() - started), max(remaining, 0.0)))
            log.info("sleeping %.0fs before the last dry-run poll", nap)
            time.sleep(nap)
            continue
        if remaining <= 0:
            break
        nap = max(30.0, min(interval - (time.monotonic() - started), remaining))
        log.info("sleeping %.0fs (%.0fs left in this watch window)", nap, remaining)
        time.sleep(nap)

    log.info("forecast %d question(s) this run. LLM usage: %s", total, USAGE.summary())
    tally = TALLY.lines() + [
        f"LLM usage: {USAGE.summary()}",
        f"failed model calls: {failure_summary()}",
    ]
    for line in tally:
        log.info("tally: %s", line)
    troubled = bool(TALLY.with_problems or TALLY.duplicates or TALLY.comment_failures or TALLY.held_back)
    report.annotate("warning" if troubled else "notice", "run tally", "\n".join(tally))
    report.summary("### Run tally\n\n" + "\n".join(f"- {line}" for line in tally))
    return 0


def _keys_present() -> list[str]:
    """Which optional keys this run holds, by name only. Never the values."""
    names = (
        "OPENROUTER_API_KEY",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "ASKNEWS_CLIENT_ID",
        "ASKNEWS_SECRET",
    )
    return [n for n in names if os.environ.get(n)]


def _self_audit(client: MetaculusClient) -> None:
    """Read the bot's own record in the scored tournaments, and publish it.

    Always the scored tournaments, whatever this run forecasts, because that
    record is what the prize depends on. BOT_AUDIT=off skips it.
    """
    if (os.environ.get("BOT_AUDIT") or "").strip().lower() in ("off", "0", "false", "no"):
        return
    started = time.monotonic()
    try:
        me, audits = run_audit(client, config.MODES["tournament"])
    except Exception as exc:  # noqa: BLE001 - never let the audit stop a run
        log.error("self-audit crashed: %s", str(exc)[:300])
        return
    if not audits:
        report.annotate("warning", "self-audit", "could not read the bot's own record")
        return
    who = me.get("username") or "the bot"
    lines = [f"{who} (id {me.get('id')}), audited in {time.monotonic() - started:.0f}s"]
    lines += [a.line() for a in audits]
    try:
        recent = latest_comments(client, me.get("id"))
        lines.append("latest comments:" + ("\n  " + "\n  ".join(recent) if recent else " none"))
    except Exception as exc:  # noqa: BLE001 - never let the audit stop a run
        lines.append(f"latest comments unreadable: {str(exc)[:160]}")
    for line in lines:
        log.info("audit: %s", line)
    clean = all(a.clean for a in audits)
    report.annotate("notice" if clean else "warning", "self-audit", "\n".join(lines))
    report.summary("### Self-audit\n\n" + "\n".join(f"- {line}" for line in lines))


if __name__ == "__main__":
    raise SystemExit(main())
