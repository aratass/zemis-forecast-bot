"""Run configuration.

Tournament ids are read from the environment with slugs as defaults, because a
slug keeps working when Metaculus renumbers a season and the official SDK's
hardcoded id does not. TOURNAMENTS can be overridden entirely for a test run.
"""

from __future__ import annotations

import os

# Fall 2026 FutureEval runs 28 Sep 2026 to 6 Jan 2027, $50k.
# MiniBench is a rolling $1k round every two weeks.
SEASONAL_SLUG = os.environ.get("SEASONAL_TOURNAMENT", "fall-futureeval-2026")
MINIBENCH_SLUG = os.environ.get("MINIBENCH_TOURNAMENT", "minibench")
TEST_SLUG = os.environ.get("TEST_TOURNAMENT", "bot-testing-area")

MODES: dict[str, list[str]] = {
    "tournament": [SEASONAL_SLUG, MINIBENCH_SLUG],
    "seasonal": [SEASONAL_SLUG],
    "minibench": [MINIBENCH_SLUG],
    "test": [TEST_SLUG],
}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Five runs per question across three model families. The bot-maker survey put
# winners at roughly 28 LLM calls per question against 7 for non-winners, so
# this is the floor rather than the ceiling; raise it once the credit
# allocation is known.
RUNS_PER_QUESTION = _int_env("RUNS_PER_QUESTION", 5)

# On a metered free tier, five runs a question exhausts the per minute budget
# before the first question finishes, and a question nobody forecasts scores
# zero. Runs that land beat runs that 429.
#
# With the ensemble spread over several Gemini Flash versions, each version has
# its own free allowance, so one run per version costs each allowance one call
# per question: about 11 a day at the season's pace (60 MiniBench questions in
# a week plus about 3 seasonal ones a day), inside the 20 a day Google gives a
# free key on its newest Flash model. Without that spread it stays at 2.
RUNS_PER_QUESTION_METERED = _int_env("RUNS_PER_QUESTION_METERED", 4)
RUNS_PER_QUESTION_UNTIERED = _int_env("RUNS_PER_QUESTION_UNTIERED", 2)
ENSEMBLE_MODELS = _int_env("ENSEMBLE_MODELS", 3)

# A forecast is only submitted once at least this many strong-model answers
# are in. Until then the question waits for the next poll, keeping the answers
# it already has, unless it closes within DEFER_MARGIN_MINUTES, in which case
# whatever answered is used, stand-ins included. Questions are open to bots
# for 1.5 hours by the resources page (the watcher logs show some Fall ones
# open for 3), and the watcher polls every 4 minutes, so a 30 minute margin
# leaves at least fifteen attempts.
MIN_STRONG_ANSWERS = _int_env("MIN_STRONG_ANSWERS", 2)
DEFER_MARGIN_MINUTES = _int_env("DEFER_MARGIN_MINUTES", 30)

# Each ensemble call gets one retry on the same model before moving on to the
# next Flash version.
ENSEMBLE_ATTEMPTS = _int_env("ENSEMBLE_ATTEMPTS", 2)

# A dry run only reads and reasons, but it spends the same free allowances as a
# live run, so it looks at a handful of questions, not the whole tournament.
DRY_RUN_LIMIT = _int_env("DRY_RUN_LIMIT", 8)

# Questions are open to bots for about three hours, so a tick that takes longer
# than a few minutes risks missing the window entirely.
MAX_QUESTIONS_PER_TICK = _int_env("MAX_QUESTIONS_PER_TICK", 25)
QUESTION_WORKERS = _int_env("QUESTION_WORKERS", 3)
