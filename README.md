# Metaculus forecasting bot

An automated forecaster for the Metaculus FutureEval bot tournaments. It polls
for newly opened questions, researches each one, asks several models from
different families, combines their answers, and submits a calibrated forecast
with a comment explaining it.

This file doubles as the "overview of how the bot works" that prize eligibility
requires.

## What it is aimed at

Tournament rank is the **sum of spot peer scores** over every question, and the
share of the prize pool is proportional to **that sum squared**. Two things
follow, and they drive every design decision here:

1. **A question not forecast scores zero**, which forfeits roughly fifteen to
   nineteen points. Because the prize rule is quadratic, the value of each
   additional point grows with the total you already have, so coverage
   compounds. Questions open at random hours and close to bots after about
   three hours.
2. **One confident blowup can undo dozens of good questions.** Saying 0.1% and
   being wrong, against a field at 50%, scores about -621. Roughly 33 well
   forecast questions to climb back.

So the bot is built to never miss a question and never blow up, in that order.
Being clever comes third.

## How a forecast is produced

1. **Poll.** Every 15 minutes, list open questions in the configured
   tournaments. Questions the bot has already forecast are skipped, using the
   API's own record rather than local state, so a lost runner cannot cause a
   double submission.
2. **Research.** A model turns the question into three search queries, then
   every free source is queried in parallel: AskNews if a key is present,
   GDELT, Wikipedia, and the open odds on Polymarket and Manifold. Sources fail
   independently; a dead endpoint costs a log line, never the forecast.
3. **Ensemble.** On free keys, four runs, one led by each of the newest
   Gemini Flash versions the key can see (3.5 and newer), with every run
   falling through the other Flash versions before anything else. Other free
   models (OpenRouter's free Nemotron 3 Ultra, GitHub Models' GPT-4.1, Groq's
   gpt-oss-120b, in that order) only stand in, and only when the question is
   within 30 minutes of closing and fewer than two Flash answers are in. Until
   then the question waits for the next poll and keeps the answers it has.
   The reason is measured: Metaculus's FutureEval model leaderboard scores
   Gemini 3.5 Flash at +12.17 and 3.6 Flash at +13.22 against GPT-4o, and
   gpt-oss-120b at -0.26, and from 21 to 30 September 63 of the 68 answers
   behind this bot's tournament forecasts came from gpt-oss-120b because the
   single Gemini member kept answering 429 or 503. With sponsored credits the
   old design applies: runs across three model families. The prompt asks for
   the status quo outcome, an explicit base rate and reference class, what the
   evidence establishes, the case each way, and how much can change in the
   time left. It does not use multi-persona or "think like a Bayesian"
   framing, both of which measured worse than plain reasoning in published
   evaluations.
4. **Aggregate.** Median, not mean, so one hallucinated number cannot move the
   answer. Numeric questions take the median percentile by percentile. When
   there are enough Flash answers, stand-in answers are left out of the median
   altogether.
5. **Calibrate.** Binary probabilities are shrunk toward even odds in log-odds
   space and then capped at 5% and 95%. Numeric distributions are widened, given
   explicit tails, and mixed with a uniform. Multiple choice is mixed toward the
   uniform, because it is the question type where bots lose most heavily to
   human professionals.
6. **Submit**, then comment. The comment records the models, the sources, the
   individual ensemble values, the calibration applied, and the reasoning of
   the run nearest the median (the rules ask for comments "so everyone can see
   their reasoning"; Metaculus makes them public when the question closes).

## The parts worth reviewing

**`bot/cdf.py`** is where a numeric forecast is most likely to be thrown away.
Metaculus rejects a malformed distribution outright, and a rejected forecast
scores zero. The constraints are exact: 201 values evenly spaced in unscaled
space, every bucket mass between 0.00005 and 0.2, and the endpoints exactly 0
and 1 when the corresponding bound is closed. This module contains a local copy
of the server's validator, every CDF is checked against it before leaving the
process, and anything that fails degrades to a legal uniform rather than being
skipped. The fuzz tests throw 3,000 malformed model outputs at it.

The distribution is given explicit tails. Interpolating outward from the
outermost elicited percentile extrapolates the steep middle slope, which
produces a box with cliff edges, where an outcome one bucket outside the
elicited range scores the same as one at the far end of the scale. Tail anchors
at the 1st and 0.1st percentiles fix that. `tests/test_backtest.py` guards it.

**`bot/aggregate.py`** holds the calibration constants and the reasoning behind
each one, including the break-even arithmetic for the 5% cap, kept as a test so
the number cannot drift away from its justification.

**`bot/scoring.py`** reimplements the tournament's own scoring from the
Metaculus server source, so the bot can be measured on the metric that pays
rather than on a proxy. It reproduces the published winning average of +18.90
peer points to within 0.6.

## Running it

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q

python -m bot.runner --check-sources          # which sources answer, which models resolved
python -m bot.runner --mode test              # the unscored bot testing area
python -m bot.runner --mode test --dry-run      # re-forecasts the testing area, submits nothing
python -m bot.runner --mode tournament --dry-run
python -m bot.runner --mode tournament        # live
```

`--watch SECONDS` keeps one process polling, for continuous coverage from a
single job.

## Secrets

Only `METACULUS_TOKEN` is required. It authenticates the API and, as a
fallback, the Metaculus LLM proxy. Everything else is optional and improves
things if present:

| secret | what it buys |
| --- | --- |
| `METACULUS_TOKEN` | required: reading questions, submitting, commenting |
| `GEMINI_API_KEY` | the ensemble on free keys: every Gemini Flash version from 3.5 up |
| `OPENROUTER_API_KEY` | a free key buys the free Nemotron 3 Ultra stand-in (`OPENROUTER_FREE_ONLY=0` for sponsored credits) |
| `GROQ_API_KEY` | gpt-oss-120b, the last stand-in |
| `ASKNEWS_CLIENT_ID`, `ASKNEWS_SECRET` | news search via the Metaculus partnership |

GitHub Models' GPT-4.1 stand-in needs no secret: the workflow passes its own
`GITHUB_TOKEN` as `GITHUB_MODELS_TOKEN` when it grants `models: read`.

No secret is ever logged; the client scrubs the token out of exception text
before printing it.

## Configuration

| variable | default | notes |
| --- | --- | --- |
| `SEASONAL_TOURNAMENT` | `fall-futureeval-2026` | a slug, so it survives renumbering |
| `MINIBENCH_TOURNAMENT` | `minibench` | |
| `RUNS_PER_QUESTION` | 5 | winners averaged about 28 LLM calls per question |
| `RUNS_PER_QUESTION_METERED` | 4 | on free keys: one run led by each Flash version |
| `MIN_STRONG_ANSWERS` | 2 | Flash answers needed before a forecast goes out early |
| `DEFER_MARGIN_MINUTES` | 30 | inside this, stand-ins are used rather than nothing |
| `PRIMARY_GEMINI_MIN_VERSION` | 3.5 | oldest Flash version allowed in the ensemble |
| `ENSEMBLE_MODELS` | 3 | one per model family |
| `BOT_MODELS` | unset | comma separated, pins models instead of resolving them |
| `MAX_QUESTIONS_PER_TICK` | 25 | a tick must finish well inside the three hour window |

Model ids are resolved at runtime from the provider catalogue rather than
pinned, because a season runs four months and model names change inside that
window.

## Measuring it

```bash
python -m backtest.run collect --limit 60 --out backtest/cache/runs.json
python -m backtest.run fit --cache backtest/cache/runs.json
```

Collect runs the ensemble over already-resolved questions and caches the raw
model output. Fit sweeps the calibration parameters over that cache, so one
round of paid calls fits everything.

**Read the output as a ranking of configurations, not as a performance
estimate.** A model can already know how a resolved question turned out;
published work measures a 52% Brier gap when models are asked to pretend
otherwise. That contamination pushes any fitted shrinkage toward zero, which is
the wrong direction. Honest calibration comes from MiniBench, which is live,
unresolved, and runs a fresh round every two weeks.

## Rules this bot is built to respect

- No human in the loop. The bot forecasts and comments through the API on its
  own; nothing is reviewed and resubmitted.
- It forecasts each question once. It does not re-run a question because the
  answer looked wrong.
- Development and testing happen on resolved questions, the bot testing area,
  or the main site, never on open tournament questions. A dry run only
  re-forecasts questions it has already answered when it is pointed at the
  bot testing area; in tournament mode it skips them like a live run.
- A question that waits for more Flash answers is not re-run because of what
  it said: nothing has been submitted, and the answers already in are kept.
- Prediction market odds are used, which the rules explicitly permit: a bot
  "may use any resources that are generally available to human forecasters.
  This includes using publicly available forecasts on questions found on other
  platforms or on Metaculus itself."
