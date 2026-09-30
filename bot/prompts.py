"""Prompts.

Shaped by what the published bot-maker surveys actually found, not by intuition:

  * Explicit base rates correlate with winning (r = +0.38; 40% of the top 15
    used them versus 7% of the bottom half).
  * Looking up how similar past questions resolved: 34% of winners, 0% of
    non-winners.
  * Status quo bias in the world, not the model, is the single most repeated
    piece of advice: most things do not change inside a short window.
  * Multi-persona prompting and "think like a Bayesian" framing both measured
    WORSE than plain reasoning, so neither appears here.
  * Date confusion is a top failure mode, so today's date, the open date and the
    resolution date are stated explicitly every time.
"""

from __future__ import annotations

from datetime import datetime, timezone

SYSTEM = (
    "You are a superforecaster producing a calibrated probability estimate. "
    "You are scored by a logarithmic rule against other forecasters, so "
    "overconfidence is punished far more harshly than it is rewarded. "
    "Reason concisely and concretely. Do not hedge in prose; put your uncertainty "
    "in the numbers."
)

_COMMON_STEPS = """Work through these steps before answering:
(a) The status quo outcome, if nothing changes between now and the resolution date.
(b) A reference class and its base rate. Say how often outcomes like this one occur, and over what sample. If you are guessing at the base rate, say so.
(c) What the evidence below actually establishes, separating reported fact from speculation. Note anything that looks stale.
(d) The strongest case for a higher answer and the strongest case for a lower one.
(e) How much can realistically change in the time remaining.

Weight the status quo heavily. Most questions resolve the boring way, and the world changes more slowly than news coverage suggests."""


# Two failure modes that bot makers report as their single most expensive
# questions of a season: one lost 90 peer points to the first, another said the
# second was his worst question by a factor of two. Both are cheap to prevent
# and neither is obvious to a model reading the question cold.
_GUARDS = """Two things about how this question works, which are the most expensive mistakes available here:

* It is open right now, which means it has NOT resolved yet. If the background or the evidence reads as though the thing already happened, assume you are looking at a similar past event, a proposal, a plan or somebody's forecast, not at resolution. Do not answer as though the outcome is settled unless the evidence is dated after today's date above and names this exact event.
* A question of the form "will X happen before <date>" covers the whole window from now until that date. It has not resolved No merely because X has not happened yet, and it has not resolved Yes merely because something resembling X happened before the question opened."""


# The rules ask for comments "so everyone can see their reasoning", and the
# comment quotes the model's answer. A reasoning model that thinks in private
# and then obeys "finish with exactly this line" answers with the line alone:
# the third dry run's Gemini comment read, in full, "PROBABILITY: 2%". So the
# reasoning is asked for in the open, briefly.
_SHOW_WORK = """Write your reasoning for these steps as short plain text before the final block: a few sentences for each step, naming the base rate and the evidence you relied on. It is published with the forecast as its explanation."""


# Looking up related markets is the single strongest free signal in the
# published surveys: 34% of winners did it against 0% of non-winners (p = 0.04),
# and it ranked second of 33 features the following season (r = +0.34).
#
# But the same data warns against leaning on it. One bot maker blended crowd
# predictions mechanically and fell from 20th to 63rd, and within the winner
# cohort more community-checking correlates with a LOWER score. So the prices
# go in as evidence to be weighed, never as an answer to copy.
_MARKET_NOTE = """About the [Manifold] and [Polymarket] lines above: those are live prices on OTHER questions that a keyword search thought were related. They are not this question.

Check the wording and the dates before you use one. A liquid market (high volume, many traders) asking almost exactly this question is strong evidence and should move you a long way. A thin market, or one whose wording or deadline differs, is weak and may be about something else entirely. Do not copy a price across a difference you have not checked, and do not average yourself with a market just to feel safe."""


def _header(ctx: dict) -> str:
    now = datetime.now(timezone.utc)
    parts = [
        f"Today's date: {now:%Y-%m-%d %H:%M} UTC.",
        f"Question: {ctx['title']}",
    ]
    if ctx.get("resolve_time"):
        parts.append(f"Scheduled resolution date: {ctx['resolve_time']}")
    if ctx.get("close_time"):
        parts.append(f"Question closes: {ctx['close_time']}")
    if ctx.get("days_left") is not None:
        parts.append(f"Time until resolution: about {ctx['days_left']} days.")
    return "\n".join(parts)


def _body(ctx: dict, research: str) -> str:
    chunks = [
        f"Resolution criteria:\n{ctx.get('resolution_criteria') or '(none given)'}",
        f"Background:\n{ctx.get('description') or '(none given)'}",
    ]
    if ctx.get("fine_print"):
        chunks.append(f"Fine print:\n{ctx['fine_print']}")
    chunks.append(f"Evidence gathered today:\n{research}")
    if "[Manifold]" in research or "[Polymarket]" in research:
        chunks.append(_MARKET_NOTE)
    return "\n\n".join(chunks)


def binary_prompt(ctx: dict, research: str) -> list[dict]:
    user = f"""{_header(ctx)}

{_body(ctx, research)}

{_COMMON_STEPS}

{_GUARDS}

{_SHOW_WORK}

Finish with exactly this line and nothing after it:
PROBABILITY: X%

where X is a number between 1 and 99."""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def numeric_prompt(ctx: dict, research: str) -> list[dict]:
    bounds = []
    unit = ctx.get("unit") or ""
    if ctx.get("open_lower_bound"):
        bounds.append(f"Values below {ctx['range_min_label']} are possible.")
    else:
        bounds.append(f"The outcome cannot be below {ctx['range_min_label']}.")
    if ctx.get("open_upper_bound"):
        bounds.append(f"Values above {ctx['range_max_label']} are possible.")
    else:
        bounds.append(f"The outcome cannot be above {ctx['range_max_label']}.")

    user = f"""{_header(ctx)}

{_body(ctx, research)}

Units: {unit or 'as stated in the question'}
{' '.join(bounds)}

{_COMMON_STEPS}
(f) What an unexpectedly low outcome would look like, and an unexpectedly high one. Real distributions have fatter tails than they feel like they should.

{_GUARDS}

{_SHOW_WORK}

Then give your distribution as percentiles. Use plain numbers in the question's units, with no commas, currency symbols, percent signs or words. Report the value on the scale the units state: if the unit is millions, answer in millions, not in ones. A bot maker lost eighty peer points on a single question to that one mistake.

Finish with exactly this block and nothing after it:
PERCENTILES:
5: <value>
10: <value>
20: <value>
40: <value>
60: <value>
80: <value>
90: <value>
95: <value>

The values must increase down the list."""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def multiple_choice_prompt(ctx: dict, research: str) -> list[dict]:
    options = ctx.get("options") or []
    listed = "\n".join(f"- {o}" for o in options)
    user = f"""{_header(ctx)}

{_body(ctx, research)}

The options are:
{listed}

{_COMMON_STEPS}

{_GUARDS}

Leave real probability on options that look unlikely but are not impossible. Do not let any option fall below 1 percent unless it is genuinely ruled out by the resolution criteria.

{_SHOW_WORK}

Finish with exactly this block, one line per option, using the option text exactly as written above, and nothing after it:
PROBABILITIES:
<option text>: X%
<option text>: X%

The percentages must sum to 100."""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def search_queries_prompt(ctx: dict) -> list[dict]:
    user = f"""Today is {datetime.now(timezone.utc):%Y-%m-%d}.

I need to research this forecasting question:
{ctx['title']}

{(ctx.get('resolution_criteria') or '')[:1200]}

Give me three news search queries that would surface the most decision-relevant
recent reporting. Make them short keyword queries, not questions, and make them
different from each other: one on the immediate subject, one on the wider
situation it sits in, and one on the mechanism or process that would decide it.

Answer with a JSON array of three strings and nothing else."""
    return [{"role": "system", "content": "You write precise search queries."}, {"role": "user", "content": user}]


SUMMARY_TEMPLATE = """Forecast produced automatically.

Models: {models}
Research sources: {sources}
Ensemble: {n_runs} runs, aggregated by {method}
{calibration}

{prediction_line}

Reasoning from one run:
{reasoning}
"""
