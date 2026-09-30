"""The checks a forecast must pass, and a running tally of what the run did.

The checks are the ones that decide whether a forecast counts at all or keeps
the bot prize eligible, not whether it is any good:

* a binary forecast at exactly 50 percent means the pipeline fell back to a
  placeholder, which scores like no forecast and still uses the question up;
* a malformed multiple choice or continuous forecast is rejected by the server;
* a question forecast twice in one run breaks the one-forecast rule;
* a comment without the reasoning fails the rule that comments show it.

Every forecast is checked before it is submitted. A problem is reported (log,
annotation, job summary) and, for the two cases the server would reject
anyway, the forecast is not sent.
"""

from __future__ import annotations

import threading
from collections import Counter
from dataclasses import dataclass, field

from .cdf import DEFAULT_INBOUND_OUTCOME_COUNT, validate_cdf

BINARY_MIN, BINARY_MAX = 0.001, 0.999


def check_payload(payload: dict, question: dict) -> list[str]:
    """Problems with a forecast payload. Empty means it is fit to submit."""
    problems: list[str] = []
    qtype = question.get("type")
    if qtype == "binary":
        p = payload.get("probability_yes")
        if not isinstance(p, (int, float)):
            return ["binary forecast has no probability"]
        if abs(float(p) - 0.5) < 1e-9:
            problems.append("binary forecast at exactly 50%")
        if not BINARY_MIN <= float(p) <= BINARY_MAX:
            problems.append(f"binary probability {p} outside [0.001, 0.999]")
    elif qtype == "multiple_choice":
        cats = payload.get("probability_yes_per_category") or {}
        options = list(question.get("options") or [])
        if set(cats) != set(options):
            problems.append("multiple choice options do not match the question")
        total = sum(float(v) for v in cats.values())
        # The server's test is numpy.isclose(total, 1), about one part in 1e5.
        if abs(total - 1.0) > 1e-5:
            problems.append(f"multiple choice probabilities sum to {total:.6f}")
        if any(not 0.001 <= float(v) <= 0.999 for v in cats.values()):
            problems.append("a multiple choice probability is outside [0.001, 0.999]")
    elif qtype in ("numeric", "discrete", "date"):
        cdf = payload.get("continuous_cdf") or []
        count = int(question.get("inbound_outcome_count") or DEFAULT_INBOUND_OUTCOME_COUNT)
        errors = validate_cdf(
            cdf,
            count,
            bool(question.get("open_lower_bound")),
            bool(question.get("open_upper_bound")),
        )
        problems.extend(f"distribution: {e}" for e in errors[:3])
    else:
        problems.append(f"unsupported question type {qtype!r}")
    return problems


def is_rejectable(problems: list[str]) -> bool:
    """Problems the server rejects outright, so there is no point sending it.

    Comment problems are reported but never hold a forecast back: a forecast
    with a thin comment still scores, and one not sent scores zero.
    """
    return any(
        not p.startswith("binary forecast at exactly 50%") and not p.startswith("comment ")
        for p in problems
    )


def check_comment(comment: str) -> list[str]:
    problems: list[str] = []
    text = comment or ""
    if len(text.strip()) < 200:
        problems.append("comment is nearly empty")
    if "Forecast:" not in text:
        problems.append("comment does not state the forecast")
    if "(not captured)" in text:
        problems.append("comment carries no reasoning")
    return problems


@dataclass
class ForecastRecord:
    question_id: int
    question_type: str
    headline: str
    models: list[str]
    sources: str
    problems: list[str]
    comment_chars: int


@dataclass
class RunTally:
    """What this process did. Shared by the worker threads, so it locks."""

    forecasts: list[ForecastRecord] = field(default_factory=list)
    waiting: set = field(default_factory=set)
    failed: list[str] = field(default_factory=list)
    comment_failures: list[int] = field(default_factory=list)
    held_back: list[int] = field(default_factory=list)
    attempts: Counter = field(default_factory=Counter)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def forecast(self, record: ForecastRecord) -> None:
        with self._lock:
            self.forecasts.append(record)
            self.attempts[record.question_id] += 1
            self.waiting.discard(record.question_id)

    def wait(self, qid: int) -> None:
        with self._lock:
            self.waiting.add(qid)

    def fail(self, qid: int, why: str) -> None:
        with self._lock:
            self.failed.append(f"q{qid}: {why[:160]}")

    def comment_failed(self, qid: int) -> None:
        with self._lock:
            self.comment_failures.append(qid)

    def hold_back(self, qid: int) -> None:
        with self._lock:
            self.held_back.append(qid)

    def forecast_ids(self) -> set:
        with self._lock:
            return {r.question_id for r in self.forecasts}

    @property
    def duplicates(self) -> list[int]:
        with self._lock:
            return sorted(q for q, n in self.attempts.items() if n > 1)

    @property
    def with_problems(self) -> list[ForecastRecord]:
        with self._lock:
            return [r for r in self.forecasts if r.problems]

    def lines(self) -> list[str]:
        by_type = Counter(r.question_type for r in self.forecasts)
        models = Counter(m for r in self.forecasts for m in r.models)
        out = [
            f"forecasts: {len(self.forecasts)} ({', '.join(f'{t} {n}' for t, n in sorted(by_type.items())) or 'none'})",
            f"forecasts each model answered in: {', '.join(f'{m} {n}' for m, n in models.most_common()) or 'none'}",
            f"forecasts with a problem: {len(self.with_problems)}",
            f"questions forecast twice in this run: {self.duplicates or 0}",
            f"comments that failed to post: {self.comment_failures or 0}",
            f"held back because the server would reject them: {self.held_back or 0}",
            f"still waiting for strong answers: {sorted(self.waiting) or 0}",
            f"failed: {len(self.failed)}" + (f" ({'; '.join(self.failed[:5])})" if self.failed else ""),
        ]
        return out
