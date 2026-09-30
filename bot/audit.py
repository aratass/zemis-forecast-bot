"""The bot checks its own record, from the server's side, at the start of a run.

Everything a person would otherwise have to log in as the bot to see:

* how many questions it has forecast in each tournament, how many of those
  have resolved, and the summed scores on the resolved ones;
* whether every forecast question carries its comment, which the prize rules
  require ("We require the bots to leave comments and forecasts so everyone
  can see their reasoning");
* whether any question was forecast more than once (the rules ask for one);
* whether any binary forecast sits at exactly 50 percent, or any continuous
  forecast fails the server's own shape rules;
* whether an open question is still waiting for its forecast;
* where the bot stands on each tournament's leaderboard.

It only reads. It costs a handful of API requests and no model calls, and a
failure anywhere in it is reported and never stops the run.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .cdf import DEFAULT_INBOUND_OUTCOME_COUNT, validate_cdf
from .client import MetaculusClient, MetaculusError, sub_questions

log = logging.getLogger(__name__)

ALL_STATUSES = ["open", "closed", "resolved"]
UNSCORED = {"annulled", "ambiguous"}


@dataclass
class TournamentAudit:
    slug: str
    posts_forecast: int = 0
    questions_forecast: int = 0
    questions_open: int = 0
    questions_resolved: int = 0
    questions_annulled: int = 0
    spot_peer_sum: float = 0.0
    peer_sum: float = 0.0
    baseline_sum: float = 0.0
    scored: int = 0
    missing_comments: list[int] = field(default_factory=list)
    duplicates: list[int] = field(default_factory=list)
    at_fifty: list[int] = field(default_factory=list)
    malformed: list[int] = field(default_factory=list)
    open_unforecast: list[int] = field(default_factory=list)
    # Closed or resolved questions the bot never forecast. Each one scored
    # zero, which under a squared prize rule is the most expensive outcome.
    missed: list[int] = field(default_factory=list)
    project_id: int | None = None
    rank: int | None = None
    ranked_entries: int | None = None
    leaderboard_score: float | None = None
    prize: float | None = None
    errors: list[str] = field(default_factory=list)
    # One line per problem question, saying when it happened, so a duplicate
    # can be matched to the run that made it.
    details: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not (self.missing_comments or self.duplicates or self.at_fifty or self.malformed)

    def line(self) -> str:
        parts = [
            f"{self.slug}: {self.questions_forecast} questions forecast "
            f"({self.questions_open} open, {self.questions_resolved} resolved"
            + (f", {self.questions_annulled} annulled" if self.questions_annulled else "")
            + ")"
        ]
        if self.scored:
            parts.append(
                f"scores on {self.scored} resolved: spot peer {self.spot_peer_sum:+.1f}, "
                f"peer {self.peer_sum:+.1f}, baseline {self.baseline_sum:+.1f}"
            )
        if self.rank is not None:
            board = f"leaderboard rank {self.rank}"
            if self.ranked_entries:
                board += f" of {self.ranked_entries}"
            if self.leaderboard_score is not None:
                board += f", score {self.leaderboard_score:+.1f}"
            if self.prize:
                board += f", projected prize {self.prize:.0f} USD"
            parts.append(board)
        elif self.project_id is not None:
            parts.append("not on the leaderboard yet")
        parts.append(
            "comments missing on "
            + (f"{len(self.missing_comments)} post(s) {self.missing_comments[:10]}" if self.missing_comments else "0 posts")
        )
        parts.append(
            "questions forecast twice or more: "
            + (f"{len(self.duplicates)} {self.duplicates[:10]}" if self.duplicates else "0")
        )
        parts.append(
            "binary at exactly 50%: " + (f"{len(self.at_fifty)} {self.at_fifty[:10]}" if self.at_fifty else "0")
        )
        parts.append(
            "malformed forecasts: " + (f"{len(self.malformed)} {self.malformed[:10]}" if self.malformed else "0")
        )
        parts.append(
            "closed without a forecast from the bot: "
            + (f"{len(self.missed)} {self.missed[:10]}" if self.missed else "0")
        )
        if self.open_unforecast:
            parts.append(f"open and not yet forecast: {self.open_unforecast[:10]}")
        if self.errors:
            parts.append("audit errors: " + "; ".join(self.errors)[:300])
        line = "; ".join(parts)
        if self.details:
            line += "\n  " + "\n  ".join(self.details[:12])
        return line


def _forecast_count(question: dict) -> int:
    mine = question.get("my_forecasts") or {}
    history = mine.get("history") or []
    if history:
        return len(history)
    latest = mine.get("latest") or {}
    return 1 if latest.get("forecast_values") else 0


def _when(epoch: Any) -> str:
    try:
        return datetime.fromtimestamp(float(epoch), timezone.utc).strftime("%m-%d %H:%M")
    except (TypeError, ValueError, OverflowError, OSError):
        return "?"


def _when_iso(value: Any) -> str:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).strftime("%m-%d %H:%M")
    except (TypeError, ValueError):
        return "?"


def _forecast_times(question: dict) -> str:
    history = (question.get("my_forecasts") or {}).get("history") or []
    times = [_when(h.get("start_time")) for h in history if isinstance(h, dict)]
    return ", ".join(times[:8]) + (f" (+{len(times) - 8} more)" if len(times) > 8 else "")


def _latest_values(question: dict) -> list[float] | None:
    latest = (question.get("my_forecasts") or {}).get("latest") or {}
    values = latest.get("forecast_values")
    return list(values) if isinstance(values, list) else None


def _is_resolved(question: dict) -> bool:
    return question.get("status") == "resolved" or question.get("resolution") not in (None, "")


def check_question(question: dict) -> list[str]:
    """What is wrong with the bot's own latest forecast on this question."""
    values = _latest_values(question)
    if values is None:
        return []
    qtype = question.get("type")
    problems: list[str] = []
    if qtype == "binary":
        if len(values) == 2 and abs(values[1] - 0.5) < 1e-9:
            problems.append("fifty")
    elif qtype == "multiple_choice":
        if abs(sum(v for v in values if v is not None) - 1.0) > 1e-5 or any(
            v is not None and not 0.0 < v < 1.0 for v in values
        ):
            problems.append("malformed")
    elif qtype in ("numeric", "discrete", "date"):
        count = int(question.get("inbound_outcome_count") or DEFAULT_INBOUND_OUTCOME_COUNT)
        errors = validate_cdf(
            values,
            count,
            bool(question.get("open_lower_bound")),
            bool(question.get("open_upper_bound")),
        )
        if errors:
            problems.append("malformed")
    return problems


def _default_project_id(post: dict) -> int | None:
    projects = post.get("projects") or {}
    default = projects.get("default_project") or {}
    pid = default.get("id") if isinstance(default, dict) else None
    return int(pid) if isinstance(pid, int) else None


def _leaderboard_standing(client: MetaculusClient, project_id: int, audit: TournamentAudit) -> None:
    boards = client.project_leaderboard(project_id)
    if not boards:
        return
    board = boards[0]
    entries = board.get("entries") or []
    ranked = [e for e in entries if e.get("rank") is not None and not e.get("excluded")]
    audit.ranked_entries = len(ranked) or None
    mine = board.get("userEntry")
    if isinstance(mine, dict):
        audit.rank = mine.get("rank")
        score = mine.get("score")
        audit.leaderboard_score = float(score) if isinstance(score, (int, float)) else None
        prize = mine.get("prize")
        audit.prize = float(prize) if isinstance(prize, (int, float)) and prize else None


def audit_tournament(client: MetaculusClient, slug: str, me_id: int) -> TournamentAudit:
    audit = TournamentAudit(slug=slug)
    try:
        forecast_posts = list(
            client.iter_posts(
                slug,
                statuses=ALL_STATUSES,
                extra=[("forecaster_id", me_id)],
                include_descriptions=False,
            )
        )
        commented = {
            p.get("id")
            for p in client.iter_posts(
                slug,
                statuses=ALL_STATUSES,
                extra=[("commented_by", me_id)],
                include_descriptions=False,
            )
        }
        open_posts = list(client.iter_posts(slug, statuses="open", include_descriptions=False))
    except MetaculusError as exc:
        audit.errors.append(str(exc)[:200])
        return audit
    try:
        unforecast_closed = list(
            client.iter_posts(
                slug,
                statuses=["closed", "resolved"],
                extra=[("not_forecaster_id", me_id)],
                include_descriptions=False,
            )
        )
    except MetaculusError as exc:
        audit.errors.append(f"missed questions: {str(exc)[:160]}")
        unforecast_closed = []
    for post in unforecast_closed:
        for question in sub_questions(post):
            if str(question.get("resolution")) in UNSCORED:
                continue
            audit.missed.append(question.get("id"))
            if len(audit.details) < 30:
                audit.details.append(
                    f"missed q{question.get('id')}: open {_when_iso(question.get('open_time'))} to "
                    f"{_when_iso(question.get('actual_close_time') or question.get('scheduled_close_time'))} UTC: "
                    f"{(question.get('title') or post.get('title') or '')[:60]}"
                )

    projects: Counter = Counter()
    for post in forecast_posts:
        pid = _default_project_id(post)
        if pid is not None:
            projects[pid] += 1
        forecast_here = False
        for question in sub_questions(post):
            n = _forecast_count(question)
            if n == 0:
                continue
            forecast_here = True
            audit.questions_forecast += 1
            qid = question.get("id")
            title = (question.get("title") or post.get("title") or "")[:70]
            if n > 1:
                audit.duplicates.append(qid)
                audit.details.append(
                    f"q{qid} forecast {n} times, UTC {_forecast_times(question)} ({question.get('type')}, "
                    f"post {post.get('id')}{', in a group' if post.get('group_of_questions') else ''}): {title}"
                )
            problems = check_question(question)
            if "fifty" in problems:
                audit.at_fifty.append(qid)
                audit.details.append(f"q{qid} at exactly 50%, UTC {_forecast_times(question)}: {title}")
            if "malformed" in problems:
                audit.malformed.append(qid)
                audit.details.append(f"q{qid} malformed ({question.get('type')}): {title}")
            if question.get("status") == "open":
                audit.questions_open += 1
            if _is_resolved(question):
                if str(question.get("resolution")) in UNSCORED:
                    audit.questions_annulled += 1
                    continue
                audit.questions_resolved += 1
                scores = (question.get("my_forecasts") or {}).get("score_data") or {}
                if any(k in scores for k in ("spot_peer_score", "peer_score", "baseline_score")):
                    audit.scored += 1
                    audit.spot_peer_sum += float(scores.get("spot_peer_score") or 0.0)
                    audit.peer_sum += float(scores.get("peer_score") or 0.0)
                    audit.baseline_sum += float(scores.get("baseline_score") or 0.0)
        if forecast_here:
            audit.posts_forecast += 1
            if post.get("id") not in commented:
                audit.missing_comments.append(post.get("id"))

    for post in open_posts:
        if not projects:
            pid = _default_project_id(post)
            if pid is not None:
                projects[pid] += 1
        for question in sub_questions(post):
            if question.get("status") not in (None, "open"):
                continue
            if "my_forecasts" in question and _forecast_count(question) == 0:
                audit.open_unforecast.append(question.get("id"))

    if projects:
        audit.project_id = projects.most_common(1)[0][0]
        try:
            _leaderboard_standing(client, audit.project_id, audit)
        except MetaculusError as exc:
            audit.errors.append(f"leaderboard: {str(exc)[:160]}")
    return audit


_MODELS_LINE = re.compile(r"^Models:\s*(.+)$", re.M)
_FORECAST_LINE = re.compile(r"^Forecast:\s*(.+)$", re.M)


def latest_comments(client: MetaculusClient, me_id: int, limit: int = 6) -> list[str]:
    """One line per recent comment: when, where, which models, what forecast.

    The comment is written from the same run that submitted the forecast, so
    this is the quickest outside check that the live pipeline works end to
    end: which models answered, and that the comment went up.
    """
    lines = []
    for c in client.my_recent_comments(me_id, limit=limit):
        text = c.get("text") or ""
        models = _MODELS_LINE.search(text)
        forecast = _FORECAST_LINE.search(text)
        when = _when_iso(c.get("created_at"))
        post = (c.get("on_post_data") or {}).get("id") or c.get("on_post")
        lines.append(
            f"{when} post {post}: {forecast.group(1)[:60] if forecast else 'no forecast line'}; "
            f"models {models.group(1)[:140] if models else 'not named'}; {len(text)} chars"
        )
    return lines


def run_audit(client: MetaculusClient, tournaments: list[str]) -> tuple[dict[str, Any], list[TournamentAudit]]:
    """Audit every tournament. Returns (who the bot is, one audit per tournament)."""
    try:
        me = client.me()
    except MetaculusError as exc:
        log.error("audit: could not read the bot's own account: %s", str(exc)[:200])
        return {}, []
    me_id = me.get("id")
    if not isinstance(me_id, int):
        log.error("audit: the account answer carried no id")
        return me, []
    audits = []
    for slug in tournaments:
        try:
            audits.append(audit_tournament(client, slug, me_id))
        except Exception as exc:  # noqa: BLE001 - an audit must never stop a run
            broken = TournamentAudit(slug=slug)
            broken.errors.append(f"{type(exc).__name__}: {str(exc)[:200]}")
            audits.append(broken)
    return me, audits
