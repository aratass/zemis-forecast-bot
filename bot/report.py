"""What a run did, readable without the run's logs.

GitHub shows the logs of a public repository's workflow runs only to a
signed-in user, and serves the log archive from a storage host that the bot
maker's other tools cannot reach. Annotations are different: the checks API
returns them to anyone holding a read token, and they sit on the run page. So
the facts that decide whether the bot is healthy are written there as well as
to the log: which models answered, what each forecast was, what the self-audit
found, and anything that went wrong. The same text goes to the job summary.

Nothing secret is ever passed in here. Every caller builds its text from
question ids, model names, counts and forecast values.
"""

from __future__ import annotations

import os
import threading

# GitHub keeps at most ten annotations of each level per step, and drops the
# rest silently. Callers that must be seen (the end-of-run tally) reserve a
# slot so that per-question notes cannot crowd them out.
PER_LEVEL_LIMIT = 10
MAX_MESSAGE_CHARS = 3800

_LEVELS = ("notice", "warning", "error")
_counts = {level: 0 for level in _LEVELS}
_lock = threading.Lock()


def in_actions() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true"


def _escape_data(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_property(text: str) -> str:
    return _escape_data(text).replace(":", "%3A").replace(",", "%2C")


def annotate(level: str, title: str, message: str, reserve: int = 0) -> bool:
    """Write one annotation. False when not on Actions or the budget is spent.

    ``reserve`` keeps that many slots of the level free for later, more
    important annotations.
    """
    if level not in _counts or not in_actions():
        return False
    body = (message or "").strip()
    if len(body) > MAX_MESSAGE_CHARS:
        body = body[: MAX_MESSAGE_CHARS - 20].rstrip() + "\n[trimmed]"
    with _lock:
        if _counts[level] >= PER_LEVEL_LIMIT - reserve:
            return False
        _counts[level] += 1
        print(f"::{level} title={_escape_property(title)}::{_escape_data(body)}", flush=True)
    return True


def remaining(level: str) -> int:
    with _lock:
        return PER_LEVEL_LIMIT - _counts.get(level, PER_LEVEL_LIMIT)


def summary(markdown: str) -> None:
    """Append to the job summary shown on the run page. Never raises."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with _lock, open(path, "a", encoding="utf-8") as fh:
            fh.write(markdown.rstrip() + "\n\n")
    except OSError:
        pass


def reset() -> None:
    """For tests: forget how many annotations have been written."""
    with _lock:
        for level in _LEVELS:
            _counts[level] = 0
