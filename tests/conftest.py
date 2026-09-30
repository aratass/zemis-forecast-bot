"""Test setup: make the package importable, and keep the tests off the network.

Without the path line, `pytest tests/` puts only the tests directory on the
path and the imports fail, while `python -m pytest` happens to work because it
adds the working directory. Pinning it here means both spellings behave the
same on any machine.

The network guard exists because of a real CI failure. A test asserted that the
model resolver falls back to a hardcoded list, and it passed for weeks, but only
because the machine it was written on could not reach the model catalogue. The
moment it ran somewhere with working network the fetch succeeded, the resolver
returned real models, and the assertion broke. A test whose result depends on
whether the network happens to be reachable is not testing anything. So a real
HTTP call inside a test now fails loudly instead of quietly changing the answer.
"""

import sys
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def no_real_http(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError(
            "a test tried to make a real HTTP request. Stub it: a test that "
            "depends on network reachability passes or fails for the wrong reason."
        )

    monkeypatch.setattr(requests, "get", blocked)
    monkeypatch.setattr(requests, "post", blocked)
    monkeypatch.setattr(requests.Session, "request", blocked)


@pytest.fixture(autouse=True)
def fresh_model_state():
    """Module-level model bookkeeping must not leak from one test into the next.

    A test that resolves a tiered ensemble fills FALLBACK_ONLY, and a test that
    forecasts question 101 leaves its answers in the per-question cache; either
    would silently change what the next test sees.
    """
    from bot import forecast, llm

    def clear():
        llm.FALLBACK_ONLY.clear()
        llm.EXHAUSTED_UNTIL.clear()
        llm._COOLDOWN_UNTIL.clear()
        llm.PRIMARY_PROVIDER = None
        llm.FAILURES.clear()
        forecast._STRONG_ANSWERS.clear()

    clear()
    yield
    clear()


@pytest.fixture(autouse=True)
def not_on_actions(monkeypatch):
    """Tests run on GitHub Actions in CI. Nothing a test does may become an
    annotation or a line in that CI run's job summary, so both are switched off
    unless a test turns them on for itself."""
    from bot import report

    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    report.reset()
    yield
    report.reset()
