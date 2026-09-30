"""The watcher loop around the free-tier ensemble.

A question that is waiting for more strong answers must not be submitted, must
not be logged as a failure, and must not pay for its research twice. And a dry
run may only re-forecast questions in the bot testing area: the rules forbid
previewing forecasts on open tournament questions.
"""

from datetime import datetime, timedelta, timezone

import pytest

from bot import forecast as fc
from bot import runner
from bot.forecast import EnsembleTooThin


class Client:
    def __init__(self, posts, dry_run=False):
        self.posts = posts
        self.dry_run = dry_run
        self.submitted = []
        self.comments = []

    def iter_posts(self, tournament):
        return list(self.posts.get(tournament, []))

    def get_post(self, post_id):
        raise AssertionError("not needed")

    def submit_forecasts(self, payloads):
        if not self.dry_run:
            self.submitted.extend(payloads)

    def post_comment(self, post_id, text):
        if not self.dry_run:
            self.comments.append((post_id, text))


def _post(pid, qid, forecast_before=False):
    close = (datetime.now(timezone.utc) + timedelta(minutes=80)).isoformat()
    history = [{"question_id": qid}] if forecast_before else []
    return {
        "id": pid,
        "question": {
            "id": qid,
            "type": "binary",
            "title": f"question {qid}",
            "scheduled_close_time": close,
            "scheduled_resolve_time": close,
            "my_forecasts": {"history": history, "latest": None},
        },
    }


@pytest.fixture(autouse=True)
def no_research(monkeypatch):
    calls = []

    def research(ctx, models):
        calls.append(ctx["question_id"])
        return "evidence", ["Fake"]

    runner._RESEARCH.clear()
    monkeypatch.setattr(runner, "search_queries", lambda ctx, models: ["q"])
    monkeypatch.setattr(runner.research_mod, "gather", lambda *a, **k: runner.research_mod.ResearchReport())
    yield calls
    runner._RESEARCH.clear()


def test_a_waiting_question_is_not_submitted_and_not_an_error(monkeypatch, caplog):
    client = Client({"t": [_post(1, 11)]})

    def thin(**kwargs):
        raise EnsembleTooThin("1 of 2 strong answers so far")

    monkeypatch.setattr(runner, "forecast_question", thin)
    with caplog.at_level("INFO"):
        done = runner.run_tick(client, ["t"], ["a/one"], runs=2, limit=5)
    assert done == 0
    assert client.submitted == [] and client.comments == []
    assert "waiting" in caplog.text
    assert "ERROR" not in caplog.text


def test_research_is_done_once_while_a_question_waits(monkeypatch):
    client = Client({"t": [_post(1, 11)]})
    gathered = []

    def gather(queries, include_markets=True, ctx=None):
        gathered.append(ctx["question_id"])
        return runner.research_mod.ResearchReport()

    monkeypatch.setattr(runner.research_mod, "gather", gather)
    attempts = {"n": 0}
    real = fc.forecast_question

    def sometimes_thin(**kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise EnsembleTooThin("waiting")
        monkeypatch.setattr(fc, "chat_with_fallback", lambda m, models, **k: ("PROBABILITY: 20%", models[0]))
        return real(**kwargs)

    monkeypatch.setattr(runner, "forecast_question", sometimes_thin)
    runner.run_tick(client, ["t"], ["a/one"], runs=1, limit=5)
    runner.run_tick(client, ["t"], ["a/one"], runs=1, limit=5)
    assert gathered == [11], "the second poll must reuse the first poll's research"
    assert len(client.submitted) == 1


def test_a_test_area_dry_run_includes_questions_already_forecast():
    client = Client({"bot-testing-area": [_post(1, 11, forecast_before=True), _post(2, 12)]}, dry_run=True)
    targets = runner.collect_targets(client, ["bot-testing-area"], include_forecast=True)
    assert [q["id"] for _, q in targets] == [11, 12]


def test_a_normal_poll_still_skips_them():
    client = Client({"minibench": [_post(1, 11, forecast_before=True), _post(2, 12)]})
    targets = runner.collect_targets(client, ["minibench"])
    assert [q["id"] for _, q in targets] == [12]


def test_only_a_dry_run_in_test_mode_re_forecasts(monkeypatch):
    seen = {}

    class FakeMetaculus(Client):
        def __init__(self, dry_run=False):
            super().__init__({}, dry_run=dry_run)

    def fake_tick(client, tournaments, models, runs, limit, include_forecast=False):
        seen[tuple(tournaments)] = (include_forecast, limit)
        return 0

    monkeypatch.setattr(runner, "MetaculusClient", FakeMetaculus)
    monkeypatch.setattr(runner, "resolve_models", lambda n: ["a/one"])
    monkeypatch.setattr(runner, "run_tick", fake_tick)
    probed = []
    monkeypatch.setattr(runner, "probe", lambda models: probed.append(list(models)) or {})

    runner.main(["--mode", "test", "--dry-run"])
    runner.main(["--mode", "tournament", "--dry-run"])
    runner.main(["--mode", "test"])

    from bot import config

    assert seen[(config.TEST_SLUG,)] == (False, config.MAX_QUESTIONS_PER_TICK), "last call: live test mode"
    runner.main(["--mode", "test", "--dry-run"])
    assert seen[(config.TEST_SLUG,)] == (True, config.DRY_RUN_LIMIT)
    assert seen[(config.SEASONAL_SLUG, config.MINIBENCH_SLUG)][0] is False, (
        "a tournament dry run must never re-forecast open tournament questions"
    )
    assert len(probed) == 3, "every dry run starts by probing the models, live runs do not"
