"""Seeing what the bot did without its logs, and refusing forecasts that cannot count.

GitHub shows a public repository's run logs only to signed-in users, so every
fact that decides whether the bot is healthy also goes into annotations, which
the API returns to any reader, and into the job summary. These tests pin what
those say and that they can never crowd out the end-of-run tally.
"""

from datetime import datetime, timedelta, timezone

import pytest

from bot import audit, llm, monitor, report, runner
from bot import forecast as fc
from bot.cdf import safe_cdf
from bot.forecast import EnsembleTooThin
from tests.test_pipeline import POST, REPLIES, q_binary, q_mc, q_numeric


# -- annotations ----------------------------------------------------------------
@pytest.fixture
def actions(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    report.reset()
    return summary


def _commands(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith("::")]


def test_annotations_escape_newlines_and_property_separators(actions, capsys):
    assert report.annotate("notice", "q1: a, b", "line one\nline two 50%")
    (line,) = _commands(capsys.readouterr().out)
    assert line == "::notice title=q1%3A a%2C b::line one%0Aline two 50%25"


def test_nothing_is_annotated_off_actions(monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    assert not report.annotate("notice", "t", "m")
    assert _commands(capsys.readouterr().out) == []


def test_a_reserved_slot_survives_a_flood(actions, capsys):
    sent = sum(report.annotate("notice", f"q{i}", "x", reserve=1) for i in range(30))
    assert sent == report.PER_LEVEL_LIMIT - 1
    assert report.annotate("notice", "run tally", "the one that must be seen")
    assert not report.annotate("notice", "late", "x")


def test_the_summary_is_appended(actions):
    report.summary("### A")
    report.summary("### B")
    assert actions.read_text() == "### A\n\n### B\n\n"


# -- checks on a payload --------------------------------------------------------
def test_a_binary_at_exactly_fifty_is_flagged_but_not_held_back():
    problems = monitor.check_payload({"probability_yes": 0.5}, q_binary())
    assert problems == ["binary forecast at exactly 50%"]
    assert not monitor.is_rejectable(problems)


def test_what_the_server_rejects_is_held_back():
    assert monitor.is_rejectable(monitor.check_payload({"probability_yes": 0.9995}, q_binary()))
    mc = q_mc()
    bad = {o: 0.5 for o in mc["options"]}
    assert monitor.is_rejectable(monitor.check_payload({"probability_yes_per_category": bad}, mc))
    q = q_numeric()
    assert monitor.is_rejectable(monitor.check_payload({"continuous_cdf": [0.0] * 201}, q))


def test_rounding_to_six_places_is_within_the_servers_tolerance():
    mc = q_mc()
    k = len(mc["options"])
    cats = {o: round(1.0 / k, 6) for o in mc["options"]}
    assert monitor.check_payload({"probability_yes_per_category": cats}, mc) == []


def test_a_legal_distribution_passes():
    q = q_numeric()
    cdf = safe_cdf(200, q["open_lower_bound"], q["open_upper_bound"])
    assert monitor.check_payload({"continuous_cdf": cdf}, q) == []


def test_a_comment_without_reasoning_is_flagged_but_never_holds_a_forecast_back():
    problems = monitor.check_comment("x" * 300 + "Forecast: 20%\n(not captured)")
    assert "comment carries no reasoning" in problems
    assert not monitor.is_rejectable(problems)
    assert monitor.check_comment("Reasoning. " * 40 + "Forecast: 20%") == []


def test_the_tally_counts_duplicates():
    tally = monitor.RunTally()
    rec = monitor.ForecastRecord(1, "binary", "20%", ["m"], "Wikipedia 2", [], 900)
    tally.forecast(rec)
    tally.forecast(rec)
    assert tally.duplicates == [1]
    assert any("forecast twice" in line and "[1]" in line for line in tally.lines())


# -- the runner refuses what cannot count, and reports what it did --------------
class Client:
    def __init__(self, posts, dry_run=False):
        self.posts, self.dry_run = posts, dry_run
        self.submitted, self.comments = [], []

    def iter_posts(self, tournament, **kwargs):
        return list(self.posts.get(tournament, []))

    def submit_forecasts(self, payloads):
        if not self.dry_run:
            self.submitted.extend(payloads)

    def post_comment(self, post_id, text):
        if not self.dry_run:
            self.comments.append((post_id, text))


def _open_post(pid, question):
    question = dict(question)
    question["scheduled_close_time"] = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    question["my_forecasts"] = {"history": [], "latest": None}
    return {"id": pid, "question": question}


@pytest.fixture
def quiet_research(monkeypatch):
    runner._RESEARCH.clear()
    runner.TALLY = monitor.RunTally()
    monkeypatch.setattr(runner, "search_queries", lambda ctx, models: ["q"])
    monkeypatch.setattr(runner.research_mod, "gather", lambda *a, **k: runner.research_mod.ResearchReport())
    yield
    runner._RESEARCH.clear()
    runner.TALLY = monitor.RunTally()


def test_a_forecast_the_server_would_reject_is_not_sent(monkeypatch, quiet_research):
    client = Client({"t": [_open_post(1, q_binary())]})

    def broken(**kwargs):
        forecast = fc.Forecast(101, 1, "binary", {"probability_yes": 1.0}, "Forecast: 100%" + "x" * 300, "100%")
        return forecast

    monkeypatch.setattr(runner, "forecast_question", broken)
    done = runner.run_tick(client, ["t"], ["a/one"], runs=1, limit=5)
    assert done == 0 and client.submitted == [] and client.comments == []
    assert runner.TALLY.held_back == [101]


def test_each_forecast_leaves_an_annotation(monkeypatch, quiet_research, actions, capsys):
    client = Client({"t": [_open_post(1, q_binary())]}, dry_run=True)
    monkeypatch.setattr(fc, "chat_with_fallback", lambda m, models, **k: ("Base rates. " * 30 + "PROBABILITY: 20%", models[0]))
    runner.run_tick(client, ["t"], ["gemini/models/gemini-3.6-flash"], runs=2, limit=5)
    (line,) = [c for c in _commands(capsys.readouterr().out) if "q101" in c]
    assert line.startswith("::notice title=dry run q101 binary::")
    assert "would submit%3A" not in line and "would submit: probability_yes=" in line
    assert "checks: all passed" in line
    assert "gemini-3.6-flash" in line


def test_a_dry_run_in_the_test_area_samples_every_type():
    posts = [({"id": i}, {"id": i, "type": t}) for i, t in enumerate(
        ["binary", "binary", "binary", "binary", "numeric", "numeric", "multiple_choice", "date", "discrete"]
    )]
    picked = runner.one_of_each_type(posts, 5)
    assert sorted({q["type"] for _, q in picked}) == ["binary", "date", "discrete", "multiple_choice", "numeric"]


# -- waiting and empty answers --------------------------------------------------
STAND_IN = "groq/openai/gpt-oss-120b"
FLASH = ["gemini/models/gemini-3.8-flash", "gemini/models/gemini-3.7-flash"]


def _soon(factory, minutes):
    q = factory()
    q["scheduled_close_time"] = (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()
    return q


@pytest.mark.parametrize("factory", [q_numeric, q_mc], ids=["numeric", "mc"])
def test_a_numeric_or_mc_question_with_no_strong_answer_waits_rather_than_erroring(factory, monkeypatch):
    llm.FALLBACK_ONLY.add(STAND_IN)

    def overloaded(messages, models, **kwargs):
        raise llm.LLMError("HTTP 503 overloaded")

    monkeypatch.setattr(fc, "chat_with_fallback", overloaded)
    with pytest.raises(EnsembleTooThin):
        fc.forecast_question(post=POST, question=_soon(factory, 80), research_text="", research_sources=[],
                             models=FLASH + [STAND_IN], runs=2)


def test_kept_answers_are_dropped_once_the_forecast_is_out(monkeypatch):
    fc._STRONG_ANSWERS[(101, "binary")] = [("PROBABILITY: 20%", FLASH[0])]
    fc._STRONG_ANSWERS[(102, "numeric")] = [("x", FLASH[0])]
    fc.forget(101)
    assert list(fc._STRONG_ANSWERS) == [(102, "numeric")]


class Resp:
    def __init__(self, content):
        self.status_code, self.text, self.headers, self._content = 200, "", {}, content

    ok = True

    def json(self):
        return {"choices": [{"message": {"content": self._content}, "finish_reason": "stop"}], "usage": {}}


def test_an_empty_answer_falls_through_to_the_next_model(monkeypatch):
    queue = [Resp(""), Resp("PROBABILITY: 30%")]
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: queue.pop(0))
    monkeypatch.setattr(llm.LIMITER, "wait", lambda key: None)
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    text, used = llm.chat_with_fallback([{"role": "user", "content": "x"}], FLASH, attempts=1)
    assert used == FLASH[1] and text == "PROBABILITY: 30%"


# -- the self-audit -------------------------------------------------------------
def _q(qid, qtype="binary", n=1, values=None, status="open", resolution=None, scores=None, **extra):
    history = [{"question_id": qid}] * n
    latest = {"forecast_values": values} if n else None
    q = {"id": qid, "type": qtype, "status": status, "resolution": resolution,
         "my_forecasts": {"history": history, "latest": latest, "score_data": scores or {}}}
    q.update(extra)
    return q


class AuditClient:
    def __init__(self, forecast_posts, commented_ids, open_posts, board=None, missed_posts=()):
        self.forecast_posts, self.commented_ids, self.open_posts, self.board = (
            forecast_posts, commented_ids, open_posts, board)
        self.missed_posts = list(missed_posts)
        self.calls = []

    def me(self):
        return {"id": 7, "username": "zemis-bot"}

    def iter_posts(self, slug, statuses="open", extra=None, include_descriptions=True, **kwargs):
        self.calls.append((slug, statuses, tuple(extra or ())))
        extra = dict(extra or [])
        if "forecaster_id" in extra:
            return list(self.forecast_posts)
        if "commented_by" in extra:
            return [p for p in self.forecast_posts if p["id"] in self.commented_ids]
        if "not_forecaster_id" in extra:
            return list(self.missed_posts)
        return list(self.open_posts)

    def project_leaderboard(self, project_id):
        return self.board or []


def test_the_audit_counts_and_flags_what_the_rules_care_about():
    good_cdf = safe_cdf(200, False, True)
    posts = [
        {"id": 1, "projects": {"default_project": {"id": 33121}},
         "question": _q(11, values=[0.8, 0.2], status="resolved", resolution="no",
                        scores={"spot_peer_score": 12.5, "peer_score": 10.0, "baseline_score": 40.0})},
        {"id": 2, "projects": {"default_project": {"id": 33121}},
         "question": _q(12, values=[0.5, 0.5], n=2)},
        {"id": 3, "projects": {"default_project": {"id": 33121}},
         "question": _q(13, "numeric", values=good_cdf, open_lower_bound=False, open_upper_bound=True,
                        inbound_outcome_count=200)},
        {"id": 4, "projects": {"default_project": {"id": 33121}},
         "question": _q(14, values=[0.3, 0.7], status="resolved", resolution="annulled")},
    ]
    open_posts = [{"id": 9, "question": _q(19, n=0)}]
    board = [{"entries": [{"rank": 1}, {"rank": 2}, {"rank": 3}],
              "userEntry": {"rank": 2, "score": 12.5, "prize": 0}}]
    missed = [{"id": 8, "question": {"id": 18, "type": "binary", "status": "closed", "title": "missed one",
                                     "open_time": "2026-09-29T10:00:00Z", "scheduled_close_time": "2026-09-29T11:30:00Z"}},
              {"id": 6, "question": {"id": 16, "type": "binary", "status": "resolved", "resolution": "annulled"}}]
    client = AuditClient(posts, commented_ids={1, 3, 4}, open_posts=open_posts, board=board, missed_posts=missed)
    me, (result,) = audit.run_audit(client, ["fall-futureeval-2026"])
    assert me["username"] == "zemis-bot"
    assert result.questions_forecast == 4
    assert result.questions_resolved == 1 and result.questions_annulled == 1
    assert result.scored == 1 and result.spot_peer_sum == pytest.approx(12.5)
    assert result.missing_comments == [2]
    assert result.duplicates == [12]
    assert result.at_fifty == [12]
    assert result.malformed == []
    assert result.open_unforecast == [19]
    assert (result.rank, result.ranked_entries) == (2, 3)
    assert result.missed == [18], "an annulled question is not a miss"
    assert any("missed q18: open 09-29 10:00 to 09-29 11:30 UTC" in d for d in result.details)
    assert not result.clean
    line = result.line()
    assert "rank 2 of 3" in line and "comments missing on 1 post(s) [2]" in line
    forecaster_calls = [c for c in client.calls if ("forecaster_id", 7) in c[2]]
    assert forecaster_calls and forecaster_calls[0][1] == ["open", "closed", "resolved"]


def test_a_malformed_distribution_is_caught():
    q = _q(13, "numeric", values=[0.0] * 201, open_lower_bound=False, open_upper_bound=False,
           inbound_outcome_count=200)
    assert audit.check_question(q) == ["malformed"]


def test_the_audit_never_raises():
    class Broken(AuditClient):
        def iter_posts(self, *a, **k):
            raise RuntimeError("boom")

    me, (result,) = audit.run_audit(Broken([], set(), []), ["minibench"])
    assert result.errors and "boom" in result.errors[0]


def test_the_catalogue_report_survives_dead_catalogues(monkeypatch):
    for name in ("GEMINI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.setenv(name, "x")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def refuse(*a, **k):
        raise llm.requests.ConnectionError("refused")

    monkeypatch.setattr(llm.requests, "get", refuse)
    lines = llm.catalogue_report()
    assert lines == ["gemini flash ids: none listed", "gemma ids: none listed", "groq ids: none listed"]


def test_a_bare_answer_brings_its_separate_reasoning_into_the_comment(monkeypatch):
    llm.FALLBACK_ONLY.add(STAND_IN)
    bare = llm.Answer(REPLIES["binary"], "Base rate: most such bills stall in committee.")

    monkeypatch.setattr(fc, "chat_with_fallback", lambda m, models, **k: (bare, STAND_IN))
    forecast = fc.forecast_question(post=POST, question=_soon(q_binary, 10), research_text="",
                                    research_sources=[], models=FLASH + [STAND_IN], runs=2)
    assert "most such bills stall in committee" in forecast.comment


def test_failures_are_counted_by_model_and_reason(monkeypatch):
    class R:
        status_code = 503
        text = '{"error": {"code": 503, "status": "UNAVAILABLE", "message": "high demand"}}'
        headers = {}
        ok = False

    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: R())
    monkeypatch.setattr(llm.LIMITER, "wait", lambda key: None)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    llm.FAILURES.clear()
    with pytest.raises(llm.LLMError):
        llm.chat([{"role": "user", "content": "x"}], FLASH[0], attempts=2)
    assert llm.failure_summary() == "gemini-3.8-flash HTTP 503 x2"


def test_a_google_quota_error_names_the_limit():
    class R:
        text = ('[{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota", "details": '
                '[{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier", '
                '"quotaValue": "20"}]}]}}]')

    brief = llm._error_brief(R())
    assert "PerDay" in brief and "limit 20" in brief


def test_a_dry_run_keeps_polling_while_questions_wait_and_never_repeats_one(monkeypatch, quiet_research):
    from bot import config

    posts = {"bot-testing-area": [_open_post(1, q_binary()), _open_post(2, dict(q_mc(), id=105))]}

    class FakeMetaculus(Client):
        def __init__(self, dry_run=False):
            super().__init__(posts, dry_run=dry_run)

    calls = []

    def forecast_question(post, question, **kwargs):
        calls.append(question["id"])
        if question["id"] == 105 and calls.count(105) == 1:
            raise EnsembleTooThin("1 of 2 strong answers so far")
        if question["type"] == "binary":
            payload = {"question": question["id"], "probability_yes": 0.3}
        else:
            k = len(question["options"])
            payload = {"question": question["id"],
                       "probability_yes_per_category": {o: 1.0 / k for o in question["options"]}}
        return fc.Forecast(question["id"], post["id"], question["type"], payload,
                           "Reasoning. " * 30 + "Forecast: x", "x", models_used=["gemini/x"])

    monkeypatch.setenv("BOT_AUDIT", "off")
    monkeypatch.setattr(runner, "MetaculusClient", FakeMetaculus)
    monkeypatch.setattr(runner, "resolve_models", lambda n: ["gemini/x"])
    monkeypatch.setattr(runner, "probe", lambda models: {})
    monkeypatch.setattr(runner, "catalogue_report", lambda: [])
    monkeypatch.setattr(runner, "forecast_question", forecast_question)
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)
    monkeypatch.setattr(config, "DRY_RUN_PATIENCE_SECONDS", 600)
    runner.main(["--mode", "test", "--dry-run"])
    assert calls == [101, 105, 105] or sorted(calls) == [101, 105, 105], calls
    assert runner.TALLY.duplicates == []
    assert not runner.TALLY.waiting


def test_latest_comments_name_the_models_and_the_forecast():
    class C:
        def my_recent_comments(self, author_id, limit=6):
            assert author_id == 7
            return [{"created_at": "2026-09-30T21:10:00Z", "on_post": 45999,
                     "text": "Forecast produced automatically.\n\nModels: gemini/models/gemini-3.6-flash\n"
                             "Research sources: AskNews\n\nForecast: 23.0%\n\nReasoning from one run:\nx"}]

    (line,) = audit.latest_comments(C(), 7)
    assert line.startswith("09-30 21:10 post 45999: 23.0%; models gemini/models/gemini-3.6-flash;")


def test_the_last_dry_run_poll_takes_waiting_questions_through_the_deadline(monkeypatch, quiet_research):
    from bot import config

    posts = {"bot-testing-area": [_open_post(1, q_binary())]}

    class FakeMetaculus(Client):
        def __init__(self, dry_run=False):
            super().__init__(posts, dry_run=dry_run)

    seen_force = []

    def forecast_question(post, question, **kwargs):
        seen_force.append(fc.FORCE_DEADLINE)
        if not fc.FORCE_DEADLINE:
            raise EnsembleTooThin("0 of 2 strong answers so far")
        return fc.Forecast(question["id"], post["id"], "binary", {"question": question["id"], "probability_yes": 0.3},
                           "Reasoning. " * 30 + "Forecast: 30%", "30%", models_used=["groq/x"])

    clock = {"t": 1000.0}
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(runner.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    monkeypatch.setenv("BOT_AUDIT", "off")
    monkeypatch.setattr(runner, "MetaculusClient", FakeMetaculus)
    monkeypatch.setattr(runner, "resolve_models", lambda n: ["gemini/x"])
    monkeypatch.setattr(runner, "probe", lambda models: {})
    monkeypatch.setattr(runner, "catalogue_report", lambda: [])
    monkeypatch.setattr(runner, "forecast_question", forecast_question)
    monkeypatch.setattr(config, "DRY_RUN_PATIENCE_SECONDS", 400)
    monkeypatch.setattr(config, "DRY_RUN_INTERVAL_SECONDS", 150)
    runner.main(["--mode", "test", "--dry-run"])
    assert seen_force[-1] is True and seen_force[0] is False
    assert [r.question_id for r in runner.TALLY.forecasts] == [101]
    assert not runner.TALLY.waiting


def test_a_gemini_daily_refusal_is_looked_at_again_within_half_an_hour(monkeypatch):
    class R:
        status_code = 429
        text = ('[{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota", "details": '
                '[{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier", "quotaValue": "20"}]}]}}]')
        headers = {}
        ok = False

    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: R())
    monkeypatch.setattr(llm.LIMITER, "wait", lambda key: None)
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    with pytest.raises(llm.ModelUnavailable):
        llm.chat([{"role": "user", "content": "x"}], FLASH[0])
    wait = llm.EXHAUSTED_UNTIL[FLASH[0]] - llm.time.time()
    assert 0 < wait <= llm.GEMINI_DAILY_BACKOFF_SECONDS + 1


@pytest.mark.parametrize("factory", [q_binary, q_numeric, q_mc], ids=["binary", "numeric", "mc"])
def test_every_prompt_asks_for_the_reasoning_in_the_open(factory):
    from bot import prompts

    ctx = fc.build_context(POST, factory())
    builder = {"binary": prompts.binary_prompt, "numeric": prompts.numeric_prompt,
               "multiple_choice": prompts.multiple_choice_prompt}[ctx["type"]]
    user = builder(ctx, "evidence")[-1]["content"]
    assert "published with the forecast as its explanation" in user
    assert user.index("published with the forecast") < user.rindex("Finish with exactly")


def test_audit_only_writes_the_audit_and_the_ensemble_and_forecasts_nothing(monkeypatch, actions, capsys):
    audited = []

    class FakeMetaculus(Client):
        def __init__(self, dry_run=False):
            super().__init__({}, dry_run=dry_run)

    monkeypatch.setattr(runner, "MetaculusClient", FakeMetaculus)
    monkeypatch.setattr(runner, "_self_audit", lambda client: audited.append(True))
    monkeypatch.setattr(runner, "resolve_models", lambda n: (llm.FALLBACK_ONLY.add(STAND_IN) or FLASH + [STAND_IN]))
    monkeypatch.setattr(runner, "run_tick", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no forecasting")))
    assert runner.main(["--audit-only"]) == 0
    assert audited == [True]
    (line,) = [c for c in _commands(capsys.readouterr().out) if "ensemble at start" in c]
    assert "gemini-3.8-flash" in line and "gpt-oss-120b" in line


def test_a_groq_request_fits_under_its_tokens_per_minute_limit(monkeypatch):
    sent = []

    class R:
        status_code, text, headers, ok = 200, "", {}, True

        def json(self):
            return {"choices": [{"message": {"content": "PROBABILITY: 20%"}, "finish_reason": "stop"}], "usage": {}}

    monkeypatch.setattr(llm.requests, "post", lambda url, headers=None, json=None, timeout=None: sent.append(json) or R())
    monkeypatch.setattr(llm.LIMITER, "wait", lambda key: None)
    monkeypatch.setenv("GROQ_API_KEY", "q")
    long_prompt = [{"role": "user", "content": "evidence " * 2500}]  # about 6,400 tokens
    llm.chat(long_prompt, STAND_IN)
    prompt_tokens = len(long_prompt[0]["content"]) / 3.5
    assert sent[0]["max_tokens"] + prompt_tokens <= 8000
    assert sent[0]["max_tokens"] >= llm.MIN_ANSWER_ROOM
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    llm.chat(long_prompt, FLASH[0])
    assert sent[1]["max_tokens"] == llm.REASONING_MAX_TOKENS, "only Groq is squeezed"
