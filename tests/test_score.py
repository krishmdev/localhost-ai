"""POST /v1/score on the fake model: in-place and teacher-forced sites, request validation, the
pin fields, and that scoring alongside generation traffic gives the same numbers as alone."""

import asyncio
import math

import httpx
import pytest
import torch
from fakes import FakeRunner, PieceTokenizer, score_service
from fastapi.testclient import TestClient

from localhost_ai.api.app import create_app
from localhost_ai.engine.score import Site, score

MSG = [{"role": "user", "content": "cite the source"}]
CONT = "the answer [1] and [2]."
TOK = PieceTokenizer()


def head(messages=MSG):
    return "".join(f"{m['role']}: {m['content']}." for m in messages) + "assistant:"


def reference(text_before: str, cand: str) -> float:
    """log P(cand | text_before) under the fake model, tokenizing cand on its own after the
    tokens of text_before (the teacher-forced definition)."""
    runner = FakeRunner()
    ids = TOK(text_before)["input_ids"]
    tail = TOK(cand)["input_ids"]
    lp = runner.logprob_rows(ids + tail, [len(ids) + i - 1 for i in range(len(tail))])
    return float(sum(lp[i, t] for i, t in enumerate(tail)))


@pytest.fixture
def client():
    with TestClient(create_app(score_service())) as c:
        yield c


def post(client, sites, continuation=CONT, **extra):
    return client.post("/v1/score", json={"model": "fake-model", "messages": MSG,
                                          "continuation": continuation, "sites": sites, **extra})


def test_in_place_site_matches_the_forward_pass(client):
    off = CONT.index("[2") + 1  # the "2": "[" is its own token here, so "2" starts a token
    r = post(client, [{"char_offset": off, "candidates": ["1", "2", "3"]}])
    assert r.status_code == 200, r.text
    site = r.json()["sites"][0]
    assert site["forced"] is False
    full = head() + CONT
    ids = TOK(full)["input_ids"]
    offs = TOK(full, return_offsets_mapping=True)["offset_mapping"]
    t = next(i for i, (s, _) in enumerate(offs) if s == len(head()) + off)
    assert site["token_index"] == t
    lp = FakeRunner().logprob_rows(ids, [t - 1])[0]
    for c in "123":
        assert site["candidates"][c] == pytest.approx(float(lp[TOK(c)["input_ids"][0]]), abs=1e-6)
    z = sum(math.exp(v) for v in site["candidates"].values())
    for c, p in site["renorm"].items():
        assert p == pytest.approx(math.exp(site["candidates"][c]) / z, abs=1e-9)


def test_site_inside_a_token_backs_off_and_teacher_forces(client):
    # with no space before the bracket, "[1" is one token, so the site is inside it
    cont = "see[1]."
    off = cont.index("1")
    r = post(client, [{"char_offset": off, "candidates": ["1", "2"]}], continuation=cont)
    assert r.status_code == 200, r.text
    site = r.json()["sites"][0]
    assert site["forced"] is True
    before = head() + cont[: cont.index("[")]
    for c in ("1", "2"):
        want = reference(before, "[" + c) - reference(before, "[")
        assert site["candidates"][c] == pytest.approx(want, abs=1e-5)


def test_multi_token_candidate_is_teacher_forced(client):
    off = CONT.index("answer")
    r = post(client, [{"char_offset": off, "candidates": ["answer", "a"]}])
    site = r.json()["sites"][0]
    assert site["forced"] is True
    before = head() + CONT[:off]
    assert site["candidates"]["answer"] == pytest.approx(reference(before, "answer"), abs=1e-5)


def test_sites_together_equal_sites_one_at_a_time(client):
    sites = [{"char_offset": CONT.index("[1") + 1, "candidates": ["1", "2", "3"]},
             {"char_offset": CONT.index("[2") + 1, "candidates": ["1", "2", "3"]},
             {"char_offset": CONT.index("answer"), "candidates": ["answer", "x"]}]
    together = post(client, sites).json()["sites"]
    alone = [post(client, [s]).json()["sites"][0] for s in sites]
    assert together == alone


def test_response_carries_the_pin(client):
    body = post(client, [{"char_offset": 0, "candidates": ["t"]}]).json()
    assert body["model"] == "fake-model"
    assert body["commit"] == "c" * 40 and body["dirty"] is False
    assert body["revision"] == "0" * 40
    assert body["tokenizer_sha"] == "f" * 64
    assert body["prompt_tokens"] == len(TOK(head() + CONT)["input_ids"])


def test_site_at_the_end_of_the_continuation_reads_the_next_token(client):
    r = post(client, [{"char_offset": len(CONT), "candidates": ["a", "b"]}])
    assert r.status_code == 200
    assert r.json()["sites"][0]["token_index"] == len(TOK(head() + CONT)["input_ids"])


def test_chat_template_kwargs_reach_the_template(client):
    s = [{"char_offset": 0, "candidates": ["t", "a"]}]
    plain = post(client, s).json()
    thinking = post(client, s, chat_template_kwargs={"enable_thinking": True}).json()
    assert thinking["prompt_tokens"] > plain["prompt_tokens"]


@pytest.mark.parametrize("sites,code", [
    ([{"char_offset": 999, "candidates": ["1"]}], "invalid_request"),
    ([{"char_offset": 0, "candidates": [""]}], "invalid_request"),
    ([{"char_offset": 0, "candidates": []}], "invalid_request"),
    ([], "invalid_request"),
])
def test_bad_sites_are_400(client, sites, code):
    r = post(client, sites)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == code


def test_context_overflow_is_400_not_truncated():
    with TestClient(create_app(score_service(max_context=20))) as c:
        r = post(c, [{"char_offset": 1, "candidates": ["1"]}])
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "context_length_exceeded"


def test_unknown_model_is_404(client):
    r = client.post("/v1/score", json={"model": "other", "messages": MSG, "continuation": "x",
                                       "sites": [{"char_offset": 0, "candidates": ["x"]}]})
    assert r.status_code == 404


def test_score_function_rejects_a_site_on_the_first_token():
    with pytest.raises(ValueError):
        score(TOK, FakeRunner().logprob_rows, "", "abc", [Site(0, ("a",))], 100)


async def test_scoring_during_generation_matches_scoring_alone():
    svc = score_service(FakeRunner(t0=0.002))
    app = create_app(svc)
    svc.start()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://localhost") as http:
            sites = [{"char_offset": CONT.index("[1") + 1, "candidates": ["1", "2", "3"]},
                     {"char_offset": CONT.index("answer"), "candidates": ["answer"]}]
            body = {"messages": MSG, "continuation": CONT, "sites": sites}
            alone = (await http.post("/v1/score", json=body)).json()
            gens = [http.post("/v1/chat/completions",
                              json={"messages": [{"role": "user", "content": f"hi {i}"}],
                                    "max_tokens": 30, "temperature": 0})
                    for i in range(6)]
            scores = [http.post("/v1/score", json=body) for _ in range(4)]
            out = await asyncio.gather(*gens, *scores)
            assert all(r.status_code == 200 for r in out)
            for r in out[6:]:
                assert r.json() == alone
    finally:
        svc.stop()


async def test_a_queued_score_is_answered_when_the_scheduler_fails_everything():
    svc = score_service()
    app = create_app(svc)  # engine not started, so the job stays queued
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://localhost") as http:
        task = asyncio.create_task(http.post("/v1/score", json={
            "messages": MSG, "continuation": CONT,
            "sites": [{"char_offset": 0, "candidates": ["t"]}]}))
        while not svc.engine.scheduler.jobs:
            await asyncio.sleep(0.01)
        svc.engine.scheduler.fail_all("model is being replaced")
        r = await task
    assert r.status_code == 503


def test_logprob_rows_is_a_log_softmax():
    lp = FakeRunner().logprob_rows([1, 2, 3], [0, 2])
    assert lp.shape == (2, 40)
    assert torch.allclose(lp.exp().sum(-1), torch.ones(2))


async def test_a_job_whose_caller_went_away_is_skipped():
    svc = score_service()  # engine not started, so the job waits until step() below
    ran = []
    task = asyncio.create_task(svc.engine.run(lambda: ran.append(1)))
    while not svc.engine.scheduler.jobs:
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    svc.engine.scheduler.step()
    assert ran == [] and not svc.engine.scheduler.jobs


def test_too_many_candidates_at_a_site_is_400(client):
    r = post(client, [{"char_offset": 1, "candidates": [f"c{i}" for i in range(65)]}])
    assert r.status_code == 400


def test_a_candidate_over_the_token_cap_is_400(client):
    r = post(client, [{"char_offset": 1, "candidates": ["x" * 33]}])
    assert r.status_code == 400
    assert "tokens" in r.json()["error"]["message"]


def test_too_many_forced_forwards_is_400_before_any_forward():
    svc = score_service()
    runner = svc.parts.runner
    # 5 sites x 60 two-token candidates = 300 teacher-forced passes, over the 256 cap
    cands = [f"{a}{b}" for a in "xyz" for b in "abcdefghijklmnopqrst"]
    sites = [{"char_offset": i, "candidates": cands} for i in range(1, 6)]
    with TestClient(create_app(svc)) as c:
        runner.calls.clear()
        r = post(c, sites)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "score_too_large"
    assert not [k for k in runner.calls if k[0] == "score"]


def test_under_the_forward_cap_is_scored(client):
    sites = [{"char_offset": 1, "candidates": [f"x{b}" for b in "abcdefghijklmnopqrst"]}]
    assert post(client, sites).status_code == 200


async def test_a_full_score_queue_is_429():
    svc = score_service(max_score_jobs=1)
    app = create_app(svc)  # engine not started, so the first job stays queued
    body = {"messages": MSG, "continuation": CONT,
            "sites": [{"char_offset": 0, "candidates": ["t"]}]}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://localhost") as http:
        first = asyncio.create_task(http.post("/v1/score", json=body))
        while not svc.engine.scheduler.jobs:
            await asyncio.sleep(0.01)
        r = await http.post("/v1/score", json=body)
        assert r.status_code == 429
        assert r.json()["error"]["code"] == "queue_full" and "retry-after" in r.headers
        svc.engine.scheduler.step()
        assert (await first).status_code == 200


def test_score_stops_between_forwards_once_the_caller_is_gone():
    from localhost_ai.engine.score import Stopped
    runner = FakeRunner()
    with pytest.raises(Stopped):
        score(TOK, runner.logprob_rows, head(), CONT, [Site(3, ("xy",))], 100, stop=lambda: True)
    assert not runner.calls
