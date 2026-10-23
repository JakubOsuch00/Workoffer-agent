import json

from conftest import RecordingMockClient, reply

from agent_project import MockClient
from agent_project.agent import EXAMPLE_GOAL, run_agent

FINAL_OK = {
    "recommended_offer_ids": ["J01", "J19", "J24"],
    "skills_to_learn": ["Django"],
    "summary": "Three offers fit your profile.",
}


def _events(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_happy_path(tmp_path) -> None:
    client = MockClient(
        [
            reply(("get_profile", {})),
            reply(("search_offers", {"query": "python data"})),
            reply(("check_match", {"offer_ids": ["J01", "J19", "J24", "J05", "J03"]})),
            reply(("final_answer", FINAL_OK)),
        ]
    )
    result = run_agent(EXAMPLE_GOAL, client, trace_path=tmp_path / "t.jsonl")

    assert result.stop_reason == "final_answer"
    assert result.steps == 4
    assert result.final_answer["recommended_offer_ids"] == ["J01", "J19", "J24"]
    assert abs(result.cost_usd - 0.004) < 1e-9

    events = _events(result.trace_path)
    assert events[0]["event"] == "run_started"
    assert events[-1]["event"] == "run_finished"
    assert events[-1]["payload"]["stop_reason"] == "final_answer"
    responses = [e for e in events if e["event"] == "model_response"]
    assert all("latency_ms" in e["payload"] for e in responses)
    assert all("cost_usd" in e["payload"] for e in responses)


def test_unchecked_offer_is_rejected_then_corrected(tmp_path) -> None:
    client = RecordingMockClient(
        [
            reply(("final_answer", {**FINAL_OK, "recommended_offer_ids": ["J01"]})),
            reply(("check_match", {"offer_ids": ["J01"]})),
            reply(("final_answer", {**FINAL_OK, "recommended_offer_ids": ["J01"]})),
        ]
    )
    result = run_agent(EXAMPLE_GOAL, client, trace_path=tmp_path / "t.jsonl")

    assert result.stop_reason == "final_answer"
    observation = json.loads(client.seen[1][-1]["content"])
    assert observation["error"] == "final_answer_rejected"
    assert observation["problems"][0]["problem"] == "not_checked"


def test_already_applied_offer_is_rejected(tmp_path) -> None:
    client = RecordingMockClient(
        [
            reply(("check_match", {"offer_ids": ["J05"]})),
            reply(("final_answer", {**FINAL_OK, "recommended_offer_ids": ["J05"]})),
            reply(("final_answer", {**FINAL_OK, "recommended_offer_ids": []})),
        ]
    )
    result = run_agent(EXAMPLE_GOAL, client, trace_path=tmp_path / "t.jsonl")

    observation = json.loads(client.seen[2][-1]["content"])
    assert observation["problems"][0]["problem"] == "already_applied"
    assert result.final_answer["recommended_offer_ids"] == []


def test_near_miss_offer_is_rejected(tmp_path) -> None:
    client = RecordingMockClient(
        [
            reply(("check_match", {"offer_ids": ["J03"]})),
            reply(("final_answer", {**FINAL_OK, "recommended_offer_ids": ["J03"]})),
            reply(("final_answer", {**FINAL_OK, "recommended_offer_ids": []})),
        ]
    )
    run_agent(EXAMPLE_GOAL, client, trace_path=tmp_path / "t.jsonl")
    problem = json.loads(client.seen[2][-1]["content"])["problems"][0]
    assert problem == {
        "offer_id": "J03",
        "problem": "not_full_match",
        "failed": ["skills_ok"],
    }


def test_three_failed_steps_stop_the_run(tmp_path) -> None:
    client = MockClient(
        [
            reply(("delete_offer", {})),
            reply(("get_offer", {"offer_id": "J99"})),
            reply(text="I think J01 is good."),
            reply(("get_profile", {})),
        ]
    )
    result = run_agent(EXAMPLE_GOAL, client, trace_path=tmp_path / "t.jsonl")
    assert result.stop_reason == "too_many_failures"
    assert result.steps == 3


def test_success_resets_failure_counter(tmp_path) -> None:
    client = MockClient(
        [
            reply(("get_offer", {"offer_id": "J99"})),
            reply(("get_offer", {"offer_id": "J98"})),
            reply(("get_profile", {})),
            reply(("get_offer", {"offer_id": "J97"})),
            reply(("final_answer", {**FINAL_OK, "recommended_offer_ids": []})),
        ]
    )
    result = run_agent(EXAMPLE_GOAL, client, trace_path=tmp_path / "t.jsonl")
    assert result.stop_reason == "final_answer"


def test_identical_call_is_not_executed_twice(tmp_path) -> None:
    client = RecordingMockClient(
        [
            reply(("get_offer", {"offer_id": "J01"})),
            reply(("get_offer", '{"offer_id":"J01"}')),
            reply(("final_answer", {**FINAL_OK, "recommended_offer_ids": []})),
        ]
    )
    run_agent(EXAMPLE_GOAL, client, trace_path=tmp_path / "t.jsonl")
    assert json.loads(client.seen[2][-1]["content"])["error"] == "duplicate_call"


def test_step_limit(tmp_path) -> None:
    queries = ["python", "data", "sql", "excel", "junior"]
    client = MockClient([reply(("search_offers", {"query": q})) for q in queries])
    result = run_agent(
        EXAMPLE_GOAL, client, max_steps=3, trace_path=tmp_path / "t.jsonl"
    )
    assert result.stop_reason == "step_limit"
    assert result.steps == 3


def test_step_limit_has_hard_cap(tmp_path) -> None:
    client = MockClient(
        [reply(("search_offers", {"query": f"q{i}"})) for i in range(20)]
    )
    result = run_agent(
        EXAMPLE_GOAL, client, max_steps=50, trace_path=tmp_path / "t.jsonl"
    )
    assert result.steps == 12


def test_budget_limit(tmp_path) -> None:
    client = MockClient(
        [
            reply(("get_profile", {}), cost=0.06),
            reply(("search_offers", {"query": "python"}), cost=0.06),
            reply(("search_offers", {"query": "data"}), cost=0.06),
        ]
    )
    result = run_agent(EXAMPLE_GOAL, client, trace_path=tmp_path / "t.jsonl")
    assert result.stop_reason == "budget_exceeded"
    assert result.steps == 2


def test_model_error_is_traced(tmp_path) -> None:
    result = run_agent(EXAMPLE_GOAL, MockClient([]), trace_path=tmp_path / "t.jsonl")
    assert result.stop_reason == "model_error"
    assert _events(result.trace_path)[1]["event"] == "model_error"
