import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_project.llm import ModelClient, OpenRouterClient
from agent_project.tools import (
    TOOLS,
    FinalAnswerArgs,
    execute_tool,
    parse_arguments,
    tool_schemas,
)
from agent_project.tracing import append_trace

TRACE_DIR = Path(__file__).resolve().parents[2] / "traces"

DEFAULT_MAX_STEPS = 10
MAX_STEPS_HARD_CAP = 12
MAX_COST_USD = 0.10
MAX_CONSECUTIVE_FAILURES = 3

EXAMPLE_GOAL = (
    "I am looking for my first job in Python or data analysis. Find offers that "
    "fit my profile, skip the ones I already applied to, and tell me which "
    "skills I am missing for offers that almost match."
)

SYSTEM_PROMPT = """You are a job search assistant for a student looking for a \
first job.
Work step by step with the tools:
1. Read the candidate profile with get_profile.
2. Search offers with search_offers (try a few relevant keywords).
3. Check promising offers with check_match. Never judge a match yourself; \
check_match does the comparison.
4. Finish with final_answer. Recommend only offers that check_match reported \
with is_full_match=true and already_applied=false. List skills_to_learn from \
missing_skills of near-miss offers.
Independent tool calls may be requested together in one turn.
If a tool returns an error, read it and correct your next request.
Always finish by calling final_answer; do not answer in plain text."""

NO_TOOL_MESSAGE = (
    "No tool was called. Continue with a tool, or call final_answer to finish."
)


@dataclass
class RunResult:
    stop_reason: str
    final_answer: dict[str, Any] | None
    steps: int
    cost_usd: float
    trace_path: Path


def check_final(
    args: FinalAnswerArgs, checked: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    problems = []
    if len(set(args.recommended_offer_ids)) != len(args.recommended_offer_ids):
        problems.append({"problem": "duplicate_offer_ids"})
    for offer_id in dict.fromkeys(args.recommended_offer_ids):
        result = checked.get(offer_id)
        if result is None:
            problems.append(
                {
                    "offer_id": offer_id,
                    "problem": "not_checked",
                    "hint": "Call check_match for this offer before recommending it.",
                }
            )
        elif not result["is_full_match"]:
            failed = [name for name, ok in result["checks"].items() if not ok]
            problems.append(
                {"offer_id": offer_id, "problem": "not_full_match", "failed": failed}
            )
        elif result["already_applied"]:
            problems.append({"offer_id": offer_id, "problem": "already_applied"})
    if not problems:
        return None
    return {"ok": False, "error": "final_answer_rejected", "problems": problems}


def _call_key(name: str, raw_arguments: Any) -> str:
    try:
        data = json.loads(raw_arguments) if isinstance(raw_arguments, str) else None
    except json.JSONDecodeError:
        data = None
    if data is None:
        data = raw_arguments
    return name + ":" + json.dumps(data, sort_keys=True, default=str)


def _assistant_message(response: dict[str, Any]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": response.get("text")}
    if response.get("tool_calls"):
        message["tool_calls"] = [
            {
                "id": call["id"],
                "type": "function",
                "function": {"name": call["name"], "arguments": call["arguments"]},
            }
            for call in response["tool_calls"]
        ]
    return message


def default_trace_path() -> Path:
    return TRACE_DIR / f"run-{datetime.now():%Y%m%d-%H%M%S}.jsonl"


def run_agent(
    goal: str,
    client: ModelClient,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_cost_usd: float = MAX_COST_USD,
    trace_path: Path | None = None,
) -> RunResult:
    max_steps = max(1, min(max_steps, MAX_STEPS_HARD_CAP))
    max_cost_usd = min(max_cost_usd, MAX_COST_USD)
    trace_path = trace_path or default_trace_path()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": goal},
    ]
    tools = tool_schemas()
    checked: dict[str, dict[str, Any]] = {}
    executed: set[str] = set()
    total_cost = 0.0
    total_latency_ms = 0
    failures = 0
    final: dict[str, Any] | None = None
    stop_reason = "step_limit"
    step = 0

    append_trace(
        trace_path,
        "run_started",
        {
            "goal": goal,
            "model": getattr(client, "model", type(client).__name__),
            "max_steps": max_steps,
            "max_cost_usd": max_cost_usd,
            "tools": [tool["function"]["name"] for tool in tools],
        },
    )

    for step in range(1, max_steps + 1):
        try:
            response = client.complete(messages, tools)
        except Exception as exc:
            append_trace(trace_path, "model_error", {"step": step, "error": repr(exc)})
            stop_reason = "model_error"
            break

        usage = response.get("usage") or {}
        step_cost = float(usage.get("cost") or 0.0)
        total_cost += step_cost
        total_latency_ms += response.get("latency_ms") or 0
        calls = response.get("tool_calls") or []
        append_trace(
            trace_path,
            "model_response",
            {
                "step": step,
                "text": response.get("text"),
                "tool_calls": calls,
                "finish_reason": response.get("finish_reason"),
                "model": response.get("model"),
                "provider": response.get("provider"),
                "latency_ms": response.get("latency_ms"),
                "usage": usage,
                "cost_usd": step_cost,
                "total_cost_usd": round(total_cost, 6),
            },
        )
        messages.append(_assistant_message(response))

        step_ok = False
        if not calls:
            messages.append({"role": "user", "content": NO_TOOL_MESSAGE})
            append_trace(
                trace_path,
                "tool_result",
                {"step": step, "name": None, "result": {"error": "no_tool_call"}},
            )

        for call in calls:
            started = time.perf_counter()
            name, raw = call["name"], call.get("arguments")
            key = _call_key(name, raw)

            if name == "final_answer":
                args, result = parse_arguments(FinalAnswerArgs, raw)
                if args is not None:
                    result = check_final(args, checked)
                    if result is None:
                        final = args.model_dump()
                        result = {"ok": True, "accepted": True}
            elif key in executed:
                result = {
                    "ok": False,
                    "error": "duplicate_call",
                    "message": "This exact call was already run. Use its result.",
                }
            else:
                result = execute_tool(name, raw)
                if result.get("ok"):
                    executed.add(key)
                if name == "check_match" and result.get("ok"):
                    for item in result["results"]:
                        if "error" not in item:
                            checked[item["offer_id"]] = item

            step_ok = step_ok or bool(result.get("ok"))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
            append_trace(
                trace_path,
                "tool_result",
                {
                    "step": step,
                    "name": name,
                    "arguments": raw,
                    "allowed": name in TOOLS or name == "final_answer",
                    "result": result,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )

        failures = 0 if step_ok else failures + 1
        if final is not None:
            stop_reason = "final_answer"
            break
        if failures >= MAX_CONSECUTIVE_FAILURES:
            stop_reason = "too_many_failures"
            break
        if total_cost >= max_cost_usd:
            stop_reason = "budget_exceeded"
            break

    append_trace(
        trace_path,
        "run_finished",
        {
            "stop_reason": stop_reason,
            "steps": step,
            "total_cost_usd": round(total_cost, 6),
            "total_model_latency_ms": total_latency_ms,
            "final_answer": final,
        },
    )
    return RunResult(stop_reason, final, step, total_cost, trace_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the job search agent.")
    parser.add_argument("goal", nargs="?", default=EXAMPLE_GOAL)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    cli = parser.parse_args()

    result = run_agent(cli.goal, OpenRouterClient(), max_steps=cli.max_steps)
    print(f"Stop reason: {result.stop_reason}")
    print(f"Steps: {result.steps}, cost: USD {result.cost_usd:.6f}")
    print(f"Trace: {result.trace_path.relative_to(TRACE_DIR.parent)}")
    if result.final_answer:
        print(json.dumps(result.final_answer, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
