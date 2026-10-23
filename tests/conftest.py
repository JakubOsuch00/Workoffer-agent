import copy
import itertools
import json
from typing import Any

from agent_project import MockClient

_ids = itertools.count(1)


def reply(
    *calls: tuple[str, dict[str, Any] | str],
    text: str | None = None,
    cost: float = 0.001,
) -> dict[str, Any]:
    return {
        "text": text,
        "tool_calls": [
            {
                "id": f"call_{next(_ids)}",
                "name": name,
                "arguments": args if isinstance(args, str) else json.dumps(args),
            }
            for name, args in calls
        ],
        "finish_reason": "tool_calls" if calls else "stop",
        "model": "mock",
        "provider": "mock",
        "latency_ms": 5,
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": cost},
    }


class RecordingMockClient(MockClient):
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        super().__init__(responses)
        self.seen: list[list[dict[str, Any]]] = []

    def complete(self, messages, tools=None):
        self.seen.append(copy.deepcopy(messages))
        return super().complete(messages, tools)
