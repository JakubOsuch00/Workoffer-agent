import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from agent_project.agent import (
    DEFAULT_MAX_STEPS,
    EXAMPLE_GOAL,
    MAX_STEPS_HARD_CAP,
    TRACE_DIR,
    default_trace_path,
    run_agent,
)
from agent_project.llm import ModelClient, OpenRouterClient

FOLDERS = {"traces": TRACE_DIR, "evidence": TRACE_DIR.parent / "evidence"}
PAGE = Path(__file__).with_name("ui.html")
MAX_GOAL_LENGTH = 2000
_TRACE_NAME = re.compile(r"^[\w.-]+\.jsonl$")
_run_lock = threading.Lock()


def list_traces() -> list[dict[str, str]]:
    return [
        {"folder": folder, "name": path.name}
        for folder, directory in FOLDERS.items()
        for path in sorted(directory.glob("*.jsonl"), reverse=True)
    ]


def read_trace(folder: str, name: str) -> list[dict[str, Any]] | None:
    if folder not in FOLDERS or not _TRACE_NAME.match(name):
        return None
    path = FOLDERS[folder] / name
    if not path.is_file():
        return None
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            break  # last line can be incomplete during a live run
    return events


def start_run(
    goal: str, max_steps: int, client: ModelClient, trace_path: Path
) -> threading.Thread:
    if not _run_lock.acquire(blocking=False):
        raise RuntimeError("Another run is still in progress.")

    def work() -> None:
        try:
            run_agent(goal, client, max_steps=max_steps, trace_path=trace_path)
        finally:
            _run_lock.release()

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    return thread


def validate_run_request(body: Any) -> tuple[str, int]:
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object.")
    goal, max_steps = body.get("goal"), body.get("max_steps", DEFAULT_MAX_STEPS)
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("Goal must be a non-empty text.")
    if len(goal) > MAX_GOAL_LENGTH:
        raise ValueError(f"Goal must be at most {MAX_GOAL_LENGTH} characters.")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise ValueError("max_steps must be a whole number.")
    if not 1 <= max_steps <= MAX_STEPS_HARD_CAP:
        raise ValueError(f"max_steps must be between 1 and {MAX_STEPS_HARD_CAP}.")
    return goal.strip(), max_steps


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: Any) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path == "/":
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/defaults":
            self._json(
                200,
                {
                    "goal": EXAMPLE_GOAL,
                    "max_steps": DEFAULT_MAX_STEPS,
                    "max_steps_cap": MAX_STEPS_HARD_CAP,
                },
            )
        elif url.path == "/api/traces":
            self._json(200, list_traces())
        elif url.path == "/api/trace":
            query = parse_qs(url.query)
            folder = query.get("folder", [""])[0]
            name = query.get("name", [""])[0]
            events = read_trace(folder, name)
            if events is None:
                self._json(404, {"error": "not_found"})
            else:
                self._json(200, events)
        else:
            self._json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/run":
            self._json(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"null")
            goal, max_steps = validate_run_request(body)
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"error": "invalid_request", "message": str(exc)})
            return
        try:
            client = OpenRouterClient()
        except ValueError as exc:
            self._json(400, {"error": "setup", "message": str(exc)})
            return
        trace_path = default_trace_path()
        try:
            start_run(goal, max_steps, client, trace_path)
        except RuntimeError as exc:
            self._json(409, {"error": "busy", "message": str(exc)})
            return
        self._json(202, {"folder": "traces", "name": trace_path.name})

    def log_message(self, format: str, *args: Any) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Local web UI for the agent.")
    parser.add_argument("--port", type=int, default=8000)
    port = parser.parse_args().port

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Job Search Agent UI: http://127.0.0.1:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
