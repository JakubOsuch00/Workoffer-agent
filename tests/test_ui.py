import json
import threading
import urllib.error
import urllib.request

import pytest
from conftest import reply

from agent_project import MockClient, ui


def test_read_trace_rejects_unknown_folder_and_path_traversal() -> None:
    assert ui.read_trace("secrets", "run.jsonl") is None
    assert ui.read_trace("traces", "../.env") is None
    assert ui.read_trace("evidence", "missing.jsonl") is None


def test_read_trace_skips_unfinished_last_line(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(ui.FOLDERS, "traces", tmp_path)
    (tmp_path / "run.jsonl").write_text('{"event": "a"}\n{"event": ', "utf-8")
    assert ui.read_trace("traces", "run.jsonl") == [{"event": "a"}]


def test_validate_run_request() -> None:
    assert ui.validate_run_request({"goal": " find jobs ", "max_steps": 3}) == (
        "find jobs",
        3,
    )
    for body in [
        None,
        {"goal": ""},
        {"goal": "x", "max_steps": 13},
        {"goal": "x", "max_steps": "5"},
        {"goal": "x", "max_steps": True},
    ]:
        with pytest.raises(ValueError):
            ui.validate_run_request(body)


def test_start_run_writes_trace_and_blocks_second_run(tmp_path) -> None:
    gate = threading.Event()

    class SlowClient(MockClient):
        def complete(self, messages, tools=None):
            gate.wait(5)
            return super().complete(messages, tools)

    final = {"recommended_offer_ids": [], "summary": "Nothing found."}
    client = SlowClient([reply(("final_answer", final))])
    thread = ui.start_run("goal", 3, client, tmp_path / "run.jsonl")
    with pytest.raises(RuntimeError):
        ui.start_run("goal", 3, client, tmp_path / "other.jsonl")
    gate.set()
    thread.join(5)

    lines = (tmp_path / "run.jsonl").read_text("utf-8").splitlines()
    assert json.loads(lines[-1])["payload"]["stop_reason"] == "final_answer"


@pytest.fixture
def server():
    httpd = ui.ThreadingHTTPServer(("127.0.0.1", 0), ui.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def test_server_serves_page_and_rejects_bad_requests(server) -> None:
    with urllib.request.urlopen(server + "/") as response:
        assert b"Job Search Agent" in response.read()

    with pytest.raises(urllib.error.HTTPError) as missing:
        urllib.request.urlopen(server + "/api/trace?folder=traces&name=../.env")
    assert missing.value.code == 404

    request = urllib.request.Request(
        server + "/api/run", data=b'{"goal": ""}', method="POST"
    )
    with pytest.raises(urllib.error.HTTPError) as bad:
        urllib.request.urlopen(request)
    assert bad.value.code == 400
