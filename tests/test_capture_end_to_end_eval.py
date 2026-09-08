import json
from pathlib import Path

import pytest

from tools.capture_end_to_end_eval import capture_report


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_capture_persists_complete_real_eval_response_atomically(tmp_path):
    output = tmp_path / "campus_end_to_end_latest.json"
    payload = {
        "total": 2,
        "passed": 1,
        "metrics": {"tool_correctness": 0.5, "task_completion": 0.5},
        "results": [{"test_id": "one"}, {"test_id": "two"}],
    }
    calls = []

    def opener(request, timeout):
        calls.append((request.full_url, request.get_method(), timeout, request.data))
        return FakeResponse(payload)

    result = capture_report(
        base_url="http://localhost:8000",
        dataset_path=tmp_path / "missing.json",
        output_path=output,
        opener=opener,
        payload_override={"dialog_cases": []},
    )

    assert result == payload
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert calls[0][:3] == ("http://localhost:8000/eval/run", "POST", 1800)


def test_capture_does_not_overwrite_previous_report_on_invalid_response(tmp_path):
    output = tmp_path / "campus_end_to_end_latest.json"
    output.write_text('{"old": true}', encoding="utf-8")

    def opener(request, timeout):
        return FakeResponse({"detail": "service unavailable"})

    with pytest.raises(ValueError, match="complete evaluation report"):
        capture_report(
            base_url="http://localhost:8000",
            dataset_path=tmp_path / "missing.json",
            output_path=output,
            opener=opener,
            payload_override={"dialog_cases": []},
        )

    assert output.read_text(encoding="utf-8") == '{"old": true}'
