import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_retrieval_golden_builder_is_deterministic_and_self_validating():
    from tools.build_retrieval_golden import build_payload, validate_payload

    documents = json.loads(
        (ROOT / "data" / "knowledge" / "campus_knowledge.json").read_text(
            encoding="utf-8"
        )
    )
    first = build_payload(documents)
    second = build_payload(documents)

    assert first == second
    validate_payload(first, documents)


def test_campus_knowledge_asset_has_twenty_unique_documents():
    path = ROOT / "data" / "knowledge" / "campus_knowledge.json"
    documents = json.loads(path.read_text(encoding="utf-8"))

    assert isinstance(documents, list)
    assert len(documents) == 20
    assert len({item["title"] for item in documents}) == 20
    assert all(item["title"].strip() for item in documents)
    assert all(len(item["content"].strip()) >= 80 for item in documents)


def test_campus_golden_asset_has_balanced_intent_and_dialog_cases():
    path = ROOT / "data" / "eval" / "campus_golden.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    intent_cases = payload["intent_cases"]
    dialog_cases = payload["dialog_cases"]

    assert len(intent_cases) >= 20
    assert len(dialog_cases) >= 20
    assert len(intent_cases) + len(dialog_cases) >= 40
    assert {case["expected_intent"] for case in intent_cases} >= {
        "greeting",
        "technical",
        "billing",
        "escalation",
    }
    assert all(
        bool(case.get("question") or case.get("turns"))
        for case in dialog_cases
    )


def test_campus_golden_asset_covers_multi_label_intent_recognition():
    path = ROOT / "data" / "eval" / "campus_golden.json"
    intent_cases = json.loads(path.read_text(encoding="utf-8"))["intent_cases"]

    multi_label_cases = [
        case
        for case in intent_cases
        if len(case.get("expected_intents", [])) > 1
    ]

    assert len(multi_label_cases) >= 3
    assert any(
        set(case["expected_intents"]) == {"technical", "billing"}
        for case in multi_label_cases
    )


def test_campus_golden_expectations_match_current_agent_capabilities():
    path = ROOT / "data" / "eval" / "campus_golden.json"
    dialog_cases = json.loads(path.read_text(encoding="utf-8"))["dialog_cases"]

    escalation_cases = [
        case
        for case in dialog_cases
        if "escalation" in case.get("expected_intents", [])
    ]
    assert escalation_cases
    assert all(case.get("expected_agents") == ["general"] for case in escalation_cases)

    ticket_cases = [
        case
        for case in dialog_cases
        if "create_ticket" in case.get("expected_tools", [])
    ]
    assert ticket_cases
    assert all(
        case.get("expected_agents", [])[-1] in {"technical", "billing"}
        for case in ticket_cases
    )


def test_campus_golden_has_rag_ground_truth_for_knowledge_cases():
    payload = json.loads(
        (ROOT / "data" / "eval" / "campus_golden.json").read_text(
            encoding="utf-8"
        )
    )
    knowledge_cases = [
        case
        for case in payload["dialog_cases"]
        if case.get("expect_knowledge") is True
    ]

    assert knowledge_cases
    assert all(
        bool(case.get("reference_answer") or case.get("reference_answers"))
        for case in knowledge_cases
    )
    assert all(case.get("reference_context_ids") for case in knowledge_cases)
    assert all(
        case.get("reference_context_relevance")
        for case in knowledge_cases
    )


def test_retrieval_golden_has_140_unique_cases_and_clear_splits():
    payload = json.loads(
        (ROOT / "data" / "eval" / "campus_retrieval_golden.json").read_text(
            encoding="utf-8"
        )
    )
    cases = payload["cases"]

    assert payload["schema_version"] == 1
    assert payload["case_count"] == 140
    assert len(cases) == 140
    assert len({case["case_id"] for case in cases}) == 140
    assert len({case["question"] for case in cases}) == 140
    assert {case["category"] for case in cases} == {
        "single_document",
        "composite",
        "unanswerable",
    }
    assert sum(case["category"] == "single_document" for case in cases) == 100
    assert sum(case["category"] == "composite" for case in cases) == 20
    assert sum(case["category"] == "unanswerable" for case in cases) == 20


def test_retrieval_golden_covers_every_document_and_has_valid_relevance():
    documents = json.loads(
        (ROOT / "data" / "knowledge" / "campus_knowledge.json").read_text(
            encoding="utf-8"
        )
    )
    payload = json.loads(
        (ROOT / "data" / "eval" / "campus_retrieval_golden.json").read_text(
            encoding="utf-8"
        )
    )
    cases = payload["cases"]
    document_ids = {item["context_id"] for item in payload["corpus"]}

    assert payload["corpus_document_count"] == len(documents) == 20
    assert len(document_ids) == 20
    assert {item["title"] for item in payload["corpus"]} == {
        item["title"] for item in documents
    }

    single_coverage = {context_id: 0 for context_id in document_ids}
    for case in cases:
        relevance = case["reference_context_relevance"]
        assert set(relevance).issubset(document_ids)
        if case["category"] == "single_document":
            assert case["answerable"] is True
            assert len(relevance) == 1
            single_coverage[next(iter(relevance))] += 1
        elif case["category"] == "composite":
            assert case["answerable"] is True
            assert 2 <= len(relevance) <= 3
        else:
            assert case["answerable"] is False
            assert relevance == {}

    assert set(single_coverage.values()) == {5}


def test_intent_calibration_golden_has_fixed_validation_and_test_splits():
    payload = json.loads(
        (ROOT / "data" / "eval" / "intent_calibration_golden.json").read_text(
            encoding="utf-8"
        )
    )
    cases = payload["cases"]

    assert payload["case_count"] == len(cases) == 100
    assert sum(case["split"] == "validation" for case in cases) == 70
    assert sum(case["split"] == "test" for case in cases) == 30
    assert len({case["case_id"] for case in cases}) == 100
    assert len({case["message"] for case in cases}) == 100
    assert all(case["expected_intent"] in case["expected_intents"] for case in cases)
    assert sum(len(case["expected_intents"]) > 1 for case in cases) == 20
    assert {case["expected_intent"] for case in cases} >= {
        "query", "complaint", "request", "greeting", "escalation",
        "technical", "billing", "account", "feedback", "other",
    }


def test_sqlite_demo_seed_is_idempotent_and_covers_ticket_states(tmp_path):
    from campus.store import CampusStore
    from tools.seed_campus_demo import seed_demo_scenarios

    store = CampusStore(tmp_path / "campus.db")
    first = seed_demo_scenarios(store)
    second = seed_demo_scenarios(store)

    assert first == second
    assert set(first) == {"open_ticket", "processing_ticket", "resolved_ticket"}
    assert first["open_ticket"]["status"] == "OPEN"
    assert first["processing_ticket"]["status"] == "PROCESSING"
    assert first["resolved_ticket"]["status"] == "RESOLVED"
    assert store.query_transactions("demo_user_01", days=30)
    assert store.query_transactions("demo_user_03", days=30) == []
    assert (
        store.get_ticket(
            first["resolved_ticket"]["id"],
            user_id="demo_user_01",
        )
        is None
    )


def test_sqlite_demo_seed_script_runs_from_project_root(tmp_path):
    environment = os.environ.copy()
    environment["CAMPUS_DB_PATH"] = str(tmp_path / "campus.db")

    completed = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "seed_campus_demo.py")],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["open_ticket"]["status"] == "OPEN"
