import json

from evaluation.retrieval_dataset import load_cases, split_cases


def test_load_cases_supports_dedicated_retrieval_schema(tmp_path):
    path = tmp_path / "retrieval.json"
    path.write_text(
        json.dumps({
            "cases": [
                {
                    "case_id": "single-01-1",
                    "question": "401怎么办",
                    "category": "single_document",
                    "answerable": True,
                    "reference_context_relevance": {"network-401": 3.0},
                },
                {
                    "case_id": "unanswerable-01",
                    "question": "食堂菜单是什么",
                    "category": "unanswerable",
                    "answerable": False,
                    "reference_context_relevance": {},
                },
            ]
        }),
        encoding="utf-8",
    )

    cases = load_cases(path)

    assert [case["case_id"] for case in cases] == [
        "single-01-1",
        "unanswerable-01",
    ]
    assert cases[0]["reference_relevance"] == {"network-401": 3.0}
    assert cases[1]["answerable"] is False


def test_split_cases_keeps_unanswerable_queries_out_of_ranking_metrics():
    answerable, unanswerable = split_cases([
        {"case_id": "a", "answerable": True, "reference_relevance": {"x": 3.0}},
        {"case_id": "b", "answerable": False, "reference_relevance": {}},
    ])

    assert [case["case_id"] for case in answerable] == ["a"]
    assert [case["case_id"] for case in unanswerable] == ["b"]
