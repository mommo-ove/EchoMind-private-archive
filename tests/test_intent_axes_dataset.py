from tools.build_intent_axes_golden import build_payload


def test_axis_dataset_has_360_validation_and_100_heldout_cases():
    payload = build_payload()
    cases = payload["cases"]

    assert payload["schema_version"] == 2
    assert len(cases) == 460
    assert sum(case["split"] == "validation" for case in cases) == 360
    assert sum(case["split"] == "test" for case in cases) == 100
    assert len({case["message"] for case in cases}) == 460


def test_axis_dataset_covers_all_axes_and_review_metadata():
    cases = build_payload()["cases"]
    validation = [case for case in cases if case["split"] == "validation"]
    test = [case for case in cases if case["split"] == "test"]

    assert {domain for case in validation for domain in case["expected"]["domains"]} == {
        "general", "technical", "billing", "account"
    }
    assert {case["expected"]["action"] for case in validation} == {
        "greeting", "query", "request", "report", "complaint", "feedback", "other"
    }
    assert {case["expected"]["escalated"] for case in validation} == {False, True}
    assert all(case["review"]["status"] == "reviewed" for case in cases)
    assert all(case["source"] == "legacy_challenge" for case in test)
