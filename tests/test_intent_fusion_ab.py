import argparse
import asyncio
import json

import pytest

from evaluation.intent_fusion_ab import (
    FusionConfig,
    compare_fusion_variants,
    fuse_predictions,
    tune_fusion,
)
from tools.run_intent_fusion_ab import collect, parse_domain_score_response


def _case(case_id, domains, *, split="validation"):
    return {
        "case_id": case_id,
        "split": split,
        "message": case_id,
        "expected": {
            "domains": domains,
            "action": "other",
            "escalated": False,
        },
    }


def test_three_way_fusion_combines_per_label_scores_before_thresholding():
    rows = {
        "one": {
            "llm": {"technical": 0.7, "billing": 0.75, "account": 0.0},
            "embedding": {"technical": 0.8, "billing": 0.1, "account": 0.0},
            "pattern": {"technical": 1.0, "billing": 0.0, "account": 0.0},
        }
    }
    config = FusionConfig(
        llm_weight=0.7,
        embedding_weight=0.2,
        pattern_weight=0.1,
        thresholds={"technical": 0.6, "billing": 0.6, "account": 0.6},
    )

    predictions = fuse_predictions(rows, config)

    assert predictions["one"]["domains"] == ["technical"]
    assert predictions["one"]["scores"]["technical"] == 0.75
    assert predictions["one"]["scores"]["billing"] == 0.545


def test_grid_search_selects_weights_and_thresholds_from_validation_only():
    cases = [
        _case("v-tech", ["technical"]),
        _case("v-billing", ["billing"]),
        _case("v-general", ["general"]),
        _case("t-account", ["account"], split="test"),
    ]
    rows = {
        "v-tech": {
            "llm": {"technical": 0.9},
            "embedding": {"technical": 0.1},
            "pattern": {"technical": 0.0},
        },
        "v-billing": {
            "llm": {"billing": 0.9},
            "embedding": {"billing": 0.1},
            "pattern": {"billing": 0.0},
        },
        "v-general": {"llm": {}, "embedding": {}, "pattern": {}},
        "t-account": {
            "llm": {},
            "embedding": {"account": 1.0},
            "pattern": {"account": 1.0},
        },
    }

    first = tune_fusion(
        cases,
        rows,
        weight_values=(0.0, 0.5, 1.0),
        threshold_values=(0.4, 0.6),
    )
    changed_test = [*cases[:-1], _case("t-account", ["general"], split="test")]
    second = tune_fusion(
        changed_test,
        rows,
        weight_values=(0.0, 0.5, 1.0),
        threshold_values=(0.4, 0.6),
    )

    assert first.selection_case_ids == ["v-tech", "v-billing", "v-general"]
    assert first.config == second.config
    assert first.config.llm_weight == 1.0


def test_comparison_uses_the_same_heldout_cases_for_every_variant():
    cases = [
        _case("v1", ["technical"]),
        _case("t1", ["technical"], split="test"),
        _case("t2", ["billing"], split="test"),
    ]
    rows = {
        "v1": {
            "llm": {"technical": 0.9},
            "embedding": {"technical": 0.8},
            "pattern": {"technical": 1.0},
            "llm_domains": ["technical"],
        },
        "t1": {
            "llm": {"technical": 0.9},
            "embedding": {"technical": 0.8},
            "pattern": {"technical": 1.0},
            "llm_domains": ["technical"],
        },
        "t2": {
            "llm": {"billing": 0.9},
            "embedding": {"billing": 0.8},
            "pattern": {"billing": 1.0},
            "llm_domains": ["billing"],
        },
    }

    report = compare_fusion_variants(
        cases,
        rows,
        weight_values=(0.0, 0.25, 0.5, 1.0),
        threshold_values=(0.4, 0.6),
    )

    assert report["test_case_ids"] == ["t1", "t2"]
    assert set(report["variants"]) >= {
        "deepseek_only",
        "runtime_85_0_15",
        "fixed_70_20_10",
        "ablation_without_pattern",
        "ablation_without_embedding",
        "ablation_without_llm",
        "grid_search_best",
        "grid_search_best_true_three_way",
    }
    assert all(row["metrics"]["total"] == 2 for row in report["variants"].values())
    assert report["variants"]["runtime_85_0_15"]["config"]["llm_weight"] == 0.85
    assert report["variants"]["fixed_70_20_10"]["config"]["embedding_weight"] == 0.2
    assert report["variants"]["ablation_without_pattern"]["config"]["pattern_weight"] == 0.0
    assert report["variants"]["ablation_without_embedding"]["config"]["embedding_weight"] == 0.0
    assert report["variants"]["ablation_without_llm"]["config"]["llm_weight"] == 0.0
    assert set(report["error_analysis"]) == {
        "runtime_85_0_15",
        "fixed_70_20_10",
        "ablation_without_pattern",
        "ablation_without_embedding",
        "ablation_without_llm",
        "grid_search_best",
        "grid_search_best_true_three_way",
    }


def test_llm_score_parser_requires_every_case_and_keeps_domain_confidence():
    raw = json.dumps({
        "predictions": [
            {
                "case_id": "a",
                "domains": ["technical", "billing"],
                "domain_scores": {
                    "technical": 0.95,
                    "billing": 0.82,
                    "account": 0.03,
                },
            }
        ]
    })

    parsed = parse_domain_score_response(raw, expected_ids={"a"})

    assert parsed["a"]["llm_domains"] == ["technical", "billing"]
    assert parsed["a"]["llm"]["billing"] == 0.82


def test_score_collection_times_out_a_stuck_batch_and_splits_it(tmp_path):
    class StuckBatchClassifier:
        async def classify_batch(self, cases):
            if len(cases) > 1:
                await asyncio.sleep(0.2)
            return {
                case["case_id"]: {
                    "llm_domains": ["technical"],
                    "llm": {"technical": 0.9, "billing": 0.0, "account": 0.0},
                }
                for case in cases
            }

    dataset = tmp_path / "cases.json"
    dataset.write_text(json.dumps({"cases": [
        _case("one", ["technical"]),
        _case("two", ["technical"]),
    ]}), encoding="utf-8")
    args = argparse.Namespace(
        dataset=dataset,
        llm_scores=tmp_path / "scores.json",
        batch_size=2,
        concurrency=1,
        max_attempts=1,
        request_timeout=0.01,
    )

    asyncio.run(asyncio.wait_for(
        collect(args, classifier=StuckBatchClassifier()),
        timeout=0.1,
    ))

    saved = json.loads(args.llm_scores.read_text(encoding="utf-8"))
    assert set(saved["predictions"]) == {"one", "two"}


def test_score_collection_rejects_a_second_writer_for_the_same_checkpoint(tmp_path):
    class BlockingClassifier:
        def __init__(self):
            self.calls = 0
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def classify_batch(self, cases):
            self.calls += 1
            if self.calls == 1:
                self.started.set()
                await self.release.wait()
            return {
                case["case_id"]: {
                    "llm_domains": ["technical"],
                    "llm": {"technical": 0.9, "billing": 0.0, "account": 0.0},
                }
                for case in cases
            }

    dataset = tmp_path / "cases.json"
    dataset.write_text(json.dumps({"cases": [_case("one", ["technical"])]}), encoding="utf-8")
    args = argparse.Namespace(
        dataset=dataset,
        llm_scores=tmp_path / "scores.json",
        batch_size=1,
        concurrency=1,
        max_attempts=1,
        request_timeout=1.0,
    )

    async def scenario():
        classifier = BlockingClassifier()
        first = asyncio.create_task(collect(args, classifier=classifier))
        await classifier.started.wait()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                await collect(args, classifier=classifier)
        finally:
            classifier.release.set()
            await first

    asyncio.run(scenario())
