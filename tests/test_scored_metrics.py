
import pandas as pd
import pytest

from src.entity_resolution.evaluation.candidate_metrics import (
    evaluate_scored_candidates,
)


def make_ground_truth():
    return pd.DataFrame({
        "source1_entity_id": ["S1-1", "S1-2"],
        "matched_entity_ids": ["S2-1", "S3-1"],
    })


def make_candidates():
    return pd.DataFrame([
        {
            "source1_entity_id": "S1-1",
            "candidate_entity_id": "S2-1",
            "candidate_source": "source2",
            "combined_score": 98.0,
        },
        {
            "source1_entity_id": "S1-2",
            "candidate_entity_id": "S3-1",
            "candidate_source": "source3",
            "combined_score": 75.0,
        },
        {
            "source1_entity_id": "S1-1",
            "candidate_entity_id": "S2-9",
            "candidate_source": "source2",
            "combined_score": 90.0,
        },
    ])


def test_evaluates_precision_recall_and_f1():
    result = evaluate_scored_candidates(
        make_ground_truth(), make_candidates(), threshold=80
    )

    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)


def test_higher_threshold_can_reduce_false_positives():
    result = evaluate_scored_candidates(
        make_ground_truth(), make_candidates(), threshold=95
    )

    assert result["predicted_pairs"] == 1
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5


def test_threshold_zero_includes_all_candidates():
    result = evaluate_scored_candidates(
        make_ground_truth(), make_candidates(), threshold=0
    )

    assert result["predicted_pairs"] == 3
    assert result["true_positives"] == 2


def test_invalid_threshold_is_rejected():
    with pytest.raises(ValueError):
        evaluate_scored_candidates(
            make_ground_truth(), make_candidates(), threshold=101
        )
