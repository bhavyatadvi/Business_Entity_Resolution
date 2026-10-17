
import pandas as pd
import pytest

from src.entity_resolution.evaluation.candidate_metrics import (
    parse_ground_truth,
    evaluate_candidate_recall,
)


def test_parses_comma_separated_matches():
    ground_truth = pd.DataFrame({
        "source1_entity_id": ["S1-1"],
        "matched_entity_ids": ["S2-1,S2-2,S3-1"],
    })

    pairs = parse_ground_truth(ground_truth)

    assert len(pairs) == 3
    assert set(pairs["candidate_source"]) == {"source2", "source3"}


def test_removes_duplicate_ground_truth_pairs():
    ground_truth = pd.DataFrame({
        "source1_entity_id": ["S1-1", "S1-1"],
        "matched_entity_ids": ["S2-1", "S2-1"],
    })

    pairs = parse_ground_truth(ground_truth)

    assert len(pairs) == 1


def test_calculates_candidate_recall():
    ground_truth = pd.DataFrame({
        "source1_entity_id": ["S1-1", "S1-2"],
        "matched_entity_ids": ["S2-1,S3-1", "S2-2"],
    })

    candidates = pd.DataFrame({
        "source1_entity_id": ["S1-1", "S1-2"],
        "candidate_entity_id": ["S2-1", "S2-2"],
        "candidate_source": ["source2", "source2"],
    })

    result = evaluate_candidate_recall(ground_truth, candidates)

    assert result["true_pairs"] == 3
    assert result["found_pairs"] == 2
    assert result["candidate_recall"] == pytest.approx(2 / 3)
    assert result["source2"]["recall"] == pytest.approx(1.0)
    assert result["source3"]["recall"] == pytest.approx(0.0)


def test_zero_recall_when_no_candidates_are_found():
    ground_truth = pd.DataFrame({
        "source1_entity_id": ["S1-1"],
        "matched_entity_ids": ["S2-1"],
    })

    candidates = pd.DataFrame(columns=[
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
    ])

    result = evaluate_candidate_recall(ground_truth, candidates)

    assert result["found_pairs"] == 0
    assert result["candidate_recall"] == 0.0


def test_handles_empty_ground_truth():
    ground_truth = pd.DataFrame({
        "source1_entity_id": pd.Series(dtype=str),
        "matched_entity_ids": pd.Series(dtype=str),
    })

    candidates = pd.DataFrame(columns=[
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
    ])

    result = evaluate_candidate_recall(ground_truth, candidates)

    assert result["true_pairs"] == 0
    assert result["candidate_recall"] == 0.0


def test_rejects_unexpected_matched_id():
    ground_truth = pd.DataFrame({
        "source1_entity_id": ["S1-1"],
        "matched_entity_ids": ["S4-1"],
    })

    with pytest.raises(ValueError):
        parse_ground_truth(ground_truth)
