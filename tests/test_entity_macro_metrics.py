import pandas as pd
import pytest
from src.entity_resolution.evaluation.candidate_metrics import entity_macro_fbeta


def test_correct_empty_prediction_for_singleton_scores_one():
    gt = pd.DataFrame({"source1_entity_id": ["S1-1", "S1-2"], "matched_entity_ids": ["", "S2-1"]})
    candidates = pd.DataFrame([{"source1_entity_id": "S1-2", "candidate_entity_id": "S2-1", "candidate_source": "source2", "match_probability": 0.99}])
    result = entity_macro_fbeta(gt, candidates, 0.9)
    assert result["macro_f0.5"] == pytest.approx(1.0)


def test_false_positive_for_singleton_scores_zero():
    gt = pd.DataFrame({"source1_entity_id": ["S1-1", "S1-2"], "matched_entity_ids": ["", "S2-1"]})
    candidates = pd.DataFrame([
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-9", "candidate_source": "source2", "match_probability": 0.95},
        {"source1_entity_id": "S1-2", "candidate_entity_id": "S2-1", "candidate_source": "source2", "match_probability": 0.95},
    ])
    result = entity_macro_fbeta(gt, candidates, 0.9)
    assert result["macro_f0.5"] == pytest.approx(0.5)


def test_missing_true_candidate_is_counted_as_false_negative():
    gt = pd.DataFrame({"source1_entity_id": ["S1-1"], "matched_entity_ids": ["S2-1"]})
    candidates = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "candidate_source", "match_probability"])
    result = entity_macro_fbeta(gt, candidates, 0.9)
    assert result["macro_f0.5"] == 0.0


def test_invalid_probability_threshold_rejected():
    gt = pd.DataFrame({"source1_entity_id": ["S1-1"], "matched_entity_ids": [""]})
    candidates = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "candidate_source", "match_probability"])
    with pytest.raises(ValueError):
        entity_macro_fbeta(gt, candidates, 1.1)
