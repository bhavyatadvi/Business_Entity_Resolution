
import pandas as pd
import pytest

from src.entity_resolution.matching.score_candidates import (
    score_candidate_pairs,
)


def make_source(records):
    return pd.DataFrame(
        records,
        columns=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )


def test_scores_and_ranks_candidate_pairs():
    source1 = make_source([
        ("S1-1", "Prime Money", "10 Ellis Road", "US"),
    ])
    source2 = make_source([
        ("S2-1", "Ocean Restaurant", "90 Lake Street", "US"),
        ("S2-2", "Prime Money", "10 Ellis Road", "US"),
    ])
    source3 = make_source([
        ("S3-1", "Prime Money", "10 Ellis Road", "US"),
    ])

    candidates = pd.DataFrame([
        {
            "source1_entity_id": "S1-1",
            "candidate_entity_id": "S2-1",
            "candidate_source": "source2",
            "shared_key_count": 1,
        },
        {
            "source1_entity_id": "S1-1",
            "candidate_entity_id": "S2-2",
            "candidate_source": "source2",
            "shared_key_count": 2,
        },
        {
            "source1_entity_id": "S1-1",
            "candidate_entity_id": "S3-1",
            "candidate_source": "source3",
            "shared_key_count": 2,
        },
    ])

    result = score_candidate_pairs(
        source1, source2, source3, candidates
    )

    exact_match = result[
        result["candidate_entity_id"] == "S2-2"
    ].iloc[0]

    unrelated = result[
        result["candidate_entity_id"] == "S2-1"
    ].iloc[0]

    assert exact_match["combined_score"] == 100.0
    assert exact_match["combined_score"] > unrelated["combined_score"]
    assert len(result) == 3


def test_empty_candidates_return_expected_columns():
    source1 = make_source([
        ("S1-1", "Prime Money", "", "US"),
    ])
    source2 = make_source([
        ("S2-1", "Prime Money", "", "US"),
    ])
    source3 = make_source([
        ("S3-1", "Prime Money", "", "US"),
    ])

    candidates = pd.DataFrame(columns=[
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
        "shared_key_count",
    ])

    result = score_candidate_pairs(
        source1, source2, source3, candidates
    )

    assert result.empty
    assert "combined_score" in result.columns
    assert "name_similarity" in result.columns


def test_missing_candidate_record_raises_error():
    source1 = make_source([
        ("S1-1", "Prime Money", "", "US"),
    ])
    source2 = make_source([
        ("S2-1", "Prime Money", "", "US"),
    ])
    source3 = make_source([
        ("S3-1", "Prime Money", "", "US"),
    ])

    candidates = pd.DataFrame([{
        "source1_entity_id": "S1-1",
        "candidate_entity_id": "S2-999",
        "candidate_source": "source2",
    }])

    with pytest.raises(ValueError, match="Candidate record not found"):
        score_candidate_pairs(
            source1, source2, source3, candidates
        )


def test_invalid_candidate_source_raises_error():
    source1 = make_source([
        ("S1-1", "Prime Money", "", "US"),
    ])
    source2 = make_source([
        ("S2-1", "Prime Money", "", "US"),
    ])
    source3 = make_source([
        ("S3-1", "Prime Money", "", "US"),
    ])

    candidates = pd.DataFrame([{
        "source1_entity_id": "S1-1",
        "candidate_entity_id": "S2-1",
        "candidate_source": "source4",
    }])

    with pytest.raises(ValueError, match="candidate_source"):
        score_candidate_pairs(
            source1, source2, source3, candidates
        )
