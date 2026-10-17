
import pandas as pd
import pytest

from src.entity_resolution.blocking.candidate_generator import (
    generate_candidate_pairs,
    generate_candidates,
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


def test_generates_matching_name_candidates():
    source1 = make_source([
        ("S1-1", "Prime Money", "10 Ellis Road", "US"),
    ])
    source2 = make_source([
        ("S2-1", "Prime Money Inc", "10 Ellis Road", "US"),
        ("S2-2", "Ocean Restaurant", "90 Lake Street", "US"),
    ])

    result = generate_candidate_pairs(source1, source2, "source2")

    assert len(result) >= 1
    assert (
        (result["source1_entity_id"] == "S1-1")
        & (result["candidate_entity_id"] == "S2-1")
    ).any()


def test_does_not_duplicate_candidate_pairs():
    source1 = make_source([
        ("S1-1", "Prime Money", "10 Ellis Road", "US"),
    ])
    source2 = make_source([
        ("S2-1", "Prime Money", "10 Ellis Road", "US"),
    ])

    result = generate_candidate_pairs(source1, source2, "source2")

    assert not result.duplicated(
        ["source1_entity_id", "candidate_entity_id"]
    ).any()


def test_skips_oversized_blocks():
    source1 = make_source([
        ("S1-1", "Prime Money", "", "US"),
    ])
    source2 = make_source([
        ("S2-1", "Prime Money", "", "US"),
        ("S2-2", "Prime Money", "", "US"),
    ])

    result = generate_candidate_pairs(
        source1, source2, "source2", max_block_size=1
    )

    assert result.empty


def test_returns_empty_result_when_no_keys_overlap():
    source1 = make_source([
        ("S1-1", "Alpha", "", "US"),
    ])
    source2 = make_source([
        ("S2-1", "Beta", "", "US"),
    ])

    result = generate_candidate_pairs(source1, source2, "source2")

    assert result.empty
    assert list(result.columns) == [
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
        "shared_key_count",
    ]


def test_generates_candidates_for_both_sources():
    source1 = make_source([
        ("S1-1", "Prime Money", "10 Ellis Road", "US"),
    ])
    source2 = make_source([
        ("S2-1", "Prime Money Inc", "10 Ellis Road", "US"),
    ])
    source3 = make_source([
        ("S3-1", "Prime Money", "10 Ellis Road", "US"),
    ])

    result = generate_candidates(source1, source2, source3)

    assert "source2" in result
    assert "source3" in result
    assert not result["source2"].empty
    assert not result["source3"].empty


def test_rejects_invalid_target_source():
    source1 = make_source([
        ("S1-1", "Prime Money", "", "US"),
    ])

    with pytest.raises(ValueError):
        generate_candidate_pairs(source1, source1, "source4")

