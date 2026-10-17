
from collections import defaultdict

import pandas as pd

from src.entity_resolution.blocking.blocking_rules import (
    generate_blocking_keys,
)


CANDIDATE_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
    "shared_key_count",
]


def generate_candidate_pairs(
    source1: pd.DataFrame,
    target: pd.DataFrame,
    target_source: str,
    max_block_size: int = 500,
) -> pd.DataFrame:
    """
    Generate candidate pairs between Source 1 and one target source.

    Blocks containing more than max_block_size target records are skipped
    to prevent extremely common keys from producing huge numbers of pairs.

    This function generates candidates, not final match decisions.
    """
    if target_source not in {"source2", "source3"}:
        raise ValueError("target_source must be 'source2' or 'source3'")

    if max_block_size < 1:
        raise ValueError("max_block_size must be at least 1")

    required_columns = {"entity_id", "business_name", "business_address"}

    for label, df in (("source1", source1), (target_source, target)):
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"{label} is missing columns: {sorted(missing)}"
            )

        if df["entity_id"].isna().any():
            raise ValueError(f"{label} contains missing entity IDs")

        if df["entity_id"].duplicated().any():
            raise ValueError(f"{label} contains duplicate entity IDs")

    # Build an inverted index: blocking key -> target record positions.
    index = defaultdict(list)
    target_keys = []

    for position, record in enumerate(target.to_dict("records")):
        keys = generate_blocking_keys(record)
        target_keys.append(keys)

        for key in keys:
            index[key].append(position)

    # Ignore overly common keys to limit candidate explosion.
    usable_keys = {
        key: positions
        for key, positions in index.items()
        if len(positions) <= max_block_size
    }

    # Map each pair to the number of distinct keys they share.
    pair_counts = defaultdict(int)
    target_ids = target["entity_id"].tolist()

    for record in source1.to_dict("records"):
        source1_id = record["entity_id"]

        for key in generate_blocking_keys(record):
            positions = usable_keys.get(key, [])

            for position in positions:
                candidate_id = target_ids[position]
                pair = (source1_id, candidate_id)

                # Each record generates unique blocking keys, so a pair
                # is counted once per shared key.
                pair_counts[pair] += 1

    if not pair_counts:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)

    result = pd.DataFrame(
        [
            {
                "source1_entity_id": source1_id,
                "candidate_entity_id": candidate_id,
                "candidate_source": target_source,
                "shared_key_count": count,
            }
            for (source1_id, candidate_id), count in pair_counts.items()
        ],
        columns=CANDIDATE_COLUMNS,
    )

    return result.sort_values(
        ["source1_entity_id", "shared_key_count"],
        ascending=[True, False],
        ignore_index=True,
    )


def generate_candidates(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    max_block_size: int = 500,
) -> dict[str, pd.DataFrame]:
    """Generate Source 1 candidates against Source 2 and Source 3."""
    source2_candidates = generate_candidate_pairs(
        source1, source2, "source2", max_block_size
    )

    source3_candidates = generate_candidate_pairs(
        source1, source3, "source3", max_block_size
    )

    return {
        "source2": source2_candidates,
        "source3": source3_candidates,
    }
