
from collections import defaultdict

import pandas as pd

from src.entity_resolution.blocking.blocking_rules import (
    generate_blocking_keys,
)


def diagnose_blocking_rules(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    max_block_size: int = 500,
) -> pd.DataFrame:
    """
    Measure how many candidate pairs each blocking key would produce.

    A key shared by n Source 1 records and m target records can
    produce n * m cross-source pairs.

    This is a diagnostic only; it does not generate or remove pairs.
    """

    if max_block_size < 1:
        raise ValueError("max_block_size must be positive")

    source_frames = {
        "source1": source1,
        "source2": source2,
        "source3": source3,
    }

    key_counts = defaultdict(
        lambda: {
            "source1_count": 0,
            "source2_count": 0,
            "source3_count": 0,
        }
    )

    for source_name, frame in source_frames.items():
        for record in frame.to_dict(orient="records"):
            # Each record contributes at most once per key.
            keys = set(generate_blocking_keys(record))

            for key in keys:
                key_counts[key][f"{source_name}_count"] += 1

    rows = []

    for key, counts in key_counts.items():
        n1 = counts["source1_count"]
        n2 = counts["source2_count"]
        n3 = counts["source3_count"]

        pairs_s2 = n1 * n2
        pairs_s3 = n1 * n3

        rows.append({
            "blocking_key": str(key),
            "source1_count": n1,
            "source2_count": n2,
            "source3_count": n3,
            "source2_pair_contribution": pairs_s2,
            "source3_pair_contribution": pairs_s3,
            "total_pair_contribution": pairs_s2 + pairs_s3,
            "oversized": (
                n1 > max_block_size
                or n2 > max_block_size
                or n3 > max_block_size
            ),
        })

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return result.sort_values(
        "total_pair_contribution",
        ascending=False,
    ).reset_index(drop=True)