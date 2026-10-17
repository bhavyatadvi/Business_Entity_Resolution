
from collections import defaultdict
from itertools import product

import pandas as pd

from src.entity_resolution.blocking.blocking_rules import (
    generate_blocking_keys,
)
from src.entity_resolution.evaluation.candidate_metrics import (
    parse_ground_truth,
)


def analyze_blocking_key_impact(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    """Measure candidate volume and known-match coverage per key."""

    true_pairs = set(
        parse_ground_truth(ground_truth)[
            [
                "source1_entity_id",
                "candidate_entity_id",
                "candidate_source",
            ]
        ].itertuples(index=False, name=None)
    )

    records_by_key = defaultdict(
        lambda: {
            "source1": set(),
            "source2": set(),
            "source3": set(),
        }
    )

    def collect_keys(frame, source_name):
        for record in frame.to_dict(orient="records"):
            entity_id = str(record["entity_id"])

            # Avoid counting a record more than once for one key.
            for key in set(generate_blocking_keys(record)):
                records_by_key[key][source_name].add(entity_id)

    collect_keys(source1, "source1")
    collect_keys(source2, "source2")
    collect_keys(source3, "source3")

    rows = []

    for key, groups in records_by_key.items():
        source1_ids = groups["source1"]

        counts = {
            "source1_count": len(source1_ids),
            "source2_count": len(groups["source2"]),
            "source3_count": len(groups["source3"]),
        }

        pair_contribution = (
            counts["source1_count"] * counts["source2_count"]
            + counts["source1_count"] * counts["source3_count"]
        )

        true_pairs_covered = 0

        # Count known true pairs this key could generate.
        for target_source in ("source2", "source3"):
            for source1_id, target_id in product(
                source1_ids,
                groups[target_source],
            ):
                pair = (
                    source1_id,
                    target_id,
                    target_source,
                )

                if pair in true_pairs:
                    true_pairs_covered += 1

        rows.append({
            "blocking_key": str(key),
            **counts,
            "pair_contribution": pair_contribution,
            "true_pairs_covered": true_pairs_covered,
            "known_match_yield": (
                true_pairs_covered / pair_contribution
                if pair_contribution else 0.0
            ),
        })

    return pd.DataFrame(rows).sort_values(
        "pair_contribution",
        ascending=False,
    ).reset_index(drop=True)