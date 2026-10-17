
from pathlib import Path

import pandas as pd

from src.entity_resolution.data.data_loader import (
    DATASET_ROOT,
    load_ground_truth,
)


def audit_ground_truth(sample_size: int = 200) -> dict:
    """
    Validate a sample of ground-truth references using chunked reads.

    Only entity_id columns are read from the source files, keeping
    memory usage lower than loading complete datasets.
    """
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")

    ground_truth = load_ground_truth(nrows=sample_size)

    required_ids = {
        "source1": set(ground_truth["source1_entity_id"].dropna()),
        "source2": set(),
        "source3": set(),
    }

    for matched_ids in ground_truth["matched_entity_ids"]:
        if not matched_ids:
            continue

        for entity_id in str(matched_ids).split(","):
            entity_id = entity_id.strip()

            if entity_id.startswith("S2-"):
                required_ids["source2"].add(entity_id)
            elif entity_id.startswith("S3-"):
                required_ids["source3"].add(entity_id)

    results = {}

    for source, expected_ids in required_ids.items():
        path = DATASET_ROOT / "train" / f"train_{source}.tsv"
        found_ids = set()

        # Read IDs in chunks rather than loading all source columns.
        for chunk in pd.read_csv(
            path,
            sep="\t",
            usecols=["entity_id"],
            dtype=str,
            keep_default_na=False,
            chunksize=100_000,
        ):
            found_ids.update(
                set(chunk["entity_id"]) & expected_ids
            )

            if found_ids == expected_ids:
                break

        missing_ids = expected_ids - found_ids

        results[source] = {
            "referenced_ids": len(expected_ids),
            "found_ids": len(found_ids),
            "missing_ids": sorted(missing_ids),
        }

    return results


if __name__ == "__main__":
    report = audit_ground_truth(sample_size=200)

    print("\nGROUND-TRUTH DATA AUDIT")
    print("=" * 45)

    for source, result in report.items():
        print(f"\n{source.upper()}")
        print(f"Referenced IDs: {result['referenced_ids']:,}")
        print(f"Found IDs:      {result['found_ids']:,}")
        print(f"Missing IDs:    {len(result['missing_ids']):,}")

        if result["missing_ids"]:
            print("Examples:", result["missing_ids"][:10])

    if all(not result["missing_ids"] for result in report.values()):
        print("\nAudit passed: all sampled references were found.")
    else:
        print("\nAudit found missing references. Investigate before proceeding.")
