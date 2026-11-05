
import csv
from itertools import zip_longest
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SOURCE1 = ROOT / "student_resource/dataset/test/test_source1.tsv"
CANDIDATES = ROOT / "output/candidate_pairs.tsv"
MATCHES = ROOT / "output/matching_results.tsv"

EXPECTED_CANDIDATE_HEADER = [
    "source1_entity_id",
    "candidate_entity_ids",
]
EXPECTED_MATCH_HEADER = [
    "source1_entity_id",
    "matched_entity_ids",
]

stats = {
    "rows": 0,
    "empty_candidates": 0,
    "empty_matches": 0,
    "candidate_pairs": 0,
    "matched_pairs": 0,
}

def parse_ids(value):
    return [x.strip() for x in value.split(",") if x.strip()]

with (
    open(SOURCE1, encoding="utf-8", newline="") as sf,
    open(CANDIDATES, encoding="utf-8", newline="") as cf,
    open(MATCHES, encoding="utf-8", newline="") as mf,
):
    source_reader = csv.DictReader(sf, delimiter="\t")
    candidate_reader = csv.DictReader(cf, delimiter="\t")
    match_reader = csv.DictReader(mf, delimiter="\t")

    assert candidate_reader.fieldnames == EXPECTED_CANDIDATE_HEADER, (
        f"Invalid candidate header: {candidate_reader.fieldnames}"
    )
    assert match_reader.fieldnames == EXPECTED_MATCH_HEADER, (
        f"Invalid matching header: {match_reader.fieldnames}"
    )

    sentinel = object()

    for source, candidate, match in zip_longest(
        source_reader,
        candidate_reader,
        match_reader,
        fillvalue=sentinel,
    ):
        if sentinel in (source, candidate, match):
            raise ValueError(
                f"Different row counts near row {stats['rows'] + 1}"
            )

        sid = source["entity_id"]
        candidate_sid = candidate["source1_entity_id"]
        match_sid = match["source1_entity_id"]

        if sid != candidate_sid or sid != match_sid:
            raise ValueError(
                f"Source 1 ID/order mismatch at row {stats['rows'] + 1}"
            )

        candidate_ids = parse_ids(candidate["candidate_entity_ids"])
        matched_ids = parse_ids(match["matched_entity_ids"])

        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"Duplicate candidate IDs for {sid}")

        if len(matched_ids) != len(set(matched_ids)):
            raise ValueError(f"Duplicate matched IDs for {sid}")

        invalid_candidates = [
            x for x in candidate_ids
            if not x.startswith(("S2-", "S3-"))
        ]

        invalid_matches = [
            x for x in matched_ids
            if not x.startswith(("S2-", "S3-"))
        ]

        if invalid_candidates or invalid_matches:
            raise ValueError(f"Invalid target ID for {sid}")

        missing = set(matched_ids) - set(candidate_ids)

        if missing:
            raise ValueError(
                f"Predicted matches not in candidate list for {sid}: "
                f"{list(missing)[:5]}"
            )

        stats["rows"] += 1
        stats["candidate_pairs"] += len(candidate_ids)
        stats["matched_pairs"] += len(matched_ids)

        if not candidate_ids:
            stats["empty_candidates"] += 1

        if not matched_ids:
            stats["empty_matches"] += 1

print("\n========== OUTPUT VALIDATION ==========")
for key, value in stats.items():
    print(f"{key}: {value:,}")

print("\nSUCCESS: Both files are structurally consistent.")
