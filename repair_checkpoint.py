import csv
import shutil
from datetime import datetime
from pathlib import Path

candidate_path = Path("output/candidate_pairs.tsv")
matching_path = Path("output/matching_results.tsv")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
candidate_backup = candidate_path.with_name(
    f"candidate_pairs_before_repair_{timestamp}.tsv"
)
matching_backup = matching_path.with_name(
    f"matching_results_before_repair_{timestamp}.tsv"
)
temp_path = candidate_path.with_name("candidate_pairs_repaired.tmp")

def read_ids(path, expected_header):
    ids = set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header != expected_header:
            raise RuntimeError(f"Unexpected header in {path}: {header}")

        for line_no, row in enumerate(reader, start=2):
            if len(row) != 2 or not row[0]:
                raise RuntimeError(
                    f"Malformed row in {path} at line {line_no}"
                )
            if row[0] in ids:
                raise RuntimeError(
                    f"Duplicate ID {row[0]} in {path}"
                )
            ids.add(row[0])
    return ids

candidate_header = ["source1_entity_id", "candidate_entity_ids"]
matching_header = ["source1_entity_id", "matched_entity_ids"]

candidate_ids = read_ids(candidate_path, candidate_header)
matching_ids = read_ids(matching_path, matching_header)

candidate_only = candidate_ids - matching_ids
matching_only = matching_ids - candidate_ids

print("Candidate rows:", len(candidate_ids))
print("Matching rows:", len(matching_ids))
print("Candidate-only IDs:", len(candidate_only))
print("Matching-only IDs:", len(matching_only))

if matching_only:
    raise RuntimeError(
        "Matching file contains extra IDs. No files were changed."
    )

if not candidate_only:
    print("Files are already synchronized. No repair needed.")
    raise SystemExit(0)

# Preserve both original files before changing anything.
shutil.copy2(candidate_path, candidate_backup)
shutil.copy2(matching_path, matching_backup)

removed = 0
kept = 0

try:
    with candidate_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as source, temp_path.open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        reader = csv.reader(source, delimiter="\t")
        writer = csv.writer(
            destination, delimiter="\t", lineterminator="\n"
        )

        header = next(reader, None)
        if header != candidate_header:
            raise RuntimeError("Candidate header changed unexpectedly.")

        writer.writerow(header)

        for row in reader:
            if len(row) != 2 or not row[0]:
                raise RuntimeError("Malformed candidate row found.")

            if row[0] in matching_ids:
                writer.writerow(row)
                kept += 1
            else:
                removed += 1

    repaired_ids = read_ids(temp_path, candidate_header)

    if repaired_ids != matching_ids:
        raise RuntimeError(
            "Verification failed. Original candidate file was preserved."
        )

    if removed != len(candidate_only):
        raise RuntimeError(
            "Unexpected number of rows removed. Original preserved."
        )

    # Replace candidate file only after verification succeeds.
    temp_path.replace(candidate_path)

except Exception:
    if temp_path.exists():
        temp_path.unlink()
    raise

print("\nRepair successful.")
print("Candidate rows retained:", kept)
print("Candidate rows removed:", removed)
print("Both files now contain:", len(matching_ids), "matching Source 1 IDs")
print("Candidate backup:", candidate_backup)
print("Matching backup:", matching_backup)
