import csv
import shutil
from datetime import datetime
from pathlib import Path

candidate = Path("output/candidate_pairs.tsv")
matching = Path("output/matching_results.tsv")

candidate_header = ["source1_entity_id", "candidate_entity_ids"]
matching_header = ["source1_entity_id", "matched_entity_ids"]

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
candidate_backup = candidate.with_name(f"candidate_before_sync_{stamp}.tsv")
matching_backup = matching.with_name(f"matching_before_sync_{stamp}.tsv")
candidate_temp = candidate.with_name("candidate_sync.tmp")
matching_temp = matching.with_name("matching_sync.tmp")

def read_ids(path, expected_header):
    ids = set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header != expected_header:
            raise RuntimeError(f"Unexpected header in {path}: {header}")

        for line_number, row in enumerate(reader, start=2):
            if len(row) != 2 or not row[0]:
                raise RuntimeError(
                    f"Malformed row in {path}, line {line_number}"
                )
            if row[0] in ids:
                raise RuntimeError(
                    f"Duplicate ID {row[0]} in {path}"
                )
            ids.add(row[0])
    return ids

candidate_ids = read_ids(candidate, candidate_header)
matching_ids = read_ids(matching, matching_header)
common_ids = candidate_ids & matching_ids

print("Candidate IDs:", len(candidate_ids))
print("Matching IDs:", len(matching_ids))
print("Candidate-only IDs:", len(candidate_ids - matching_ids))
print("Matching-only IDs:", len(matching_ids - candidate_ids))
print("IDs retained in both:", len(common_ids))

if candidate_ids == matching_ids:
    print("Files already synchronized. No repair needed.")
    raise SystemExit(0)

# Preserve the originals before making any changes.
shutil.copy2(candidate, candidate_backup)
shutil.copy2(matching, matching_backup)

try:
    with candidate.open("r", encoding="utf-8-sig", newline="") as src, \
         candidate_temp.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.reader(src, delimiter="\t")
        writer = csv.writer(dst, delimiter="\t", lineterminator="\n")
        writer.writerow(next(reader))
        for row in reader:
            if row[0] in common_ids:
                writer.writerow(row)

    with matching.open("r", encoding="utf-8-sig", newline="") as src, \
         matching_temp.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.reader(src, delimiter="\t")
        writer = csv.writer(dst, delimiter="\t", lineterminator="\n")
        writer.writerow(next(reader))
        for row in reader:
            if row[0] in common_ids:
                writer.writerow(row)

    # Verify the temporary files before replacing the originals.
    if read_ids(candidate_temp, candidate_header) != common_ids:
        raise RuntimeError("Candidate verification failed.")
    if read_ids(matching_temp, matching_header) != common_ids:
        raise RuntimeError("Matching verification failed.")

    candidate_temp.replace(candidate)
    matching_temp.replace(matching)

except Exception:
    print("Repair failed. Backups are available; do not resume inference.")
    raise
finally:
    for path in (candidate_temp, matching_temp):
        if path.exists():
            path.unlink()

print("\nCheckpoint synchronized successfully.")
print("Rows retained in each file:", len(common_ids))
print("Candidate backup:", candidate_backup)
print("Matching backup:", matching_backup)
