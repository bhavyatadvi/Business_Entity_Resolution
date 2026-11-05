import csv
from pathlib import Path

files = {
    "candidates": Path("output/candidate_pairs.tsv"),
    "matches": Path("output/matching_results.tsv"),
}

def inspect(path):
    ids = set()
    duplicates = []
    malformed = []
    row_count = 0

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        print(f"\nFile: {path}")
        print("Header:", header)

        expected = ["source1_entity_id",
                    "candidate_entity_ids" if "candidate" in path.name
                    else "matched_entity_ids"]

        if header != expected:
            print("ERROR: Unexpected header!")
            return None

        for line_number, row in enumerate(reader, start=2):
            row_count += 1
            if len(row) != 2 or not row[0]:
                malformed.append(line_number)
                continue
            entity_id = row[0]
            if entity_id in ids:
                duplicates.append(entity_id)
            ids.add(entity_id)

    print("Data rows:", row_count)
    print("Unique Source 1 IDs:", len(ids))
    print("Duplicate IDs:", len(duplicates))
    print("Malformed rows:", len(malformed))
    if duplicates:
        print("Duplicate examples:", duplicates[:5])
    if malformed:
        print("Malformed line examples:", malformed[:5])
    print("First 5 IDs:", list(ids)[:5])
    return ids

candidate_ids = inspect(files["candidates"])
matching_ids = inspect(files["matches"])

if candidate_ids is not None and matching_ids is not None:
    print("\n--- COMPARISON ---")
    print("Only in candidate file:", len(candidate_ids - matching_ids))
    print("Only in matching file:", len(matching_ids - candidate_ids))
    print("Common IDs:", len(candidate_ids & matching_ids))
    print("Candidate-only examples:", list(candidate_ids - matching_ids)[:10])
    print("Matching-only examples:", list(matching_ids - candidate_ids)[:10])
