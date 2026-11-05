
"""Memory-conscious inference for the Amazon ML Challenge.

Run from the project root:
    python -m src.entity_resolution.pipeline.run_inference

Smoke test:
    python -m src.entity_resolution.pipeline.run_inference \
        --max-source1 100 --max-target-rows 10000

The smoke test is only for checking the pipeline. Do not submit its output.
"""

import argparse
import csv
import sqlite3
from collections import defaultdict
from pathlib import Path

import joblib
import pandas as pd

from src.entity_resolution.data.data_loader import (
    PROJECT_ROOT,
    DATASET_ROOT,
    SOURCE_COLUMNS,
)
from src.entity_resolution.blocking.blocking_rules import (
    generate_blocking_keys,
)
from src.entity_resolution.matching.similarity import compare_records
from src.entity_resolution.matching.model import FEATURE_COLUMNS


CHUNK_SIZE = 20_000
SQL_BATCH_SIZE = 500

SOURCE_FILES = {
    "source2": "test_source2.tsv",
    "source3": "test_source3.tsv",
}


def read_chunks(path, max_rows=None):
    """Read a TSV in bounded chunks, optionally limiting rows for smoke tests."""
    kwargs = {
        "sep": "\t",
        "dtype": str,
        "keep_default_na": False,
        "chunksize": CHUNK_SIZE,
    }
    if max_rows is not None:
        kwargs["nrows"] = max_rows

    yield from pd.read_csv(path, **kwargs)


def create_database(connection):
    connection.executescript("""
        DROP TABLE IF EXISTS blocking_keys;
        DROP TABLE IF EXISTS records;
        DROP TABLE IF EXISTS key_counts;

        CREATE TABLE key_counts (
            source TEXT NOT NULL,
            key_type TEXT NOT NULL,
            key_value TEXT NOT NULL,
            frequency INTEGER NOT NULL,
            PRIMARY KEY (source, key_type, key_value)
        );

        CREATE TABLE records (
            source TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            business_name TEXT NOT NULL,
            business_address TEXT NOT NULL,
            country TEXT NOT NULL,
            PRIMARY KEY (source, entity_id)
        );

        CREATE TABLE blocking_keys (
            source TEXT NOT NULL,
            key_type TEXT NOT NULL,
            key_value TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            PRIMARY KEY (source, key_type, key_value, entity_id)
        );
    """)


def count_target_keys(connection, source, path, max_rows=None):
    """First pass: count each blocking key across the entire target source."""
    print(f"Counting blocking-key frequencies: {path.name}")

    upsert_sql = """
        INSERT INTO key_counts(source, key_type, key_value, frequency)
        VALUES (?, ?, ?, 1)
        ON CONFLICT(source, key_type, key_value)
        DO UPDATE SET frequency = frequency + 1
    """

    for chunk_number, chunk in enumerate(read_chunks(path, max_rows)):
        rows = []

        for record in chunk.to_dict("records"):
            for key_type, key_value in generate_blocking_keys(record):
                rows.append((source, key_type, key_value))

        if rows:
            connection.executemany(upsert_sql, rows)
            connection.commit()

        print(
            f"  Frequency pass chunk {chunk_number + 1}: "
            f"{len(chunk):,} records"
        )


def index_target_records(
    connection,
    source,
    path,
    max_block_size,
    max_rows=None,
):
    """Second pass: store records and only non-oversized blocking keys."""
    print(f"Building target index: {path.name}")

    record_sql = """
        INSERT INTO records(
            source, entity_id, business_name, business_address, country
        ) VALUES (?, ?, ?, ?, ?)
    """
    key_sql = """
        INSERT OR IGNORE INTO blocking_keys(
            source, key_type, key_value, entity_id
        ) VALUES (?, ?, ?, ?)
    """

    for chunk_number, chunk in enumerate(read_chunks(path, max_rows)):
        record_rows = []
        key_rows = []
        frequency_cache = {}

        for record in chunk.to_dict("records"):
            entity_id = str(record["entity_id"])

            record_rows.append((
                source,
                entity_id,
                str(record["business_name"]),
                str(record["business_address"]),
                str(record["country"]),
            ))

            for key_type, key_value in generate_blocking_keys(record):
                cache_key = (key_type, key_value)

                if cache_key not in frequency_cache:
                    result = connection.execute(
                        """
                        SELECT frequency
                        FROM key_counts
                        WHERE source = ? AND key_type = ? AND key_value = ?
                        """,
                        (source, key_type, key_value),
                    ).fetchone()

                    frequency_cache[cache_key] = (
                        result[0] if result else 0
                    )

                if frequency_cache[cache_key] <= max_block_size:
                    key_rows.append((
                        source,
                        key_type,
                        key_value,
                        entity_id,
                    ))

        connection.executemany(record_sql, record_rows)
        connection.executemany(key_sql, key_rows)
        connection.commit()

        print(
            f"  Index pass chunk {chunk_number + 1}: "
            f"{len(record_rows):,} records, "
            f"{len(key_rows):,} usable keys"
        )


def build_target_index(
    connection,
    max_block_size,
    max_target_rows=None,
):
    """Build a disk-backed index over both target sources."""
    create_database(connection)

    for source, filename in SOURCE_FILES.items():
        path = DATASET_ROOT / "test" / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing test dataset: {path}")

        count_target_keys(
            connection, source, path, max_target_rows
        )

    for source, filename in SOURCE_FILES.items():
        path = DATASET_ROOT / "test" / filename
        index_target_records(
            connection,
            source,
            path,
            max_block_size,
            max_target_rows,
        )

    connection.executescript("""
        CREATE INDEX IF NOT EXISTS idx_block_lookup
        ON blocking_keys(source, key_type, key_value);

        CREATE INDEX IF NOT EXISTS idx_record_source
        ON records(source, entity_id);
    """)
    connection.commit()

    for source in SOURCE_FILES:
        count = connection.execute(
            "SELECT COUNT(*) FROM records WHERE source = ?",
            (source,),
        ).fetchone()[0]
        print(f"Indexed {source}: {count:,} records")


def retrieve_candidate_counts(connection, source1_record):
    """Return candidate ID -> shared blocking-key count for both sources."""
    counts = {
        "source2": defaultdict(int),
        "source3": defaultdict(int),
    }

    keys = generate_blocking_keys(source1_record)

    for source in ("source2", "source3"):
        for key_type, key_value in keys:
            rows = connection.execute(
                """
                SELECT entity_id
                FROM blocking_keys
                WHERE source = ? AND key_type = ? AND key_value = ?
                """,
                (source, key_type, key_value),
            )

            for (candidate_id,) in rows:
                counts[source][candidate_id] += 1

    return counts


def fetch_records(connection, source, entity_ids):
    """Retrieve candidate records in small SQL batches."""
    records = {}

    for start in range(0, len(entity_ids), SQL_BATCH_SIZE):
        batch = entity_ids[start:start + SQL_BATCH_SIZE]
        placeholders = ",".join("?" for _ in batch)

        query = f"""
            SELECT entity_id, business_name, business_address, country
            FROM records
            WHERE source = ? AND entity_id IN ({placeholders})
        """

        for row in connection.execute(query, [source, *batch]):
            records[row["entity_id"]] = dict(row)

    return records


def score_one_source1(
    connection,
    source1_record,
    model,
    threshold,
):
    """Generate, score and classify candidates for one Source 1 entity."""
    source1_id = str(source1_record["entity_id"])
    candidate_counts = retrieve_candidate_counts(
        connection, source1_record
    )

    all_candidate_ids = []
    matched_ids = []

    for source in ("source2", "source3"):
        counts = candidate_counts[source]
        candidate_ids = sorted(counts)

        if not candidate_ids:
            continue

        target_records = fetch_records(
            connection, source, candidate_ids
        )

        feature_rows = []
        valid_ids = []

        for candidate_id in candidate_ids:
            target_record = target_records.get(candidate_id)
            if target_record is None:
                continue

            features = compare_records(
                source1_record,
                target_record,
            )
            features["shared_key_count"] = counts[candidate_id]

            feature_rows.append(features)
            valid_ids.append(candidate_id)

        if not feature_rows:
            continue

        feature_frame = pd.DataFrame(feature_rows)

        # Keep inference features in exactly the model's training order.
        feature_frame = feature_frame.reindex(
            columns=FEATURE_COLUMNS
        ).apply(pd.to_numeric, errors="coerce")

        probabilities = model.predict_proba(
            feature_frame
        )[:, 1]

        all_candidate_ids.extend(valid_ids)

        for candidate_id, probability in zip(
            valid_ids, probabilities
        ):
            if probability >= threshold:
                matched_ids.append(
                    (candidate_id, float(probability))
                )

    # Highest confidence first; IDs are unique because each candidate
    # belongs to exactly one target source.
    matched_ids.sort(key=lambda item: (-item[1], item[0]))

    return (
        all_candidate_ids,
        [candidate_id for candidate_id, _ in matched_ids],
    )


def count_tsv_data_rows(path):
    """Count data rows without loading the full TSV into memory."""
    newline_count = 0
    last_byte = b""

    with open(path, "rb") as file:
        while True:
            block = file.read(8 * 1024 * 1024)
            if not block:
                break
            newline_count += block.count(b"\n")
            last_byte = block[-1:]

    line_count = newline_count
    if last_byte and last_byte != b"\n":
        line_count += 1

    return max(0, line_count - 1)


def index_is_complete(connection, max_block_size):
    """Check whether the existing target index can be reused."""
    required_tables = {"records", "blocking_keys", "key_counts"}
    existing_tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }

    if not required_tables.issubset(existing_tables):
        return False

    existing_indexes = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
    }

    if not {"idx_block_lookup", "idx_record_source"}.issubset(
        existing_indexes
    ):
        return False

    metadata_exists = "index_metadata" in existing_tables

    if metadata_exists:
        metadata = dict(
            connection.execute(
                "SELECT key, value FROM index_metadata"
            ).fetchall()
        )
        if metadata.get("max_block_size") != str(max_block_size):
            print("Existing index uses a different block-size setting.")
            return False
    elif max_block_size != 500:
        print(
            "Legacy index has no saved configuration. "
            "Only reuse it with the original default max-block-size of 500."
        )
        return False

    for source, filename in SOURCE_FILES.items():
        path = DATASET_ROOT / "test" / filename
        if not path.exists():
            return False

        expected_rows = count_tsv_data_rows(path)
        indexed_rows = connection.execute(
            "SELECT COUNT(*) FROM records WHERE source = ?",
            (source,),
        ).fetchone()[0]

        print(
            f"Index check {source}: "
            f"database={indexed_rows:,}, file={expected_rows:,}"
        )

        if indexed_rows != expected_rows:
            print(f"Index for {source} is incomplete.")
            return False

        key_count = connection.execute(
            "SELECT COUNT(*) FROM blocking_keys WHERE source = ?",
            (source,),
        ).fetchone()[0]

        if expected_rows > 0 and key_count == 0:
            print(f"Index for {source} contains no blocking keys.")
            return False

    return True


def load_completed_ids(path, expected_header):
    """Read IDs already written to a partial output file."""
    if not path.exists() or path.stat().st_size == 0:
        return None

    completed = set()

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t")

        header = next(reader, None)
        if header != expected_header:
            raise ValueError(
                f"Unexpected header in {path}: {header!r}"
            )

        for line_number, row in enumerate(reader, start=2):
            if len(row) != 2 or not row[0]:
                raise ValueError(
                    f"Invalid row in {path} at line {line_number}. "
                    "Preserve the backup before repairing this file."
                )

            source1_id = row[0]
            if source1_id in completed:
                raise ValueError(
                    f"Duplicate Source 1 ID {source1_id} in {path}"
                )

            completed.add(source1_id)

    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=str(
            PROJECT_ROOT / "artifacts" / "entity_match_model.joblib"
        ),
    )
    parser.add_argument("--max-block-size", type=int, default=500)
    parser.add_argument("--max-source1", type=int, default=None)
    parser.add_argument("--max-target-rows", type=int, default=None)
    args = parser.parse_args()

    if args.max_block_size < 1:
        raise ValueError("--max-block-size must be at least 1")
    if args.max_source1 is not None and args.max_source1 < 1:
        raise ValueError("--max-source1 must be positive")
    if args.max_target_rows is not None and args.max_target_rows < 1:
        raise ValueError("--max-target-rows must be positive")

    # Limited smoke tests must not be mixed with a full inference run.
    if args.max_source1 is not None or args.max_target_rows is not None:
        raise ValueError(
            "This resumable runner is for full inference. "
            "Use a separate output directory for smoke tests."
        )

    artifact_path = Path(args.model_path)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {artifact_path}\n"
            "Run train_inference_model.py first."
        )

    artifact = joblib.load(artifact_path)
    model = artifact["model"]
    threshold = float(artifact["threshold"])

    if not 0 <= threshold <= 1:
        raise ValueError("Saved threshold must be between 0 and 1")

    print(f"Model: {artifact['model_name']}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Maximum target block size: {args.max_block_size}")

    cache_dir = PROJECT_ROOT / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    database_path = cache_dir / "test_entity_index.sqlite"

    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidate_pairs.tsv"
    matching_path = output_dir / "matching_results.tsv"

    source1_path = DATASET_ROOT / "test" / "test_source1.tsv"
    if not source1_path.exists():
        raise FileNotFoundError(f"Missing test dataset: {source1_path}")

    candidate_header = [
        "source1_entity_id",
        "candidate_entity_ids",
    ]
    matching_header = [
        "source1_entity_id",
        "matched_entity_ids",
    ]

    # Do not silently mix output files from different runs.
    candidate_ids = load_completed_ids(candidate_path, candidate_header)
    matching_ids = load_completed_ids(matching_path, matching_header)

    if (candidate_ids is None) != (matching_ids is None):
        raise RuntimeError(
            "Only one output file exists. Restore both matching and "
            "candidate files from the same checkpoint before resuming."
        )

    if candidate_ids is not None and candidate_ids != matching_ids:
        raise RuntimeError(
            "Candidate and matching files contain different Source 1 IDs. "
            "Do not append to them until the checkpoint is repaired."
        )

    completed_ids = candidate_ids or set()
    print(f"Previously completed Source 1 entities: {len(completed_ids):,}")

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA cache_size = -50000")
    connection.execute("PRAGMA temp_store = FILE")

    try:
        # Reuse the database if its records and indexes are complete.
        if database_path.exists() and index_is_complete(
            connection, args.max_block_size
        ):
            print("Reusing the existing SQLite target index.")
        else:
            print("A complete compatible index was not found.")
            print("Building the target index from scratch...")
            build_target_index(
                connection,
                args.max_block_size,
                None,
            )

            connection.execute("""
                CREATE TABLE IF NOT EXISTS index_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            connection.execute(
                "INSERT OR REPLACE INTO index_metadata VALUES (?, ?)",
                ("max_block_size", str(args.max_block_size)),
            )
            connection.commit()

            if not index_is_complete(connection, args.max_block_size):
                raise RuntimeError(
                    "The index failed its post-build completeness check."
                )

        mode = "a" if candidate_ids is not None else "w"

        with (
            candidate_path.open(
                mode, encoding="utf-8", newline=""
            ) as candidate_file,
            matching_path.open(
                mode, encoding="utf-8", newline=""
            ) as matching_file,
        ):
            candidate_writer = csv.writer(
                candidate_file,
                delimiter="\t",
                lineterminator="\n",
            )
            matching_writer = csv.writer(
                matching_file,
                delimiter="\t",
                lineterminator="\n",
            )

            if mode == "w":
                candidate_writer.writerow(candidate_header)
                matching_writer.writerow(matching_header)

            processed_this_run = 0

            for chunk in read_chunks(source1_path):
                missing = set(SOURCE_COLUMNS) - set(chunk.columns)
                if missing:
                    raise ValueError(
                        f"Source 1 is missing columns: {sorted(missing)}"
                    )

                for record in chunk.to_dict("records"):
                    source1_id = str(record["entity_id"])

                    if source1_id in completed_ids:
                        continue

                    candidates, matches = score_one_source1(
                        connection,
                        record,
                        model,
                        threshold,
                    )

                    # Each completed entity gets one row in each output.
                    candidate_writer.writerow([
                        source1_id,
                        ",".join(sorted(set(candidates))),
                    ])
                    matching_writer.writerow([
                        source1_id,
                        ",".join(matches),
                    ])

                    completed_ids.add(source1_id)
                    processed_this_run += 1

                    # Flush periodically so interruptions lose little work.
                    if processed_this_run % 100 == 0:
                        candidate_file.flush()
                        matching_file.flush()
                        print(
                            f"Completed this run: {processed_this_run:,}; "
                            f"total completed: {len(completed_ids):,}"
                        )

                candidate_file.flush()
                matching_file.flush()
                print(
                    f"Checkpoint saved. Total completed: "
                    f"{len(completed_ids):,}"
                )

        expected_source1_rows = count_tsv_data_rows(source1_path)
        if len(completed_ids) != expected_source1_rows:
            print(
                f"WARNING: {len(completed_ids):,} Source 1 entities "
                f"completed; expected {expected_source1_rows:,}. "
                "Resume again to finish the remaining entities."
            )
        else:
            print("\nInference finished for every Source 1 entity.")

        print(f"Candidate file: {candidate_path}")
        print(f"Matching file: {matching_path}")
        print(f"SQLite index: {database_path}")

    except KeyboardInterrupt:
        print(
            "\nInterrupted. Completed output rows have been flushed "
            "periodically. Run this command again to resume."
        )
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    main()
