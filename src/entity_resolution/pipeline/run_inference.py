
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

    if database_path.exists():
        database_path.unlink()

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA cache_size = -50000")
    connection.execute("PRAGMA temp_store = FILE")

    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_path = output_dir / "candidate_pairs.tsv"
    matching_path = output_dir / "matching_results.tsv"

    try:
        build_target_index(
            connection,
            args.max_block_size,
            args.max_target_rows,
        )

        source1_path = DATASET_ROOT / "test" / "test_source1.tsv"
        if not source1_path.exists():
            raise FileNotFoundError(
                f"Missing test dataset: {source1_path}"
            )

        processed = 0

        with (
            candidate_path.open(
                "w", encoding="utf-8", newline=""
            ) as candidate_file,
            matching_path.open(
                "w", encoding="utf-8", newline=""
            ) as matching_file,
        ):
            candidate_writer = csv.writer(
                candidate_file, delimiter="\t",
                lineterminator="\n",
            )
            matching_writer = csv.writer(
                matching_file, delimiter="\t",
                lineterminator="\n",
            )

            candidate_writer.writerow([
                "source1_entity_id", "candidate_entity_ids"
            ])
            matching_writer.writerow([
                "source1_entity_id", "matched_entity_ids"
            ])

            for chunk in read_chunks(
                source1_path, args.max_source1
            ):
                missing = set(SOURCE_COLUMNS) - set(chunk.columns)
                if missing:
                    raise ValueError(
                        f"Source 1 is missing columns: {sorted(missing)}"
                    )

                for record in chunk.to_dict("records"):
                    source1_id = str(record["entity_id"])

                    candidates, matches = score_one_source1(
                        connection,
                        record,
                        model,
                        threshold,
                    )

                    # Every Source 1 entity gets exactly one row,
                    # including entities with no candidates or matches.
                    candidate_writer.writerow([
                        source1_id, ",".join(sorted(set(candidates)))
                    ])
                    matching_writer.writerow([
                        source1_id, ",".join(matches)
                    ])

                    processed += 1

                candidate_file.flush()
                matching_file.flush()
                print(f"Processed Source 1 entities: {processed:,}")

        print("\nInference finished.")
        print(f"Source 1 entities processed: {processed:,}")
        print(f"Candidate file: {candidate_path}")
        print(f"Matching file: {matching_path}")
        print(f"SQLite index: {database_path}")

        if args.max_source1 is not None or args.max_target_rows is not None:
            print(
                "\nWARNING: This was a limited smoke test. "
                "Do not submit these output files."
            )

    finally:
        connection.close()


if __name__ == "__main__":
    main()
