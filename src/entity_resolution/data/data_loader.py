
from pathlib import Path

import pandas as pd


# Find the project root automatically.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PROJECT_ROOT / "student_resource" / "dataset"

SOURCE_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

GROUND_TRUTH_COLUMNS = [
    "source1_entity_id",
    "matched_entity_ids",
]

VALID_SPLITS = {"train", "test"}
VALID_SOURCES = {"source1", "source2", "source3"}


def load_source(
    split: str,
    source: str,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load and validate one source dataset."""

    if split not in VALID_SPLITS:
        raise ValueError(
            f"Invalid split: {split}. Choose 'train' or 'test'."
        )

    if source not in VALID_SOURCES:
        raise ValueError(
            f"Invalid source: {source}. Choose source1, source2, or source3."
        )

    file_path = DATASET_ROOT / split / f"{split}_{source}.tsv"

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=nrows,
    )

    missing_columns = set(SOURCE_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{file_path.name} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    # Keep the expected columns in a consistent order.
    df = df[SOURCE_COLUMNS]

    print(
        f"Loaded {file_path.name}: "
        f"{len(df):,} rows, {len(df.columns)} columns"
    )

    return df


def load_ground_truth(
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load and validate the training ground-truth dataset."""

    file_path = DATASET_ROOT / "train" / "train_ground_truth.tsv"

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=nrows,
    )

    missing_columns = set(GROUND_TRUTH_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{file_path.name} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    df = df[GROUND_TRUTH_COLUMNS]

    print(
        f"Loaded {file_path.name}: "
        f"{len(df):,} rows, {len(df.columns)} columns"
    )

    return df
