
import unicodedata

import pandas as pd


def normalize_text(value: object) -> str:
    """
    Normalize text while preserving Unicode letters, combining marks,
    and numbers, including Indic scripts such as Hindi.
    """
    if value is None or pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = text.casefold()

    normalized_chars = []

    for char in text:
        category = unicodedata.category(char)

        # Keep letters, combining marks, numbers, and whitespace.
        if (
            category.startswith("L")
            or category.startswith("M")
            or category.startswith("N")
            or char.isspace()
        ):
            normalized_chars.append(char)
        else:
            # Replace punctuation and symbols with spaces.
            normalized_chars.append(" ")

    text = "".join(normalized_chars)

    # Collapse repeated whitespace.
    return " ".join(text.split())


def normalize_name(value: object) -> str:
    """Normalize a business name."""
    return normalize_text(value)


def normalize_address(value: object) -> str:
    """Normalize a business address."""
    return normalize_text(value)


def normalize_country(value: object) -> str:
    """Normalize country text without guessing country aliases."""
    return normalize_text(value)


def preprocess_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of business records with normalized columns.
    Original columns and values remain unchanged.
    """
    required_columns = {
        "entity_id",
        "business_name",
        "business_address",
        "country",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    result = df.copy()

    result["name_normalized"] = result["business_name"].map(
        normalize_name
    )
    result["address_normalized"] = result["business_address"].map(
        normalize_address
    )
    result["country_normalized"] = result["country"].map(
        normalize_country
    )

    return result
