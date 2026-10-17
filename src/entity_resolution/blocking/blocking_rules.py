
from typing import Any

from src.entity_resolution.preprocessing.normalize import (
    normalize_name,
    normalize_address,
)


# Legal suffixes and generic terms that often add little
# value when identifying a business.
GENERIC_NAME_TOKENS = {
    "inc",
    "incorporated",
    "llc",
    "ltd",
    "limited",
    "corp",
    "corporation",
    "company",
    "co",
    "llp",
    "plc",
    "gmbh",
    "private",
    "pvt",
}

# Common address words that are usually not distinctive.
GENERIC_ADDRESS_TOKENS = {
    "street",
    "st",
    "road",
    "rd",
    "avenue",
    "ave",
    "drive",
    "dr",
    "lane",
    "ln",
    "boulevard",
    "blvd",
    "highway",
    "hwy",
    "unit",
    "suite",
    "ste",
    "apartment",
    "apt",
    "floor",
    "building",
    "block",
}


def _get_normalized_value(
    record: Any,
    normalized_column: str,
    original_column: str,
    normalizer,
) -> str:
    """Use preprocessed text when available; otherwise normalize it."""
    value = record.get(normalized_column, "")

    if value is None or not str(value).strip():
        value = normalizer(record.get(original_column, ""))

    return str(value).strip()


def generate_blocking_keys(record: Any) -> set[tuple[str, str]]:
    """
    Generate multiple keys for retrieving plausible business matches.

    The function does not decide whether two records match.
    It only creates keys that candidate generation can use.
    """
    name = _get_normalized_value(
        record,
        "name_normalized",
        "business_name",
        normalize_name,
    )

    address = _get_normalized_value(
        record,
        "address_normalized",
        "business_address",
        normalize_address,
    )

    keys: set[tuple[str, str]] = set()

    # Name-based keys.
    name_tokens = [
        token
        for token in name.split()
        if len(token) >= 3
        and token not in GENERIC_NAME_TOKENS
    ]

    for token in name_tokens:
        # Exact token helps retrieve records sharing a name word.
        keys.add(("name_token", token))

        # Prefix can retrieve some spelling variants.
        if len(token) >= 4:
            keys.add(("name_prefix", token[:4]))

    # Address-based keys.
    address_tokens = address.split()

    for token in address_tokens:
        if token in GENERIC_ADDRESS_TOKENS:
            continue

        # Keep distinctive words and numeric address components.
        if len(token) >= 4:
            keys.add(("address_token", token))

        if any(char.isdigit() for char in token) and len(token) >= 2:
            keys.add(("address_number", token))

    return keys
