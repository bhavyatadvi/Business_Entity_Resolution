
import pandas as pd
import pytest

from src.entity_resolution.preprocessing.normalize import (
    normalize_text,
    normalize_name,
    normalize_address,
    normalize_country,
    preprocess_records,
)


def test_normalize_text_case_and_whitespace():
    assert normalize_text("  PRIME   MONEY  ") == "prime money"


def test_normalize_name_punctuation():
    assert normalize_name("Orelee's Barbershop") == "orelee s barbershop"


def test_normalize_address():
    assert normalize_address(
        "17560 Ellis Road, Tahlequah, OK"
    ) == "17560 ellis road tahlequah ok"


def test_normalize_empty_values():
    assert normalize_text("") == ""
    assert normalize_text(None) == ""
    assert normalize_text(float("nan")) == ""


def test_normalize_country():
    assert normalize_country("  US  ") == "us"


def test_preserves_non_english_text():
    assert normalize_name(
        "राम मार्केटिंग प्राइवेट लिमिटेड"
    ) == "राम मार्केटिंग प्राइवेट लिमिटेड"


def test_preprocess_records_preserves_original_data():
    df = pd.DataFrame({
        "entity_id": ["S1-123"],
        "business_name": ["PRIME MONEY"],
        "business_address": ["10 Main Street, TX"],
        "country": ["US"],
    })

    result = preprocess_records(df)

    assert result.loc[0, "business_name"] == "PRIME MONEY"
    assert result.loc[0, "name_normalized"] == "prime money"
    assert result.loc[0, "address_normalized"] == "10 main street tx"
    assert result.loc[0, "country_normalized"] == "us"


def test_preprocess_records_rejects_missing_columns():
    df = pd.DataFrame({"entity_id": ["S1-123"]})

    with pytest.raises(ValueError):
        preprocess_records(df)
