
import pytest

from src.entity_resolution.matching.similarity import (
    text_similarity,
    compare_records,
)


def test_identical_text_has_full_similarity():
    assert text_similarity("prime money", "prime money") == 100.0


def test_similarity_handles_word_order():
    score = text_similarity(
        "prime money services",
        "services prime money",
        method="token_sort",
    )
    assert score == 100.0


def test_empty_text_has_zero_similarity():
    assert text_similarity("", "prime money") == 0.0


def test_invalid_similarity_method_is_rejected():
    with pytest.raises(ValueError):
        text_similarity("alpha", "beta", method="unknown")


def test_identical_businesses_score_high():
    record1 = {
        "business_name": "Prime Money Inc",
        "business_address": "10 Ellis Road",
        "country": "US",
    }
    record2 = {
        "business_name": "Prime Money Inc",
        "business_address": "10 Ellis Road",
        "country": "US",
    }

    result = compare_records(record1, record2)

    assert result["name_similarity"] == 100.0
    assert result["address_similarity"] == 100.0
    assert result["country_match"] == 1.0
    assert result["combined_score"] == 100.0


def test_missing_address_does_not_break_scoring():
    record1 = {
        "business_name": "Prime Money",
        "business_address": "",
        "country": "US",
    }
    record2 = {
        "business_name": "Prime Money",
        "business_address": "",
        "country": "US",
    }

    result = compare_records(record1, record2)

    assert result["address_similarity"] is None
    assert result["combined_score"] == 100.0


def test_different_businesses_have_lower_name_similarity():
    record1 = {
        "business_name": "Prime Money",
        "business_address": "10 Ellis Road",
        "country": "US",
    }
    record2 = {
        "business_name": "Ocean Restaurant",
        "business_address": "90 Lake Street",
        "country": "US",
    }

    result = compare_records(record1, record2)

    assert result["name_similarity"] < 100.0
    assert result["combined_score"] < 100.0


def test_non_english_names_are_supported():
    record1 = {
        "business_name": "राम मार्केटिंग प्राइवेट लिमिटेड",
        "business_address": "",
        "country": "India",
    }
    record2 = {
        "business_name": "राम मार्केटिंग प्राइवेट लिमिटेड",
        "business_address": "",
        "country": "India",
    }

    result = compare_records(record1, record2)

    assert result["name_similarity"] == 100.0
    assert result["combined_score"] == 100.0
