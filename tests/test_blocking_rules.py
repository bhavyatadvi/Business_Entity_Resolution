
from src.entity_resolution.blocking.blocking_rules import (
    generate_blocking_keys,
)


def test_generates_name_token_keys():
    record = {
        "business_name": "Prime Money Inc",
        "business_address": "",
    }

    keys = generate_blocking_keys(record)

    assert ("name_token", "prime") in keys
    assert ("name_token", "money") in keys
    assert ("name_token", "inc") not in keys


def test_generates_name_prefix_keys():
    record = {
        "business_name": "Prime Money",
        "business_address": "",
    }

    keys = generate_blocking_keys(record)

    assert ("name_prefix", "prim") in keys
    assert ("name_prefix", "mone") in keys


def test_generates_address_keys():
    record = {
        "business_name": "Prime Money",
        "business_address": "17560 Ellis Road, Tahlequah, OK",
    }

    keys = generate_blocking_keys(record)

    assert ("address_number", "17560") in keys
    assert ("address_token", "ellis") in keys
    assert ("address_token", "tahlequah") in keys


def test_uses_preprocessed_columns():
    record = {
        "name_normalized": "prime money",
        "address_normalized": "17560 ellis road",
        "business_name": "PRIME MONEY",
        "business_address": "17560 Ellis Road",
    }

    keys = generate_blocking_keys(record)

    assert ("name_token", "prime") in keys
    assert ("address_token", "ellis") in keys


def test_handles_empty_values():
    record = {
        "business_name": "",
        "business_address": "",
    }

    assert generate_blocking_keys(record) == set()


def test_preserves_non_english_name_tokens():
    record = {
        "business_name": "राम मार्केटिंग प्राइवेट लिमिटेड",
        "business_address": "",
    }

    keys = generate_blocking_keys(record)

    assert ("name_token", "राम") in keys
    assert ("name_token", "मार्केटिंग") in keys
