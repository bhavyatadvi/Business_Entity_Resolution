"""Feature engineering for noisy business-name and address comparisons."""
from typing import Any
from rapidfuzz import fuzz
from src.entity_resolution.preprocessing.normalize import normalize_name, normalize_address, normalize_country


def _get_text(record: Any, normalized_column: str, original_column: str, normalizer) -> str:
    value = record.get(normalized_column, "")
    if value is None or not str(value).strip():
        value = record.get(original_column, "")
    return normalizer(value)


def text_similarity(left: str, right: str, method: str = "token_sort") -> float:
    left, right = str(left or "").strip(), str(right or "").strip()
    if not left or not right:
        return 0.0
    methods = {"ratio": fuzz.ratio, "token_sort": fuzz.token_sort_ratio, "token_set": fuzz.token_set_ratio, "wratio": fuzz.WRatio, "partial": fuzz.partial_ratio}
    if method not in methods:
        raise ValueError("method must be 'ratio', 'token_sort', 'token_set', 'wratio', or 'partial'")
    return float(methods[method](left, right))


def _jaccard(left: str, right: str) -> float:
    a, b = set(left.split()), set(right.split())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compare_records(record1: Any, record2: Any) -> dict:
    name1 = _get_text(record1, "name_normalized", "business_name", normalize_name)
    name2 = _get_text(record2, "name_normalized", "business_name", normalize_name)
    address1 = _get_text(record1, "address_normalized", "business_address", normalize_address)
    address2 = _get_text(record2, "address_normalized", "business_address", normalize_address)
    country1 = _get_text(record1, "country_normalized", "country", normalize_country)
    country2 = _get_text(record2, "country_normalized", "country", normalize_country)

    name_ratio = text_similarity(name1, name2, "ratio")
    name_token_sort = text_similarity(name1, name2, "token_sort")
    name_token_set = text_similarity(name1, name2, "token_set")
    name_wratio = text_similarity(name1, name2, "wratio")
    address_present = bool(address1 and address2)
    address_ratio = text_similarity(address1, address2, "ratio") if address_present else 0.0
    address_token_sort = text_similarity(address1, address2, "token_sort") if address_present else 0.0
    address_token_set = text_similarity(address1, address2, "token_set") if address_present else 0.0
    address_jaccard = _jaccard(address1, address2) if address_present else 0.0
    country_known = bool(country1 and country2)
    country_match = float(country1 == country2) if country_known else 0.0

    available = [(name_wratio, 0.65)]
    if address_present:
        available.append((address_token_sort, 0.30))
    if country_known:
        available.append((country_match * 100.0, 0.05))
    combined = sum(score * weight for score, weight in available) / sum(weight for _, weight in available)
    return {
        "name_similarity": round(name_wratio, 2),
        "name_ratio": round(name_ratio, 2),
        "name_token_sort": round(name_token_sort, 2),
        "name_token_set": round(name_token_set, 2),
        "address_similarity": round(address_token_sort, 2) if address_present else None,
        "address_ratio": round(address_ratio, 2),
        "address_token_set": round(address_token_set, 2),
        "address_jaccard": round(address_jaccard, 5),
        "address_missing_either": float(not address_present),
        "country_match": country_match if country_known else None,
        "country_missing_either": float(not country_known),
        "combined_score": round(combined, 2),
    }
