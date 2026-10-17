"""Compute comparison features for candidate pairs."""
import pandas as pd
from src.entity_resolution.matching.similarity import compare_records

SCORE_COLUMNS = ["name_similarity", "name_ratio", "name_token_sort", "name_token_set", "address_similarity", "address_ratio", "address_token_set", "address_jaccard", "address_missing_either", "country_match", "country_missing_either", "combined_score"]


def score_candidate_pairs(source1: pd.DataFrame, source2: pd.DataFrame, source3: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    required_source = {"entity_id", "business_name", "business_address", "country"}
    for label, frame in (("source1", source1), ("source2", source2), ("source3", source3)):
        missing = required_source - set(frame.columns)
        if missing:
            raise ValueError(f"{label} is missing columns: {sorted(missing)}")
        if frame["entity_id"].duplicated().any():
            raise ValueError(f"{label} contains duplicate entity IDs")
    required_candidates = {"source1_entity_id", "candidate_entity_id", "candidate_source"}
    missing = required_candidates - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidates are missing columns: {sorted(missing)}")
    if not candidates["candidate_source"].isin({"source2", "source3"}).all():
        raise ValueError("candidate_source must contain only 'source2' or 'source3'")

    source1_records = source1.set_index("entity_id").to_dict("index")
    target_records = {"source2": source2.set_index("entity_id").to_dict("index"), "source3": source3.set_index("entity_id").to_dict("index")}
    scored_rows = []
    for candidate in candidates.to_dict("records"):
        sid, cid, source = candidate["source1_entity_id"], candidate["candidate_entity_id"], candidate["candidate_source"]
        if sid not in source1_records:
            raise ValueError(f"Source 1 record not found: {sid}")
        if cid not in target_records[source]:
            raise ValueError(f"Candidate record not found in {source}: {cid}")
        features = compare_records(source1_records[sid], target_records[source][cid])
        scored_rows.append({**candidate, **features})
    columns = list(candidates.columns) + [c for c in SCORE_COLUMNS if c not in candidates.columns]
    if not scored_rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(scored_rows).sort_values(["source1_entity_id", "candidate_source", "combined_score"], ascending=[True, True, False], ignore_index=True)
