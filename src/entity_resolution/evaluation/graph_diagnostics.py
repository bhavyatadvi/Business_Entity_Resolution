
import pandas as pd


def diagnose_candidate_graph(
    candidates: pd.DataFrame,
) -> dict:
    required = [
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
    ]

    missing = set(required) - set(candidates.columns)
    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if candidates.empty:
        raise ValueError("Candidates dataframe is empty")

    source1_degree = (
        candidates.groupby("source1_entity_id")
        .size()
    )

    target_degree = (
        candidates.groupby(
            ["candidate_source", "candidate_entity_id"]
        )
        .size()
    )

    return {
        "candidate_pairs": len(candidates),
        "unique_source1_entities": (
            candidates["source1_entity_id"].nunique()
        ),
        "unique_target_entities": (
            candidates[
                ["candidate_source", "candidate_entity_id"]
            ].drop_duplicates().shape[0]
        ),
        "source1_degree_median": float(
            source1_degree.median()
        ),
        "source1_degree_p95": float(
            source1_degree.quantile(0.95)
        ),
        "source1_degree_max": int(
            source1_degree.max()
        ),
        "target_degree_median": float(
            target_degree.median()
        ),
        "target_degree_p95": float(
            target_degree.quantile(0.95)
        ),
        "target_degree_max": int(
            target_degree.max()
        ),
        "source1_top_10": (
            source1_degree.sort_values(
                ascending=False
            ).head(10).to_dict()
        ),
    }