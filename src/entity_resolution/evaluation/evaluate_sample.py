
import pandas as pd

from src.entity_resolution.data.data_loader import DATASET_ROOT
from src.entity_resolution.blocking.candidate_generator import (
    generate_candidates,
)
from src.entity_resolution.evaluation.candidate_metrics import (
    parse_ground_truth,
    evaluate_candidate_recall,
    evaluate_scored_candidates,
)
from src.entity_resolution.matching.score_candidates import (
    score_candidate_pairs,
)
from src.entity_resolution.matching.model import (
    evaluate_match_model,
)
from src.entity_resolution.evaluation.split_audit import (
    audit_candidate_components,
)
from src.entity_resolution.evaluation.graph_diagnostics import (
    diagnose_candidate_graph,
)
from src.entity_resolution.evaluation.blocking_diagnostics import (
    diagnose_blocking_rules,
)
from src.entity_resolution.evaluation.blocking_impact import (
    analyze_blocking_key_impact,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CHUNK_SIZE = 100_000
DISTRACTORS_PER_CHUNK = 100
GROUND_TRUTH_SAMPLE_SIZE = 500

# Preserve the original configuration as the baseline.
BLOCK_MAX_SIZE = 500

# Experiment with alternative block-size limits.
BLOCK_SIZE_OPTIONS = [500, 400, 300, 250, 200, 150, 100]

HEURISTIC_THRESHOLDS = (70, 80, 85, 90, 95)


# ---------------------------------------------------------
# Ground-truth sampling
# ---------------------------------------------------------

def load_random_ground_truth_sample(
    sample_size: int = GROUND_TRUTH_SAMPLE_SIZE,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Draw a reproducible sample from the ground-truth file
    using chunked reads instead of loading the entire file.
    """
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")

    path = DATASET_ROOT / "train" / "train_ground_truth.tsv"
    sampled_chunks = []

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=CHUNK_SIZE,
        )
    ):
        count = min(sample_size, len(chunk))

        if count:
            sampled_chunks.append(
                chunk.sample(
                    n=count,
                    random_state=random_state + chunk_number,
                )
            )

    if not sampled_chunks:
        return pd.DataFrame(
            columns=[
                "source1_entity_id",
                "matched_entity_ids",
            ]
        )

    combined = pd.concat(
        sampled_chunks,
        ignore_index=True,
    )

    return combined.sample(
        n=min(sample_size, len(combined)),
        random_state=random_state,
    ).reset_index(drop=True)


# ---------------------------------------------------------
# Record loading
# ---------------------------------------------------------

def load_evaluation_records(
    source: str,
    required_ids: set[str],
    include_distractors: bool = False,
) -> pd.DataFrame:
    """
    Retrieve required records with chunked reads.

    For target sources, optionally sample additional records
    from every chunk as distractors.

    This is a diagnostic sample, not the complete target dataset.
    """
    if source not in {"source1", "source2", "source3"}:
        raise ValueError(
            "source must be source1, source2, or source3"
        )

    path = DATASET_ROOT / "train" / f"train_{source}.tsv"

    required_columns = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ]

    found_frames = []
    sampled_frames = []
    found_ids = set()

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            path,
            sep="\t",
            usecols=required_columns,
            dtype=str,
            keep_default_na=False,
            chunksize=CHUNK_SIZE,
        )
    ):
        matches = chunk[
            chunk["entity_id"].isin(required_ids)
        ]

        if not matches.empty:
            found_frames.append(matches)
            found_ids.update(matches["entity_id"])

        if include_distractors:
            sample_size = min(
                DISTRACTORS_PER_CHUNK,
                len(chunk),
            )

            if sample_size:
                sampled_frames.append(
                    chunk.sample(
                        n=sample_size,
                        random_state=42 + chunk_number,
                    )
                )

        elif found_ids == required_ids:
            # Source 1 only needs referenced ground-truth records.
            break

    missing_ids = required_ids - found_ids

    if missing_ids:
        raise ValueError(
            f"{source}: could not find {len(missing_ids)} "
            f"required IDs. Examples: {sorted(missing_ids)[:5]}"
        )

    frames = []

    if found_frames:
        frames.append(
            pd.concat(found_frames, ignore_index=True)
        )

    if sampled_frames:
        frames.append(
            pd.concat(sampled_frames, ignore_index=True)
        )

    if not frames:
        return pd.DataFrame(columns=required_columns)

    result = pd.concat(frames, ignore_index=True)

    # Remove duplicate records found in both collections.
    return result.drop_duplicates(
        subset=["entity_id"],
        ignore_index=True,
    )


# ---------------------------------------------------------
# Helper: combine candidates from both target sources
# ---------------------------------------------------------

def combine_candidates(
    candidates: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Combine Source 2 and Source 3 candidate pairs."""
    return pd.concat(
        [
            candidates["source2"],
            candidates["source3"],
        ],
        ignore_index=True,
    )


# ---------------------------------------------------------
# Experiment: compare block-size limits
# ---------------------------------------------------------

def run_block_size_experiment(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Compare block-size limits using the same evaluation records.

    The baseline candidate set is returned separately so the
    existing scoring and model evaluation remain unchanged.
    """
    print("\nBLOCK SIZE EXPERIMENT")
    print("=" * 90)

    experiment_results = []
    baseline_candidates = None

    for block_size in BLOCK_SIZE_OPTIONS:
        print(f"\nTesting max_block_size={block_size}...")

        trial_candidates = generate_candidates(
            source1,
            source2,
            source3,
            max_block_size=block_size,
        )

        trial_all = combine_candidates(trial_candidates)

        metrics = evaluate_candidate_recall(
            ground_truth,
            trial_all,
        )

        experiment_results.append({
            "max_block_size": block_size,
            "candidate_pairs": len(trial_all),
            "candidate_recall": metrics["candidate_recall"],
            "source2_recall": metrics["source2"]["recall"],
            "source3_recall": metrics["source3"]["recall"],
        })

        print(f"Candidate pairs: {len(trial_all):,}")
        print(
            f"Overall candidate recall: "
            f"{metrics['candidate_recall']:.2%}"
        )
        print(
            f"Source 2 recall: "
            f"{metrics['source2']['recall']:.2%}"
        )
        print(
            f"Source 3 recall: "
            f"{metrics['source3']['recall']:.2%}"
        )

        if block_size == BLOCK_MAX_SIZE:
            baseline_candidates = trial_candidates

    comparison = pd.DataFrame(experiment_results)

    print("\nBLOCK SIZE COMPARISON")
    print(
        comparison.to_string(
            index=False,
            formatters={
                "candidate_recall": "{:.2%}".format,
                "source2_recall": "{:.2%}".format,
                "source3_recall": "{:.2%}".format,
            },
        )
    )

    if baseline_candidates is None:
        raise RuntimeError(
            "The original baseline block size was not tested"
        )

    return baseline_candidates, comparison


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------

def main() -> None:
    print("Loading random ground-truth sample...")

    ground_truth = load_random_ground_truth_sample(
        sample_size=GROUND_TRUTH_SAMPLE_SIZE,
        random_state=42,
    )

    truth_pairs = parse_ground_truth(ground_truth)

    required_ids = {
        "source1": set(
            ground_truth["source1_entity_id"]
        ),
        "source2": set(
            truth_pairs.loc[
                truth_pairs["candidate_source"] == "source2",
                "candidate_entity_id",
            ]
        ),
        "source3": set(
            truth_pairs.loc[
                truth_pairs["candidate_source"] == "source3",
                "candidate_entity_id",
            ]
        ),
    }

    print("\nRetrieving Source 1 records...")
    source1 = load_evaluation_records(
        "source1",
        required_ids["source1"],
    )

    print("\nSampling Source 2 records...")
    source2 = load_evaluation_records(
        "source2",
        required_ids["source2"],
        include_distractors=True,
    )

    print("\nSampling Source 3 records...")
    source3 = load_evaluation_records(
        "source3",
        required_ids["source3"],
        include_distractors=True,
    )

    # -----------------------------------------------------
    # Dataset summary
    # -----------------------------------------------------

    print("\nEvaluation dataset sizes")
    print(f"Source 1: {len(source1):,}")
    print(f"Source 2: {len(source2):,}")
    print(f"Source 3: {len(source3):,}")
    print(f"Ground-truth rows: {len(ground_truth):,}")
    print(f"Ground-truth pairs: {len(truth_pairs):,}")

    # -----------------------------------------------------
    # Blocking key impact analysis
    # -----------------------------------------------------

    print("\nBLOCKING KEY IMPACT ANALYSIS")

    impact = analyze_blocking_key_impact(
        source1,
        source2,
        source3,
        ground_truth,
    )

    print("\nTop 20 keys by candidate-pair contribution:")
    print(impact.head(20).to_string(index=False))

    print("\nTop 20 keys by known true matches recovered:")
    print(
        impact.sort_values(
            "true_pairs_covered",
            ascending=False,
        ).head(20).to_string(index=False)
    )

    # -----------------------------------------------------
    # Blocking rule diagnostics
    # -----------------------------------------------------

    print("\nBLOCKING RULE DIAGNOSTICS")

    blocking_report = diagnose_blocking_rules(
        source1,
        source2,
        source3,
        max_block_size=BLOCK_MAX_SIZE,
    )

    print(blocking_report.head(20).to_string(index=False))

    print("\nOversized blocking keys:")
    print(
        blocking_report["oversized"]
        .value_counts()
        .to_string()
    )

    # -----------------------------------------------------
    # Block-size experiment
    # -----------------------------------------------------

    baseline_candidates, block_size_comparison = (
        run_block_size_experiment(
            source1,
            source2,
            source3,
            ground_truth,
        )
    )

    # Keep downstream model evaluation on the original
    # max_block_size=500 baseline for a fair comparison.
    candidates = baseline_candidates
    all_candidates = combine_candidates(candidates)

    # -----------------------------------------------------
    # Candidate graph degree diagnostics
    # -----------------------------------------------------

    print("\nCANDIDATE GRAPH DEGREE DIAGNOSTICS")

    degree_report = diagnose_candidate_graph(all_candidates)

    for metric, value in degree_report.items():
        print(f"{metric}: {value}")

    # -----------------------------------------------------
    # Candidate graph component audit
    # -----------------------------------------------------

    print("\nCANDIDATE GRAPH COMPONENT AUDIT")

    component_report = audit_candidate_components(
        all_candidates
    )

    for metric, value in component_report.items():
        print(f"{metric}: {value}")

    # -----------------------------------------------------
    # Baseline candidate recall
    # -----------------------------------------------------

    print("\nBASELINE CANDIDATE RECALL")

    candidate_metrics = evaluate_candidate_recall(
        ground_truth,
        all_candidates,
    )

    print(
        f"Overall recall: "
        f"{candidate_metrics['candidate_recall']:.2%}"
    )

    for source_name in ("source2", "source3"):
        metrics = candidate_metrics[source_name]

        print(
            f"{source_name}: "
            f"{metrics['found_pairs']:,}/"
            f"{metrics['true_pairs']:,} "
            f"({metrics['recall']:.2%})"
        )

    print(f"Total candidates: {len(all_candidates):,}")

    # -----------------------------------------------------
    # Score baseline candidate pairs
    # -----------------------------------------------------

    print("\nScoring candidate pairs...")

    scored_candidates = score_candidate_pairs(
        source1,
        source2,
        source3,
        all_candidates,
    )

    # -----------------------------------------------------
    # Heuristic score threshold comparison
    # -----------------------------------------------------

    print("\nHEURISTIC THRESHOLD COMPARISON")
    print("=" * 100)

    print(
        f"{'Threshold':>10} "
        f"{'Predicted':>12} "
        f"{'TP':>8} "
        f"{'FP':>8} "
        f"{'FN':>8} "
        f"{'Precision':>12} "
        f"{'Recall':>10} "
        f"{'F1':>10}"
    )

    for threshold in HEURISTIC_THRESHOLDS:
        result = evaluate_scored_candidates(
            ground_truth,
            scored_candidates,
            threshold=threshold,
        )

        print(
            f"{threshold:>10} "
            f"{result['predicted_pairs']:>12,} "
            f"{result['true_positives']:>8,} "
            f"{result['false_positives']:>8,} "
            f"{result['false_negatives']:>8,} "
            f"{result['precision']:>11.2%} "
            f"{result['recall']:>9.2%} "
            f"{result['f1']:>9.2%}"
        )

    print("\nCandidate scoring summary")
    print(f"Candidates scored: {len(scored_candidates):,}")
    print(
        "Candidates with missing address similarity: "
        f"{scored_candidates['address_similarity'].isna().sum():,}"
    )

    # -----------------------------------------------------
    # Top heuristic candidates
    # -----------------------------------------------------

    print("\nTop 10 candidates by heuristic score")

    display_columns = [
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
        "name_similarity",
        "address_similarity",
        "country_match",
        "combined_score",
    ]

    print(
        scored_candidates.sort_values(
            "combined_score",
            ascending=False,
        )[display_columns].head(10).to_string(index=False)
    )

    # -----------------------------------------------------
    # Supervised Logistic Regression baseline
    # -----------------------------------------------------

    print("\nTRAINING SUPERVISED MATCH MODEL...")

    model_result = evaluate_match_model(
        scored_candidates,
        ground_truth,
    )

    print("\nVALIDATION THRESHOLD RESULTS")

    print(
        model_result["validation_results"].to_string(
            index=False,
            formatters={
                "precision": "{:.2%}".format,
                "recall": "{:.2%}".format,
                "f1": "{:.2%}".format,
            },
        )
    )

    print("\nHELD-OUT TEST RESULTS")

    for metric, value in model_result["test_metrics"].items():
        if metric in {
            "precision",
            "recall",
            "f1",
            "average_precision",
        }:
            print(f"{metric}: {value:.2%}")
        else:
            print(f"{metric}: {value}")

    # -----------------------------------------------------
    # Split overlap diagnostics
    # -----------------------------------------------------

    print("\nSPLIT OVERLAP DIAGNOSTICS")

    for name, count in model_result["split_diagnostics"].items():
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()