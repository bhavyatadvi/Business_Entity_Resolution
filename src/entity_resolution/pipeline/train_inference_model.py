
"""Train and save a model for the full-test inference pipeline.

Run from the project root:
    python -m src.entity_resolution.pipeline.train_inference_model
"""

import argparse
import json
from pathlib import Path

import joblib

from src.entity_resolution.data.data_loader import PROJECT_ROOT
from src.entity_resolution.blocking.candidate_generator import (
    generate_candidates,
)
from src.entity_resolution.evaluation.candidate_metrics import (
    parse_ground_truth,
)
from src.entity_resolution.evaluation.evaluate_sample import (
    load_random_ground_truth_sample,
    load_evaluation_records,
)
from src.entity_resolution.matching.score_candidates import (
    score_candidate_pairs,
)
from src.entity_resolution.matching.model import (
    evaluate_match_model,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "hist_gradient_boosting"],
        default="logistic_regression",
    )
    parser.add_argument("--max-block-size", type=int, default=500)
    args = parser.parse_args()

    if args.sample_size < 50:
        raise ValueError("Use at least 50 ground-truth rows.")

    print("1. Sampling labelled training entities...")
    ground_truth = load_random_ground_truth_sample(
        sample_size=args.sample_size,
        random_state=42,
    )

    truth_pairs = parse_ground_truth(ground_truth)

    required_ids = {
        "source1": set(ground_truth["source1_entity_id"].astype(str)),
        "source2": set(
            truth_pairs.loc[
                truth_pairs["candidate_source"] == "source2",
                "candidate_entity_id",
            ].astype(str)
        ),
        "source3": set(
            truth_pairs.loc[
                truth_pairs["candidate_source"] == "source3",
                "candidate_entity_id",
            ].astype(str)
        ),
    }

    print("2. Loading required records and sampled distractors...")
    source1 = load_evaluation_records(
        "source1", required_ids["source1"]
    )
    source2 = load_evaluation_records(
        "source2",
        required_ids["source2"],
        include_distractors=True,
    )
    source3 = load_evaluation_records(
        "source3",
        required_ids["source3"],
        include_distractors=True,
    )

    print("3. Generating candidate pairs...")
    candidate_dict = generate_candidates(
        source1,
        source2,
        source3,
        max_block_size=args.max_block_size,
    )

    import pandas as pd

    candidates = pd.concat(
        [candidate_dict["source2"], candidate_dict["source3"]],
        ignore_index=True,
    )

    print(f"Candidate pairs: {len(candidates):,}")

    if candidates.empty:
        raise RuntimeError("No candidates generated; cannot train a model.")

    print("4. Computing comparison features...")
    scored = score_candidate_pairs(
        source1,
        source2,
        source3,
        candidates,
    )

    print("5. Training and selecting the threshold...")
    result = evaluate_match_model(
        scored,
        ground_truth,
        random_state=42,
        model_name=args.model,
    )

    print("\nValidation threshold results:")
    print(
        result["validation_results"].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nHeld-out diagnostic metrics:")
    print(
        json.dumps(result["test_metrics"], indent=2, default=float)
    )

    artifact_dir = PROJECT_ROOT / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = artifact_dir / "entity_match_model.joblib"
    metadata_path = artifact_dir / "entity_match_model_metadata.json"

    artifact = {
        "model": result["model"],
        "threshold": float(result["threshold"]),
        "feature_columns": result["feature_columns"],
        "model_name": result["model_name"],
        "sample_size": len(ground_truth),
        "max_block_size": args.max_block_size,
        "random_state": 42,
    }

    joblib.dump(artifact, artifact_path)

    metadata = {
        "model_name": result["model_name"],
        "threshold": float(result["threshold"]),
        "ground_truth_sample_size": len(ground_truth),
        "candidate_pairs": len(candidates),
        "max_block_size": args.max_block_size,
        "note": (
            "Prototype artifact trained using a sampled workflow. "
            "Validate on larger, entity-disjoint data before final submission."
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved model: {artifact_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Selected threshold: {result['threshold']:.3f}")


if __name__ == "__main__":
    main()
