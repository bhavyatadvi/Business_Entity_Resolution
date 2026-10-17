"""Train and evaluate supervised business-pair match classifiers.

Threshold selection uses the challenge's macro F0.5 per Source 1 entity,
not pair-level F1. The returned model is fitted on the training split only;
use its selected threshold only after reviewing validation/test diagnostics.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.entity_resolution.evaluation.candidate_metrics import parse_ground_truth, entity_macro_fbeta

FEATURE_COLUMNS = [
    "name_similarity", "name_ratio", "name_token_sort", "name_token_set",
    "address_similarity", "address_ratio", "address_token_set", "address_jaccard",
    "address_missing_either", "country_match", "country_missing_either", "shared_key_count",
]
THRESHOLDS = tuple(round(x, 2) for x in np.arange(0.50, 0.991, 0.025))


def _make_model(model_name: str, random_state: int):
    if model_name == "logistic_regression":
        classifier = LogisticRegression(max_iter=1200, class_weight="balanced", random_state=random_state)
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0, add_indicator=True)), ("scaler", StandardScaler()), ("classifier", classifier)])
    if model_name == "hist_gradient_boosting":
        classifier = HistGradientBoostingClassifier(max_iter=180, learning_rate=0.08, max_leaf_nodes=20, l2_regularization=1.0, random_state=random_state)
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0, add_indicator=True)), ("classifier", classifier)])
    raise ValueError("model_name must be 'logistic_regression' or 'hist_gradient_boosting'")


def _calculate_split_overlap(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> dict:
    def target_ids(frame):
        if frame.empty:
            return set()
        return set(zip(frame["candidate_source"].astype(str), frame["candidate_entity_id"].astype(str)))
    train_targets, val_targets, test_targets = target_ids(train), target_ids(validation), target_ids(test)
    return {
        "source1_overlap_train_validation": len(set(train.source1_entity_id) & set(validation.source1_entity_id)),
        "source1_overlap_train_test": len(set(train.source1_entity_id) & set(test.source1_entity_id)),
        "source1_overlap_validation_test": len(set(validation.source1_entity_id) & set(test.source1_entity_id)),
        "target_overlap_train_validation": len(train_targets & val_targets),
        "target_overlap_train_test": len(train_targets & test_targets),
        "target_overlap_validation_test": len(val_targets & test_targets),
    }


def _add_labels(scored_candidates: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    data = scored_candidates.copy()
    truth = set(parse_ground_truth(ground_truth).itertuples(index=False, name=None))
    data["label"] = [int(row in truth) for row in data[["source1_entity_id", "candidate_entity_id", "candidate_source"]].itertuples(index=False, name=None)]
    return data


def evaluate_match_model(scored_candidates: pd.DataFrame, ground_truth: pd.DataFrame, random_state: int = 42, model_name: str = "logistic_regression") -> dict:
    required = {"source1_entity_id", "candidate_entity_id", "candidate_source", *FEATURE_COLUMNS}
    missing = required - set(scored_candidates.columns)
    if missing:
        raise ValueError(f"Scored candidates are missing: {sorted(missing)}")
    if scored_candidates.empty:
        raise ValueError("No scored candidates were provided")
    if ground_truth["source1_entity_id"].duplicated().any():
        raise ValueError("Ground truth must contain one row per Source 1 entity")
    data = _add_labels(scored_candidates, ground_truth)
    for col in FEATURE_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        if np.isinf(data[col].dropna()).any():
            raise ValueError(f"Feature '{col}' contains infinite values")
    if not data["candidate_source"].isin({"source2", "source3"}).all():
        raise ValueError("candidate_source must contain only source2 and source3")

    # Split the complete Source 1 universe, not just IDs that happened to
    # produce candidates. This retains singleton/no-candidate entities in scoring.
    all_ids = ground_truth["source1_entity_id"].astype(str).to_numpy()
    if len(all_ids) < 10:
        raise ValueError("At least 10 Source 1 entities are needed for a train/validation/test split")
    splitter1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=random_state)
    train_id_idx, temp_id_idx = next(splitter1.split(all_ids, groups=all_ids))
    temp_ids = all_ids[temp_id_idx]
    splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=random_state + 1)
    val_local, test_local = next(splitter2.split(temp_ids, groups=temp_ids))
    train_ids, val_ids, test_ids = set(all_ids[train_id_idx]), set(temp_ids[val_local]), set(temp_ids[test_local])

    train = data[data.source1_entity_id.astype(str).isin(train_ids)].copy()
    validation = data[data.source1_entity_id.astype(str).isin(val_ids)].copy()
    test = data[data.source1_entity_id.astype(str).isin(test_ids)].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("A split has no candidate pairs; inspect candidate generation and dataset size")
    if train.label.nunique() < 2:
        raise ValueError("Training split needs both positive and negative candidate pairs")

    model = _make_model(model_name, random_state)
    model.fit(train[FEATURE_COLUMNS].astype(float), train.label.astype(int))
    val_prob = model.predict_proba(validation[FEATURE_COLUMNS].astype(float))[:, 1]
    val_scored = validation[["source1_entity_id", "candidate_entity_id", "candidate_source"]].copy()
    val_scored["match_probability"] = val_prob

    threshold_rows = []
    gt_by_id = ground_truth[ground_truth.source1_entity_id.astype(str).isin(val_ids)]
    for threshold in THRESHOLDS:
        macro = entity_macro_fbeta(gt_by_id, val_scored, threshold, source1_ids=val_ids)
        pair_pred = (val_prob >= threshold).astype(int)
        threshold_rows.append({**macro,
            "pair_precision": precision_score(validation.label, pair_pred, zero_division=0),
            "pair_recall": recall_score(validation.label, pair_pred, zero_division=0),
            "pair_f1": f1_score(validation.label, pair_pred, zero_division=0),
        })
    best = max(threshold_rows, key=lambda row: (row["macro_f0.5"], row["macro_precision"], row["threshold"]))

    test_prob = model.predict_proba(test[FEATURE_COLUMNS].astype(float))[:, 1]
    test_scored = test[["source1_entity_id", "candidate_entity_id", "candidate_source"]].copy()
    test_scored["match_probability"] = test_prob
    test_gt = ground_truth[ground_truth.source1_entity_id.astype(str).isin(test_ids)]
    test_macro = entity_macro_fbeta(test_gt, test_scored, best["threshold"], source1_ids=test_ids)
    test_pair_pred = (test_prob >= best["threshold"]).astype(int)
    test_metrics = {
        **test_macro,
        "pair_precision": precision_score(test.label, test_pair_pred, zero_division=0),
        "pair_recall": recall_score(test.label, test_pair_pred, zero_division=0),
        "pair_f1": f1_score(test.label, test_pair_pred, zero_division=0),
        "average_precision": average_precision_score(test.label, test_prob),
        "test_pairs": len(test), "test_positive_pairs": int(test.label.sum()),
        "train_pairs": len(train), "validation_pairs": len(validation),
        "model_name": model_name,
    }
    return {
        "model": model,
        "model_name": model_name,
        "feature_columns": FEATURE_COLUMNS.copy(),
        "threshold": best["threshold"],
        "validation_results": pd.DataFrame(threshold_rows),
        "test_metrics": test_metrics,
        "split_diagnostics": _calculate_split_overlap(train, validation, test),
    }


def compare_match_models(scored_candidates: pd.DataFrame, ground_truth: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Run the same entity-level evaluation for both baseline and tree model."""
    rows = []
    for name in ("logistic_regression", "hist_gradient_boosting"):
        result = evaluate_match_model(scored_candidates, ground_truth, random_state, name)
        rows.append(result["test_metrics"])
    return pd.DataFrame(rows).sort_values("macro_f0.5", ascending=False, ignore_index=True)
