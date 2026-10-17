"""Metrics for candidate generation and entity-level match evaluation."""

import pandas as pd

GROUND_TRUTH_COLUMNS = {"source1_entity_id", "matched_entity_ids"}
CANDIDATE_COLUMNS = {"source1_entity_id", "candidate_entity_id", "candidate_source"}
PAIR_COLUMNS = ["source1_entity_id", "candidate_entity_id", "candidate_source"]


def parse_ground_truth(ground_truth: pd.DataFrame) -> pd.DataFrame:
    missing = GROUND_TRUTH_COLUMNS - set(ground_truth.columns)
    if missing:
        raise ValueError(f"Ground truth is missing columns: {sorted(missing)}")
    pairs = []
    for source1_id, matched_ids in ground_truth[["source1_entity_id", "matched_entity_ids"]].itertuples(index=False, name=None):
        if not str(source1_id).strip():
            raise ValueError("Ground truth contains an empty Source 1 ID")
        if not str(matched_ids).strip():
            continue
        for candidate_id in str(matched_ids).split(","):
            candidate_id = candidate_id.strip()
            if not candidate_id:
                continue
            if candidate_id.startswith("S2-"):
                source = "source2"
            elif candidate_id.startswith("S3-"):
                source = "source3"
            else:
                raise ValueError(f"Unexpected matched entity ID: {candidate_id}")
            pairs.append((str(source1_id), candidate_id, source))
    return pd.DataFrame(pairs, columns=PAIR_COLUMNS).drop_duplicates(ignore_index=True)


def _f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta * beta
    denominator = b2 * precision + recall
    return (1 + b2) * precision * recall / denominator if denominator else 0.0


def entity_macro_fbeta(
    ground_truth: pd.DataFrame,
    scored_candidates: pd.DataFrame,
    threshold: float,
    source1_ids=None,
    score_column: str = "match_probability",
    beta: float = 0.5,
) -> dict:
    """Macro-average F-beta per Source 1 entity, including empty/singleton entities.

    Empty prediction + empty truth scores 1.0, per challenge rules. A non-empty
    prediction for an entity with empty truth scores 0.0.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    required_gt = GROUND_TRUTH_COLUMNS
    missing = required_gt - set(ground_truth.columns)
    if missing:
        raise ValueError(f"Ground truth is missing columns: {sorted(missing)}")
    required_candidates = set(PAIR_COLUMNS) | {score_column}
    missing = required_candidates - set(scored_candidates.columns)
    if missing:
        raise ValueError(f"Scored candidates are missing columns: {sorted(missing)}")

    gt_map = {}
    for sid, matches in ground_truth[["source1_entity_id", "matched_entity_ids"]].itertuples(index=False, name=None):
        values = {x.strip() for x in str(matches).split(",") if x.strip()} if str(matches).strip() else set()
        gt_map[str(sid)] = values
    if source1_ids is None:
        ids = list(gt_map)
    else:
        ids = list(dict.fromkeys(str(x) for x in source1_ids))
    if not ids:
        raise ValueError("At least one Source 1 entity is required for macro evaluation")

    predicted = {}
    selected = scored_candidates[scored_candidates[score_column] >= threshold]
    for sid, cid in selected[["source1_entity_id", "candidate_entity_id"]].itertuples(index=False, name=None):
        predicted.setdefault(str(sid), set()).add(str(cid))

    scores, precisions, recalls = [], [], []
    for sid in ids:
        truth = gt_map.get(sid, set())
        pred = predicted.get(sid, set())
        if not truth and not pred:
            p = r = f = 1.0
        elif not truth or not pred:
            p = 0.0 if pred else 1.0
            r = 0.0 if truth else 1.0
            f = 0.0
        else:
            tp = len(truth & pred)
            p = tp / len(pred)
            r = tp / len(truth)
            f = _f_beta(p, r, beta)
        precisions.append(p)
        recalls.append(r)
        scores.append(f)

    return {
        "threshold": threshold,
        "macro_f0.5" if beta == 0.5 else f"macro_f{beta}": sum(scores) / len(scores),
        "macro_precision": sum(precisions) / len(precisions),
        "macro_recall": sum(recalls) / len(recalls),
        "source1_entities": len(ids),
    }


def evaluate_candidate_recall(ground_truth: pd.DataFrame, candidates: pd.DataFrame) -> dict:
    missing = CANDIDATE_COLUMNS - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidates are missing columns: {sorted(missing)}")
    truth = set(parse_ground_truth(ground_truth).itertuples(index=False, name=None))
    candidate_rows = candidates[PAIR_COLUMNS].drop_duplicates()
    candidate_keys = set(candidate_rows.itertuples(index=False, name=None))
    found = truth & candidate_keys
    def source_metrics(source):
        source_truth = {p for p in truth if p[2] == source}
        source_found = source_truth & candidate_keys
        return {"true_pairs": len(source_truth), "found_pairs": len(source_found), "recall": len(source_found) / len(source_truth) if source_truth else 0.0}
    return {"true_pairs": len(truth), "found_pairs": len(found), "candidate_recall": len(found) / len(truth) if truth else 0.0, "source2": source_metrics("source2"), "source3": source_metrics("source3")}


def evaluate_scored_candidates(ground_truth: pd.DataFrame, scored_candidates: pd.DataFrame, threshold: float) -> dict:
    """Legacy heuristic-score evaluation (threshold is on 0-100 scale)."""
    if not 0 <= threshold <= 100:
        raise ValueError("threshold must be between 0 and 100")
    required = set(PAIR_COLUMNS) | {"combined_score"}
    missing = required - set(scored_candidates.columns)
    if missing:
        raise ValueError(f"Scored candidates are missing columns: {sorted(missing)}")
    truth = set(parse_ground_truth(ground_truth).itertuples(index=False, name=None))
    selected = scored_candidates[scored_candidates["combined_score"] >= threshold]
    predicted = set(selected[PAIR_COLUMNS].itertuples(index=False, name=None))
    tp, fp, fn = len(truth & predicted), len(predicted - truth), len(truth - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": threshold, "predicted_pairs": len(predicted), "true_positives": tp, "false_positives": fp, "false_negatives": fn, "precision": precision, "recall": recall, "f1": f1}
