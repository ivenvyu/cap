from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TRAINING_SNAPSHOT = Path("data/review/phase1b/v2_2_3/training_snapshot_classifier_v2_2_3.jsonl")
FEATURE_GLOB = "data/feature_snapshots/v2_2_2/phase1b_dinov2_anchor/*.jsonl"

MODEL_OUT = Path("models/phase1b_smoke/classifier_smoke_model_v2_2_3.joblib")
REPORT_OUT = Path("audit/phase_1b/phase_1b_classifier_smoke_v2_2_3_jsonl.json")

FEATURE_COLUMNS = [
    "clip_positive_max_sim",
    "clip_positive_mean_sim",
    "clip_negative_max_sim",
    "clip_negative_mean_sim",
    "clip_margin",
    "clip_rank_percentile",

    "dinov2_campaign_pos_nn_sim",
    "dinov2_campaign_neg_nn_sim",
    "dinov2_campaign_margin",
    "dinov2_campaign_pos_count",
    "dinov2_campaign_neg_count",

    "dinov2_family_pos_nn_sim",
    "dinov2_family_neg_nn_sim",
    "dinov2_family_margin",
    "dinov2_family_support_count",

    "dinov2_cluster_positive_rate_coarse",
    "dinov2_cluster_positive_rate_mid",
    "dinov2_cluster_positive_rate_fine",
    "dinov2_cluster_review_count_coarse",
    "dinov2_cluster_review_count_mid",
    "dinov2_cluster_review_count_fine",
    "dinov2_duplicate_group_seen",

    "required_region_safe_mean",
    "required_region_safe_min",
    "title_region_safe_score",
    "info_region_safe_score",
    "edge_density",
    "contrast",
    "brightness",
    "saturation",

    "image_category_gallery",
    "image_category_tree",
    "image_category_flower",
    "path_has_architecture",
    "path_has_garden",

    "campaign_is_summer",
    "campaign_is_walking_program",
    "campaign_is_garden",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_feature_snapshots(pattern: str) -> dict[str, dict[str, Any]]:
    feature_by_id: dict[str, dict[str, Any]] = {}
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"no feature snapshot files matched: {pattern}")

    for file in files:
        for row in read_jsonl(Path(file)):
            fid = row.get("feature_snapshot_id")
            if not fid:
                continue
            feature_by_id[str(fid)] = row

    return feature_by_id


def make_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def main() -> None:
    training_rows_all = read_jsonl(TRAINING_SNAPSHOT)
    feature_by_id = load_feature_snapshots(FEATURE_GLOB)

    usable_rows = []
    excluded_rows = []

    for row in training_rows_all:
        if row.get("use_for_training") is False:
            excluded_rows.append({
                "pair_id": row.get("pair_id"),
                "campaign_id": row.get("campaign_id"),
                "reason": "use_for_training_false",
            })
            continue

        fid = row.get("feature_snapshot_id")
        if fid not in feature_by_id:
            excluded_rows.append({
                "pair_id": row.get("pair_id"),
                "campaign_id": row.get("campaign_id"),
                "feature_snapshot_id": fid,
                "reason": "missing_feature_snapshot",
            })
            continue

        label = int(row["label"])
        if label not in (0, 1):
            excluded_rows.append({
                "pair_id": row.get("pair_id"),
                "campaign_id": row.get("campaign_id"),
                "label": row.get("label"),
                "reason": "unsupported_label",
            })
            continue

        usable_rows.append(row)

    if not usable_rows:
        raise RuntimeError("no usable training rows")

    used_feature_columns = []
    for col in FEATURE_COLUMNS:
        values = []
        for row in usable_rows:
            f = feature_by_id[row["feature_snapshot_id"]].get("features", {})
            values.append(f.get(col))
        non_null = [v for v in values if v is not None]
        if non_null:
            used_feature_columns.append(col)

    X = []
    y = []
    campaigns = []
    decision_labels = []
    source_snapshots = []

    for row in usable_rows:
        f = feature_by_id[row["feature_snapshot_id"]].get("features", {})
        X.append([f.get(col) for col in used_feature_columns])
        y.append(int(row["label"]))
        campaigns.append(str(row.get("campaign_id", "")))
        decision_labels.append(str(row.get("decision_label", "")))
        source_snapshots.append(str(row.get("source_snapshot", "")))

    X_np = np.array(X, dtype=float)
    y_np = np.array(y, dtype=int)
    groups = np.array(campaigns)

    logo = LeaveOneGroupOut()
    oof_score = np.zeros(len(y_np), dtype=float)
    fold_reports = []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X_np, y_np, groups), start=1):
        held_out = sorted(set(groups[test_idx].tolist()))

        # train/test 어느 한쪽이라도 label이 하나뿐이면 logistic training/eval이 불안정하므로 기록만 하고 skip
        if len(set(y_np[train_idx].tolist())) < 2 or len(set(y_np[test_idx].tolist())) < 2:
            fold_reports.append({
                "fold": fold_idx,
                "held_out_campaigns": held_out,
                "test_n": int(len(test_idx)),
                "train_label_counts": dict(Counter(map(str, y_np[train_idx].tolist()))),
                "test_label_counts": dict(Counter(map(str, y_np[test_idx].tolist()))),
                "skipped": True,
                "reason": "single_class_train_or_test",
            })
            oof_score[test_idx] = float(np.mean(y_np[train_idx]))
            continue

        model = make_pipeline()
        model.fit(X_np[train_idx], y_np[train_idx])
        score = model.predict_proba(X_np[test_idx])[:, 1]
        pred = (score >= 0.5).astype(int)
        oof_score[test_idx] = score

        fold_reports.append({
            "fold": fold_idx,
            "held_out_campaigns": held_out,
            "test_n": int(len(test_idx)),
            "train_label_counts": dict(Counter(map(str, y_np[train_idx].tolist()))),
            "test_label_counts": dict(Counter(map(str, y_np[test_idx].tolist()))),
            "balanced_accuracy": float(balanced_accuracy_score(y_np[test_idx], pred)),
            "roc_auc": safe_auc(y_np[test_idx], score),
            "skipped": False,
        })

    oof_pred = (oof_score >= 0.5).astype(int)

    final_model = make_pipeline()
    final_model.fit(X_np, y_np)

    clf = final_model.named_steps["clf"]
    coefs = clf.coef_[0]
    top = sorted(
        [
            {"feature": col, "coefficient": float(coef)}
            for col, coef in zip(used_feature_columns, coefs)
        ],
        key=lambda x: abs(x["coefficient"]),
        reverse=True,
    )[:15]

    report = {
        "report_status": "classifier_smoke_training_diagnostic_v2_2_3_jsonl",
        "score_status": "diagnostic_only",
        "threshold_status": "no_calibrated_threshold",
        "model_out": str(MODEL_OUT),
        "report_out": str(REPORT_OUT),
        "dataset": {
            "rows_all": len(training_rows_all),
            "rows_used_for_training": len(usable_rows),
            "rows_excluded": len(excluded_rows),
            "excluded_reason_counts": dict(Counter(x["reason"] for x in excluded_rows)),
            "campaign_counts": dict(Counter(campaigns)),
            "label_counts": dict(Counter(map(str, y_np.tolist()))),
            "decision_label_counts": dict(Counter(decision_labels)),
            "source_snapshot_counts": dict(Counter(source_snapshots)),
            "configured_feature_count": len(FEATURE_COLUMNS),
            "used_feature_count": len(used_feature_columns),
            "used_feature_columns": used_feature_columns,
        },
        "oof": {
            "n": int(len(y_np)),
            "label_counts": dict(Counter(map(str, y_np.tolist()))),
            "accuracy": float(accuracy_score(y_np, oof_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_np, oof_pred)),
            "confusion_matrix_labels": ["0", "1"],
            "confusion_matrix": confusion_matrix(y_np, oof_pred, labels=[0, 1]).tolist(),
            "diagnostic_threshold": 0.5,
            "threshold_status": "sklearn_default_diagnostic_only_not_calibrated",
            "per_class": classification_report(y_np, oof_pred, labels=[0, 1], output_dict=True, zero_division=0),
            "roc_auc": safe_auc(y_np, oof_score),
            "average_precision": safe_ap(y_np, oof_score),
        },
        "folds": fold_reports,
        "top_coefficients": top,
        "excluded_rows_sample": excluded_rows[:20],
        "non_claims": [
            "calibrated accept/reject threshold가 아니다.",
            "production model quality 주장이 아니다.",
            "campaign 간 score 직접 비교에 사용하지 않는다.",
            "coverage_gap campaign은 일반 quality claim에서 분리한다.",
        ],
    }

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "feature_columns": used_feature_columns,
            "model_id": "classifier_smoke_v2_2_3",
            "score_status": "diagnostic_only",
            "threshold_status": "no_calibrated_threshold",
        },
        MODEL_OUT,
    )
    REPORT_OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({
        "event": "done",
        "model_out": str(MODEL_OUT),
        "report_out": str(REPORT_OUT),
        "rows_all": len(training_rows_all),
        "rows_used_for_training": len(usable_rows),
        "rows_excluded": len(excluded_rows),
        "label_counts": report["dataset"]["label_counts"],
        "campaign_counts": report["dataset"]["campaign_counts"],
        "source_snapshot_counts": report["dataset"]["source_snapshot_counts"],
        "oof_balanced_accuracy": report["oof"]["balanced_accuracy"],
        "oof_roc_auc": report["oof"]["roc_auc"],
        "oof_average_precision": report["oof"]["average_precision"],
        "score_status": "diagnostic_only",
        "threshold_status": "no_calibrated_threshold",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
