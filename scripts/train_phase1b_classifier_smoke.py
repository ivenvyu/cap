from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def flatten_feature_row(row: dict[str, Any]) -> dict[str, Any]:
    out = {
        "feature_snapshot_id": row["feature_snapshot_id"],
        "pair_id": row["pair_id"],
        "campaign_id": row["campaign_id"],
        "image_id": row["image_id"],
        "layout_spec_id": row.get("layout_spec_id"),
        "feature_status": row.get("feature_status"),
    }

    features = row.get("features", {})
    if not isinstance(features, dict):
        raise RuntimeError(f"features is not dict for {row['feature_snapshot_id']}")

    for k, v in features.items():
        out[k] = v

    return out


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def metric_block(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    # Diagnostic-only default classifier threshold.
    # This is not a calibrated accept/reject threshold.
    y_pred = (y_prob >= 0.5).astype(int)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "n": int(len(y_true)),
        "label_counts": {str(k): int(v) for k, v in Counter(y_true.tolist()).items()},
        "diagnostic_threshold": 0.5,
        "threshold_status": "sklearn_default_diagnostic_only_not_calibrated",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": safe_auc(y_true, y_prob),
        "average_precision": safe_average_precision(y_true, y_prob),
        "confusion_matrix_labels": ["0", "1"],
        "confusion_matrix": cm.astype(int).tolist(),
        "per_class": {
            "0": {
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
                "support": int(support[0]),
            },
            "1": {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1": float(f1[1]),
                "support": int(support[1]),
            },
        },
    }


def make_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--training",
        default="data/review/phase1b/filtered/training_snapshot_phase1b_classifier_v1__filtered.jsonl",
    )
    ap.add_argument(
        "--feature-dir",
        default="data/feature_snapshots/v2_2_1/phase1b",
    )
    ap.add_argument(
        "--model-out",
        default="models/phase1b_smoke/classifier_smoke_model.joblib",
    )
    ap.add_argument(
        "--report-out",
        default="audit/phase_1b/phase_1b_classifier_smoke_report.json",
    )
    args = ap.parse_args()

    training_path = Path(args.training)
    feature_dir = Path(args.feature_dir)

    training_rows = read_jsonl(training_path)
    if not training_rows:
        raise RuntimeError(f"empty training rows: {training_path}")

    feature_files = sorted(feature_dir.glob("pair_feature_snapshots__*.jsonl"))
    if not feature_files:
        raise RuntimeError(f"no feature snapshot files found: {feature_dir}")

    feature_rows = []
    for p in feature_files:
        feature_rows.extend(read_jsonl(p))

    feature_df = pd.DataFrame([flatten_feature_row(r) for r in feature_rows])
    train_df = pd.DataFrame(training_rows)

    if "feature_snapshot_id" not in train_df.columns:
        raise RuntimeError("training rows missing feature_snapshot_id")
    if "label" not in train_df.columns:
        raise RuntimeError("training rows missing label")

    merged = train_df.merge(
        feature_df,
        on=["feature_snapshot_id", "pair_id", "campaign_id", "image_id"],
        how="left",
        suffixes=("_train", "_feature"),
        validate="many_to_one",
    )

    if merged.isna().all(axis=1).any():
        raise RuntimeError("bad merge produced empty rows")

    missing_features = merged[merged["feature_status"].isna()]
    if len(missing_features):
        raise RuntimeError(f"missing feature rows after join: {len(missing_features)}")

    excluded_non_feature_cols = {
        "training_snapshot_id",
        "snapshot_version",
        "created_at",
        "snapshot_kind",
        "label_policy",
        "aggregation_policy",
        "source_review_event_ids",
        "pair_id",
        "campaign_id",
        "image_id",
        "layout_spec_id_train",
        "layout_spec_id_feature",
        "preview_renderer_version",
        "duplicate_group_id",
        "feature_snapshot_id",
        "label",
        "label_status",
        "decision_label",
        "issue_tags",
        "preference_rank",
        "group_id",
        "relevance_grade",
        "disagreement",
        "audit",
        "feature_status",
    }

    candidate_feature_cols = [
        c for c in merged.columns
        if c not in excluded_non_feature_cols
    ]

    numeric_features = []
    for c in candidate_feature_cols:
        converted = pd.to_numeric(merged[c], errors="coerce")
        if converted.notna().any():
            merged[c] = converted
            numeric_features.append(c)

    if not numeric_features:
        raise RuntimeError("no numeric features found")

    X = merged[numeric_features].astype(float)
    y = merged["label"].astype(int).to_numpy()
    groups = merged["campaign_id"].astype(str).to_numpy()

    if set(np.unique(y).tolist()) != {0, 1}:
        raise RuntimeError(f"classifier labels must be binary 0/1, got {sorted(set(y.tolist()))}")

    logo = LeaveOneGroupOut()
    folds = []

    all_oof_prob = np.zeros(len(y), dtype=float)
    all_oof_seen = np.zeros(len(y), dtype=bool)

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        heldout_campaign = sorted(set(groups[test_idx].tolist()))
        if len(heldout_campaign) != 1:
            raise RuntimeError(f"unexpected heldout campaigns in fold {fold_idx}: {heldout_campaign}")

        model = make_pipeline()
        model.fit(X.iloc[train_idx], y[train_idx])

        prob = model.predict_proba(X.iloc[test_idx])[:, 1]
        all_oof_prob[test_idx] = prob
        all_oof_seen[test_idx] = True

        folds.append(
            {
                "fold": fold_idx,
                "heldout_campaign": heldout_campaign[0],
                "train_n": int(len(train_idx)),
                "test_n": int(len(test_idx)),
                "train_label_counts": {
                    str(k): int(v) for k, v in Counter(y[train_idx].tolist()).items()
                },
                "test_label_counts": {
                    str(k): int(v) for k, v in Counter(y[test_idx].tolist()).items()
                },
                "metrics": metric_block(y[test_idx], prob),
            }
        )

    if not all_oof_seen.all():
        raise RuntimeError("not all rows received out-of-fold predictions")

    oof_metrics = metric_block(y, all_oof_prob)

    final_model = make_pipeline()
    final_model.fit(X, y)

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    model_payload = {
        "model": final_model,
        "feature_columns": numeric_features,
        "training_path": str(training_path),
        "feature_dir": str(feature_dir),
        "score_status": "diagnostic_only",
        "model_kind": "sklearn_logistic_regression_smoke_classifier",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    joblib.dump(model_payload, model_out)

    clf = final_model.named_steps["classifier"]
    coefs = clf.coef_[0]
    feature_coefficients = sorted(
        [
            {
                "feature": name,
                "coefficient": float(coef),
                "abs_coefficient": float(abs(coef)),
            }
            for name, coef in zip(numeric_features, coefs)
        ],
        key=lambda x: x["abs_coefficient"],
        reverse=True,
    )

    report = {
        "metadata": {
            "spec_version": "v2.2.1",
            "phase": "phase_1b",
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "report_status": "classifier_smoke_training_diagnostic",
            "score_status": "diagnostic_only",
            "threshold_status": "no_calibrated_threshold",
        },
        "inputs": {
            "training": str(training_path),
            "feature_dir": str(feature_dir),
        },
        "outputs": {
            "model_out": str(model_out),
            "report_out": args.report_out,
        },
        "dataset": {
            "rows": int(len(merged)),
            "campaign_counts": {
                str(k): int(v) for k, v in Counter(groups.tolist()).items()
            },
            "label_counts": {
                str(k): int(v) for k, v in Counter(y.tolist()).items()
            },
            "feature_count": len(numeric_features),
            "feature_columns": numeric_features,
        },
        "model": {
            "kind": "sklearn_logistic_regression",
            "purpose": "smoke_training_feature_plumbing_check",
            "class_weight": "balanced",
            "calibration": "uncalibrated",
        },
        "leave_one_campaign_out": {
            "folds": folds,
            "out_of_fold_metrics": oof_metrics,
        },
        "top_abs_coefficients": feature_coefficients[:30],
        "interpretation": {
            "ko": (
                "이 결과는 classifier smoke training 진단이다. "
                "feature join, feature matrix 생성, campaign-held-out loop, model serialization이 "
                "동작하는지 확인하기 위한 것이며, production quality 또는 calibrated threshold를 주장하지 않는다."
            ),
            "excluded_campaign_note": (
                "phase1b_indoor_gallery_winter_art는 raw pool coverage gap diagnostic으로 제외된 filtered set을 사용했다."
            ),
        },
        "non_claims": [
            "production reranker quality를 주장하지 않는다.",
            "calibrated accept/reject threshold를 주장하지 않는다.",
            "이 결과로 자동 accept/reject를 수행하지 않는다.",
            "ranker generalization 성능을 주장하지 않는다.",
            "support explanation은 여전히 Phase 1b에서 deferred 상태다.",
        ],
    }

    write_json(Path(args.report_out), report)

    print(json.dumps({
        "event": "done",
        "rows": len(merged),
        "campaigns": len(set(groups.tolist())),
        "label_counts": {
            str(k): int(v) for k, v in Counter(y.tolist()).items()
        },
        "feature_count": len(numeric_features),
        "oof_metrics": oof_metrics,
        "model_out": str(model_out),
        "report_out": args.report_out,
        "score_status": "diagnostic_only",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
