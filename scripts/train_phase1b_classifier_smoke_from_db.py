from __future__ import annotations

import argparse
import json
import sqlite3
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise RuntimeError(f"missing DB: {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def load_training_set_from_db(conn: sqlite3.Connection, training_set_id: str) -> list[dict[str, Any]]:
    training_set = conn.execute(
        """
        SELECT *
        FROM training_sets
        WHERE training_set_id = ?
        """,
        (training_set_id,),
    ).fetchone()

    if training_set is None:
        raise RuntimeError(f"missing training_set_id: {training_set_id}")

    rows = conn.execute(
        """
        SELECT
            i.item_order,
            t.training_snapshot_id,
            t.snapshot_kind,
            t.snapshot_version,
            t.campaign_id,
            t.image_id,
            t.pair_id,
            t.feature_snapshot_id,
            t.layout_spec_id,
            t.label,
            t.decision_label,
            t.group_id,
            t.label_status,
            t.issue_tags_json,
            p.features_json,
            p.feature_status,
            p.duplicate_group_id,
            p.clip_margin,
            p.clip_positive_max_sim,
            p.clip_negative_max_sim,
            p.clip_rank_percentile,
            p.required_region_safe_min,
            p.required_region_safe_mean,
            p.edge_density,
            p.brightness
        FROM training_set_items i
        JOIN training_snapshots t
          ON i.training_snapshot_id = t.training_snapshot_id
        JOIN pair_features p
          ON t.feature_snapshot_id = p.feature_snapshot_id
        WHERE i.training_set_id = ?
        ORDER BY i.item_order
        """,
        (training_set_id,),
    ).fetchall()

    if not rows:
        raise RuntimeError(f"training set has no rows: {training_set_id}")

    return [dict(r) for r in rows]


def flatten_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    flat = []

    for row in rows:
        features = json.loads(row["features_json"])
        if not isinstance(features, dict):
            raise RuntimeError(f"features_json is not object for {row['feature_snapshot_id']}")

        out = {
            "item_order": row["item_order"],
            "training_snapshot_id": row["training_snapshot_id"],
            "snapshot_kind": row["snapshot_kind"],
            "snapshot_version": row["snapshot_version"],
            "campaign_id": row["campaign_id"],
            "image_id": row["image_id"],
            "pair_id": row["pair_id"],
            "feature_snapshot_id": row["feature_snapshot_id"],
            "layout_spec_id": row["layout_spec_id"],
            "label": int(row["label"]),
            "decision_label": row["decision_label"],
            "group_id": row["group_id"],
            "label_status": row["label_status"],
            "feature_status": row["feature_status"],
            "duplicate_group_id": row["duplicate_group_id"],
        }

        for k, v in features.items():
            out[k] = v

        flat.append(out)

    return pd.DataFrame(flat)


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


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def metric_block(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
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


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "item_order",
        "training_snapshot_id",
        "snapshot_kind",
        "snapshot_version",
        "campaign_id",
        "image_id",
        "pair_id",
        "feature_snapshot_id",
        "layout_spec_id",
        "label",
        "decision_label",
        "group_id",
        "label_status",
        "feature_status",
        "duplicate_group_id",
    }

    cols = []
    for c in df.columns:
        if c in excluded:
            continue

        converted = pd.to_numeric(df[c], errors="coerce")
        if converted.notna().any():
            df[c] = converted
            cols.append(c)

    return cols


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/db/cap_reranker_ontology.db")
    ap.add_argument("--training-set-id", default="phase1b_filtered_classifier_v1")
    ap.add_argument("--model-out", default="models/phase1b_smoke/classifier_smoke_model_db.joblib")
    ap.add_argument("--report-out", default="audit/phase_1b/phase_1b_classifier_smoke_report_db.json")
    args = ap.parse_args()

    conn = connect(Path(args.db))
    rows = load_training_set_from_db(conn, args.training_set_id)
    df = flatten_rows(rows)

    if len(df) != 120:
        raise RuntimeError(f"expected 120 DB training rows, got {len(df)}")

    label_counts = Counter(df["label"].astype(int).tolist())
    if dict(label_counts) != {0: 70, 1: 50}:
        raise RuntimeError(f"unexpected label counts: {label_counts}")

    feature_cols = numeric_feature_columns(df)
    if not feature_cols:
        raise RuntimeError("no numeric feature columns found")

    X = df[feature_cols].astype(float)
    y = df["label"].astype(int).to_numpy()
    groups = df["campaign_id"].astype(str).to_numpy()

    logo = LeaveOneGroupOut()

    folds = []
    all_oof_prob = np.zeros(len(y), dtype=float)
    all_seen = np.zeros(len(y), dtype=bool)

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        heldout = sorted(set(groups[test_idx].tolist()))
        if len(heldout) != 1:
            raise RuntimeError(f"unexpected heldout campaigns: {heldout}")

        model = make_pipeline()
        model.fit(X.iloc[train_idx], y[train_idx])

        prob = model.predict_proba(X.iloc[test_idx])[:, 1]
        all_oof_prob[test_idx] = prob
        all_seen[test_idx] = True

        folds.append(
            {
                "fold": fold_idx,
                "heldout_campaign": heldout[0],
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

    if not all_seen.all():
        raise RuntimeError("not all rows received OOF predictions")

    oof_metrics = metric_block(y, all_oof_prob)

    final_model = make_pipeline()
    final_model.fit(X, y)

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": final_model,
            "feature_columns": feature_cols,
            "db": args.db,
            "training_set_id": args.training_set_id,
            "score_status": "diagnostic_only",
            "model_kind": "sklearn_logistic_regression_smoke_classifier_db_first",
            "created_at": utc_now(),
        },
        model_out,
    )

    coefs = final_model.named_steps["classifier"].coef_[0]
    top_coeffs = sorted(
        [
            {
                "feature": name,
                "coefficient": float(coef),
                "abs_coefficient": float(abs(coef)),
            }
            for name, coef in zip(feature_cols, coefs)
        ],
        key=lambda x: x["abs_coefficient"],
        reverse=True,
    )

    report = {
        "metadata": {
            "spec_version": "v2.2.1",
            "phase": "phase_1b",
            "created_at": utc_now(),
            "report_status": "classifier_smoke_training_diagnostic_db_first",
            "score_status": "diagnostic_only",
            "threshold_status": "no_calibrated_threshold",
            "db_role": "operational_source_of_truth",
        },
        "inputs": {
            "db": args.db,
            "training_set_id": args.training_set_id,
        },
        "outputs": {
            "model_out": str(model_out),
            "report_out": args.report_out,
        },
        "dataset": {
            "rows": int(len(df)),
            "campaign_counts": {
                str(k): int(v) for k, v in Counter(groups.tolist()).items()
            },
            "label_counts": {
                str(k): int(v) for k, v in Counter(y.tolist()).items()
            },
            "feature_count": len(feature_cols),
            "feature_columns": feature_cols,
        },
        "model": {
            "kind": "sklearn_logistic_regression",
            "purpose": "db_first_smoke_training_feature_plumbing_check",
            "class_weight": "balanced",
            "calibration": "uncalibrated",
        },
        "leave_one_campaign_out": {
            "folds": folds,
            "out_of_fold_metrics": oof_metrics,
        },
        "top_abs_coefficients": top_coeffs[:30],
        "interpretation": {
            "ko": (
                "이 결과는 DB source-of-truth 기준 classifier smoke training 진단이다. "
                "DB의 named training set에서 120 rows를 읽어 feature matrix, campaign-held-out loop, "
                "model serialization이 동작하는지 확인한다. production quality 또는 calibrated threshold를 주장하지 않는다."
            )
        },
        "non_claims": [
            "production reranker quality를 주장하지 않는다.",
            "calibrated accept/reject threshold를 주장하지 않는다.",
            "automatic accept/reject에 사용하지 않는다.",
            "candidate-level support explanation을 생성하지 않는다.",
        ],
    }

    write_json(Path(args.report_out), report)

    print(json.dumps({
        "event": "done",
        "db": args.db,
        "training_set_id": args.training_set_id,
        "rows": len(df),
        "campaigns": len(set(groups.tolist())),
        "label_counts": {
            str(k): int(v) for k, v in Counter(y.tolist()).items()
        },
        "feature_count": len(feature_cols),
        "oof_metrics": oof_metrics,
        "model_out": str(model_out),
        "report_out": args.report_out,
        "score_status": "diagnostic_only",
        "db_role": "operational_source_of_truth",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
