from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCORE_STATUS = "diagnostic_only"
THRESHOLD_STATUS = "no_calibrated_threshold"

EXPECTED_PHASE1B_FEATURE_COLUMNS = [
    "brightness",
    "campaign_is_garden",
    "campaign_is_summer",
    "campaign_is_walking_program",
    "clip_margin",
    "clip_negative_max_sim",
    "clip_negative_mean_sim",
    "clip_positive_max_sim",
    "clip_positive_mean_sim",
    "clip_rank_percentile",
    "contrast",
    "dinov2_campaign_anchor_missing",
    "dinov2_campaign_neg_count",
    "dinov2_campaign_pos_count",
    "dinov2_cluster_review_count_coarse",
    "dinov2_cluster_review_count_fine",
    "dinov2_cluster_review_count_mid",
    "dinov2_duplicate_group_seen",
    "dinov2_family_anchor_missing",
    "dinov2_family_support_count",
    "edge_density",
    "image_category_course",
    "image_category_flower",
    "image_category_gallery",
    "image_category_tree",
    "image_season_unknown",
    "info_region_safe_score",
    "path_has_architecture",
    "path_has_garden",
    "required_region_safe_mean",
    "required_region_safe_min",
    "saturation",
    "title_region_safe_score",
]


ID_LIKE_COLUMNS = {
    "training_set_id",
    "training_snapshot_id",
    "snapshot_kind",
    "snapshot_version",
    "campaign_id",
    "image_id",
    "pair_id",
    "feature_snapshot_id",
    "layout_spec_id",
    "decision_label",
    "group_id",
    "label_status",
    "issue_tags_json",
    "artifact_id",
    "raw_json",
    "created_at",
    "score_status",
    "metadata_status",
}


def connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise RuntimeError(f"missing DB: {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            """
            SELECT COUNT(*)
            FROM sqlite_master
            WHERE type IN ('table', 'view')
              AND name = ?
            """,
            (table,),
        ).fetchone()[0]
        > 0
    )


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(r["name"])
        for r in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


def load_effective_classifier_dataset(conn: sqlite3.Connection) -> pd.DataFrame:
    required_views = [
        "v_effective_training_set_items_v1",
        "v_effective_pair_features_v1",
    ]
    for view in required_views:
        if not table_exists(conn, view):
            raise RuntimeError(f"missing required view: {view}")

    tsi_cols = table_columns(conn, "v_effective_training_set_items_v1")
    pf_cols = table_columns(conn, "v_effective_pair_features_v1")

    join_keys = ["campaign_id", "image_id"]
    if "layout_spec_id" in tsi_cols and "layout_spec_id" in pf_cols:
        join_keys.append("layout_spec_id")

    join_clause = " AND ".join([f"tsi.{k} = pf.{k}" for k in join_keys])

    sql = f"""
    SELECT
        tsi.training_set_id,
        tsi.training_snapshot_id,
        tsi.snapshot_kind,
        tsi.campaign_id,
        tsi.image_id,
        tsi.layout_spec_id,
        tsi.label,
        tsi.decision_label,
        pf.*
    FROM v_effective_training_set_items_v1 tsi
    JOIN v_effective_pair_features_v1 pf
      ON {join_clause}
    WHERE tsi.training_set_id = 'phase1b_filtered_classifier_v1'
      AND tsi.snapshot_kind = 'classifier'
    ORDER BY tsi.campaign_id, tsi.training_snapshot_id
    """

    df = pd.read_sql_query(sql, conn)

    if df.empty:
        raise RuntimeError("effective classifier dataset is empty")

    # If pf.* duplicated key columns, pandas may suffix columns; remove duplicate names robustly.
    df = df.loc[:, ~df.columns.duplicated()]

    # The complete Phase 1b feature set lives in pair_features.features_json.
    # Direct SQL columns contain only a small diagnostic subset.
    df = expand_features_json(df)

    return df


def expand_features_json(df: pd.DataFrame) -> pd.DataFrame:
    if "features_json" not in df.columns:
        raise RuntimeError("effective dataset missing features_json column")

    feature_rows: list[dict[str, Any]] = []

    for i, raw in enumerate(df["features_json"].tolist()):
        if raw is None:
            raise RuntimeError(f"features_json is null at row {i}")

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"invalid features_json at row {i}") from e

        if not isinstance(parsed, dict):
            raise RuntimeError(f"features_json must decode to object at row {i}")

        # pair_features.features_json stores the feature map directly.
        # raw_json stores {"features": {...}}, but we intentionally use features_json.
        feature_rows.append(parsed)

    features_df = pd.DataFrame(feature_rows, index=df.index)

    # Do not overwrite existing identifier/label columns. Feature columns with
    # the same name as direct pair_features columns are replaced by the canonical
    # values from features_json to match the original DB-first trainer behavior.
    non_feature_cols = [c for c in df.columns if c not in features_df.columns]
    out = pd.concat([df[non_feature_cols], features_df], axis=1)

    return out


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    missing = [c for c in EXPECTED_PHASE1B_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(
            "effective dataset is missing expected Phase 1b feature columns: "
            + ", ".join(missing)
        )
    return list(EXPECTED_PHASE1B_FEATURE_COLUMNS)


def coerce_feature_frame(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    for col in feature_cols:
        s = df[col]

        if s.dtype == bool:
            out[col] = s.astype(float)
            continue

        # DB imports may store booleans as text.
        if s.dtype == object:
            lowered = s.astype(str).str.strip().str.lower()
            bool_map = {
                "true": 1.0,
                "false": 0.0,
                "yes": 1.0,
                "no": 0.0,
                "none": np.nan,
                "null": np.nan,
                "nan": np.nan,
                "": np.nan,
            }
            mapped = lowered.map(bool_map)
            numeric = pd.to_numeric(s, errors="coerce")
            out[col] = numeric.where(numeric.notna(), mapped)
        else:
            out[col] = pd.to_numeric(s, errors="coerce")

    return out


def train_oof(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    y = df["label"].astype(int).to_numpy()
    groups = df["campaign_id"].astype(str).to_numpy()
    X = coerce_feature_frame(df, feature_cols)

    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        raise RuntimeError("need at least 2 campaign groups for grouped OOF diagnostic")

    n_splits = len(unique_groups)
    splitter = GroupKFold(n_splits=n_splits)

    oof_prob = np.zeros(len(df), dtype=float)
    oof_pred = np.zeros(len(df), dtype=int)
    folds: list[dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        test_campaigns = sorted(set(groups[test_idx]))
        if len(test_campaigns) != 1:
            raise RuntimeError(f"expected one held-out campaign, got {test_campaigns}")

        pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=13,
                    ),
                ),
            ]
        )

        pipe.fit(X.iloc[train_idx], y[train_idx])
        prob = pipe.predict_proba(X.iloc[test_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)

        oof_prob[test_idx] = prob
        oof_pred[test_idx] = pred

        y_test = y[test_idx]
        labels = {
            str(k): int(v)
            for k, v in pd.Series(y_test).value_counts().sort_index().to_dict().items()
        }

        try:
            auc = float(roc_auc_score(y_test, prob))
        except ValueError:
            auc = None

        folds.append(
            {
                "fold": fold_idx,
                "held_out_campaign": test_campaigns[0],
                "test_n": int(len(test_idx)),
                "labels": labels,
                "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                "roc_auc": auc,
            }
        )

    metrics: dict[str, Any] = {
        "n": int(len(df)),
        "label_counts": {
            str(k): int(v)
            for k, v in pd.Series(y).value_counts().sort_index().to_dict().items()
        },
        "accuracy": float(accuracy_score(y, oof_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, oof_pred)),
        "confusion_matrix_labels": ["0", "1"],
        "confusion_matrix": confusion_matrix(y, oof_pred, labels=[0, 1]).tolist(),
        "diagnostic_threshold": 0.5,
        "threshold_status": "sklearn_default_diagnostic_only_not_calibrated",
        "per_class": classification_report(
            y,
            oof_pred,
            labels=[0, 1],
            output_dict=True,
            zero_division=0,
        ),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y, oof_prob))
    except ValueError:
        metrics["roc_auc"] = None

    try:
        metrics["average_precision"] = float(average_precision_score(y, oof_prob))
    except ValueError:
        metrics["average_precision"] = None

    return {
        "out_of_fold_metrics": metrics,
        "folds": folds,
    }


def fit_full_model_coefficients(df: pd.DataFrame, feature_cols: list[str]) -> list[dict[str, Any]]:
    y = df["label"].astype(int).to_numpy()
    X = coerce_feature_frame(df, feature_cols)

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=13,
                ),
            ),
        ]
    )
    pipe.fit(X, y)

    clf = pipe.named_steps["clf"]
    coefs = clf.coef_[0]

    rows = [
        {"feature": feature, "coefficient": float(coef)}
        for feature, coef in zip(feature_cols, coefs)
    ]
    rows.sort(key=lambda r: abs(r["coefficient"]), reverse=True)
    return rows[:20]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/db/cap_reranker_ontology.db")
    ap.add_argument("--report-out", default="audit/phase_1b/phase_1b_classifier_smoke_effective_db.json")
    args = ap.parse_args()

    conn = connect(Path(args.db))

    df = load_effective_classifier_dataset(conn)
    feature_cols = infer_feature_columns(df)

    result = train_oof(df, feature_cols)
    top_coefficients = fit_full_model_coefficients(df, feature_cols)

    report = {
        "report_status": "classifier_smoke_training_diagnostic_effective_db_first",
        "db_role": "operational_source_of_truth",
        "score_status": SCORE_STATUS,
        "threshold_status": THRESHOLD_STATUS,
        "effective_input_policy": {
            "training_items_view": "v_effective_training_set_items_v1",
            "pair_features_view": "v_effective_pair_features_v1",
            "filter": "flower_season_exclusion_filter_v1",
            "interpretation": (
                "known flower images whose bloom season conflicts with the campaign season "
                "are removed before diagnostic smoke training"
            ),
        },
        "dataset": {
            "rows": int(len(df)),
            "campaign_counts": {
                str(k): int(v)
                for k, v in df["campaign_id"].value_counts().sort_index().to_dict().items()
            },
            "label_counts": {
                str(k): int(v)
                for k, v in df["label"].astype(int).value_counts().sort_index().to_dict().items()
            },
            "decision_label_counts": {
                str(k): int(v)
                for k, v in df["decision_label"].value_counts().sort_index().to_dict().items()
            },
            "feature_count": len(feature_cols),
            "feature_columns": feature_cols,
        },
        **result,
        "top_coefficients": top_coefficients,
    }

    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(jdump(report) + "\n", encoding="utf-8")

    print(json.dumps(
        {
            "report_status": report["report_status"],
            "db_role": report["db_role"],
            "score_status": report["score_status"],
            "threshold_status": report["threshold_status"],
            "dataset": report["dataset"],
            "oof": report["out_of_fold_metrics"],
            "folds": report["folds"],
            "top_coefficients": report["top_coefficients"][:10],
            "report_out": str(out),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
