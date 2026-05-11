from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
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
MODEL_FAMILY = "sklearn_logistic_regression_smoke"
EXCLUDED_FEATURE_GROUPS = {"critic"}
EXCLUDED_EXACT_FEATURES = {
    "dinov2_cluster_id_coarse",
    "dinov2_cluster_id_mid",
    "dinov2_cluster_id_fine",
}


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_feature_profile(path: Path) -> tuple[list[str], dict[str, str]]:
    profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    groups = profile.get("feature_groups", {})
    feature_cols: list[str] = []
    feature_actions: dict[str, str] = {}

    for group_name, group in groups.items():
        if group_name in EXCLUDED_FEATURE_GROUPS:
            continue
        action = str(group.get("training_action", "unspecified"))
        for feature in group.get("features", []):
            if feature in EXCLUDED_EXACT_FEATURES:
                continue
            if feature not in feature_cols:
                feature_cols.append(feature)
            feature_actions[feature] = action

    if not feature_cols:
        raise RuntimeError(f"no usable features in profile: {path}")
    return feature_cols, feature_actions


def load_features(paths: list[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            fid = row.get("feature_snapshot_id")
            if not isinstance(fid, str) or not fid:
                raise RuntimeError(f"feature row missing feature_snapshot_id: {path}")
            if fid in out:
                raise RuntimeError(f"duplicate feature_snapshot_id: {fid}")
            features = row.get("features")
            if not isinstance(features, dict):
                raise RuntimeError(f"feature row missing features object: {fid}")
            out[fid] = row
    return out


def build_dataset(
    *,
    training_snapshot_path: Path,
    feature_paths: list[Path],
    feature_cols: list[str],
    feature_actions: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    snapshots = read_jsonl(training_snapshot_path)
    if not snapshots:
        raise RuntimeError(f"empty training snapshot file: {training_snapshot_path}")

    features_by_id = load_features(feature_paths)
    rows: list[dict[str, Any]] = []
    missing_feature_ids: list[str] = []

    for snap in snapshots:
        if snap.get("snapshot_kind") != "classifier":
            continue
        if snap.get("label_status") != "human_reviewed":
            continue
        fid = snap.get("feature_snapshot_id")
        feat_row = features_by_id.get(fid)
        if feat_row is None:
            missing_feature_ids.append(str(fid))
            continue

        features = feat_row["features"]
        row: dict[str, Any] = {
            "training_snapshot_id": snap["training_snapshot_id"],
            "snapshot_version": snap["snapshot_version"],
            "campaign_id": snap["campaign_id"],
            "image_id": snap["image_id"],
            "pair_id": snap["pair_id"],
            "layout_spec_id": snap.get("layout_spec_id"),
            "feature_snapshot_id": fid,
            "decision_label": snap["decision_label"],
            "label": snap["label"],
        }
        for col in feature_cols:
            row[col] = features.get(col)
        rows.append(row)

    if missing_feature_ids:
        sample = ", ".join(missing_feature_ids[:10])
        raise RuntimeError(f"training snapshots reference missing feature ids: {sample}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("joined classifier dataset is empty")

    X_raw = coerce_feature_frame(df, feature_cols)
    diagnostics = feature_diagnostics(X_raw, feature_cols, feature_actions)
    usable_cols = [
        r["feature"]
        for r in diagnostics
        if r["numeric_non_null_count"] > 0 and not r["is_constant_non_null"]
    ]
    if not usable_cols:
        raise RuntimeError("no usable non-constant numeric features for smoke training")

    return df, X_raw[usable_cols], diagnostics


def coerce_feature_frame(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in feature_cols:
        s = df[col] if col in df.columns else pd.Series([np.nan] * len(df), index=df.index)
        if s.dtype == bool:
            out[col] = s.astype(float)
            continue
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


def feature_diagnostics(
    X: pd.DataFrame,
    feature_cols: list[str],
    feature_actions: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = int(len(X))
    for col in feature_cols:
        s = X[col]
        non_null = s.dropna()
        unique_non_null = int(non_null.nunique(dropna=True))
        rows.append(
            {
                "feature": col,
                "configured_training_action": feature_actions.get(col),
                "row_count": n,
                "numeric_non_null_count": int(non_null.shape[0]),
                "numeric_null_count": int(s.isna().sum()),
                "numeric_null_rate": float(s.isna().mean()) if n else None,
                "unique_non_null_count": unique_non_null,
                "is_constant_non_null": bool(unique_non_null <= 1),
                "used_in_smoke_training": bool(non_null.shape[0] > 0 and unique_non_null > 1),
                "diagnostic_only": True,
            }
        )
    return rows


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
                    random_state=13,
                ),
            ),
        ]
    )


def train_oof(df: pd.DataFrame, X: pd.DataFrame) -> dict[str, Any]:
    y = df["label"].astype(int).to_numpy()
    groups = df["campaign_id"].astype(str).to_numpy()
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        raise RuntimeError("need at least 2 campaign groups for grouped OOF diagnostic")

    splitter = GroupKFold(n_splits=len(unique_groups))
    oof_prob = np.zeros(len(df), dtype=float)
    oof_pred = np.zeros(len(df), dtype=int)
    folds: list[dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        held_out = sorted(set(groups[test_idx]))
        pipe = make_pipeline()
        pipe.fit(X.iloc[train_idx], y[train_idx])
        prob = pipe.predict_proba(X.iloc[test_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        oof_prob[test_idx] = prob
        oof_pred[test_idx] = pred

        y_test = y[test_idx]
        try:
            auc = float(roc_auc_score(y_test, prob))
        except ValueError:
            auc = None
        folds.append(
            {
                "fold": fold_idx,
                "held_out_campaigns": held_out,
                "test_n": int(len(test_idx)),
                "labels": {str(k): int(v) for k, v in pd.Series(y_test).value_counts().sort_index().to_dict().items()},
                "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                "roc_auc": auc,
            }
        )

    metrics: dict[str, Any] = {
        "n": int(len(df)),
        "label_counts": {str(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().to_dict().items()},
        "accuracy": float(accuracy_score(y, oof_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, oof_pred)),
        "confusion_matrix_labels": ["0", "1"],
        "confusion_matrix": confusion_matrix(y, oof_pred, labels=[0, 1]).tolist(),
        "diagnostic_threshold": 0.5,
        "threshold_status": "sklearn_default_diagnostic_only_not_calibrated",
        "per_class": classification_report(y, oof_pred, labels=[0, 1], output_dict=True, zero_division=0),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y, oof_prob))
    except ValueError:
        metrics["roc_auc"] = None
    try:
        metrics["average_precision"] = float(average_precision_score(y, oof_prob))
    except ValueError:
        metrics["average_precision"] = None

    return {"out_of_fold_metrics": metrics, "folds": folds}


def fit_full_model(df: pd.DataFrame, X: pd.DataFrame) -> tuple[Pipeline, list[dict[str, Any]]]:
    y = df["label"].astype(int).to_numpy()
    pipe = make_pipeline()
    pipe.fit(X, y)
    coefs = pipe.named_steps["clf"].coef_[0]
    rows = [{"feature": col, "coefficient": float(coef)} for col, coef in zip(X.columns, coefs)]
    rows.sort(key=lambda r: abs(r["coefficient"]), reverse=True)
    return pipe, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--training-snapshots",
        default="data/review/phase1b/v2_2_2/training_snapshot_phase1b_classifier_v2_2_2.jsonl",
    )
    ap.add_argument(
        "--feature-snapshots-glob",
        default="data/feature_snapshots/v2_2_2/phase1b_dinov2_anchor/*.jsonl",
    )
    ap.add_argument("--feature-profile", default="configs/feature_profile_v2_2_2.yaml")
    ap.add_argument("--model-out", default="models/phase1b_smoke/classifier_smoke_model_v2_2_2.joblib")
    ap.add_argument("--report-out", default="audit/phase_1b/phase_1b_classifier_smoke_v2_2_2_jsonl.json")
    args = ap.parse_args()

    feature_paths = sorted(Path().glob(args.feature_snapshots_glob))
    if not feature_paths:
        raise RuntimeError(f"no feature snapshot files matched: {args.feature_snapshots_glob}")

    feature_cols, feature_actions = load_feature_profile(Path(args.feature_profile))
    df, X, feature_diag = build_dataset(
        training_snapshot_path=Path(args.training_snapshots),
        feature_paths=feature_paths,
        feature_cols=feature_cols,
        feature_actions=feature_actions,
    )

    result = train_oof(df, X)
    pipe, top_coefficients = fit_full_model(df, X)

    report = {
        "report_status": "classifier_smoke_training_diagnostic_v2_2_2_jsonl",
        "score_status": SCORE_STATUS,
        "threshold_status": THRESHOLD_STATUS,
        "model_family": MODEL_FAMILY,
        "input_policy": {
            "training_snapshots": args.training_snapshots,
            "feature_snapshots_glob": args.feature_snapshots_glob,
            "feature_profile": args.feature_profile,
            "feature_selection": "numeric_non_null_and_non_constant_diagnostic_only",
            "excluded_feature_groups": sorted(EXCLUDED_FEATURE_GROUPS),
            "excluded_exact_features": sorted(EXCLUDED_EXACT_FEATURES),
            "interpretation": "Diagnostic smoke model only. Do not report as calibrated production reranker performance.",
        },
        "dataset": {
            "rows": int(len(df)),
            "campaign_counts": {str(k): int(v) for k, v in df["campaign_id"].value_counts().sort_index().to_dict().items()},
            "label_counts": {str(k): int(v) for k, v in df["label"].astype(int).value_counts().sort_index().to_dict().items()},
            "decision_label_counts": {str(k): int(v) for k, v in df["decision_label"].value_counts().sort_index().to_dict().items()},
            "configured_feature_count": len(feature_cols),
            "used_feature_count": int(X.shape[1]),
            "used_feature_columns": list(X.columns),
        },
        **result,
        "feature_diagnostics": feature_diag,
        "top_coefficients": top_coefficients[:30],
    }

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipe,
            "feature_columns": list(X.columns),
            "score_status": SCORE_STATUS,
            "threshold_status": THRESHOLD_STATUS,
            "model_family": MODEL_FAMILY,
            "training_snapshots": args.training_snapshots,
            "feature_snapshots_glob": args.feature_snapshots_glob,
        },
        model_out,
    )

    report_out = Path(args.report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(jdump(report) + "\n", encoding="utf-8")

    print(json.dumps({
        "report_status": report["report_status"],
        "score_status": SCORE_STATUS,
        "threshold_status": THRESHOLD_STATUS,
        "model_out": str(model_out),
        "report_out": str(report_out),
        "dataset": report["dataset"],
        "oof": report["out_of_fold_metrics"],
        "folds": report["folds"],
        "top_coefficients": report["top_coefficients"][:10],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
