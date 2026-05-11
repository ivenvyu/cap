from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read feature profile YAML")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def flatten_profile(profile: dict[str, Any]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for group_name, group in profile.get("feature_groups", {}).items():
        action = str(group.get("training_action", "unspecified"))
        for feature in group.get("features", []):
            out[str(feature)] = {"group": str(group_name), "training_action": action}
    return out


def canonicalize(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        return round(value, 12)
    return value


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit active/placeholder feature signals from CAP ontology DB.")
    ap.add_argument("--db", default="data/db/cap_reranker_ontology.db")
    ap.add_argument("--feature-profile", default="configs/feature_profile_v2_2_2.yaml")
    ap.add_argument("--source", default="pair_features", choices=["pair_features", "v_effective_pair_features_v1"])
    ap.add_argument("--jsonl-glob", default=None, help="Audit PairFeatureSnapshot JSONL files instead of DB rows.")
    ap.add_argument("--out", default="audit/phase_1b/active_feature_audit_v2_2_2.json")
    ap.add_argument("--fail-on-missing", action="store_true")
    args = ap.parse_args()

    profile = load_yaml(Path(args.feature_profile))
    feature_meta = flatten_profile(profile)
    expected = list(feature_meta.keys())

    if args.jsonl_glob:
        rows = []
        for path in sorted(Path(".").glob(args.jsonl_glob)):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                rows.append({
                    "feature_snapshot_id": obj.get("feature_snapshot_id"),
                    "campaign_id": obj.get("campaign_id"),
                    "image_id": obj.get("image_id"),
                    "layout_spec_id": obj.get("layout_spec_id"),
                    "features_json": json.dumps(obj.get("features") or {}, ensure_ascii=False),
                })
    else:
        conn = sqlite3.connect(args.db)
        conn.row_factory = sqlite3.Row
        if not table_exists(conn, args.source):
            raise RuntimeError(f"source table/view not found: {args.source}")
        rows = conn.execute(f"SELECT feature_snapshot_id, campaign_id, image_id, layout_spec_id, features_json FROM {args.source}").fetchall()
    stats: dict[str, dict[str, Any]] = {}
    values: dict[str, list[Any]] = defaultdict(list)
    missing_key_count = {f: 0 for f in expected}
    malformed_rows: list[str] = []

    for row in rows:
        try:
            features = json.loads(row["features_json"])
        except Exception:
            malformed_rows.append(str(row["feature_snapshot_id"]))
            continue
        if not isinstance(features, dict):
            malformed_rows.append(str(row["feature_snapshot_id"]))
            continue
        for f in expected:
            if f not in features:
                missing_key_count[f] += 1
                values[f].append(None)
            else:
                values[f].append(features.get(f))

    n = len(rows)
    for f in expected:
        vals = values.get(f, [])
        non_null = [v for v in vals if v is not None]
        unique_non_null = sorted({canonicalize(v) for v in non_null}, key=lambda x: str(x))
        meta = feature_meta[f]
        stats[f] = {
            "feature": f,
            "group": meta["group"],
            "training_action": meta["training_action"],
            "row_count": n,
            "present_key_count": n - missing_key_count[f],
            "missing_key_count": missing_key_count[f],
            "non_null_count": len(non_null),
            "null_count": n - len(non_null),
            "null_rate": None if n == 0 else (n - len(non_null)) / n,
            "unique_non_null_count": len(unique_non_null),
            "is_constant_non_null": len(unique_non_null) <= 1 if non_null else True,
            "example_non_null_values": unique_non_null[:5],
            "diagnostic_only": True,
        }

    by_group: dict[str, dict[str, Any]] = {}
    for group_name, group in profile.get("feature_groups", {}).items():
        fs = [str(f) for f in group.get("features", [])]
        by_group[group_name] = {
            "training_action": group.get("training_action"),
            "feature_count": len(fs),
            "all_keys_present": all(stats[f]["missing_key_count"] == 0 for f in fs),
            "non_null_features": [f for f in fs if stats[f]["non_null_count"] > 0],
            "all_null_features": [f for f in fs if stats[f]["non_null_count"] == 0],
            "constant_non_null_features": [f for f in fs if stats[f]["non_null_count"] > 0 and stats[f]["is_constant_non_null"]],
        }

    hard_errors = []
    if malformed_rows:
        hard_errors.append({"type": "malformed_features_json", "count": len(malformed_rows), "examples": malformed_rows[:10]})
    missing_features = [f for f, c in missing_key_count.items() if c > 0]
    if missing_features:
        hard_errors.append({"type": "missing_required_feature_key", "features": missing_features})

    report = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "db": args.db,
            "source": args.source if not args.jsonl_glob else f"jsonl_glob:{args.jsonl_glob}",
            "feature_profile": args.feature_profile,
            "row_count": n,
            "threshold_status": profile.get("policy", {}).get("threshold_status", "diagnostic_only"),
            "diagnostic_only": True,
        },
        "hard_errors": hard_errors,
        "summary_by_group": by_group,
        "features": [stats[f] for f in expected],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "done", "out": str(out), "row_count": n, "hard_error_count": len(hard_errors)}, ensure_ascii=False))

    if args.fail_on_missing and hard_errors:
        raise RuntimeError(f"feature audit hard errors: {hard_errors[:3]}")


if __name__ == "__main__":
    main()
