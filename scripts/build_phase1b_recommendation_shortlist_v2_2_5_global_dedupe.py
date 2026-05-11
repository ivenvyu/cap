from __future__ import annotations

import csv
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path


MODULE_PATH = Path("scripts/build_phase1b_recommendation_shortlist_v2_2_5.py")
OUT_DIR = Path("data/review/phase1b/v2_2_5/shortlist_global_dedupe")


def load_module():
    spec = importlib.util.spec_from_file_location("shortlist_v2_2_5", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {MODULE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def select_for_campaign_with_global_dedupe(
    m,
    rows: list[dict],
    *,
    feature_by_id: dict,
    campaign_cfg: dict,
    per_campaign: int,
    global_used_dups: set[str],
) -> tuple[list[dict], list[dict]]:
    selected: list[dict] = []
    audit: list[dict] = []
    used_images: set[str] = set()

    def score(row: dict) -> float:
        return m.get_score(row)

    def clip_signal(row: dict) -> float:
        return max(
            m.get_feature(row, feature_by_id, "clip_positive_max_sim"),
            m.get_feature(row, feature_by_id, "clip_positive_mean_sim"),
            m.get_feature(row, feature_by_id, "clip_margin"),
        )

    def dino_signal(row: dict) -> float:
        return max(
            m.get_feature(row, feature_by_id, "dinov2_campaign_margin"),
            m.get_feature(row, feature_by_id, "dinov2_family_margin"),
            m.get_feature(row, feature_by_id, "dinov2_campaign_pos_nn_sim"),
            m.get_feature(row, feature_by_id, "dinov2_family_pos_nn_sim"),
        )

    def clip_neg_signal(row: dict) -> float:
        return m.get_feature(row, feature_by_id, "clip_negative_max_sim")

    model_top = sorted(rows, key=score, reverse=True)

    added = m.add_unique(
        selected,
        model_top,
        bucket="model_top_shortlist",
        budget=5,
        used_images=used_images,
        used_dups=global_used_dups,
        feature_by_id=feature_by_id,
        campaign_cfg=campaign_cfg,
    )
    audit.append({"bucket": "model_top_shortlist", "requested": 5, "added": added})

    prompt_conflict = sorted(rows, key=lambda r: (score(r), clip_neg_signal(r)), reverse=True)
    added = m.add_unique(
        selected,
        prompt_conflict,
        bucket="prompt_conflict_audit",
        budget=1,
        used_images=used_images,
        used_dups=global_used_dups,
        feature_by_id=feature_by_id,
        campaign_cfg=campaign_cfg,
    )
    audit.append({"bucket": "prompt_conflict_audit", "requested": 1, "added": added})

    clip_disagreement = sorted(rows, key=lambda r: (clip_signal(r), -score(r)), reverse=True)
    added = m.add_unique(
        selected,
        clip_disagreement,
        bucket="clip_model_disagreement_audit",
        budget=1,
        used_images=used_images,
        used_dups=global_used_dups,
        feature_by_id=feature_by_id,
        campaign_cfg=campaign_cfg,
    )
    audit.append({"bucket": "clip_model_disagreement_audit", "requested": 1, "added": added})

    dino_disagreement = sorted(rows, key=lambda r: (dino_signal(r), -score(r)), reverse=True)
    added = m.add_unique(
        selected,
        dino_disagreement,
        bucket="dinov2_model_disagreement_audit",
        budget=1,
        used_images=used_images,
        used_dups=global_used_dups,
        feature_by_id=feature_by_id,
        campaign_cfg=campaign_cfg,
    )
    audit.append({"bucket": "dinov2_model_disagreement_audit", "requested": 1, "added": added})

    if len(selected) < per_campaign:
        need = per_campaign - len(selected)
        added = m.add_unique(
            selected,
            model_top,
            bucket="fill_model_top_after_global_dedupe",
            budget=need,
            used_images=used_images,
            used_dups=global_used_dups,
            feature_by_id=feature_by_id,
            campaign_cfg=campaign_cfg,
        )
        audit.append({"bucket": "fill_model_top_after_global_dedupe", "requested": need, "added": added})

    selected.sort(
        key=lambda r: (
            str(r.get("campaign_id")),
            -m.to_float(r.get("diagnostic_model_score")),
            str(r.get("shortlist_bucket")),
        )
    )
    for i, r in enumerate(selected, start=1):
        r["shortlist_rank"] = i
        r["global_duplicate_suppression"] = "enabled_v2_2_5"
        r["review_batch_duplicate_policy"] = "suppress_duplicate_group_across_active_campaigns"

    return selected, audit


def main() -> None:
    m = load_module()

    score_rows = m.read_jsonl(Path("data/retrieval/phase1b/v2_2_5/candidate_score_snapshot_v2_2_5.jsonl"))
    feature_by_id = m.load_features("data/feature_snapshots/v2_2_5/phase1b_duplicate_canonicalized/*.jsonl")
    briefs_doc = m.read_yaml(Path("configs/campaign_review_briefs_v1.yaml"))
    campaigns_cfg = briefs_doc.get("campaigns", {})
    active_campaigns = sorted(m.active_campaign_ids(briefs_doc))

    rows_by_campaign: dict[str, list[dict]] = defaultdict(list)
    excluded_campaigns = Counter()

    for row in score_rows:
        cid = str(row.get("campaign_id", ""))
        if cid not in set(active_campaigns):
            excluded_campaigns[cid] += 1
            continue
        rows_by_campaign[cid].append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    global_used_dups: set[str] = set()
    all_selected: list[dict] = []
    audit_by_campaign: dict[str, dict] = {}

    for cid in active_campaigns:
        rows = rows_by_campaign.get(cid, [])
        selected, bucket_audit = select_for_campaign_with_global_dedupe(
            m,
            rows,
            feature_by_id=feature_by_id,
            campaign_cfg=campaigns_cfg.get(cid, {}),
            per_campaign=8,
            global_used_dups=global_used_dups,
        )

        for r in selected:
            r["campaign_id"] = cid

        all_selected.extend(selected)
        audit_by_campaign[cid] = {
            "candidate_rows": len(rows),
            "selected_rows": len(selected),
            "bucket_audit": bucket_audit,
            "bucket_counts": dict(Counter(str(r.get("shortlist_bucket")) for r in selected)),
            "score_min": min(m.to_float(r.get("diagnostic_model_score")) for r in selected) if selected else None,
            "score_median": sorted(m.to_float(r.get("diagnostic_model_score")) for r in selected)[len(selected)//2] if selected else None,
            "score_max": max(m.to_float(r.get("diagnostic_model_score")) for r in selected) if selected else None,
        }

    preferred = [
        "shortlist_version",
        "campaign_id",
        "campaign_title_ko",
        "campaign_status",
        "shortlist_rank",
        "shortlist_bucket",
        "global_duplicate_suppression",
        "review_batch_duplicate_policy",
        "bucket_review_focus_ko",
        "review_question_ko",
        "image_id",
        "canonical_image_id",
        "duplicate_group_id",
        "pair_id",
        "feature_snapshot_id",
        "layout_spec_id",
        "diagnostic_model_score",
        "clip_margin",
        "clip_negative_max_sim",
        "dinov2_campaign_margin",
        "dinov2_family_margin",
        "required_region_safe_min",
        "campaign_brief_ko",
        "positive_visual_criteria_ko",
        "negative_visual_criteria_ko",
        "score_status",
        "threshold_status",
        "decision",
        "issue_tags",
        "notes",
    ]

    fieldnames = []
    for f in preferred:
        if f not in fieldnames:
            fieldnames.append(f)
    for row in all_selected:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)

    csv_out = OUT_DIR / "recommendation_shortlist_v2_2_5_global_dedupe.csv"
    labeled_template = OUT_DIR / "recommendation_shortlist_v2_2_5_global_dedupe.labeled_template.csv"

    out_rows = []
    for r in all_selected:
        rr = dict(r)
        rr.setdefault("decision", "")
        rr.setdefault("issue_tags", "")
        rr.setdefault("notes", "")
        out_rows.append(rr)

    write_csv(csv_out, out_rows, fieldnames)
    write_csv(labeled_template, out_rows, fieldnames)

    repo_root = Path(".").resolve()
    manifest_paths = m.load_manifest_paths(repo_root)

    by_campaign: dict[str, list[dict]] = defaultdict(list)
    for r in all_selected:
        by_campaign[str(r.get("campaign_id"))].append(r)

    campaign_pages = []
    for cid, cr in sorted(by_campaign.items()):
        title = f"{cid} · {campaigns_cfg.get(cid, {}).get('title_ko', '')}"
        page_name = f"recommendation_shortlist_v2_2_5_global_dedupe__{cid}.html"
        page_path = OUT_DIR / page_name
        page_path.write_text(
            m.render_page(title, cr, repo_root=repo_root, html_dir=OUT_DIR, manifest_paths=manifest_paths),
            encoding="utf-8",
        )
        campaign_pages.append({
            "campaign_id": cid,
            "title_ko": campaigns_cfg.get(cid, {}).get("title_ko", ""),
            "rows": len(cr),
            "page": page_name,
        })

    links = "\n".join(
        f'<li><a href="{m.esc(x["page"])}">{m.esc(x["campaign_id"])} · {m.esc(x["title_ko"])}</a> ({x["rows"]} rows)</li>'
        for x in campaign_pages
    )

    index_path = OUT_DIR / "index.html"
    index_path.write_text(
        f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>v2.2.5 Global Dedupe Shortlist</title>
</head>
<body>
  <h1>v2.2.5 Global Dedupe Shortlist</h1>
  <p>active campaign 전체 review batch에서 duplicate_group_id 반복을 억제한 shortlist입니다.</p>
  <p>이 파일은 추가 대량 라벨링용이 아니라 추천 후보 확인용입니다.</p>
  <p>CSV: <code>{m.esc(str(csv_out))}</code></p>
  <p>Template: <code>{m.esc(str(labeled_template))}</code></p>
  <ul>{links}</ul>
</body>
</html>
""",
        encoding="utf-8",
    )

    # global duplicate check
    by_dup = defaultdict(list)
    for r in all_selected:
        dup = r.get("duplicate_group_id") or r.get("canonical_image_id") or r.get("image_id")
        by_dup[str(dup)].append(r)
    global_dups = {k: v for k, v in by_dup.items() if len(v) > 1}

    summary = {
        "event": "done",
        "version": "v2.2.5_global_dedupe",
        "purpose": "recommendation_shortlist_not_bulk_labeling",
        "selected_rows": len(all_selected),
        "active_campaigns": active_campaigns,
        "excluded_campaign_counts": dict(excluded_campaigns),
        "bucket_counts": dict(Counter(str(r.get("shortlist_bucket")) for r in all_selected)),
        "campaign_pages": campaign_pages,
        "audit_by_campaign": audit_by_campaign,
        "global_duplicate_group_repeats": len(global_dups),
        "global_duplicate_examples": {
            k: [
                {
                    "campaign_id": x.get("campaign_id"),
                    "image_id": x.get("image_id"),
                    "bucket": x.get("shortlist_bucket"),
                    "score": x.get("diagnostic_model_score"),
                }
                for x in v
            ]
            for k, v in list(global_dups.items())[:20]
        },
        "csv_out": str(csv_out),
        "labeled_template": str(labeled_template),
        "index_html": str(index_path),
        "score_status": "diagnostic_only",
        "threshold_status": "no_calibrated_threshold",
        "non_claims": [
            "production model quality를 주장하지 않는다.",
            "global duplicate suppression은 review batch 다양성 확보용이며 품질 threshold가 아니다.",
            "campaign 간 score를 직접 비교하지 않는다.",
        ],
    }

    summary_out = OUT_DIR / "recommendation_shortlist_v2_2_5_global_dedupe_summary.json"
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
