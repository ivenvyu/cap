from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import pandas as pd

QUEUE_VERSION = "review_queue_v2_2_2"
TRIAGE_VERSION = "triage_25_v1"
SCORE_STATUS = "diagnostic_only"
THRESHOLD_STATUS = "no_calibrated_threshold"

# Human-facing campaign briefs. These are review instructions, not model labels.
CAMPAIGN_BRIEFS: dict[str, dict[str, str]] = {
    "phase1b_summer_garden_walk": {
        "title_ko": "여름 정원 산책 홍보물",
        "brief_ko": "여름의 초록 정원과 산책로가 느껴지는 프로그램 안내 이미지.",
        "positive_ko": "초록 정원, 나무, 산책길, 밝은 여름 계절감, 고요한 야외 배경.",
        "negative_ko": "실내/겨울/가을 느낌, 꽃 클로즈업만 있는 사진, 건축물이 주인공인 사진, 너무 복잡한 배경.",
    },
    "phase1b_autumn_garden_walk": {
        "title_ko": "가을 정원 산책 홍보물",
        "brief_ko": "가을 정원 산책 또는 단풍 산책 프로그램에 어울리는 안내 이미지.",
        "positive_ko": "단풍, 노랑·갈색 계열, 가을 산책로, 차분한 정원 풍경, 계절감 있는 야외 배경.",
        "negative_ko": "짙은 여름 초록만 보이는 사진, 봄꽃 클로즈업, 실내 갤러리, 겨울/눈 느낌.",
    },
    "phase1b_architecture_exhibition_visit": {
        "title_ko": "건축/전시 방문 홍보물",
        "brief_ko": "사유원 건축물, 전시 공간, 방문 동선 또는 구조적 공간감을 보여주는 안내 이미지.",
        "positive_ko": "건축물, 벽면, 실내외 구조물, 전시 공간, 방문 동선, 공간성이 드러나는 사진.",
        "negative_ko": "정원 풍경만 있는 사진, 꽃/식물 클로즈업, 건축/전시 느낌이 약한 일반 자연 풍경.",
    },
    "phase1b_botanical_spring_program": {
        "title_ko": "봄 식물 프로그램 홍보물",
        "brief_ko": "봄 식물 관찰, 생태 프로그램, 식물 체험 안내에 어울리는 이미지.",
        "positive_ko": "봄 느낌, 꽃, 식물, 새싹, 식물 관찰, 생태 프로그램 분위기.",
        "negative_ko": "가을 단풍, 겨울/눈, 건축물 중심, 식물 프로그램보다 산책 배경에 가까운 사진, 너무 여름 같은 짙은 초록.",
    },
    "phase1b_indoor_gallery_winter_art": {
        "title_ko": "겨울 실내 갤러리 전시 홍보물",
        "brief_ko": "겨울 실내 갤러리 또는 예술 전시 안내에 어울리는 차분한 이미지.",
        "positive_ko": "실내, 갤러리, 전시 공간, 겨울 분위기, 차분함, 예술적 공간감.",
        "negative_ko": "야외 정원, 여름 초록 풍경, 봄꽃, 산책로, 계절감이 강한 야외 자연 사진.",
    },
}

BUCKET_FOCUS_KO: dict[str, str] = {
    "model_top_diagnostic": "모델이 좋다고 본 후보입니다. 실제로도 이 홍보물 배경으로 좋은지 확인하세요.",
    "clip_high_model_low": "CLIP은 높게 봤지만 모델은 낮게 본 후보입니다. 모델이 놓친 좋은 semantic 후보인지 확인하세요.",
    "dinov2_high_model_low": "DINOv2 visual-anchor는 높게 봤지만 모델은 낮게 본 후보입니다. 시각적으로 비슷해서 실제로도 좋은지 확인하세요.",
    "model_high_clip_negative_high": "모델 점수가 높지만 negative prompt와도 가까운 후보입니다. 사진은 좋아 보여도 주제와 충돌하는 hard negative인지 확인하세요.",
    "cluster_diversity": "다양한 DINOv2 cluster에서 뽑은 후보입니다. 이미지 pool coverage와 예상 밖 좋은 후보를 확인하세요.",
    "layout_safe_coverage": "텍스트 영역 안전도가 높은 후보입니다. 실제로 제목/정보 박스를 얹어도 괜찮을지 확인하세요.",
    "random_control": "무작위 control 후보입니다. 모델/CLIP/DINO가 놓친 예상 밖 좋은 이미지가 있는지 확인하세요.",
    "fill_remaining_diagnostic_mixed": "부족분을 채운 fallback 후보입니다. 품질 claim이 아니라 workflow 확인용입니다.",
}

# Exactly five rows per campaign by default. Fallback buckets are used only if the preferred bucket row is unavailable.
TRIAGE_BUCKET_PLAN: list[tuple[str, list[str]]] = [
    ("clip_high_model_low", ["clip_high_model_low", "cluster_diversity", "model_top_diagnostic"]),
    ("dinov2_high_model_low", ["dinov2_high_model_low", "cluster_diversity", "model_top_diagnostic"]),
    ("model_high_clip_negative_high", ["model_high_clip_negative_high", "model_top_diagnostic", "cluster_diversity"]),
    ("model_top_diagnostic", ["model_top_diagnostic", "layout_safe_coverage", "cluster_diversity"]),
    ("random_or_cluster_control", ["random_control", "cluster_diversity", "layout_safe_coverage", "model_top_diagnostic"]),
]

DECISION_LABELS = ["reject", "acceptable", "accept", "best", "skip"]
ISSUE_TAGS = [
    "semantic_mismatch",
    "season_mismatch",
    "mood_mismatch",
    "brand_tone_mismatch",
    "text_region_conflict",
    "low_contrast",
    "too_busy_background",
    "visual_hierarchy_weak",
    "poor_composition",
    "duplicate_or_too_similar",
    "already_used_in_recent_campaign",
    "low_resolution",
]

DISPLAY_COLUMNS = [
    "triage_row_id",
    "campaign_id",
    "campaign_title_ko",
    "campaign_brief_ko",
    "source_bucket",
    "bucket_review_focus_ko",
    "image_id",
    "diagnostic_model_score",
    "model_rank_desc",
    "clip_margin",
    "dinov2_campaign_margin",
    "required_region_safe_min",
    "image_path",
    "resolved_path",
    "decision",
    "issue_tags",
    "preference_rank",
    "notes",
]


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value)


def safe_float(value: Any, digits: int = 4) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if pd.isna(v):
        return ""
    return f"{v:.{digits}f}"


def img_src(row: pd.Series) -> str:
    preview = safe_str(row.get("preview_path"))
    if preview:
        return preview
    resolved = safe_str(row.get("resolved_path"))
    if resolved:
        return f"file://{resolved}"
    image_path = safe_str(row.get("image_path"))
    if image_path:
        return image_path
    return ""


def add_campaign_brief_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "campaign_title_ko",
        "campaign_brief_ko",
        "positive_visual_criteria_ko",
        "negative_visual_criteria_ko",
        "review_question_ko",
        "bucket_review_focus_ko",
    ]:
        if col not in out.columns:
            out[col] = ""

    for idx, row in out.iterrows():
        cid = safe_str(row.get("campaign_id"))
        brief = CAMPAIGN_BRIEFS.get(cid, {})
        out.at[idx, "campaign_title_ko"] = brief.get("title_ko", cid)
        out.at[idx, "campaign_brief_ko"] = brief.get("brief_ko", "이 campaign의 홍보물 배경으로 자연스러운지 판단하세요.")
        out.at[idx, "positive_visual_criteria_ko"] = brief.get("positive_ko", "campaign 주제와 맞고 홍보물 배경으로 쓰기 좋은 이미지.")
        out.at[idx, "negative_visual_criteria_ko"] = brief.get("negative_ko", "campaign 주제와 맞지 않거나 홍보물 배경으로 쓰기 어려운 이미지.")
        out.at[idx, "review_question_ko"] = "이 사진을 이 제목의 홍보물 배경으로 썼을 때 자연스러운가?"
        out.at[idx, "bucket_review_focus_ko"] = BUCKET_FOCUS_KO.get(safe_str(row.get("source_bucket")), "이 후보가 실제로 쓸 만한지 판단하세요.")
    return out


def choose_one(
    cdf: pd.DataFrame,
    preferred_label: str,
    bucket_fallbacks: list[str],
    used_pair_ids: set[str],
    used_images: set[str],
) -> dict[str, Any] | None:
    for bucket in bucket_fallbacks:
        bdf = cdf[cdf["source_bucket"].astype(str) == bucket].copy()
        if bdf.empty:
            continue
        sort_cols = [c for c in ["model_rank_desc", "clip_rank_desc", "dinov2_anchor_rank_desc", "queue_row_id"] if c in bdf.columns]
        if sort_cols:
            bdf = bdf.sort_values(sort_cols, ascending=True)
        for _, row in bdf.iterrows():
            pair_id = safe_str(row.get("pair_id"))
            image_id = safe_str(row.get("image_id"))
            if pair_id in used_pair_ids or image_id in used_images:
                continue
            out = row.to_dict()
            out["triage_requested_bucket"] = preferred_label
            out["triage_selected_bucket"] = bucket
            out["triage_fallback_used"] = bucket != bucket_fallbacks[0]
            out["triage_reason"] = BUCKET_FOCUS_KO.get(bucket, "triage sample")
            used_pair_ids.add(pair_id)
            used_images.add(image_id)
            return out
    return None


def build_triage(df: pd.DataFrame, max_per_campaign: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []

    for campaign_id, cdf in df.groupby("campaign_id", sort=True):
        used_pair_ids: set[str] = set()
        used_images: set[str] = set()
        campaign_rows: list[dict[str, Any]] = []
        bucket_audit: list[dict[str, Any]] = []

        for preferred_label, fallbacks in TRIAGE_BUCKET_PLAN[:max_per_campaign]:
            selected = choose_one(cdf, preferred_label, fallbacks, used_pair_ids, used_images)
            if selected is None:
                bucket_audit.append(
                    {
                        "requested": preferred_label,
                        "selected": None,
                        "fallback_used": None,
                        "status": "not_filled",
                    }
                )
                continue
            campaign_rows.append(selected)
            bucket_audit.append(
                {
                    "requested": preferred_label,
                    "selected": selected.get("source_bucket"),
                    "fallback_used": bool(selected.get("triage_fallback_used")),
                    "status": "filled",
                }
            )

        # If max_per_campaign is larger than the plan or a bucket was unavailable, fill from remaining rows.
        if len(campaign_rows) < max_per_campaign:
            remaining = cdf.copy()
            sort_cols = [c for c in ["model_rank_desc", "clip_rank_desc", "dinov2_anchor_rank_desc", "queue_row_id"] if c in remaining.columns]
            if sort_cols:
                remaining = remaining.sort_values(sort_cols, ascending=True)
            for _, row in remaining.iterrows():
                if len(campaign_rows) >= max_per_campaign:
                    break
                pair_id = safe_str(row.get("pair_id"))
                image_id = safe_str(row.get("image_id"))
                if pair_id in used_pair_ids or image_id in used_images:
                    continue
                out = row.to_dict()
                out["triage_requested_bucket"] = "fill_remaining"
                out["triage_selected_bucket"] = safe_str(row.get("source_bucket"))
                out["triage_fallback_used"] = True
                out["triage_reason"] = "부족한 triage slot을 채운 fallback 후보입니다."
                used_pair_ids.add(pair_id)
                used_images.add(image_id)
                campaign_rows.append(out)

        rows.extend(campaign_rows)
        audit.append(
            {
                "campaign_id": campaign_id,
                "requested_rows": max_per_campaign,
                "selected_rows": len(campaign_rows),
                "bucket_audit": bucket_audit,
                "source_bucket_counts": dict(pd.Series([r.get("source_bucket") for r in campaign_rows]).value_counts()),
            }
        )

    triage = pd.DataFrame(rows)
    if not triage.empty:
        triage = add_campaign_brief_columns(triage)
        triage.insert(0, "triage_row_id", [f"triage_{i:03d}" for i in range(1, len(triage) + 1)])
        for col in ["decision", "issue_tags", "preference_rank", "notes"]:
            if col not in triage.columns:
                triage[col] = ""
            else:
                triage[col] = ""
    return triage, audit


def render_metric_grid(row: pd.Series) -> str:
    metrics = [
        ("model", safe_float(row.get("diagnostic_model_score"))),
        ("model_rank", safe_str(row.get("model_rank_desc"))),
        ("clip_margin", safe_float(row.get("clip_margin"))),
        ("clip-", safe_float(row.get("clip_negative_max_sim"))),
        ("dino_campaign_margin", safe_float(row.get("dinov2_campaign_margin"))),
        ("dino_family_margin", safe_float(row.get("dinov2_family_margin"))),
        ("safe_min", safe_float(row.get("required_region_safe_min"))),
        ("cluster_mid", safe_str(row.get("dinov2_cluster_id_mid"))),
    ]
    cells = []
    for key, value in metrics:
        cells.append(
            f'<div class="metric"><span class="metric-key">{html.escape(key)}</span>'
            f'<span class="metric-value">{html.escape(value)}</span></div>'
        )
    return "\n".join(cells)


def render_card(row: pd.Series) -> str:
    src = img_src(row)
    issue_text = "; ".join(ISSUE_TAGS)
    decision_text = " / ".join(DECISION_LABELS)
    return f"""
    <article class="card">
      <div class="campaign-box">
        <div class="campaign-title">{html.escape(safe_str(row.get('campaign_title_ko')))}</div>
        <div>{html.escape(safe_str(row.get('campaign_brief_ko')))}</div>
        <div><b>좋은 후보</b>: {html.escape(safe_str(row.get('positive_visual_criteria_ko')))}</div>
        <div><b>거절 후보</b>: {html.escape(safe_str(row.get('negative_visual_criteria_ko')))}</div>
      </div>
      <div class="image-wrap"><img src="{html.escape(src)}" loading="lazy" /></div>
      <div class="meta">
        <div><b>triage row</b>: {html.escape(safe_str(row.get('triage_row_id')))}</div>
        <div><b>bucket</b>: {html.escape(safe_str(row.get('source_bucket')))}</div>
        <div class="bucket-focus">{html.escape(safe_str(row.get('bucket_review_focus_ko')))}</div>
        <div><b>image</b>: {html.escape(safe_str(row.get('image_id')))}</div>
        <div><b>layout</b>: {html.escape(safe_str(row.get('layout_spec_id')))}</div>
        <div><b>path</b>: {html.escape(safe_str(row.get('image_path')))}</div>
      </div>
      <div class="metrics">{render_metric_grid(row)}</div>
      <div class="review-box">
        <div><b>질문</b>: {html.escape(safe_str(row.get('review_question_ko')))}</div>
        <div><b>decision</b>: <span class="hint">{html.escape(decision_text)}</span></div>
        <div><b>issue_tags</b>: <span class="hint">{html.escape(issue_text)}</span></div>
        <div><b>notes</b>: 짧게만 적어도 됩니다.</div>
      </div>
    </article>
    """


def render_page(title: str, df: pd.DataFrame, summary: dict[str, Any]) -> str:
    cards = "\n".join(render_card(row) for _, row in df.iterrows())
    bucket_counts = df["source_bucket"].value_counts().to_dict() if "source_bucket" in df.columns else {}
    campaign_counts = df["campaign_id"].value_counts().to_dict() if "campaign_id" in df.columns else {}
    bucket_lines = "".join(f"<li>{html.escape(str(k))}: {int(v)}</li>" for k, v in bucket_counts.items())
    campaign_lines = "".join(f"<li>{html.escape(str(k))}: {int(v)}</li>" for k, v in campaign_counts.items())
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>{html.escape(title)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; line-height: 1.45; }}
h1 {{ margin-bottom: 6px; }}
.summary {{ border: 1px solid #ddd; border-radius: 10px; padding: 14px; margin-bottom: 18px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 16px; }}
.card {{ border: 1px solid #ddd; border-radius: 12px; padding: 12px; break-inside: avoid; }}
.campaign-box {{ border: 1px solid #eee; border-radius: 10px; padding: 10px; margin-bottom: 10px; font-size: 13px; }}
.campaign-title {{ font-size: 17px; font-weight: 700; margin-bottom: 4px; }}
.image-wrap {{ width: 100%; height: 280px; display: flex; align-items: center; justify-content: center; background: #f5f5f5; border-radius: 8px; overflow: hidden; }}
.image-wrap img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
.meta {{ font-size: 13px; line-height: 1.45; margin-top: 10px; word-break: break-word; }}
.bucket-focus {{ margin: 4px 0 6px 0; font-size: 12px; }}
.metrics {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 4px 8px; margin-top: 10px; font-size: 12px; }}
.metric {{ display: flex; justify-content: space-between; gap: 8px; border-bottom: 1px solid #eee; }}
.metric-key {{ font-weight: 600; }}
.review-box {{ margin-top: 10px; padding-top: 10px; border-top: 1px solid #ddd; min-height: 90px; font-size: 13px; line-height: 1.55; }}
.hint {{ font-size: 12px; }}
@media print {{ .card {{ page-break-inside: avoid; }} }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div class="summary">
  <div><b>목적</b>: 전체 150개를 다 보지 않고, 캠페인별 5개씩만 빠르게 검토하는 triage set입니다.</div>
  <div><b>rows</b>: {len(df)}</div>
  <div><b>score status</b>: {html.escape(str(summary.get('score_status', SCORE_STATUS)))}; <b>threshold</b>: {html.escape(str(summary.get('threshold_status', THRESHOLD_STATUS)))}</div>
  <div><b>중요</b>: 이 점수는 pass/fail 확률이 아닙니다. campaign 내부 순위와 bucket 검토용입니다.</div>
  <div><b>campaign counts</b>:</div><ul>{campaign_lines}</ul>
  <div><b>bucket counts</b>:</div><ul>{bucket_lines}</ul>
</div>
<div class="grid">{cards}</div>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", default="data/review/phase1b/v2_2_2/review_queue_v2_2_2_all_campaigns.csv")
    ap.add_argument("--out-dir", default="data/review/phase1b/v2_2_2/triage")
    ap.add_argument("--max-per-campaign", type=int, default=5)
    args = ap.parse_args()

    combined = Path(args.combined)
    if not combined.exists():
        raise RuntimeError(f"combined review queue not found: {combined}")

    df = pd.read_csv(combined)
    required = {"campaign_id", "source_bucket", "image_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"combined review queue missing required columns: {missing}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    triage, audit = build_triage(df, max_per_campaign=int(args.max_per_campaign))
    if triage.empty:
        raise RuntimeError("triage selection produced no rows")

    csv_out = out_dir / "review_queue_v2_2_2_triage_25.csv"
    labeled_out = out_dir / "review_queue_v2_2_2_triage_25.labeled_template.csv"
    html_out = out_dir / "review_queue_v2_2_2_triage_25.html"
    summary_out = out_dir / "review_queue_v2_2_2_triage_25_summary.json"

    # Put human-facing columns first, then preserve all machine columns.
    first_cols = [c for c in DISPLAY_COLUMNS if c in triage.columns]
    rest_cols = [c for c in triage.columns if c not in first_cols]
    triage = triage[first_cols + rest_cols]
    triage.to_csv(csv_out, index=False)
    triage.to_csv(labeled_out, index=False)

    summary = {
        "event": "done",
        "queue_version": QUEUE_VERSION,
        "triage_version": TRIAGE_VERSION,
        "input_rows": int(len(df)),
        "selected_rows": int(len(triage)),
        "campaign_count": int(triage["campaign_id"].nunique()),
        "max_per_campaign": int(args.max_per_campaign),
        "score_status": SCORE_STATUS,
        "threshold_status": THRESHOLD_STATUS,
        "interpretation": "Triage sample for workflow validation. It is not a final evaluation set and not a calibrated quality threshold.",
        "csv_out": str(csv_out),
        "labeled_template_out": str(labeled_out),
        "html_out": str(html_out),
        "campaign_counts": {str(k): int(v) for k, v in triage["campaign_id"].value_counts().sort_index().items()},
        "source_bucket_counts": {str(k): int(v) for k, v in triage["source_bucket"].value_counts().sort_index().items()},
        "campaign_audit": audit,
    }
    summary_out.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=str),
    encoding="utf-8",
)
    html_out.write_text(render_page("CAP v2.2.2 Triage Review — 25 rows", triage, summary), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
