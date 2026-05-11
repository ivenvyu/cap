from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import pandas as pd

QUEUE_VERSION = "review_queue_v2_2_2"
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
DECISION_LABELS = ["reject", "acceptable", "accept", "best", "skip"]


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


def render_metric_grid(row: pd.Series) -> str:
    metrics = [
        ("model", safe_float(row.get("diagnostic_model_score"))),
        ("model_rank", safe_str(row.get("model_rank_desc"))),
        ("clip_margin", safe_float(row.get("clip_margin"))),
        ("clip+", safe_float(row.get("clip_positive_max_sim"))),
        ("clip-", safe_float(row.get("clip_negative_max_sim"))),
        ("dino_campaign_margin", safe_float(row.get("dinov2_campaign_margin"))),
        ("dino_family_margin", safe_float(row.get("dinov2_family_margin"))),
        ("safe_min", safe_float(row.get("required_region_safe_min"))),
        ("title_safe", safe_float(row.get("title_region_safe_score"))),
        ("info_safe", safe_float(row.get("info_region_safe_score"))),
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
    source_bucket = safe_str(row.get("source_bucket"))
    bucket_note = safe_str(row.get("bucket_semantics_actual"))
    src = img_src(row)
    issue_text = ", ".join(ISSUE_TAGS)
    decision_text = " / ".join(DECISION_LABELS)
    return f"""
    <article class="card">
      <div class="image-wrap">
        <img src="{html.escape(src)}" loading="lazy" />
      </div>
      <div class="meta">
        <div><b>row</b>: {html.escape(safe_str(row.get('queue_row_id')))}</div>
        <div><b>bucket</b>: {html.escape(source_bucket)}</div>
        <div class="note">{html.escape(bucket_note)}</div>
        <div><b>campaign</b>: {html.escape(safe_str(row.get('campaign_id')))}</div>
        <div><b>image</b>: {html.escape(safe_str(row.get('image_id')))}</div>
        <div><b>duplicate_group</b>: {html.escape(safe_str(row.get('duplicate_group_id')))}</div>
        <div><b>layout</b>: {html.escape(safe_str(row.get('layout_spec_id')))}</div>
        <div><b>path</b>: {html.escape(safe_str(row.get('image_path')))}</div>
        <div><b>category</b>: {html.escape(safe_str(row.get('category')))} / {html.escape(safe_str(row.get('source_group')))}</div>
      </div>
      <div class="metrics">{render_metric_grid(row)}</div>
      <div class="review-box">
        <div><b>decision</b>: <span class="hint">{html.escape(decision_text)}</span></div>
        <div><b>issue_tags</b>: <span class="hint">{html.escape(issue_text)}</span></div>
        <div><b>preference_rank</b>:</div>
        <div><b>notes</b>:</div>
      </div>
    </article>
    """


def render_page(title: str, df: pd.DataFrame, summary: dict[str, Any] | None = None) -> str:
    bucket_counts = df["source_bucket"].value_counts().to_dict() if "source_bucket" in df.columns else {}
    bucket_lines = "".join(
        f"<li>{html.escape(str(k))}: {int(v)}</li>" for k, v in bucket_counts.items()
    )
    cards = "\n".join(render_card(row) for _, row in df.iterrows())
    summary_note = ""
    if summary:
        summary_note = html.escape(str(summary.get("interpretation", "")))
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>{html.escape(title)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; }}
h1 {{ margin-bottom: 6px; }}
.summary {{ border: 1px solid #ddd; border-radius: 10px; padding: 12px; margin-bottom: 18px; line-height: 1.45; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 16px; }}
.card {{ border: 1px solid #ddd; border-radius: 12px; padding: 12px; break-inside: avoid; }}
.image-wrap {{ width: 100%; height: 280px; display: flex; align-items: center; justify-content: center; background: #f5f5f5; border-radius: 8px; overflow: hidden; }}
.image-wrap img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
.meta {{ font-size: 13px; line-height: 1.45; margin-top: 10px; word-break: break-word; }}
.note {{ margin: 4px 0 6px 0; font-size: 12px; }}
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
  <div><b>rows</b>: {len(df)}</div>
  <div><b>unique images</b>: {df['image_id'].nunique() if 'image_id' in df.columns else ''}</div>
  <div><b>unique duplicate groups</b>: {df['duplicate_group_id'].nunique() if 'duplicate_group_id' in df.columns else ''}</div>
  <div><b>score status</b>: diagnostic_only; <b>threshold</b>: no_calibrated_threshold</div>
  <div>{summary_note}</div>
  <div><b>bucket counts</b>:</div>
  <ul>{bucket_lines}</ul>
</div>
<div class="grid">
{cards}
</div>
</body>
</html>
"""


def load_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue-dir", default="data/review/phase1b/v2_2_2/queues")
    ap.add_argument("--combined", default="data/review/phase1b/v2_2_2/review_queue_v2_2_2_all_campaigns.csv")
    ap.add_argument("--summary", default="audit/phase_1b/review_queue_v2_2_2_summary.json")
    ap.add_argument("--out-dir", default="data/review/phase1b/v2_2_2/html")
    ap.add_argument("--labeled-out", default="data/review/phase1b/v2_2_2/review_queue_v2_2_2_all_campaigns.labeled_template.csv")
    args = ap.parse_args()

    queue_dir = Path(args.queue_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = load_summary(Path(args.summary))

    queue_files = sorted(queue_dir.glob("review_queue_v2_2_2__*.csv"))
    if not queue_files:
        raise RuntimeError(f"no v2.2.2 queue CSV files found in {queue_dir}")

    index_links: list[str] = []
    total_rows = 0
    for csv_path in queue_files:
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        campaign_id = safe_str(df["campaign_id"].iloc[0])
        title = f"CAP v2.2.2 Review Queue — {campaign_id}"
        html_text = render_page(title, df, summary)
        out_path = out_dir / f"{csv_path.stem}.html"
        out_path.write_text(html_text, encoding="utf-8")
        index_links.append(f'<li><a href="{html.escape(out_path.name)}">{html.escape(campaign_id)}</a> — {len(df)} rows</li>')
        total_rows += len(df)

    combined_path = Path(args.combined)
    if combined_path.exists():
        combined = pd.read_csv(combined_path)
        for col in ["decision", "issue_tags", "preference_rank", "notes"]:
            if col not in combined.columns:
                combined[col] = ""
        labeled_out = Path(args.labeled_out)
        labeled_out.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(labeled_out, index=False)
    else:
        labeled_out = None

    issue_items = "".join(f"<li>{html.escape(tag)}</li>" for tag in ISSUE_TAGS)
    decision_items = "".join(f"<li>{html.escape(label)}</li>" for label in DECISION_LABELS)
    labeled_note = f"<p>Labeled template: {html.escape(str(labeled_out))}</p>" if labeled_out else ""
    index = f"""<!doctype html>
<html>
<head><meta charset="utf-8" /><title>CAP v2.2.2 Review Queues</title></head>
<body>
<h1>CAP v2.2.2 Review Queues</h1>
<p>Rows: {total_rows}. Scores are diagnostic-only; no calibrated pass/fail threshold is used.</p>
{labeled_note}
<h2>Campaign sheets</h2>
<ul>{''.join(index_links)}</ul>
<h2>Decision labels</h2>
<ul>{decision_items}</ul>
<h2>Issue tags</h2>
<ul>{issue_items}</ul>
</body>
</html>
"""
    index_path = out_dir / "index.html"
    index_path.write_text(index, encoding="utf-8")

    result = {
        "event": "done",
        "queue_version": QUEUE_VERSION,
        "queue_files": [str(p) for p in queue_files],
        "html_out_dir": str(out_dir),
        "index": str(index_path),
        "labeled_template": str(labeled_out) if labeled_out else None,
        "rows": int(total_rows),
        "score_status": "diagnostic_only",
        "threshold_status": "no_calibrated_threshold",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
