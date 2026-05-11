from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd


def render_card(row: pd.Series) -> str:
    img_src = str(row["resolved_path"])
    return f"""
    <div class="card">
      <img src="file://{html.escape(img_src)}" />
      <div class="meta">
        <div><b>row:</b> {html.escape(str(row["queue_row_id"]))}</div>
        <div><b>bucket:</b> {html.escape(str(row["source_bucket"]))}</div>
        <div><b>image_id:</b> {html.escape(str(row["image_id"]))}</div>
        <div><b>duplicate_group:</b> {html.escape(str(row["duplicate_group_id"]))}</div>
        <div><b>category:</b> {html.escape(str(row["category"]))} / {html.escape(str(row["source_group"]))}</div>
        <div><b>path:</b> {html.escape(str(row["image_path"]))}</div>
        <div><b>clip_margin:</b> {float(row["clip_margin"]):.4f}</div>
        <div><b>clip_positive:</b> {float(row["clip_positive_max_sim"]):.4f}</div>
        <div><b>clip_negative:</b> {float(row["clip_negative_max_sim"]):.4f}</div>
        <div><b>safe_min:</b> {float(row["required_region_safe_min"]):.4f}</div>
        <div><b>safe_mean:</b> {float(row["required_region_safe_mean"]):.4f}</div>
        <div class="review">
          <b>decision:</b><br/>
          <b>issue_tags:</b><br/>
          <b>preference_rank:</b><br/>
          <b>notes:</b>
        </div>
      </div>
    </div>
    """


def render_page(title: str, df: pd.DataFrame) -> str:
    bucket_counts = df["source_bucket"].value_counts().to_dict()
    bucket_text = "<br/>".join(
        f"{html.escape(str(k))}: {v}" for k, v in bucket_counts.items()
    )

    cards = "\n".join(render_card(row) for _, row in df.iterrows())

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>{html.escape(title)}</title>
<style>
body {{
  font-family: -apple-system, BlinkMacSystemFont, sans-serif;
  margin: 24px;
}}
h1 {{
  margin-bottom: 8px;
}}
.summary {{
  margin-bottom: 20px;
  padding: 12px;
  background: #f6f6f6;
  border-radius: 10px;
  line-height: 1.5;
}}
.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
  gap: 18px;
}}
.card {{
  border: 1px solid #ddd;
  border-radius: 12px;
  padding: 12px;
  break-inside: avoid;
}}
.card img {{
  width: 100%;
  height: 260px;
  object-fit: contain;
  background: #f5f5f5;
  border-radius: 8px;
}}
.meta {{
  font-size: 13px;
  line-height: 1.5;
  margin-top: 10px;
  word-break: break-all;
}}
.review {{
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid #eee;
  min-height: 72px;
}}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div class="summary">
  <b>rows:</b> {len(df)}<br/>
  <b>unique duplicate groups:</b> {df["duplicate_group_id"].nunique()}<br/>
  <b>bucket counts:</b><br/>
  {bucket_text}
</div>
<div class="grid">
{cards}
</div>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-dir", default="data/review/phase1b")
    ap.add_argument("--out-dir", default="data/review/phase1b/html")
    args = ap.parse_args()

    review_dir = Path(args.review_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(review_dir.glob("review_queue__*.csv"))
    if not files:
        raise RuntimeError(f"no review queue csv files found in {review_dir}")

    for csv_path in files:
        df = pd.read_csv(csv_path)
        campaign_id = str(df["campaign_id"].iloc[0])
        title = f"Phase 1b Review Queue — {campaign_id}"
        html_text = render_page(title, df)

        out_path = out_dir / f"{csv_path.stem}.html"
        out_path.write_text(html_text, encoding="utf-8")
        print(f"wrote {out_path} rows={len(df)}")

    index_links = []
    for p in sorted(out_dir.glob("review_queue__*.html")):
        index_links.append(f'<li><a href="{html.escape(p.name)}">{html.escape(p.stem)}</a></li>')

    index = f"""<!doctype html>
<html>
<head><meta charset="utf-8" /><title>Phase 1b Review Queues</title></head>
<body>
<h1>Phase 1b Review Queues</h1>
<ul>
{chr(10).join(index_links)}
</ul>
</body>
</html>
"""
    (out_dir / "index.html").write_text(index, encoding="utf-8")
    print(f"wrote {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
