from __future__ import annotations

import argparse
import csv
import html
import json
import sqlite3
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


def connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise RuntimeError(f"missing DB: {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def safe_id(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(s))


def resolve_image_path(row: dict[str, Any]) -> Path:
    candidates = [
        row.get("resolved_path"),
        row.get("path"),
    ]
    for c in candidates:
        if not c:
            continue
        p = Path(str(c))
        if p.exists():
            return p
    raise RuntimeError(f"cannot resolve image path for {row.get('image_id')}: {candidates}")


def make_thumb(src: Path, out: Path, size: tuple[int, int]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(src)
    img = ImageOps.exif_transpose(img).convert("RGB")
    fitted = ImageOps.contain(img, size)

    canvas = Image.new("RGB", size, "white")
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    canvas.save(out, quality=90)


def load_requirements(conn: sqlite3.Connection, score_version: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT DISTINCT
            s.campaign_id,
            s.cue_id,
            r.requirement_role,
            r.source_value,
            c.cue_group,
            c.cue_type,
            c.prompts_json,
            c.ontology_write_allowed_if_human_verified,
            c.verified_ontology_tags_json
        FROM campaign_image_cue_scores s
        JOIN campaign_visual_cue_requirements r
          ON s.campaign_id = r.campaign_id
         AND s.cue_id = r.cue_id
        JOIN visual_cues c
          ON s.cue_id = c.cue_id
        WHERE s.score_version = ?
        ORDER BY s.campaign_id, s.cue_id
        """,
        (score_version,),
    ).fetchall()

    out = []
    for r in rows:
        d = dict(r)
        d["prompts"] = json.loads(d.pop("prompts_json") or "[]")
        d["verified_ontology_tags"] = json.loads(d.pop("verified_ontology_tags_json") or "{}")
        out.append(d)

    return out


def load_top_rows(
    conn: sqlite3.Connection,
    campaign_id: str,
    cue_id: str,
    score_version: str,
    top_k: int,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            s.campaign_id,
            s.cue_id,
            s.score,
            s.score_status,
            s.score_version,
            i.image_id,
            i.path,
            i.resolved_path,
            i.category,
            i.source_group,
            i.place_name,
            i.subject_name,
            i.metadata_status
        FROM campaign_image_cue_scores s
        JOIN images i
          ON s.image_id = i.image_id
        WHERE s.campaign_id = ?
          AND s.cue_id = ?
          AND s.score_version = ?
        ORDER BY s.score DESC, i.image_id
        LIMIT ?
        """,
        (campaign_id, cue_id, score_version, top_k),
    ).fetchall()

    return [dict(r) for r in rows]


def load_existing_tags(conn: sqlite3.Connection, image_id: str) -> dict[str, list[str]]:
    rows = conn.execute(
        """
        SELECT
            v.axis_id,
            v.tag_name,
            a.label_source,
            a.confidence_status
        FROM image_tag_assertions a
        JOIN tag_values v
          ON a.tag_id = v.tag_id
        WHERE a.image_id = ?
        ORDER BY v.axis_id, v.tag_name
        """,
        (image_id,),
    ).fetchall()

    out: dict[str, list[str]] = {}
    for r in rows:
        axis = str(r["axis_id"])
        val = f'{r["tag_name"]} [{r["label_source"]}/{r["confidence_status"]}]'
        out.setdefault(axis, []).append(val)

    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "campaign_id",
        "cue_id",
        "rank",
        "score",
        "image_id",
        "path",
        "category",
        "source_group",
        "place_name",
        "subject_name",
        "existing_space_tags",
        "existing_subject_tags",
        "existing_mood_tags",
        "human_decision",
        "issue_tags",
        "notes",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def render_requirement_html(
    requirement: dict[str, Any],
    rows: list[dict[str, Any]],
    out_dir: Path,
) -> str:
    cards = []

    for row in rows:
        rel_thumb = Path(row["thumb_path"]).relative_to(out_dir)

        existing_tags_html = ""
        for axis, tags in row["existing_tags"].items():
            existing_tags_html += (
                f"<div><b>{html.escape(axis)}:</b> "
                + ", ".join(html.escape(t) for t in tags)
                + "</div>"
            )

        cards.append(
            f"""
            <section class="card">
              <div class="rank">#{row["rank"]} · score={row["score"]:.6f}</div>
              <img src="{html.escape(str(rel_thumb))}" />
              <div class="meta">
                <b>image_id:</b> {html.escape(row["image_id"])}<br/>
                <b>path:</b> {html.escape(row["path"])}<br/>
                <b>category:</b> {html.escape(str(row.get("category", "")))}<br/>
                <b>source_group:</b> {html.escape(str(row.get("source_group", "")))}<br/>
                <b>place_name:</b> {html.escape(str(row.get("place_name", "")))}<br/>
                <b>subject_name:</b> {html.escape(str(row.get("subject_name", "")))}<br/>
              </div>
              <div class="tags">
                <b>existing propagated ontology tags</b>
                {existing_tags_html or "<div>none</div>"}
              </div>
              <div class="review">
                <b>review fields:</b><br/>
                human_decision: accept / reject / unsure<br/>
                issue_tags:<br/>
                notes:
              </div>
            </section>
            """
        )

    prompts = "".join(f"<li>{html.escape(p)}</li>" for p in requirement["prompts"])

    verified_tags = json.dumps(
        requirement["verified_ontology_tags"],
        ensure_ascii=False,
        sort_keys=True,
    )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>{html.escape(requirement["campaign_id"])} — {html.escape(requirement["cue_id"])}</title>
<style>
body {{
  font-family: -apple-system, BlinkMacSystemFont, sans-serif;
  margin: 24px;
}}
.summary {{
  background: #f7f7f7;
  padding: 12px;
  border-radius: 10px;
  line-height: 1.5;
  margin-bottom: 18px;
}}
.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 16px;
}}
.card {{
  border: 1px solid #ddd;
  border-radius: 12px;
  padding: 12px;
}}
.card img {{
  width: 100%;
  height: 190px;
  object-fit: contain;
  background: #f0f0f0;
  border-radius: 8px;
}}
.rank {{
  font-weight: 700;
  margin-bottom: 8px;
}}
.meta, .tags, .review {{
  font-size: 12px;
  line-height: 1.5;
  margin-top: 10px;
  word-break: break-word;
}}
.review {{
  background: #fafafa;
  padding: 8px;
  border-radius: 8px;
}}
code {{
  background: #eee;
  padding: 2px 4px;
  border-radius: 4px;
}}
</style>
</head>
<body>
<h1>{html.escape(requirement["campaign_id"])} — {html.escape(requirement["cue_id"])}</h1>

<div class="summary">
  <b>cue_group:</b> {html.escape(requirement["cue_group"])}<br/>
  <b>cue_type:</b> {html.escape(requirement["cue_type"])}<br/>
  <b>campaign season/source:</b> {html.escape(str(requirement["source_value"]))}<br/>
  <b>ontology_write_allowed_if_human_verified:</b> {requirement["ontology_write_allowed_if_human_verified"]}<br/>
  <b>verified_ontology_tags:</b> <code>{html.escape(verified_tags)}</code><br/>
  <b>주의:</b> 이 score는 calibrated threshold가 아니다. cue별 상대 순위 검토용이다.
</div>

<h2>Prompts</h2>
<ul>
{prompts}
</ul>

<h2>Top candidates</h2>
<div class="grid">
{''.join(cards)}
</div>
</body>
</html>
"""


def render_index(requirements: list[dict[str, Any]], links: list[dict[str, str]]) -> str:
    items = []
    for link in links:
        items.append(
            f'<li><a href="{html.escape(link["html"])}">'
            f'{html.escape(link["campaign_id"])} — {html.escape(link["cue_id"])}</a> '
            f'(<a href="{html.escape(link["csv"])}">csv</a>)</li>'
        )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Campaign Visual Cue Score Review Sheets</title>
</head>
<body>
<h1>Campaign Visual Cue Score Review Sheets</h1>
<p>
These sheets are diagnostic-only review artifacts.
Scores are not calibrated thresholds and should be used only for cue ranking inspection.
</p>
<ul>
{''.join(items)}
</ul>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/db/cap_reranker_ontology.db")
    ap.add_argument("--score-version", default="campaign_visual_cue_clip_v1")
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--out-dir", default="data/review/ontology/campaign_visual_cue_scores_v1")
    args = ap.parse_args()

    conn = connect(Path(args.db))
    out_dir = Path(args.out_dir)
    thumb_dir = out_dir / "thumbs"
    out_dir.mkdir(parents=True, exist_ok=True)

    requirements = load_requirements(conn, args.score_version)
    if not requirements:
        raise RuntimeError(f"no requirements found for score_version={args.score_version}")

    links = []

    for req in requirements:
        rows = load_top_rows(
            conn=conn,
            campaign_id=req["campaign_id"],
            cue_id=req["cue_id"],
            score_version=args.score_version,
            top_k=args.top_k,
        )

        if not rows:
            continue

        enriched = []

        page_id = f'{safe_id(req["campaign_id"])}__{safe_id(req["cue_id"])}'

        for rank, row in enumerate(rows, start=1):
            image_path = resolve_image_path(row)
            thumb_path = thumb_dir / f"{page_id}__rank_{rank:03d}__{safe_id(row['image_id'])}.jpg"
            make_thumb(image_path, thumb_path, size=(320, 220))

            existing_tags = load_existing_tags(conn, row["image_id"])

            enriched_row = dict(row)
            enriched_row["rank"] = rank
            enriched_row["thumb_path"] = str(thumb_path)
            enriched_row["existing_tags"] = existing_tags
            enriched_row["existing_space_tags"] = "|".join(existing_tags.get("space_axis", []))
            enriched_row["existing_subject_tags"] = "|".join(existing_tags.get("subject_axis", []))
            enriched_row["existing_mood_tags"] = "|".join(existing_tags.get("mood_axis", []))
            enriched_row["human_decision"] = ""
            enriched_row["issue_tags"] = ""
            enriched_row["notes"] = ""
            enriched.append(enriched_row)

        html_name = f"{page_id}.html"
        csv_name = f"{page_id}.csv"

        html_path = out_dir / html_name
        csv_path = out_dir / csv_name

        html_path.write_text(
            render_requirement_html(req, enriched, out_dir),
            encoding="utf-8",
        )
        write_csv(csv_path, enriched)

        links.append(
            {
                "campaign_id": req["campaign_id"],
                "cue_id": req["cue_id"],
                "html": html_name,
                "csv": csv_name,
            }
        )

        print(f"wrote {html_path} rows={len(enriched)}")
        print(f"wrote {csv_path} rows={len(enriched)}")

    index_path = out_dir / "index.html"
    index_path.write_text(render_index(requirements, links), encoding="utf-8")

    summary = {
        "event": "done",
        "db": args.db,
        "score_version": args.score_version,
        "top_k": args.top_k,
        "requirements": len(requirements),
        "sheets": len(links),
        "out_dir": str(out_dir),
        "index_html": str(index_path),
        "score_status": "diagnostic_only",
        "threshold_status": "no_calibrated_threshold",
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
