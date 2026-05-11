from __future__ import annotations

import argparse
import csv
import html
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


DEFAULT_INPUT = "data/review/phase1b/v2_2_2/review_queue_v2_2_2_all_campaigns.csv"
DEFAULT_BRIEFS = "configs/campaign_review_briefs_v1.yaml"
DEFAULT_POLICY = "configs/review_queue_policy_v2_2_3.yaml"
DEFAULT_OUT_DIR = "data/review/phase1b/v2_2_3/html"


BUCKET_FOCUS_KO = {
    "model_top_diagnostic": "모델이 높게 본 후보입니다. 실제로 좋은지, 점수에 속은 후보는 아닌지 확인합니다.",
    "clip_high_model_low": "CLIP은 좋게 봤지만 모델은 낮게 본 후보입니다. 모델이 놓친 semantic positive인지 확인합니다.",
    "dinov2_high_model_low": "DINOv2 visual-anchor는 높지만 모델은 낮게 본 후보입니다. 시각적으로 비슷하지만 주제에는 안 맞는지 확인합니다.",
    "model_high_clip_negative_high": "구 bucket명입니다. v2.2.3에서는 prompt_conflict_audit으로 해석합니다. hard negative라고 단정하지 말고 negative prompt 충돌 여부를 봅니다.",
    "prompt_conflict_audit": "모델 점수는 높지만 negative prompt와도 가까운 후보입니다. hard negative가 아니라 prompt-conflict 점검용입니다.",
    "cluster_diversity": "시각 cluster 다양성 확보용 후보입니다. 새로운 좋은 유형이나 coverage gap이 있는지 확인합니다.",
    "layout_safe_coverage": "텍스트 배치가 안전해 보이는 후보입니다. 실제로 제목/정보 박스를 얹어도 되는지 확인합니다.",
    "random_control": "retrieval candidate pool 내부 random control입니다. 모델이 놓친 의외의 좋은 후보가 있는지 확인합니다.",
}


def read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"yaml root must be object: {path}")
    return data


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def esc(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def join_criteria(xs: Any) -> str:
    if isinstance(xs, list):
        return " / ".join(str(x) for x in xs)
    if xs is None:
        return ""
    return str(xs)


def normalize_bucket(bucket: str) -> str:
    if bucket == "model_high_clip_negative_high":
        return "prompt_conflict_audit"
    return bucket


def load_manifest_paths(repo_root: Path) -> dict[str, str]:
    manifest = repo_root / "data/ontology/raw_image_manifest_v2_2_1.jsonl"
    out: dict[str, str] = {}
    if not manifest.exists():
        return out

    candidate_keys = [
        "path",
        "file_path",
        "raw_path",
        "relative_path",
        "image_path",
        "source_path",
        "local_path",
    ]

    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            image_id = row.get("image_id") or row.get("raw_image_id") or row.get("id")
            if not image_id:
                continue
            for k in candidate_keys:
                v = row.get(k)
                if v:
                    out[str(image_id)] = str(v)
                    break
    return out


def path_to_html_src(path_value: str, *, repo_root: Path, html_dir: Path) -> str:
    if not path_value:
        return ""

    p = Path(path_value)
    candidates = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(repo_root / p)
        candidates.append(repo_root / "data" / "raw" / p)
        candidates.append(p)

    for c in candidates:
        try:
            if c.exists():
                return os.path.relpath(c, html_dir)
        except OSError:
            pass

    # HTML에서 상대경로로라도 시도
    return path_value


def find_image_src(row: dict[str, Any], *, repo_root: Path, html_dir: Path, manifest_paths: dict[str, str]) -> str:
    candidate_cols = [
        "preview_path",
        "preview_file",
        "preview_image_path",
        "thumbnail_path",
        "thumb_path",
        "image_path",
        "raw_path",
        "file_path",
        "path",
        "image_file",
    ]

    for col in candidate_cols:
        v = row.get(col)
        if v:
            return path_to_html_src(str(v), repo_root=repo_root, html_dir=html_dir)

    image_id = row.get("image_id") or row.get("raw_image_id")
    if image_id and str(image_id) in manifest_paths:
        return path_to_html_src(manifest_paths[str(image_id)], repo_root=repo_root, html_dir=html_dir)

    return ""


def enrich_rows(
    rows: list[dict[str, str]],
    *,
    briefs: dict[str, Any],
    policy: dict[str, Any],
) -> list[dict[str, Any]]:
    campaigns = briefs.get("campaigns", {})
    if not isinstance(campaigns, dict):
        raise RuntimeError("campaign_review_briefs_v1.yaml must contain campaigns object")

    bucket_cfg = policy.get("buckets", {}) if isinstance(policy.get("buckets", {}), dict) else {}

    enriched = []
    for idx, row in enumerate(rows, start=1):
        r: dict[str, Any] = dict(row)

        campaign_id = r.get("campaign_id", "")
        c = campaigns.get(campaign_id, {})

        source_bucket = r.get("source_bucket") or r.get("bucket") or ""
        bucket_display = normalize_bucket(str(source_bucket))
        bucket_policy = bucket_cfg.get(bucket_display, {})

        r["review_row_number"] = idx
        r["campaign_title_ko"] = c.get("title_ko", "")
        r["campaign_status"] = c.get("campaign_status", "")
        r["campaign_brief_ko"] = c.get("brief_ko", "")
        r["positive_visual_criteria_ko"] = join_criteria(c.get("positive_visual_criteria_ko", []))
        r["negative_visual_criteria_ko"] = join_criteria(c.get("negative_visual_criteria_ko", []))
        r["review_question_ko"] = f"이 이미지를 '{c.get('title_ko', campaign_id)}' 홍보물 배경으로 쓸 수 있는가?"

        r["bucket_display_name_v2_2_3"] = bucket_display
        r["bucket_review_focus_ko"] = BUCKET_FOCUS_KO.get(
            bucket_display,
            str(bucket_policy.get("interpretation") or bucket_policy.get("purpose") or "이 bucket의 후보가 실제로 적합한지 확인합니다."),
        )

        r["score_status"] = r.get("score_status") or "diagnostic_only"
        r["threshold_status"] = r.get("threshold_status") or "no_calibrated_threshold"

        enriched.append(r)

    return enriched


def render_card(row: dict[str, Any], *, image_src: str) -> str:
    decision = row.get("decision", "")
    issue_tags = row.get("issue_tags", "")
    score = row.get("diagnostic_score") or row.get("score") or row.get("model_score") or row.get("classifier_score") or ""

    image_html = (
        f'<img src="{esc(image_src)}" loading="lazy" />'
        if image_src
        else f'<div class="missing-image">이미지 경로 없음<br>{esc(row.get("image_id", ""))}</div>'
    )

    return f"""
    <article class="card">
      <div class="imgbox">{image_html}</div>
      <div class="meta">
        <div class="row-id">#{esc(row.get("review_row_number"))} · {esc(row.get("image_id", ""))}</div>
        <h3>{esc(row.get("campaign_title_ko", row.get("campaign_id", "")))}</h3>
        <p class="brief">{esc(row.get("campaign_brief_ko", ""))}</p>

        <div class="criteria good"><b>좋은 기준</b><br>{esc(row.get("positive_visual_criteria_ko", ""))}</div>
        <div class="criteria bad"><b>거절 기준</b><br>{esc(row.get("negative_visual_criteria_ko", ""))}</div>

        <div class="question">{esc(row.get("review_question_ko", ""))}</div>

        <dl>
          <dt>source_bucket</dt><dd>{esc(row.get("source_bucket", row.get("bucket", "")))}</dd>
          <dt>v2.2.3 bucket</dt><dd>{esc(row.get("bucket_display_name_v2_2_3", ""))}</dd>
          <dt>bucket 검토 포인트</dt><dd>{esc(row.get("bucket_review_focus_ko", ""))}</dd>
          <dt>score</dt><dd>{esc(score)}</dd>
          <dt>decision</dt><dd>{esc(decision)}</dd>
          <dt>issue_tags</dt><dd>{esc(issue_tags)}</dd>
        </dl>
      </div>
    </article>
    """


def render_page(title: str, rows: list[dict[str, Any]], *, repo_root: Path, html_dir: Path, manifest_paths: dict[str, str]) -> str:
    cards = []
    for row in rows:
        image_src = find_image_src(row, repo_root=repo_root, html_dir=html_dir, manifest_paths=manifest_paths)
        cards.append(render_card(row, image_src=image_src))

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>{esc(title)}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Noto Sans KR", sans-serif;
      margin: 24px;
      background: #f7f7f7;
      color: #222;
    }}
    h1 {{ margin-bottom: 8px; }}
    .notice {{
      background: #fff7d6;
      border: 1px solid #e0c45c;
      padding: 12px 14px;
      margin: 16px 0 24px;
      border-radius: 10px;
      line-height: 1.5;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: white;
      border: 1px solid #ddd;
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }}
    .imgbox {{
      background: #eee;
      width: 100%;
      aspect-ratio: 4 / 3;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }}
    .imgbox img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }}
    .missing-image {{
      color: #777;
      text-align: center;
      line-height: 1.5;
    }}
    .meta {{
      padding: 14px;
    }}
    .row-id {{
      color: #777;
      font-size: 12px;
      margin-bottom: 6px;
    }}
    h3 {{
      margin: 0 0 8px;
      font-size: 18px;
    }}
    .brief {{
      margin: 0 0 10px;
      line-height: 1.45;
    }}
    .criteria {{
      padding: 8px;
      border-radius: 8px;
      margin: 8px 0;
      line-height: 1.4;
      font-size: 13px;
    }}
    .good {{ background: #eef8ef; }}
    .bad {{ background: #fff0f0; }}
    .question {{
      margin: 10px 0;
      padding: 10px;
      background: #eef3ff;
      border-radius: 8px;
      font-weight: 600;
    }}
    dl {{
      display: grid;
      grid-template-columns: 130px 1fr;
      gap: 6px 10px;
      font-size: 13px;
      margin: 10px 0 0;
    }}
    dt {{ color: #666; }}
    dd {{ margin: 0; word-break: break-word; }}
    a {{ color: #1558d6; }}
  </style>
</head>
<body>
  <h1>{esc(title)}</h1>
  <div class="notice">
    <b>v2.2.3 Review Sheet</b><br>
    이 화면은 사람이 campaign_id만 보고 판단하지 않도록 한국어 campaign 설명과 bucket 검토 포인트를 포함합니다.<br>
    diagnostic score는 최종 품질 점수나 pass/fail threshold가 아닙니다.
  </div>
  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--campaign-briefs", default=DEFAULT_BRIEFS)
    ap.add_argument("--policy", default=DEFAULT_POLICY)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--prefix", default="review_queue_v2_2_3")
    args = ap.parse_args()

    repo_root = Path(".").resolve()
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(input_path)
    briefs = read_yaml(Path(args.campaign_briefs))
    policy = read_yaml(Path(args.policy))
    enriched = enrich_rows(rows, briefs=briefs, policy=policy)

    existing_fields = list(rows[0].keys()) if rows else []
    added_fields = [
        "review_row_number",
        "campaign_title_ko",
        "campaign_status",
        "campaign_brief_ko",
        "positive_visual_criteria_ko",
        "negative_visual_criteria_ko",
        "review_question_ko",
        "bucket_display_name_v2_2_3",
        "bucket_review_focus_ko",
        "score_status",
        "threshold_status",
    ]
    label_fields = ["decision", "issue_tags", "preference_rank", "notes", "human_review_status"]
    fieldnames = []
    for f in existing_fields + added_fields + label_fields:
        if f not in fieldnames:
            fieldnames.append(f)

    enriched_csv = Path(args.out_dir) / f"{args.prefix}_enriched.csv"
    labeled_template = Path(args.out_dir) / f"{args.prefix}_enriched.labeled_template.csv"
    write_csv(enriched_csv, enriched, fieldnames)
    write_csv(labeled_template, enriched, fieldnames)

    manifest_paths = load_manifest_paths(repo_root)

    by_campaign: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        by_campaign[str(row.get("campaign_id", "unknown"))].append(row)

    campaign_pages = []
    for campaign_id, cr in sorted(by_campaign.items()):
        title = f"{campaign_id} · {cr[0].get('campaign_title_ko', '')}"
        page_name = f"{args.prefix}__{campaign_id}.html"
        page_path = out_dir / page_name
        page_path.write_text(
            render_page(title, cr, repo_root=repo_root, html_dir=out_dir, manifest_paths=manifest_paths),
            encoding="utf-8",
        )
        campaign_pages.append({
            "campaign_id": campaign_id,
            "title_ko": cr[0].get("campaign_title_ko", ""),
            "rows": len(cr),
            "page": page_name,
        })

    index_rows = "\n".join(
        f'<li><a href="{esc(x["page"])}">{esc(x["campaign_id"])} · {esc(x["title_ko"])}</a> ({x["rows"]} rows)</li>'
        for x in campaign_pages
    )
    index_html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>v2.2.3 Review Queue Index</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Noto Sans KR", sans-serif;
      margin: 32px;
      line-height: 1.6;
    }}
    code {{ background: #eee; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>v2.2.3 Review Queue Index</h1>
  <p>입력 파일: <code>{esc(str(input_path))}</code></p>
  <p>라벨 입력 템플릿: <code>{esc(str(labeled_template))}</code></p>
  <p>diagnostic score는 최종 품질 점수나 threshold가 아닙니다.</p>
  <ul>
    {index_rows}
  </ul>
</body>
</html>
"""
    index_path = out_dir / "index.html"
    index_path.write_text(index_html, encoding="utf-8")

    summary = {
        "event": "done",
        "input": str(input_path),
        "out_dir": str(out_dir),
        "index": str(index_path),
        "enriched_csv": str(enriched_csv),
        "labeled_template": str(labeled_template),
        "rows": len(enriched),
        "campaign_count": len(by_campaign),
        "campaign_pages": campaign_pages,
        "campaign_status_counts": dict(Counter(str(r.get("campaign_status", "")) for r in enriched)),
        "bucket_counts": dict(Counter(str(r.get("bucket_display_name_v2_2_3", "")) for r in enriched)),
        "score_status": "diagnostic_only",
        "threshold_status": "no_calibrated_threshold",
    }
    summary_path = out_dir / f"{args.prefix}_summary.json"
    write_json(summary_path, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
