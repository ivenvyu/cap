from __future__ import annotations

import argparse
import csv
import glob
import html
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SCORE_SNAPSHOT = "data/retrieval/phase1b/v2_2_5/candidate_score_snapshot_v2_2_5.jsonl"
DEFAULT_FEATURE_GLOB = "data/feature_snapshots/v2_2_5/phase1b_duplicate_canonicalized/*.jsonl"
DEFAULT_CAMPAIGN_BRIEFS = "configs/campaign_review_briefs_v1.yaml"
DEFAULT_OUT_DIR = "data/review/phase1b/v2_2_5/shortlist"


MODEL_SCORE_KEYS = [
    "diagnostic_score",
    "model_score",
    "classifier_score",
    "score",
    "predicted_probability",
    "predicted_positive_probability",
    "probability",
    "prob_accept",
]


BUCKET_PLAN = {
    "model_top_shortlist": 5,
    "prompt_conflict_audit": 1,
    "clip_model_disagreement_audit": 1,
    "dinov2_model_disagreement_audit": 1,
}


BUCKET_FOCUS_KO = {
    "model_top_shortlist": "모델이 높게 본 추천 후보입니다. 실제 홍보물 배경으로 쓸 만한지 확인합니다.",
    "prompt_conflict_audit": "모델 점수는 높지만 negative prompt와도 가까운 후보입니다. hard negative라고 단정하지 말고 prompt 충돌 여부를 봅니다.",
    "clip_model_disagreement_audit": "CLIP은 높게 봤지만 모델은 낮게 본 후보입니다. 모델이 낮게 본 판단이 타당한지 확인합니다.",
    "dinov2_model_disagreement_audit": "DINOv2 visual-anchor는 높지만 모델은 낮게 본 후보입니다. 시각적으로 비슷하지만 주제 적합성은 약한지 확인합니다.",
    "fill_model_top": "중복 제거 후 부족분을 모델 상위 후보로 채운 항목입니다.",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise RuntimeError(f"missing jsonl: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"missing yaml: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"yaml root must be object: {path}")
    return data


def to_float(x: Any, default: float = 0.0) -> float:
    if x is None or x == "":
        return default
    try:
        return float(x)
    except Exception:
        return default


def esc(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def join_criteria(xs: Any) -> str:
    if isinstance(xs, list):
        return " / ".join(str(x) for x in xs)
    if xs is None:
        return ""
    return str(xs)


def load_features(pattern: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"no feature files matched: {pattern}")

    for file in files:
        for row in read_jsonl(Path(file)):
            fid = row.get("feature_snapshot_id")
            if fid:
                out[str(fid)] = row
    return out


def get_feature(row: dict[str, Any], feature_by_id: dict[str, dict[str, Any]], name: str, default: float = 0.0) -> float:
    features = row.get("features")
    if isinstance(features, dict) and name in features:
        return to_float(features.get(name), default)

    fid = row.get("feature_snapshot_id")
    if fid and str(fid) in feature_by_id:
        f = feature_by_id[str(fid)].get("features", {})
        if isinstance(f, dict):
            return to_float(f.get(name), default)

    return default





def get_score(row: dict[str, Any]) -> float:
    # v2.2.5 canonical score snapshot의 주 점수
    scores = row.get("scores")
    if isinstance(scores, dict):
        if "diagnostic_accept_score" in scores:
            return to_float(scores.get("diagnostic_accept_score"), 0.0)

    # top-level known score keys
    for k in MODEL_SCORE_KEYS:
        if k in row:
            return to_float(row.get(k), 0.0)

    # nested score containers
    for container_key in ["scores", "model_scores", "score_snapshot"]:
        obj = row.get(container_key)
        if isinstance(obj, dict):
            for k in [
                "diagnostic_accept_score",
                "positive_probability",
                "predicted_positive_probability",
                "prob_positive",
                "probability_positive",
                "classifier_probability",
                "classifier_score",
                "diagnostic_score",
                "score",
            ]:
                if k in obj:
                    return to_float(obj.get(k), 0.0)

            numeric_items = []
            for k, v in obj.items():
                if isinstance(v, (int, float)):
                    numeric_items.append((k, float(v)))
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, (int, float)):
                            numeric_items.append((f"{k}.{kk}", float(vv)))

            preferred = [
                item for item in numeric_items
                if any(token in item[0].lower() for token in ["diagnostic_accept", "positive", "prob", "accept"])
            ]
            if preferred:
                return float(preferred[0][1])
            if numeric_items:
                return float(numeric_items[0][1])

    for k, v in row.items():
        lk = str(k).lower()
        if ("score" in lk or "prob" in lk) and isinstance(v, (int, float)):
            return float(v)

    raise RuntimeError(
        "Could not find diagnostic model score column in candidate_score_snapshot. "
        f"Available keys sample: {sorted(row.keys())[:30]} "
        f"scores={row.get('scores')}"
    )


def get_id(row: dict[str, Any], *keys: str, default: str = "") -> str:
    for k in keys:
        v = row.get(k)
        if v not in (None, ""):
            return str(v)
    return default


def load_manifest_paths(repo_root: Path) -> dict[str, str]:
    manifest = repo_root / "data/ontology/raw_image_manifest_v2_2_1.jsonl"
    out: dict[str, str] = {}
    if not manifest.exists():
        return out

    candidate_path_keys = [
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
            row = json.loads(line)
            image_id = row.get("image_id") or row.get("raw_image_id") or row.get("id")
            if not image_id:
                continue
            for k in candidate_path_keys:
                v = row.get(k)
                if v:
                    out[str(image_id)] = str(v)
                    break
    return out


def path_to_html_src(path_value: str, *, repo_root: Path, html_dir: Path) -> str:
    if not path_value:
        return ""

    p = Path(path_value)
    candidates: list[Path] = []

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

    return path_value


def find_image_src(row: dict[str, Any], *, repo_root: Path, html_dir: Path, manifest_paths: dict[str, str]) -> str:
    for col in [
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
    ]:
        v = row.get(col)
        if v:
            return path_to_html_src(str(v), repo_root=repo_root, html_dir=html_dir)

    image_id = row.get("image_id") or row.get("raw_image_id")
    if image_id and str(image_id) in manifest_paths:
        return path_to_html_src(manifest_paths[str(image_id)], repo_root=repo_root, html_dir=html_dir)

    return ""


def active_campaign_ids(briefs: dict[str, Any]) -> list[str]:
    campaigns = briefs.get("campaigns", {})
    out = []
    for cid, cfg in campaigns.items():
        if cfg.get("campaign_status") == "coverage_gap_diagnostic":
            continue
        if cfg.get("exclude_from_model_quality_claims") is True:
            continue
        out.append(str(cid))
    return sorted(out)


def enrich_row(
    row: dict[str, Any],
    *,
    feature_by_id: dict[str, dict[str, Any]],
    campaign_cfg: dict[str, Any],
    bucket: str,
    shortlist_rank: int,
) -> dict[str, Any]:
    score = get_score(row)

    out = dict(row)
    out["shortlist_version"] = "v2_2_5"
    out["shortlist_rank"] = shortlist_rank
    out["shortlist_bucket"] = bucket
    out["bucket_review_focus_ko"] = BUCKET_FOCUS_KO.get(bucket, "")
    out["diagnostic_model_score"] = score
    out["score_status"] = "diagnostic_only"
    out["threshold_status"] = "no_calibrated_threshold"

    visual_dup_id = get_duplicate_group_id(row, feature_by_id, fallback=get_id(row, "image_id", "raw_image_id"))
    out["visual_duplicate_group_id"] = visual_dup_id
    if not out.get("duplicate_group_id"):
        out["duplicate_group_id"] = visual_dup_id

    out["campaign_title_ko"] = campaign_cfg.get("title_ko", "")
    out["campaign_status"] = campaign_cfg.get("campaign_status", "")
    out["campaign_brief_ko"] = campaign_cfg.get("brief_ko", "")
    out["positive_visual_criteria_ko"] = join_criteria(campaign_cfg.get("positive_visual_criteria_ko", []))
    out["negative_visual_criteria_ko"] = join_criteria(campaign_cfg.get("negative_visual_criteria_ko", []))
    out["review_question_ko"] = f"이 이미지를 '{campaign_cfg.get('title_ko', out.get('campaign_id', ''))}' 홍보물 후보로 둘 만한가?"

    out["clip_positive_max_sim"] = get_feature(row, feature_by_id, "clip_positive_max_sim")
    out["clip_positive_mean_sim"] = get_feature(row, feature_by_id, "clip_positive_mean_sim")
    out["clip_negative_max_sim"] = get_feature(row, feature_by_id, "clip_negative_max_sim")
    out["clip_margin"] = get_feature(row, feature_by_id, "clip_margin")
    out["dinov2_campaign_margin"] = get_feature(row, feature_by_id, "dinov2_campaign_margin")
    out["dinov2_family_margin"] = get_feature(row, feature_by_id, "dinov2_family_margin")
    out["required_region_safe_min"] = get_feature(row, feature_by_id, "required_region_safe_min")
    out["required_region_safe_mean"] = get_feature(row, feature_by_id, "required_region_safe_mean")

    return out



def get_duplicate_group_id(row: dict[str, Any], feature_by_id: dict[str, dict[str, Any]], *, fallback: str = "") -> str:
    # Candidate score snapshot에 duplicate_group_id가 있으면 우선 사용
    for key in ["duplicate_group_id", "exact_duplicate_group_id", "visual_duplicate_group_id"]:
        v = row.get(key)
        if v not in (None, ""):
            return str(v)

    # 없으면 feature_snapshot에서 duplicate_group_id를 조회
    fid = row.get("feature_snapshot_id")
    if fid and str(fid) in feature_by_id:
        fs = feature_by_id[str(fid)]
        for key in ["duplicate_group_id", "exact_duplicate_group_id", "visual_duplicate_group_id"]:
            v = fs.get(key)
            if v not in (None, ""):
                return str(v)

    return str(fallback or row.get("image_id") or row.get("raw_image_id") or "")


def add_unique(
    selected: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    bucket: str,
    budget: int,
    used_images: set[str],
    used_dups: set[str],
    feature_by_id: dict[str, dict[str, Any]],
    campaign_cfg: dict[str, Any],
) -> int:
    added = 0
    for row in candidates:
        image_id = get_id(row, "image_id", "raw_image_id")
        dup_id = get_duplicate_group_id(row, feature_by_id, fallback=image_id)

        if image_id in used_images:
            continue
        if dup_id and dup_id in used_dups:
            continue

        selected.append(
            enrich_row(
                row,
                feature_by_id=feature_by_id,
                campaign_cfg=campaign_cfg,
                bucket=bucket,
                shortlist_rank=len(selected) + 1,
            )
        )
        used_images.add(image_id)
        if dup_id:
            used_dups.add(dup_id)

        added += 1
        if added >= budget:
            break

    return added


def select_for_campaign(
    rows: list[dict[str, Any]],
    *,
    feature_by_id: dict[str, dict[str, Any]],
    campaign_cfg: dict[str, Any],
    per_campaign: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    used_images: set[str] = set()
    used_dups: set[str] = set()

    def score(row: dict[str, Any]) -> float:
        return get_score(row)

    def clip_signal(row: dict[str, Any]) -> float:
        return max(
            get_feature(row, feature_by_id, "clip_positive_max_sim"),
            get_feature(row, feature_by_id, "clip_positive_mean_sim"),
            get_feature(row, feature_by_id, "clip_margin"),
        )

    def dino_signal(row: dict[str, Any]) -> float:
        return max(
            get_feature(row, feature_by_id, "dinov2_campaign_margin"),
            get_feature(row, feature_by_id, "dinov2_family_margin"),
            get_feature(row, feature_by_id, "dinov2_campaign_pos_nn_sim"),
            get_feature(row, feature_by_id, "dinov2_family_pos_nn_sim"),
        )

    def clip_neg_signal(row: dict[str, Any]) -> float:
        return get_feature(row, feature_by_id, "clip_negative_max_sim")

    # 1. model top
    model_top = sorted(rows, key=score, reverse=True)
    n = add_unique(
        selected,
        model_top,
        bucket="model_top_shortlist",
        budget=BUCKET_PLAN["model_top_shortlist"],
        used_images=used_images,
        used_dups=used_dups,
        feature_by_id=feature_by_id,
        campaign_cfg=campaign_cfg,
    )
    audit.append({"bucket": "model_top_shortlist", "requested": BUCKET_PLAN["model_top_shortlist"], "added": n})

    # 2. prompt conflict: high model + high negative prompt
    prompt_conflict = sorted(rows, key=lambda r: (score(r), clip_neg_signal(r)), reverse=True)
    n = add_unique(
        selected,
        prompt_conflict,
        bucket="prompt_conflict_audit",
        budget=BUCKET_PLAN["prompt_conflict_audit"],
        used_images=used_images,
        used_dups=used_dups,
        feature_by_id=feature_by_id,
        campaign_cfg=campaign_cfg,
    )
    audit.append({"bucket": "prompt_conflict_audit", "requested": BUCKET_PLAN["prompt_conflict_audit"], "added": n})

    # 3. CLIP high / model low
    clip_disagreement = sorted(rows, key=lambda r: (clip_signal(r), -score(r)), reverse=True)
    n = add_unique(
        selected,
        clip_disagreement,
        bucket="clip_model_disagreement_audit",
        budget=BUCKET_PLAN["clip_model_disagreement_audit"],
        used_images=used_images,
        used_dups=used_dups,
        feature_by_id=feature_by_id,
        campaign_cfg=campaign_cfg,
    )
    audit.append({"bucket": "clip_model_disagreement_audit", "requested": BUCKET_PLAN["clip_model_disagreement_audit"], "added": n})

    # 4. DINO high / model low
    dino_disagreement = sorted(rows, key=lambda r: (dino_signal(r), -score(r)), reverse=True)
    n = add_unique(
        selected,
        dino_disagreement,
        bucket="dinov2_model_disagreement_audit",
        budget=BUCKET_PLAN["dinov2_model_disagreement_audit"],
        used_images=used_images,
        used_dups=used_dups,
        feature_by_id=feature_by_id,
        campaign_cfg=campaign_cfg,
    )
    audit.append({"bucket": "dinov2_model_disagreement_audit", "requested": BUCKET_PLAN["dinov2_model_disagreement_audit"], "added": n})

    # 부족분은 model top으로 채움
    if len(selected) < per_campaign:
        n = add_unique(
            selected,
            model_top,
            bucket="fill_model_top",
            budget=per_campaign - len(selected),
            used_images=used_images,
            used_dups=used_dups,
            feature_by_id=feature_by_id,
            campaign_cfg=campaign_cfg,
        )
        audit.append({"bucket": "fill_model_top", "requested": per_campaign - len(selected), "added": n})

    # rank 재정렬: model top 성격이므로 score desc 우선, 같은 bucket 내 안정 정렬
    selected.sort(key=lambda r: (str(r.get("campaign_id")), -to_float(r.get("diagnostic_model_score")), str(r.get("shortlist_bucket"))))
    for i, r in enumerate(selected, start=1):
        r["shortlist_rank"] = i

    return selected, audit


def render_card(row: dict[str, Any], *, image_src: str) -> str:
    image_html = (
        f'<img src="{esc(image_src)}" loading="lazy" />'
        if image_src
        else f'<div class="missing-image">이미지 경로 없음<br>{esc(row.get("image_id", ""))}</div>'
    )

    return f"""
    <article class="card">
      <div class="imgbox">{image_html}</div>
      <div class="meta">
        <div class="rank">#{esc(row.get("shortlist_rank"))} · {esc(row.get("shortlist_bucket"))}</div>
        <h3>{esc(row.get("campaign_title_ko"))}</h3>
        <p class="brief">{esc(row.get("campaign_brief_ko"))}</p>
        <div class="criteria good"><b>좋은 기준</b><br>{esc(row.get("positive_visual_criteria_ko"))}</div>
        <div class="criteria bad"><b>거절 기준</b><br>{esc(row.get("negative_visual_criteria_ko"))}</div>
        <div class="focus"><b>검토 포인트</b><br>{esc(row.get("bucket_review_focus_ko"))}</div>
        <div class="question">{esc(row.get("review_question_ko"))}</div>
        <dl>
          <dt>image</dt><dd>{esc(row.get("image_id", ""))}</dd>
          <dt>score</dt><dd>{esc(round(to_float(row.get("diagnostic_model_score")), 4))}</dd>
          <dt>clip_margin</dt><dd>{esc(round(to_float(row.get("clip_margin")), 4))}</dd>
          <dt>clip_negative</dt><dd>{esc(round(to_float(row.get("clip_negative_max_sim")), 4))}</dd>
          <dt>dino_campaign_margin</dt><dd>{esc(round(to_float(row.get("dinov2_campaign_margin")), 4))}</dd>
          <dt>dino_family_margin</dt><dd>{esc(round(to_float(row.get("dinov2_family_margin")), 4))}</dd>
          <dt>safe_min</dt><dd>{esc(round(to_float(row.get("required_region_safe_min")), 4))}</dd>
          <dt>path</dt><dd>{esc(row.get("path") or row.get("raw_path") or row.get("file_path") or "")}</dd>
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
    .meta {{ padding: 14px; }}
    .rank {{ color: #666; font-size: 13px; margin-bottom: 6px; }}
    h3 {{ margin: 0 0 8px; font-size: 18px; }}
    .brief {{ margin: 0 0 10px; line-height: 1.45; }}
    .criteria, .focus {{
      padding: 8px;
      border-radius: 8px;
      margin: 8px 0;
      line-height: 1.4;
      font-size: 13px;
    }}
    .good {{ background: #eef8ef; }}
    .bad {{ background: #fff0f0; }}
    .focus {{ background: #eef3ff; }}
    .question {{
      margin: 10px 0;
      padding: 10px;
      background: #f3f3f3;
      border-radius: 8px;
      font-weight: 600;
    }}
    dl {{
      display: grid;
      grid-template-columns: 150px 1fr;
      gap: 6px 10px;
      font-size: 13px;
      margin: 10px 0 0;
    }}
    dt {{ color: #666; }}
    dd {{ margin: 0; word-break: break-word; }}
  </style>
</head>
<body>
  <h1>{esc(title)}</h1>
  <div class="notice">
    <b>v2.2.5 Recommendation Shortlist</b><br>
    이 화면은 추가 대량 라벨링용이 아니라 추천 후보 확인용입니다.<br>
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
    ap.add_argument("--score-snapshot", default=DEFAULT_SCORE_SNAPSHOT)
    ap.add_argument("--feature-glob", default=DEFAULT_FEATURE_GLOB)
    ap.add_argument("--campaign-briefs", default=DEFAULT_CAMPAIGN_BRIEFS)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--per-campaign", type=int, default=8)
    args = ap.parse_args()

    repo_root = Path(".").resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    score_rows = read_jsonl(Path(args.score_snapshot))
    feature_by_id = load_features(args.feature_glob)
    briefs_doc = read_yaml(Path(args.campaign_briefs))
    campaigns_cfg = briefs_doc.get("campaigns", {})
    active_campaigns = set(active_campaign_ids(briefs_doc))

    rows_by_campaign: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded_campaigns = Counter()

    for row in score_rows:
        cid = str(row.get("campaign_id", ""))
        if cid not in active_campaigns:
            excluded_campaigns[cid] += 1
            continue
        rows_by_campaign[cid].append(row)

    all_selected: list[dict[str, Any]] = []
    audit_by_campaign: dict[str, Any] = {}

    for cid in sorted(active_campaigns):
        rows = rows_by_campaign.get(cid, [])
        if not rows:
            audit_by_campaign[cid] = {"candidate_rows": 0, "selected_rows": 0, "bucket_audit": []}
            continue

        selected, bucket_audit = select_for_campaign(
            rows,
            feature_by_id=feature_by_id,
            campaign_cfg=campaigns_cfg.get(cid, {}),
            per_campaign=args.per_campaign,
        )

        for r in selected:
            r["campaign_id"] = cid

        all_selected.extend(selected)
        audit_by_campaign[cid] = {
            "candidate_rows": len(rows),
            "selected_rows": len(selected),
            "bucket_audit": bucket_audit,
            "bucket_counts": dict(Counter(str(r.get("shortlist_bucket")) for r in selected)),
            "score_min": min(to_float(r.get("diagnostic_model_score")) for r in selected) if selected else None,
            "score_median": sorted(to_float(r.get("diagnostic_model_score")) for r in selected)[len(selected)//2] if selected else None,
            "score_max": max(to_float(r.get("diagnostic_model_score")) for r in selected) if selected else None,
        }

    # 출력 CSV
    fieldnames: list[str] = []
    preferred = [
        "shortlist_version",
        "campaign_id",
        "campaign_title_ko",
        "campaign_status",
        "shortlist_rank",
        "shortlist_bucket",
        "bucket_review_focus_ko",
        "review_question_ko",
        "image_id",
        "raw_image_id",
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
        "path",
        "raw_path",
        "file_path",
        "campaign_brief_ko",
        "positive_visual_criteria_ko",
        "negative_visual_criteria_ko",
        "score_status",
        "threshold_status",
        "decision",
        "issue_tags",
        "notes",
    ]

    for f in preferred:
        if f not in fieldnames:
            fieldnames.append(f)

    for row in all_selected:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    csv_out = out_dir / "recommendation_shortlist_v2_2_5.csv"
    labeled_template = out_dir / "recommendation_shortlist_v2_2_5.labeled_template.csv"

    # decision columns blank
    out_rows = []
    for r in all_selected:
        rr = dict(r)
        rr.setdefault("decision", "")
        rr.setdefault("issue_tags", "")
        rr.setdefault("notes", "")
        out_rows.append(rr)

    write_csv(csv_out, out_rows, fieldnames)
    write_csv(labeled_template, out_rows, fieldnames)

    # HTML
    manifest_paths = load_manifest_paths(repo_root)
    by_campaign: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in all_selected:
        by_campaign[str(r.get("campaign_id"))].append(r)

    campaign_pages = []
    for cid, cr in sorted(by_campaign.items()):
        title = f"{cid} · {campaigns_cfg.get(cid, {}).get('title_ko', '')}"
        page_name = f"recommendation_shortlist_v2_2_5__{cid}.html"
        page_path = out_dir / page_name
        page_path.write_text(
            render_page(title, cr, repo_root=repo_root, html_dir=out_dir, manifest_paths=manifest_paths),
            encoding="utf-8",
        )
        campaign_pages.append({
            "campaign_id": cid,
            "title_ko": campaigns_cfg.get(cid, {}).get("title_ko", ""),
            "rows": len(cr),
            "page": page_name,
        })

    links = "\n".join(
        f'<li><a href="{esc(x["page"])}">{esc(x["campaign_id"])} · {esc(x["title_ko"])}</a> ({x["rows"]} rows)</li>'
        for x in campaign_pages
    )

    index_html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>v2.2.5 Recommendation Shortlist</title>
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
  <h1>v2.2.5 Recommendation Shortlist</h1>
  <p>겨울 실내 갤러리 전시는 coverage-gap campaign으로 제외했습니다.</p>
  <p>campaign별 모델 상위 후보 중심으로 8개씩 뽑았습니다.</p>
  <p>CSV: <code>{esc(str(csv_out))}</code></p>
  <p>검토 템플릿: <code>{esc(str(labeled_template))}</code></p>
  <p>diagnostic score는 최종 품질 점수나 threshold가 아닙니다.</p>
  <ul>{links}</ul>
</body>
</html>
"""
    index_path = out_dir / "index.html"
    index_path.write_text(index_html, encoding="utf-8")

    summary = {
        "event": "done",
        "version": "v2.2.5",
        "purpose": "recommendation_shortlist_not_bulk_labeling",
        "score_snapshot": args.score_snapshot,
        "feature_glob": args.feature_glob,
        "campaign_briefs": args.campaign_briefs,
        "out_dir": str(out_dir),
        "csv_out": str(csv_out),
        "labeled_template": str(labeled_template),
        "index_html": str(index_path),
        "active_campaigns": sorted(active_campaigns),
        "excluded_campaign_counts": dict(excluded_campaigns),
        "per_campaign": args.per_campaign,
        "selected_rows": len(all_selected),
        "bucket_counts": dict(Counter(str(r.get("shortlist_bucket")) for r in all_selected)),
        "campaign_pages": campaign_pages,
        "audit_by_campaign": audit_by_campaign,
        "score_status": "diagnostic_only",
        "threshold_status": "no_calibrated_threshold",
        "non_claims": [
            "production model quality를 주장하지 않는다.",
            "campaign 간 score를 직접 비교하지 않는다.",
            "diagnostic score를 pass/fail threshold로 쓰지 않는다.",
            "이 shortlist는 추가 대량 라벨링 작업이 아니다.",
        ],
    }

    summary_out = out_dir / "recommendation_shortlist_v2_2_5_summary.json"
    write_json(summary_out, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
