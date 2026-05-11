from __future__ import annotations

import argparse
import csv
import html
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


QUEUE_VERSION = "cluster_label_queue_v1"
REPRESENTATIVE_METHOD = "dinov2_centroid_medoid_plus_greedy_diversity"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise RuntimeError(f"missing DB: {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def safe_id(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(s))


def ensure_queue_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cluster_label_queue (
            queue_id TEXT PRIMARY KEY,
            cluster_version TEXT NOT NULL,
            cluster_level TEXT NOT NULL,
            cluster_id TEXT NOT NULL,
            cluster_size INTEGER NOT NULL,
            representative_image_ids_json TEXT NOT NULL,
            representative_image_paths_json TEXT NOT NULL,
            representative_method TEXT NOT NULL,
            queue_status TEXT NOT NULL,
            score_status TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )


def cluster_column(level: str) -> str:
    mapping = {
        "coarse": "dinov2_cluster_id_coarse",
        "mid": "dinov2_cluster_id_mid",
        "fine": "dinov2_cluster_id_fine",
    }
    if level not in mapping:
        raise RuntimeError(f"unsupported cluster level: {level}")
    return mapping[level]


def load_cluster_members(conn: sqlite3.Connection, level: str) -> dict[str, list[dict[str, Any]]]:
    col = cluster_column(level)

    rows = conn.execute(
        f"""
        SELECT
            c.image_id,
            c.cluster_version,
            c.{col} AS cluster_id,
            i.path,
            i.resolved_path,
            e.embedding_row,
            e.npy_path
        FROM image_clusters c
        JOIN images i
          ON c.image_id = i.image_id
        JOIN image_embeddings e
          ON c.image_id = e.image_id
         AND e.model_type = 'dinov2'
        WHERE c.{col} IS NOT NULL
        ORDER BY c.{col}, c.image_id
        """
    ).fetchall()

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(r["cluster_id"])].append(dict(r))

    return dict(groups)


def choose_representatives(
    members: list[dict[str, Any]],
    embeddings: np.ndarray,
    max_representatives: int,
) -> list[dict[str, Any]]:
    if not members:
        return []

    rows = np.array([int(m["embedding_row"]) for m in members], dtype=int)
    vecs = embeddings[rows].astype(np.float32)

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    vecs = vecs / norms

    centroid = vecs.mean(axis=0, keepdims=True)
    centroid = centroid / np.maximum(np.linalg.norm(centroid), 1e-12)

    centroid_sims = (vecs @ centroid.T).reshape(-1)
    first = int(np.argmax(centroid_sims))

    selected = [first]
    target_n = min(max_representatives, len(members))

    while len(selected) < target_n:
        selected_vecs = vecs[selected]
        sims_to_selected = vecs @ selected_vecs.T
        max_sim_to_selected = sims_to_selected.max(axis=1)
        diversity_score = 1.0 - max_sim_to_selected

        for idx in selected:
            diversity_score[idx] = -1.0

        selected.append(int(np.argmax(diversity_score)))

    out = []
    for order, idx in enumerate(selected):
        item = dict(members[idx])
        item["representative_order"] = order
        item["representative_role"] = "centroid_medoid" if order == 0 else f"diversity_{order}"
        item["centroid_similarity"] = float(centroid_sims[idx])
        out.append(item)

    return out


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
    raise RuntimeError(f"cannot resolve image path for image_id={row['image_id']}: {candidates}")


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


def load_tag_vocab(conn: sqlite3.Connection) -> dict[str, list[str]]:
    rows = conn.execute(
        """
        SELECT axis_id, tag_name
        FROM tag_values
        WHERE status = 'active'
        ORDER BY axis_id, tag_name
        """
    ).fetchall()

    vocab: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        vocab[str(r["axis_id"])].append(str(r["tag_name"]))
    return dict(vocab)


def render_html(queue_rows: list[dict[str, Any]], tag_vocab: dict[str, list[str]], out_dir: Path) -> str:
    axis_blocks = []
    for axis_id, tags in tag_vocab.items():
        axis_blocks.append(
            f"<div><b>{html.escape(axis_id)}</b>: "
            + ", ".join(f"<code>{html.escape(t)}</code>" for t in tags)
            + "</div>"
        )

    cards = []
    for row in queue_rows:
        thumbs = []
        for thumb_path, image_id, role in zip(
            row["thumb_paths"],
            row["representative_image_ids"],
            row["representative_roles"],
        ):
            rel = Path(thumb_path).relative_to(out_dir)
            thumbs.append(
                f"""
                <div class="thumb">
                  <img src="{html.escape(str(rel))}" />
                  <div>{html.escape(image_id)}</div>
                  <div><code>{html.escape(role)}</code></div>
                </div>
                """
            )

        cards.append(
            f"""
            <section class="card">
              <h2>{html.escape(row["cluster_id"])}</h2>
              <div class="meta">
                <b>queue_id:</b> {html.escape(row["queue_id"])}<br/>
                <b>level:</b> {html.escape(row["cluster_level"])}<br/>
                <b>size:</b> {row["cluster_size"]}<br/>
                <b>method:</b> {html.escape(row["representative_method"])}
              </div>
              <div class="thumbs">
                {''.join(thumbs)}
              </div>
              <div class="label-box">
                <b>라벨링 메모:</b><br/>
                space_axis_tags:<br/>
                temporal_axis_tags:<br/>
                weather_light_axis_tags:<br/>
                subject_axis_tags:<br/>
                mood_axis_tags:<br/>
                usage_axis_tags:<br/>
                notes:
              </div>
            </section>
            """
        )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Ontology Cluster Label Queue v1</title>
<style>
body {{
  font-family: -apple-system, BlinkMacSystemFont, sans-serif;
  margin: 24px;
}}
.summary {{
  padding: 12px;
  background: #f7f7f7;
  border-radius: 10px;
  margin-bottom: 20px;
  line-height: 1.5;
}}
.card {{
  border: 1px solid #ddd;
  border-radius: 12px;
  padding: 16px;
  margin-bottom: 22px;
}}
.meta {{
  font-size: 13px;
  line-height: 1.5;
  color: #333;
}}
.thumbs {{
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 12px;
}}
.thumb {{
  width: 220px;
  font-size: 12px;
  word-break: break-all;
}}
.thumb img {{
  width: 220px;
  height: 160px;
  object-fit: contain;
  background: #f0f0f0;
  border-radius: 8px;
}}
.label-box {{
  margin-top: 14px;
  padding: 10px;
  background: #fafafa;
  border-radius: 8px;
  font-size: 13px;
  line-height: 1.7;
}}
code {{
  background: #eee;
  padding: 2px 4px;
  border-radius: 4px;
}}
</style>
</head>
<body>
<h1>Ontology Cluster Label Queue v1</h1>
<div class="summary">
  <b>purpose:</b> DINOv2 cluster 대표 이미지를 보고 cluster-level ontology tag를 붙이기 위한 queue<br/>
  <b>rows:</b> {len(queue_rows)}<br/>
  <b>score_status:</b> diagnostic_only<br/>
  <b>주의:</b> 이 queue는 pass/fail 평가가 아니라 ontology content를 채우기 위한 human labeling queue다.
</div>
<h2>Active tag vocabulary</h2>
<div class="summary">
{''.join(axis_blocks)}
</div>
{''.join(cards)}
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/db/cap_reranker_ontology.db")
    ap.add_argument("--cluster-level", default="coarse", choices=["coarse", "mid", "fine"])
    ap.add_argument("--max-representatives", type=int, default=5)
    ap.add_argument("--out-dir", default="data/review/ontology/cluster_label_queue_v1")
    ap.add_argument("--reset-level", action="store_true", default=True)
    args = ap.parse_args()

    conn = connect(Path(args.db))
    ensure_queue_table(conn)

    groups = load_cluster_members(conn, args.cluster_level)
    if not groups:
        raise RuntimeError(f"no cluster groups found for level={args.cluster_level}")

    npy_paths = {
        str(member["npy_path"])
        for members in groups.values()
        for member in members
    }
    if len(npy_paths) != 1:
        raise RuntimeError(f"expected one dinov2 npy path, got {sorted(npy_paths)}")

    npy_path = Path(next(iter(npy_paths)))
    if not npy_path.exists():
        raise RuntimeError(f"missing embedding npy: {npy_path}")

    embeddings = np.load(npy_path)

    if args.reset_level:
        conn.execute(
            "DELETE FROM cluster_label_queue WHERE cluster_level = ?",
            (args.cluster_level,),
        )

    out_dir = Path(args.out_dir)
    thumb_dir = out_dir / "thumbs"
    out_dir.mkdir(parents=True, exist_ok=True)

    queue_rows: list[dict[str, Any]] = []

    for cluster_id, members in sorted(groups.items()):
        reps = choose_representatives(
            members=members,
            embeddings=embeddings,
            max_representatives=args.max_representatives,
        )

        cluster_version = str(reps[0].get("cluster_version") or "dinov2_clusters_v1")
        queue_id = f"{QUEUE_VERSION}__{args.cluster_level}__{safe_id(cluster_id)}"

        rep_ids = [str(r["image_id"]) for r in reps]
        rep_paths = []
        rep_roles = []
        thumb_paths = []

        for r in reps:
            image_path = resolve_image_path(r)
            rep_paths.append(str(image_path))
            rep_roles.append(str(r["representative_role"]))

            thumb_path = thumb_dir / f"{queue_id}__{safe_id(str(r['image_id']))}.jpg"
            make_thumb(image_path, thumb_path, size=(220, 160))
            thumb_paths.append(str(thumb_path))

        conn.execute(
            """
            INSERT OR REPLACE INTO cluster_label_queue
            (queue_id, cluster_version, cluster_level, cluster_id, cluster_size,
             representative_image_ids_json, representative_image_paths_json,
             representative_method, queue_status, score_status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                queue_id,
                cluster_version,
                args.cluster_level,
                cluster_id,
                len(members),
                jdump(rep_ids),
                jdump(rep_paths),
                REPRESENTATIVE_METHOD,
                "pending_human_cluster_label",
                "diagnostic_only",
                utc_now(),
            ),
        )

        queue_rows.append(
            {
                "queue_id": queue_id,
                "cluster_version": cluster_version,
                "cluster_level": args.cluster_level,
                "cluster_id": cluster_id,
                "cluster_size": len(members),
                "representative_image_ids": rep_ids,
                "representative_image_paths": rep_paths,
                "representative_roles": rep_roles,
                "thumb_paths": thumb_paths,
                "representative_method": REPRESENTATIVE_METHOD,
            }
        )

    conn.commit()

    label_csv = out_dir / f"cluster_label_queue__{args.cluster_level}.csv"
    with label_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "queue_id",
            "cluster_level",
            "cluster_id",
            "cluster_size",
            "representative_image_ids",
            "space_axis_tags",
            "temporal_axis_tags",
            "weather_light_axis_tags",
            "subject_axis_tags",
            "mood_axis_tags",
            "usage_axis_tags",
            "design_affordance_axis_tags",
            "confidence_status",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in queue_rows:
            writer.writerow(
                {
                    "queue_id": row["queue_id"],
                    "cluster_level": row["cluster_level"],
                    "cluster_id": row["cluster_id"],
                    "cluster_size": row["cluster_size"],
                    "representative_image_ids": "|".join(row["representative_image_ids"]),
                    "space_axis_tags": "",
                    "temporal_axis_tags": "",
                    "weather_light_axis_tags": "",
                    "subject_axis_tags": "",
                    "mood_axis_tags": "",
                    "usage_axis_tags": "",
                    "design_affordance_axis_tags": "",
                    "confidence_status": "human_cluster_label_pending",
                    "notes": "",
                }
            )

    tag_vocab = load_tag_vocab(conn)
    index_html = out_dir / f"cluster_label_queue__{args.cluster_level}.html"
    index_html.write_text(render_html(queue_rows, tag_vocab, out_dir), encoding="utf-8")

    summary = {
        "event": "done",
        "db": args.db,
        "cluster_level": args.cluster_level,
        "queue_rows": len(queue_rows),
        "total_representative_images": sum(len(r["representative_image_ids"]) for r in queue_rows),
        "max_representatives": args.max_representatives,
        "representative_method": REPRESENTATIVE_METHOD,
        "label_csv": str(label_csv),
        "index_html": str(index_html),
        "score_status": "diagnostic_only",
    }

    summary_path = out_dir / f"cluster_label_queue__{args.cluster_level}.summary.json"
    summary_path.write_text(jdump(summary) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
