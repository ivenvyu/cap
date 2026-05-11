from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCORE_VERSION = "campaign_visual_cue_clip_v1"
SCORE_STATUS = "diagnostic_only"
DEFAULT_TEXT_MODEL = "clip-ViT-B-32"


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


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return x / norm


def load_clip_image_embeddings(conn: sqlite3.Connection) -> tuple[np.ndarray, list[dict[str, Any]], str]:
    rows = conn.execute(
        """
        SELECT
            e.image_id,
            e.embedding_row,
            e.npy_path,
            e.model_name,
            e.embedding_dim,
            i.path,
            i.category,
            i.source_group,
            i.place_name,
            i.subject_name
        FROM image_embeddings e
        JOIN images i
          ON e.image_id = i.image_id
        WHERE e.model_type = 'clip'
        ORDER BY e.embedding_row
        """
    ).fetchall()

    if not rows:
        raise RuntimeError("no CLIP image embeddings found in DB")

    npy_paths = {str(r["npy_path"]) for r in rows}
    if len(npy_paths) != 1:
        raise RuntimeError(f"expected one CLIP npy_path, got {sorted(npy_paths)}")

    npy_path = Path(next(iter(npy_paths)))
    if not npy_path.exists():
        raise RuntimeError(f"missing CLIP npy file: {npy_path}")

    image_embeddings = np.load(npy_path).astype(np.float32)
    image_embeddings = l2_normalize(image_embeddings, axis=1)

    model_names = {str(r["model_name"]) for r in rows if r["model_name"]}
    model_name = sorted(model_names)[0] if model_names else DEFAULT_TEXT_MODEL

    meta = [dict(r) for r in rows]

    return image_embeddings, meta, model_name


def load_requirements(conn: sqlite3.Connection, policy_id: str | None) -> list[dict[str, Any]]:
    if policy_id:
        rows = conn.execute(
            """
            SELECT
                r.campaign_id,
                r.cue_id,
                r.policy_id,
                r.requirement_role,
                r.source_field,
                r.source_value,
                c.cue_group,
                c.cue_type,
                c.prompts_json,
                c.ontology_write_allowed_if_human_verified,
                c.verified_ontology_tags_json
            FROM campaign_visual_cue_requirements r
            JOIN visual_cues c
              ON r.cue_id = c.cue_id
            WHERE r.policy_id = ?
            ORDER BY r.campaign_id, r.cue_id
            """,
            (policy_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT
                r.campaign_id,
                r.cue_id,
                r.policy_id,
                r.requirement_role,
                r.source_field,
                r.source_value,
                c.cue_group,
                c.cue_type,
                c.prompts_json,
                c.ontology_write_allowed_if_human_verified,
                c.verified_ontology_tags_json
            FROM campaign_visual_cue_requirements r
            JOIN visual_cues c
              ON r.cue_id = c.cue_id
            ORDER BY r.campaign_id, r.cue_id
            """
        ).fetchall()

    if not rows:
        raise RuntimeError("no campaign visual cue requirements found")

    out = []
    for r in rows:
        d = dict(r)
        d["prompts"] = json.loads(d.pop("prompts_json"))
        d["verified_ontology_tags"] = json.loads(d.pop("verified_ontology_tags_json") or "{}")
        out.append(d)

    return out


class TransformersClipTextEncoder:
    def __init__(self, model_name: str):
        import torch
        from transformers import CLIPModel, CLIPTokenizer

        self.torch = torch
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.eval()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)

    def encode(self, prompts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            tokens = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}

            # Use the explicit CLIP text path instead of get_text_features().
            # Some transformers versions return BaseModelOutputWithPooling from
            # get_text_features(), while others return a tensor.
            text_outputs = self.model.text_model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens.get("attention_mask"),
                return_dict=True,
            )
            pooled_output = text_outputs.pooler_output
            text_features = self.model.text_projection(pooled_output)
            text_features = text_features.detach().cpu().numpy().astype(np.float32)

        return l2_normalize(text_features, axis=1)


class SentenceTransformersTextEncoder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, prompts: list[str]) -> np.ndarray:
        emb = self.model.encode(
            prompts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        emb = np.asarray(emb, dtype=np.float32)
        return l2_normalize(emb, axis=1)


def load_text_encoder(model_name: str):
    errors = []

    try:
        return SentenceTransformersTextEncoder(model_name)
    except Exception as e:
        errors.append(f"sentence_transformers backend failed: {type(e).__name__}: {e}")

    try:
        return TransformersClipTextEncoder(model_name)
    except Exception as e:
        errors.append(f"transformers CLIP backend failed: {type(e).__name__}: {e}")

    raise RuntimeError(
        "failed to load any CLIP text encoder backend for campaign visual cue scoring. "
        "Tried sentence_transformers and transformers.CLIPModel. "
        "Errors: " + " | ".join(errors)
    )


def encode_prompts(model, prompts: list[str]) -> np.ndarray:
    emb = model.encode(prompts)
    emb = np.asarray(emb, dtype=np.float32)
    emb = l2_normalize(emb, axis=1)
    return emb


def infer_text_model_name(db_model_name: str, explicit: str | None) -> str:
    if explicit:
        return explicit

    # Existing DB may store either "clip-ViT-B-32" or a sentence-transformers path.
    if db_model_name:
        name = db_model_name.strip()
        if name:
            return name

    return DEFAULT_TEXT_MODEL


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/db/cap_reranker_ontology.db")
    ap.add_argument("--policy-id", default="seasonal_prompt_cue_policy_v1")
    ap.add_argument("--text-model", default=None)
    ap.add_argument("--score-version", default=SCORE_VERSION)
    ap.add_argument("--reset-score-version", action="store_true", default=True)
    ap.add_argument("--summary-out", default="audit/ontology/campaign_visual_cue_scores_clip_v1.summary.json")
    args = ap.parse_args()

    conn = connect(Path(args.db))

    image_embeddings, image_meta, db_clip_model_name = load_clip_image_embeddings(conn)
    text_model_name = infer_text_model_name(db_clip_model_name, args.text_model)

    model = load_text_encoder(text_model_name)

    requirements = load_requirements(conn, args.policy_id)

    if args.reset_score_version:
        conn.execute(
            """
            DELETE FROM campaign_image_cue_scores
            WHERE score_version = ?
            """,
            (args.score_version,),
        )

    inserted = 0
    requirement_summaries = []

    for req in requirements:
        prompts = req["prompts"]
        if not prompts:
            raise RuntimeError(f"cue has no prompts: {req['cue_id']}")

        prompt_embeddings = encode_prompts(model, prompts)

        if prompt_embeddings.shape[1] != image_embeddings.shape[1]:
            raise RuntimeError(
                f"text/image embedding dim mismatch for cue={req['cue_id']}: "
                f"text_dim={prompt_embeddings.shape[1]}, image_dim={image_embeddings.shape[1]}. "
                f"Use the same CLIP model as the existing image embeddings."
            )

        # max prompt cosine: cue is considered present if any positive prompt matches well.
        sims = image_embeddings @ prompt_embeddings.T
        cue_scores = sims.max(axis=1)

        top_idx = np.argsort(-cue_scores)[:10]
        top_images = [
            {
                "rank": int(rank + 1),
                "image_id": image_meta[int(idx)]["image_id"],
                "score": float(cue_scores[int(idx)]),
                "path": image_meta[int(idx)]["path"],
                "place_name": image_meta[int(idx)]["place_name"],
                "source_group": image_meta[int(idx)]["source_group"],
                "subject_name": image_meta[int(idx)]["subject_name"],
            }
            for rank, idx in enumerate(top_idx)
        ]

        for idx, meta in enumerate(image_meta):
            raw = {
                "campaign_id": req["campaign_id"],
                "image_id": meta["image_id"],
                "cue_id": req["cue_id"],
                "policy_id": req["policy_id"],
                "cue_group": req["cue_group"],
                "cue_type": req["cue_type"],
                "score_method": "max_prompt_cosine",
                "prompts": prompts,
                "score_version": args.score_version,
                "model_name": text_model_name,
                "score_status": SCORE_STATUS,
            }

            conn.execute(
                """
                INSERT OR REPLACE INTO campaign_image_cue_scores
                (campaign_id, image_id, cue_id, model_name, score_version,
                 score, score_status, created_at, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    req["campaign_id"],
                    meta["image_id"],
                    req["cue_id"],
                    text_model_name,
                    args.score_version,
                    float(cue_scores[idx]),
                    SCORE_STATUS,
                    utc_now(),
                    jdump(raw),
                ),
            )
            inserted += 1

        requirement_summaries.append(
            {
                "campaign_id": req["campaign_id"],
                "cue_id": req["cue_id"],
                "cue_group": req["cue_group"],
                "cue_type": req["cue_type"],
                "prompts": prompts,
                "score_method": "max_prompt_cosine",
                "image_count_scored": len(image_meta),
                "score_min": float(np.min(cue_scores)),
                "score_mean": float(np.mean(cue_scores)),
                "score_max": float(np.max(cue_scores)),
                "top_images": top_images,
                "threshold_status": "no_calibrated_threshold",
                "score_status": SCORE_STATUS,
            }
        )

    conn.commit()

    summary = {
        "event": "done",
        "db": args.db,
        "policy_id": args.policy_id,
        "model_name": text_model_name,
        "score_version": args.score_version,
        "score_status": SCORE_STATUS,
        "threshold_status": "no_calibrated_threshold",
        "requirements_scored": len(requirements),
        "image_count": len(image_meta),
        "campaign_image_cue_scores_inserted": inserted,
        "score_interpretation": (
            "diagnostic retrieval cue score only; not a calibrated threshold, "
            "not an ontology label, and not a production quality claim"
        ),
        "requirements": requirement_summaries,
    }

    out = Path(args.summary_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(jdump(summary) + "\n", encoding="utf-8")

    print(json.dumps(
        {
            "event": "done",
            "db": args.db,
            "policy_id": args.policy_id,
            "model_name": text_model_name,
            "score_version": args.score_version,
            "requirements_scored": len(requirements),
            "image_count": len(image_meta),
            "campaign_image_cue_scores_inserted": inserted,
            "summary_out": str(out),
            "score_status": SCORE_STATUS,
            "threshold_status": "no_calibrated_threshold",
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
