from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import CLIPModel, CLIPProcessor


def choose_device(requested: str) -> str:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return requested


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def as_list(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x if v is not None]
    return [str(x)]


def dedupe_keep_order(xs: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in xs:
        x = " ".join(str(x).strip().split())
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def stable_hash(value: Any, n: int = 12) -> str:
    s = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def percentile_rank_higher_is_better(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=np.float64)
    if len(values) == 1:
        ranks[order] = 1.0
    else:
        ranks[order] = np.linspace(0.0, 1.0, len(values))
    return ranks.astype("float32")


def get_nested_template(bank: dict[str, Any], purpose: str, space: str, season: str) -> dict[str, list[str]] | None:
    root = bank.get("purpose_space_season_templates", {})
    try:
        node = root[purpose][space][season]
    except KeyError:
        return None
    return {
        "positive": as_list(node.get("positive")),
        "negative": as_list(node.get("negative")),
    }


def fallback_prompts(bank: dict[str, Any], space: str) -> dict[str, list[str]]:
    fallback = bank.get("fallback_templates", {})
    by_space = fallback.get("by_space_type", {})
    if space in by_space:
        node = by_space[space]
    else:
        node = fallback.get("generic", {})
    return {
        "positive": as_list(node.get("positive")),
        "negative": as_list(node.get("negative")),
    }


def build_prompt_set(bank: dict[str, Any], campaign: dict[str, Any]) -> dict[str, Any]:
    purpose = str(campaign.get("purpose_type") or "").strip()
    space = str(campaign.get("space_type") or "").strip()
    season = str(campaign.get("season") or "").strip()
    mood_tags = as_list(campaign.get("mood_tags"))

    exact = get_nested_template(bank, purpose, space, season)
    source = "exact_purpose_space_season"

    if exact is None:
        exact = fallback_prompts(bank, space)
        source = "fallback_by_space_or_generic"

    positive = []
    negative = []

    positive.extend(exact["positive"])
    negative.extend(exact["negative"])

    positive.extend(as_list(bank.get("global_positive_modifiers", {}).get("sayuwon_brand")))
    negative.extend(as_list(bank.get("global_negative_prompts")))

    mood_modifiers = bank.get("mood_prompt_modifiers", {})
    for mood in mood_tags:
        node = mood_modifiers.get(mood, {})
        positive.extend(as_list(node.get("positive")))
        negative.extend(as_list(node.get("negative")))

    positive = dedupe_keep_order(positive)
    negative = dedupe_keep_order(negative)

    if not positive:
        raise RuntimeError("positive prompt set is empty")
    if not negative:
        raise RuntimeError("negative prompt set is empty")

    prompt_set = {
        "campaign_id": campaign["campaign_id"],
        "campaign_version": campaign.get("campaign_version"),
        "prompt_bank_version": bank["prompt_bank_version"],
        "prompt_source": source,
        "purpose_type": purpose,
        "space_type": space,
        "season": season,
        "mood_tags": mood_tags,
        "clip_positive_prompts": positive,
        "clip_negative_prompts": negative,
    }
    prompt_set["prompt_set_hash"] = stable_hash(prompt_set)
    return prompt_set


def extract_text_features(
    model: CLIPModel,
    processor: CLIPProcessor,
    prompts: list[str],
    device: str,
) -> np.ndarray:
    inputs = processor(text=prompts, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.get_text_features(**inputs)

    if isinstance(out, torch.Tensor):
        feats = out
    elif hasattr(out, "text_embeds") and out.text_embeds is not None:
        feats = out.text_embeds
    elif hasattr(out, "pooler_output") and out.pooler_output is not None:
        feats = out.pooler_output
    else:
        raise TypeError(f"Unsupported CLIP text feature output type: {type(out)}")

    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.detach().cpu().numpy().astype("float32")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--prompt-bank", default="configs/prompt_template_bank_v1.yaml")
    ap.add_argument("--clip-embeddings", default="data/embeddings/clip_image_embeddings.npy")
    ap.add_argument("--clip-index", default="data/embeddings/clip_image_index.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-set-out", required=True)
    ap.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--retrieval-version", default="clip_retrieval_v1")
    args = ap.parse_args()

    campaign = read_json(Path(args.campaign))
    bank = yaml.safe_load(Path(args.prompt_bank).read_text(encoding="utf-8"))
    if bank is None:
        raise RuntimeError(f"empty prompt bank: {args.prompt_bank}")

    prompt_set = build_prompt_set(bank, campaign)

    img_emb = np.load(args.clip_embeddings).astype("float32")
    idx = pd.read_csv(args.clip_index)

    if img_emb.shape[0] != len(idx):
        raise RuntimeError("CLIP image embedding rows != index rows")

    device = choose_device(args.device)
    print(json.dumps({
        "event": "load_clip_text_model",
        "model_name": args.model_name,
        "device": device,
        "campaign_id": campaign["campaign_id"],
        "positive_prompts": len(prompt_set["clip_positive_prompts"]),
        "negative_prompts": len(prompt_set["clip_negative_prompts"]),
    }, ensure_ascii=False))

    processor = CLIPProcessor.from_pretrained(args.model_name)
    model = CLIPModel.from_pretrained(args.model_name)
    model.eval()
    model.to(device)

    pos_text = extract_text_features(model, processor, prompt_set["clip_positive_prompts"], device)
    neg_text = extract_text_features(model, processor, prompt_set["clip_negative_prompts"], device)

    pos_sims = img_emb @ pos_text.T
    neg_sims = img_emb @ neg_text.T

    pos_max = pos_sims.max(axis=1)
    pos_mean = pos_sims.mean(axis=1)
    neg_max = neg_sims.max(axis=1)
    neg_mean = neg_sims.mean(axis=1)
    margin = pos_max - neg_max
    rank_pct = percentile_rank_higher_is_better(pos_max)

    records = []
    retrieval_batch_id = f"{args.retrieval_version}__{campaign['campaign_id']}__{prompt_set['prompt_set_hash']}"

    for i, row in idx.iterrows():
        records.append({
            "retrieval_batch_id": retrieval_batch_id,
            "retrieval_version": args.retrieval_version,
            "campaign_id": campaign["campaign_id"],
            "campaign_version": campaign.get("campaign_version"),
            "prompt_bank_version": prompt_set["prompt_bank_version"],
            "prompt_set_hash": prompt_set["prompt_set_hash"],

            "image_id": row["image_id"],
            "path": row["path"],
            "resolved_path": row["resolved_path"],
            "category": None if pd.isna(row.get("category")) else row.get("category"),
            "source_group": None if pd.isna(row.get("source_group")) else row.get("source_group"),
            "place_name": None if pd.isna(row.get("place_name")) else row.get("place_name"),
            "subject_name": None if pd.isna(row.get("subject_name")) else row.get("subject_name"),

            "clip_positive_max_sim": float(pos_max[i]),
            "clip_positive_mean_sim": float(pos_mean[i]),
            "clip_negative_max_sim": float(neg_max[i]),
            "clip_negative_mean_sim": float(neg_mean[i]),
            "clip_margin": float(margin[i]),
            "clip_rank_percentile": float(rank_pct[i]),

            "score_status": "diagnostic_only",
            "threshold_policy": "no pass/fail threshold; rank candidates by CLIP similarity diagnostics",
        })

    records.sort(key=lambda r: (r["clip_margin"], r["clip_positive_max_sim"]), reverse=True)

    score_distribution = {
        "all_count": int(len(pos_max)),
        "clip_positive_max_sim_min": float(pos_max.min()),
        "clip_positive_max_sim_max": float(pos_max.max()),
        "clip_positive_max_sim_mean": float(pos_max.mean()),
        "clip_positive_max_sim_std": float(pos_max.std()),
        "clip_negative_max_sim_min": float(neg_max.min()),
        "clip_negative_max_sim_max": float(neg_max.max()),
        "clip_negative_max_sim_mean": float(neg_max.mean()),
        "clip_negative_max_sim_std": float(neg_max.std()),
        "clip_margin_min": float(margin.min()),
        "clip_margin_max": float(margin.max()),
        "clip_margin_mean": float(margin.mean()),
        "clip_margin_std": float(margin.std()),
        "score_status": "diagnostic_only",
        "threshold_policy": "distribution audit only; no pass/fail threshold"
    }

    top_k = args.top_k if args.top_k > 0 else len(records)
    records = records[:top_k]

    top_margins = np.asarray([r["clip_margin"] for r in records], dtype=np.float32)
    top_pos = np.asarray([r["clip_positive_max_sim"] for r in records], dtype=np.float32)

    score_distribution.update({
        "top_k": int(len(records)),
        "top_k_clip_margin_min": float(top_margins.min()) if len(top_margins) else None,
        "top_k_clip_margin_max": float(top_margins.max()) if len(top_margins) else None,
        "top_k_clip_margin_mean": float(top_margins.mean()) if len(top_margins) else None,
        "top_k_clip_margin_std": float(top_margins.std()) if len(top_margins) else None,
        "top_k_clip_positive_max_sim_min": float(top_pos.min()) if len(top_pos) else None,
        "top_k_clip_positive_max_sim_max": float(top_pos.max()) if len(top_pos) else None,
        "top_k_clip_positive_max_sim_mean": float(top_pos.mean()) if len(top_pos) else None,
        "top_k_clip_positive_max_sim_std": float(top_pos.std()) if len(top_pos) else None,
    })

    out_path = Path(args.out)
    prompt_out_path = Path(args.prompt_set_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_out_path.parent.mkdir(parents=True, exist_ok=True)

    with prompt_out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(prompt_set, ensure_ascii=False, sort_keys=True) + "\n")

    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

    print(json.dumps({
        "event": "done",
        "campaign_id": campaign["campaign_id"],
        "retrieval_batch_id": retrieval_batch_id,
        "prompt_set_out": str(prompt_out_path),
        "out": str(out_path),
        "rows": len(records),
        "total_images_scored": int(img_emb.shape[0]),
        "model_name": args.model_name,
        "device": device,
        "score_status": "diagnostic_only",
        "score_distribution": score_distribution,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
