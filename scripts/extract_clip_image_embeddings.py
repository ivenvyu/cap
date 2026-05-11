from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


ImageFile.LOAD_TRUNCATED_IMAGES = True


def choose_device(requested: str) -> str:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return requested


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def batched(xs: list[Any], batch_size: int):
    for i in range(0, len(xs), batch_size):
        yield i, xs[i : i + batch_size]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-embeddings", required=True)
    ap.add_argument("--out-index", required=True)
    ap.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--embedding-version", default="clip_openai_vit_base_patch32_v1")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    out_embeddings = Path(args.out_embeddings)
    out_index = Path(args.out_index)
    out_embeddings.parent.mkdir(parents=True, exist_ok=True)
    out_index.parent.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(manifest_path)
    if not rows:
        raise RuntimeError(f"empty manifest: {manifest_path}")

    device = choose_device(args.device)
    print(json.dumps({
        "event": "load_model",
        "model_name": args.model_name,
        "device": device,
        "rows": len(rows),
        "batch_size": args.batch_size,
    }, ensure_ascii=False))

    processor = CLIPProcessor.from_pretrained(args.model_name)
    model = CLIPModel.from_pretrained(args.model_name)
    model.eval()
    model.to(device)

    all_embeddings: list[np.ndarray] = []
    index_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    with torch.no_grad():
        for start_idx, batch_rows in tqdm(
            batched(rows, args.batch_size),
            total=(len(rows) + args.batch_size - 1) // args.batch_size,
            desc="CLIP image embeddings",
        ):
            images = []
            good_rows = []

            for r in batch_rows:
                path = Path(r["resolved_path"])
                try:
                    images.append(load_image(path))
                    good_rows.append(r)
                except Exception as e:
                    failures.append({
                        "image_id": r.get("image_id", ""),
                        "path": str(path),
                        "error": f"{type(e).__name__}: {e}",
                    })

            if not images:
                continue

            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            feats_out = model.get_image_features(**inputs)

            # transformers version compatibility:
            # - Some versions return a Tensor directly.
            # - Some versions may return a model output object with pooler_output.
            if isinstance(feats_out, torch.Tensor):
                feats = feats_out
            elif hasattr(feats_out, "image_embeds") and feats_out.image_embeds is not None:
                feats = feats_out.image_embeds
            elif hasattr(feats_out, "pooler_output") and feats_out.pooler_output is not None:
                feats = feats_out.pooler_output
            else:
                raise TypeError(f"Unsupported CLIP image feature output type: {type(feats_out)}")

            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats_np = feats.detach().cpu().numpy().astype("float32")

            row_offset = len(all_embeddings)

            for j, (r, emb) in enumerate(zip(good_rows, feats_np)):
                all_embeddings.append(emb)

                index_rows.append({
                    "embedding_row": row_offset + j,
                    "image_id": r["image_id"],
                    "path": r["path"],
                    "resolved_path": r["resolved_path"],
                    "category": r.get("category"),
                    "source_group": r.get("source_group"),
                    "place_name": r.get("place_name"),
                    "subject_name": r.get("subject_name"),
                    "model_name": args.model_name,
                    "embedding_version": args.embedding_version,
                    "embedding_dim": int(emb.shape[0]),
                })

    if failures:
        failure_path = out_index.with_suffix(".failures.jsonl")
        with failure_path.open("w", encoding="utf-8") as f:
            for r in failures:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        raise RuntimeError(f"{len(failures)} image load failures. See {failure_path}")

    emb_matrix = np.vstack(all_embeddings).astype("float32")

    np.save(out_embeddings, emb_matrix)

    with out_index.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "embedding_row",
            "image_id",
            "path",
            "resolved_path",
            "category",
            "source_group",
            "place_name",
            "subject_name",
            "model_name",
            "embedding_version",
            "embedding_dim",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    print(json.dumps({
        "event": "done",
        "manifest": str(manifest_path),
        "out_embeddings": str(out_embeddings),
        "out_index": str(out_index),
        "rows": len(index_rows),
        "embedding_shape": list(emb_matrix.shape),
        "model_name": args.model_name,
        "embedding_version": args.embedding_version,
        "device": device,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
