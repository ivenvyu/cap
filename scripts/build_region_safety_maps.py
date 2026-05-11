from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True


REGIONS = {
    "full": (0.0, 0.0, 1.0, 1.0),

    "top": (0.0, 0.0, 1.0, 1.0 / 3.0),
    "middle": (0.0, 1.0 / 3.0, 1.0, 2.0 / 3.0),
    "bottom": (0.0, 2.0 / 3.0, 1.0, 1.0),

    "left": (0.0, 0.0, 1.0 / 3.0, 1.0),
    "center": (1.0 / 3.0, 0.0, 2.0 / 3.0, 1.0),
    "right": (2.0 / 3.0, 0.0, 1.0, 1.0),

    "top_left": (0.0, 0.0, 1.0 / 3.0, 1.0 / 3.0),
    "top_center": (1.0 / 3.0, 0.0, 2.0 / 3.0, 1.0 / 3.0),
    "top_right": (2.0 / 3.0, 0.0, 1.0, 1.0 / 3.0),

    "middle_left": (0.0, 1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0),
    "middle_center": (1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0),
    "middle_right": (2.0 / 3.0, 1.0 / 3.0, 1.0, 2.0 / 3.0),

    "bottom_left": (0.0, 2.0 / 3.0, 1.0 / 3.0, 1.0),
    "bottom_center": (1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 1.0),
    "bottom_right": (2.0 / 3.0, 2.0 / 3.0, 1.0, 1.0),
}


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return arr


def rgb_to_luma(rgb: np.ndarray) -> np.ndarray:
    # Rec. 709 luma coefficients. This is a standard luminance transform,
    # not a project-specific aesthetic threshold.
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def crop_region(arr: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = box

    ix0 = int(round(x0 * w))
    iy0 = int(round(y0 * h))
    ix1 = int(round(x1 * w))
    iy1 = int(round(y1 * h))

    ix0 = max(0, min(ix0, w - 1))
    iy0 = max(0, min(iy0, h - 1))
    ix1 = max(ix0 + 1, min(ix1, w))
    iy1 = max(iy0 + 1, min(iy1, h))

    return arr[iy0:iy1, ix0:ix1]


def edge_density(luma: np.ndarray) -> float:
    if luma.shape[0] < 2 or luma.shape[1] < 2:
        return 0.0

    gx = np.abs(np.diff(luma, axis=1))
    gy = np.abs(np.diff(luma, axis=0))

    # No hard edge threshold. Mean gradient magnitude is retained as a continuous diagnostic feature.
    return float((gx.mean() + gy.mean()) / 2.0)


def region_features(rgb: np.ndarray) -> dict[str, float]:
    luma = rgb_to_luma(rgb)

    return {
        "brightness_mean": float(luma.mean()),
        "brightness_std": float(luma.std()),
        "contrast_std": float(luma.std()),
        "edge_density": edge_density(luma),
        "saturation_mean": float((rgb.max(axis=2) - rgb.min(axis=2)).mean()),
        "region_pixel_count": int(luma.size),
    }


def percentile_ranks(values: list[float], reverse: bool = False) -> list[float]:
    """
    Empirical percentile rank in [0, 1].
    reverse=True means lower raw values get higher percentile scores.
    """
    arr = np.asarray(values, dtype=np.float64)
    if reverse:
        arr = -arr

    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)

    if len(arr) == 1:
        ranks[order] = 1.0
        return ranks.tolist()

    ranks[order] = np.linspace(0.0, 1.0, len(arr))
    return ranks.tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--region-version", default="region_safety_maps_v1")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(manifest_path)
    if not rows:
        raise RuntimeError(f"empty manifest: {manifest_path}")

    raw_records = []

    for r in tqdm(rows, desc="region features"):
        image_path = Path(r["resolved_path"])
        rgb = load_rgb(image_path)

        h, w = rgb.shape[:2]
        regions = {}

        for region_name, box in REGIONS.items():
            crop = crop_region(rgb, box)
            regions[region_name] = {
                "box_normalized": list(box),
                **region_features(crop),
            }

        raw_records.append({
            "image_id": r["image_id"],
            "region_version": args.region_version,
            "score_status": "diagnostic_only",
            "threshold_policy": "no pass/fail threshold; empirical percentile features only",
            "path": r["path"],
            "resolved_path": r["resolved_path"],
            "category": r.get("category"),
            "source_group": r.get("source_group"),
            "place_name": r.get("place_name"),
            "subject_name": r.get("subject_name"),
            "width": int(w),
            "height": int(h),
            "regions": regions,
        })

    # Add empirical percentile features per region across the current corpus.
    # These are reference percentiles, not final pass/fail thresholds.
    for region_name in REGIONS:
        edge_vals = [rec["regions"][region_name]["edge_density"] for rec in raw_records]
        contrast_vals = [rec["regions"][region_name]["contrast_std"] for rec in raw_records]
        brightness_std_vals = [rec["regions"][region_name]["brightness_std"] for rec in raw_records]

        low_edge_pct = percentile_ranks(edge_vals, reverse=True)
        low_contrast_pct = percentile_ranks(contrast_vals, reverse=True)
        low_texture_pct = percentile_ranks(brightness_std_vals, reverse=True)

        for rec, p_edge, p_contrast, p_texture in zip(
            raw_records,
            low_edge_pct,
            low_contrast_pct,
            low_texture_pct,
        ):
            region = rec["regions"][region_name]
            region["low_edge_density_percentile"] = float(p_edge)
            region["low_contrast_std_percentile"] = float(p_contrast)
            region["low_texture_percentile"] = float(p_texture)

            # Conservative diagnostic score: the weakest of the empirical clarity percentiles.
            # This is not a calibrated threshold and must not be used as final pass/fail.
            region["region_safe_score_diagnostic"] = float(
                min(p_edge, p_contrast, p_texture)
            )

    with out_path.open("w", encoding="utf-8") as f:
        for rec in raw_records:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")

    print(json.dumps({
        "event": "done",
        "manifest": str(manifest_path),
        "out": str(out_path),
        "rows": len(raw_records),
        "regions_per_image": len(REGIONS),
        "region_version": args.region_version,
        "score_status": "diagnostic_only",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
