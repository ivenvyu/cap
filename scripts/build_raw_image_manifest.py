from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def norm_text(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def stable_id(prefix: str, value: str, n: int = 10) -> str:
    h = hashlib.sha1(value.encode("utf-8")).hexdigest()[:n]
    return f"{prefix}_{h}"


def infer_fields(raw_root: Path, image_path: Path) -> dict[str, Any]:
    rel = image_path.relative_to(raw_root)
    parts = [norm_text(p) for p in rel.parts]

    top_category = parts[0] if len(parts) >= 1 else None
    filename = parts[-1]
    stem = norm_text(Path(filename).stem)
    ext = Path(filename).suffix.lower()

    source_group = None
    place_name = None
    subject_name = None

    if top_category == "gallery":
        # gallery/건축/명정/space10_1_1.jpg
        # gallery/정원/풍설기천년/space2_1_1.jpg
        # gallery/치허문.jpg
        if len(parts) >= 4:
            source_group = parts[1]
            place_name = parts[2]
        elif len(parts) == 2:
            source_group = "gallery_root"
            place_name = stem
        else:
            source_group = parts[1] if len(parts) >= 2 else "gallery_unknown"
            place_name = stem

    elif top_category == "arbor":
        # arbor/tree/모과나무/nature1_1.jpg
        source_group = parts[1] if len(parts) >= 3 else None
        subject_name = parts[2] if len(parts) >= 4 else None

    elif top_category == "course":
        # course/고송길.jpg
        source_group = "course"
        place_name = stem

    return {
        "relative_path": norm_text(rel.as_posix()),
        "filename": filename,
        "stem": stem,
        "extension": ext,
        "top_category": top_category,
        "source_group": source_group,
        "place_name": place_name,
        "subject_name": subject_name,
        "path_parts": parts,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest-version", default="raw_image_manifest_v2_2_1")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_root.exists():
        raise FileNotFoundError(raw_root)

    rows: list[dict[str, Any]] = []

    for p in sorted(raw_root.rglob("*")):
        if not p.is_file():
            continue

        ext = p.suffix.lower()
        if ext not in IMAGE_EXTENSIONS:
            continue

        fields = infer_fields(raw_root, p)
        rel_path = fields["relative_path"]

        image_id = stable_id("raw", rel_path)

        row = {
            "image_id": image_id,
            "manifest_version": args.manifest_version,
            "path": rel_path,
            "raw_root": str(raw_root),
            "resolved_path": str(p.resolve()),
            "category": fields["top_category"],
            "source_group": fields["source_group"],
            "place_name": fields["place_name"],
            "subject_name": fields["subject_name"],
            "filename": fields["filename"],
            "stem": fields["stem"],
            "extension": fields["extension"],
            "path_parts": fields["path_parts"],
            "metadata": {
                "metadata_status": "machine_suggested",
                "source": "filesystem_manifest_v1",
                "spec_version": "v2.2.1",
            },
        }

        rows.append(row)

    seen = set()
    duplicate_ids = []
    for r in rows:
        if r["image_id"] in seen:
            duplicate_ids.append(r["image_id"])
        seen.add(r["image_id"])

    if duplicate_ids:
        raise RuntimeError(f"duplicate image_id detected: {duplicate_ids[:10]}")

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

    print(json.dumps({
        "raw_root": str(raw_root),
        "out": str(out_path),
        "rows": len(rows),
        "categories": sorted(set(r["category"] for r in rows)),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
