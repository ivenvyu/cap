from __future__ import annotations

import argparse
import csv
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps


PREVIEW_VERSION = "layout_treatment_preview_v1"

VARIANTS = [
    "full_bleed_plain",
    "full_bleed_dark_scrim",
    "full_bleed_light_scrim",
    "local_gradient_or_scrim",
    "split_image_text_panel",
    "text_color_alternative",
]

TREATMENT_BY_VARIANT = {
    "full_bleed_plain": "use_as_is",
    "full_bleed_dark_scrim": "needs_dark_scrim",
    "full_bleed_light_scrim": "needs_light_scrim",
    "local_gradient_or_scrim": "needs_gradient_scrim",
    "split_image_text_panel": "needs_split_panel",
    "text_color_alternative": "needs_text_color_swap",
}


def safe_id(value: str) -> str:
    out = []
    for ch in str(value):
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def fit_image_to_canvas(img: Image.Image, size: tuple[int, int]) -> tuple[Image.Image, tuple[int, int, int, int]]:
    canvas_w, canvas_h = size
    fitted = ImageOps.contain(img, size)
    x0 = (canvas_w - fitted.width) // 2
    y0 = (canvas_h - fitted.height) // 2
    canvas = Image.new("RGB", size, "white")
    canvas.paste(fitted, (x0, y0))
    return canvas, (x0, y0, x0 + fitted.width, y0 + fitted.height)


def layout_regions(content_box: tuple[int, int, int, int]) -> dict[str, tuple[int, int, int, int]]:
    # layout_top_left_bottom_left:
    # title = top-left half cell
    # info  = bottom-left half cell
    x0, y0, x1, y1 = content_box
    w = x1 - x0
    h = y1 - y0

    mid_x = x0 + w // 2
    mid_y = y0 + h // 2

    return {
        "title": (x0, y0, mid_x, mid_y),
        "info": (x0, mid_y, mid_x, y1),
    }


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: str) -> None:
    font = ImageFont.load_default()
    draw.text(xy, text, fill=fill, font=font)


def draw_layout_boxes(img: Image.Image, content_box: tuple[int, int, int, int], text_fill: str = "white") -> Image.Image:
    out = img.convert("RGB")
    draw = ImageDraw.Draw(out)
    regions = layout_regions(content_box)

    for name, rect in regions.items():
        draw.rectangle(rect, outline=text_fill, width=3)
        draw_label(draw, (rect[0] + 8, rect[1] + 8), name.upper(), text_fill)

    return out


def overlay_rectangles(
    img: Image.Image,
    rects: list[tuple[int, int, int, int]],
    fill: tuple[int, int, int, int],
) -> Image.Image:
    out = img.convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for rect in rects:
        draw.rectangle(rect, fill=fill)
    return Image.alpha_composite(out, overlay).convert("RGB")


def render_variant(
    original: Image.Image,
    variant: str,
    panel_size: tuple[int, int],
) -> Image.Image:
    img, content_box = fit_image_to_canvas(original, panel_size)
    regions = layout_regions(content_box)

    if variant == "full_bleed_plain":
        return draw_layout_boxes(img, content_box, text_fill="white")

    if variant == "full_bleed_dark_scrim":
        dark = ImageEnhance.Brightness(img).enhance(0.55)
        return draw_layout_boxes(dark, content_box, text_fill="white")

    if variant == "full_bleed_light_scrim":
        light = Image.blend(img, Image.new("RGB", img.size, "white"), 0.45)
        return draw_layout_boxes(light, content_box, text_fill="black")

    if variant == "local_gradient_or_scrim":
        treated = overlay_rectangles(
            img,
            [regions["title"], regions["info"]],
            fill=(0, 0, 0, 115),
        )
        return draw_layout_boxes(treated, content_box, text_fill="white")

    if variant == "split_image_text_panel":
        canvas_w, canvas_h = panel_size
        out = Image.new("RGB", panel_size, "white")

        left_w = canvas_w // 2
        left_img = ImageOps.fit(original, (left_w, canvas_h))
        out.paste(left_img, (0, 0))

        draw = ImageDraw.Draw(out)
        draw.line((left_w, 0, left_w, canvas_h), fill="black", width=2)
        draw.rectangle((left_w + 18, 24, canvas_w - 18, 92), outline="black", width=3)
        draw.rectangle((left_w + 18, canvas_h - 110, canvas_w - 18, canvas_h - 24), outline="black", width=3)
        draw_label(draw, (left_w + 28, 34), "TITLE PANEL", "black")
        draw_label(draw, (left_w + 28, canvas_h - 100), "INFO PANEL", "black")
        return out

    if variant == "text_color_alternative":
        out = img.copy()
        draw = ImageDraw.Draw(out)
        title = regions["title"]
        info = regions["info"]

        draw.rectangle(title, outline="white", width=3)
        draw_label(draw, (title[0] + 8, title[1] + 8), "WHITE TEXT OPTION", "white")

        draw.rectangle(info, outline="black", width=3)
        draw_label(draw, (info[0] + 8, info[1] + 8), "BLACK TEXT OPTION", "black")
        return out

    raise RuntimeError(f"unknown variant: {variant}")


def add_variant_header(img: Image.Image, variant: str) -> Image.Image:
    header_h = 34
    out = Image.new("RGB", (img.width, img.height + header_h), "white")
    out.paste(img, (0, header_h))

    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, out.width, header_h), fill="white")
    draw_label(draw, (8, 10), f"{variant} → {TREATMENT_BY_VARIANT[variant]}", "black")
    return out


def make_composite(
    original: Image.Image,
    panel_size: tuple[int, int],
    columns: int,
) -> Image.Image:
    panels = []
    for variant in VARIANTS:
        panel = render_variant(original, variant, panel_size)
        panels.append(add_variant_header(panel, variant))

    rows = (len(panels) + columns - 1) // columns
    panel_w, panel_h = panels[0].size

    composite = Image.new("RGB", (columns * panel_w, rows * panel_h), "white")

    for i, panel in enumerate(panels):
        x = (i % columns) * panel_w
        y = (i // columns) * panel_h
        composite.paste(panel, (x, y))

    return composite


def render_html(index_rows: list[dict[str, Any]], out_dir: Path) -> str:
    cards = []

    for row in index_rows:
        rel = Path(row["composite_preview_path"]).name
        cards.append(
            f"""
            <div class="card">
              <img src="{html.escape(rel)}" />
              <div class="meta">
                <div><b>campaign:</b> {html.escape(row["campaign_id"])}</div>
                <div><b>queue_row_id:</b> {html.escape(row["queue_row_id"])}</div>
                <div><b>decision:</b> {html.escape(str(row["campaign_decision"]))} / {html.escape(row["decision_label"])}</div>
                <div><b>bucket:</b> {html.escape(row["source_bucket"])}</div>
                <div><b>image_id:</b> {html.escape(row["image_id"])}</div>
                <div><b>path:</b> {html.escape(row["image_path"])}</div>
                <div class="review">
                  <b>layout_treatment_decision:</b><br/>
                  <b>layout_issue_tags:</b><br/>
                  <b>reviewer_notes:</b>
                </div>
              </div>
            </div>
            """
        )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Phase 1b Layout Treatment Preview v1</title>
<style>
body {{
  font-family: -apple-system, BlinkMacSystemFont, sans-serif;
  margin: 24px;
}}
.summary {{
  padding: 12px;
  background: #f6f6f6;
  border-radius: 10px;
  margin-bottom: 20px;
  line-height: 1.5;
}}
.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(760px, 1fr));
  gap: 20px;
}}
.card {{
  border: 1px solid #ddd;
  border-radius: 12px;
  padding: 12px;
}}
.card img {{
  width: 100%;
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
code {{
  background: #eee;
  padding: 2px 4px;
  border-radius: 4px;
}}
</style>
</head>
<body>
<h1>Phase 1b Layout Treatment Preview v1</h1>
<div class="summary">
  <b>preview_version:</b> {PREVIEW_VERSION}<br/>
  <b>rows:</b> {len(index_rows)}<br/>
  <b>scope:</b> campaign_decision 1/2 only, indoor/winter coverage-gap campaign excluded<br/>
  <b>allowed treatment decisions:</b>
  <code>use_as_is</code>,
  <code>needs_dark_scrim</code>,
  <code>needs_light_scrim</code>,
  <code>needs_gradient_scrim</code>,
  <code>needs_split_panel</code>,
  <code>needs_text_color_swap</code>,
  <code>needs_manual_design</code>,
  <code>unusable_even_with_treatment</code>
</div>
<div class="grid">
{''.join(cards)}
</div>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-dir", default="data/review/phase1b")
    ap.add_argument("--out-dir", default="data/previews/phase1b/layout_treatment_v1")
    ap.add_argument("--exclude-campaign", action="append", default=["phase1b_indoor_gallery_winter_art"])
    ap.add_argument("--include-rejects", action="store_true")
    ap.add_argument("--panel-width", type=int, default=420)
    ap.add_argument("--panel-height", type=int, default=280)
    ap.add_argument("--columns", type=int, default=2)
    args = ap.parse_args()

    review_dir = Path(args.review_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(review_dir.glob("review_queue__*.labeled.csv"))
    if not csv_paths:
        raise RuntimeError(f"no labeled review csv files found in {review_dir}")

    index_rows: list[dict[str, Any]] = []
    exclude_campaigns = set(args.exclude_campaign)

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            campaign_id = str(row["campaign_id"])
            decision = int(row["decision"])

            if campaign_id in exclude_campaigns:
                continue
            if not args.include_rejects and decision == 0:
                continue

            resolved_path = Path(str(row["resolved_path"]))
            if not resolved_path.exists():
                raise RuntimeError(f"missing image: {resolved_path}")

            img = Image.open(resolved_path)
            img = ImageOps.exif_transpose(img).convert("RGB")

            preview_id = f"{PREVIEW_VERSION}__{safe_id(str(row['queue_row_id']))}"
            out_path = out_dir / f"{preview_id}.jpg"

            composite = make_composite(
                img,
                panel_size=(args.panel_width, args.panel_height),
                columns=args.columns,
            )
            composite.save(out_path, quality=92)

            index_rows.append(
                {
                    "preview_id": preview_id,
                    "preview_version": PREVIEW_VERSION,
                    "campaign_id": campaign_id,
                    "queue_row_id": str(row["queue_row_id"]),
                    "campaign_decision": decision,
                    "decision_label": str(row["decision_label"]),
                    "source_bucket": str(row["source_bucket"]),
                    "image_id": str(row["image_id"]),
                    "pair_id": str(row["pair_id"]),
                    "layout_spec_id": str(row["layout_spec_id"]),
                    "feature_snapshot_id": str(row["feature_snapshot_id"]),
                    "image_path": str(row["image_path"]),
                    "resolved_path": str(row["resolved_path"]),
                    "composite_preview_path": str(out_path),
                    "variants": "|".join(VARIANTS),
                    "treatment_options": "|".join(TREATMENT_BY_VARIANT[v] for v in VARIANTS),
                }
            )

    index_csv = out_dir / "layout_treatment_preview_index_v1.csv"
    with index_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "preview_id",
                "preview_version",
                "campaign_id",
                "queue_row_id",
                "campaign_decision",
                "decision_label",
                "source_bucket",
                "image_id",
                "pair_id",
                "layout_spec_id",
                "feature_snapshot_id",
                "image_path",
                "resolved_path",
                "composite_preview_path",
                "variants",
                "treatment_options",
            ],
        )
        writer.writeheader()
        writer.writerows(index_rows)

    index_html = out_dir / "index.html"
    index_html.write_text(render_html(index_rows, out_dir), encoding="utf-8")

    run_summary = {
        "preview_version": PREVIEW_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "score_status": "diagnostic_only",
        "renderer_role": "reviewer_audit_preview",
        "final_design_renderer": False,
        "candidate_support_explanation_status": "deferred_from_phase_1b",
        "rows": len(index_rows),
        "variants": VARIANTS,
        "excluded_campaigns": sorted(exclude_campaigns),
        "include_rejects": bool(args.include_rejects),
        "index_csv": str(index_csv),
        "index_html": str(index_html),
    }

    summary_path = out_dir / "layout_treatment_preview_run_summary_v1.json"
    summary_path.write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(run_summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
