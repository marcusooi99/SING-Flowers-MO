"""
detect_flowers.py — minimal flower-presence detector for herbarium sheets.

Runs LeafMachine2's Plant Component Detector (PCD), a YOLOv5x6 model, via
torch.hub. By default a sheet is "flowering" if a flower_one (5) or
flower_many (6) detection clears --conf. Pass --include-buds to also count
bud (7) detections (LM2's bud class lumps flower buds with vegetative buds).

Writes sample_outputs/results.csv (one row per image) and annotated JPEGs.
See README.md for setup, CLAUDE.md for methodology and constraints.

Usage:
    python src/detect_flowers.py --input sample_images --output sample_outputs \
        --weights weights/LeafPriority.pt --conf 0.1 --imgsz 1280 --device cpu
"""

import argparse
import csv
import json
import os
import warnings
from pathlib import Path

# A few ops in the YOLOv5x6 / NMS path have no MPS (Apple Silicon GPU)
# implementation. Without this, `--device mps` crashes instead of falling back
# to CPU for those ops. Must be set before torch is imported. Inert no-op on
# Windows/Linux and on cpu/cuda runs (PyTorch only reads it when MPS is active).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import torch

# yolov5 v7.0 is 2022-era code running against a modern torch/CUDA stack, so
# it triggers a stream of FutureWarning/UserWarning noise (pkg_resources,
# torch.load default, torch.cuda.amp.autocast) that has nothing to do with
# whether detection is working correctly. Silencing them here, not upstream,
# since we don't control that vendored code.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# PCD classes treated as "flowering structures". By default only open flowers
# (5, 6) count: LM2's `bud` class (7) lumps flower buds with vegetative buds, so
# counting it inflates "flowering". Pass --include-buds to add class 7. Per-class
# counts and all_flower_boxes always cover 5/6/7 in the CSV, so a run can be
# re-scored with buds either way after the fact.
FLOWER_CLASS_NAMES = {5: "flower_one", 6: "flower_many", 7: "bud"}
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser(description="Detect flower presence on herbarium sheets")
    p.add_argument("--input", required=True, help="Folder of herbarium sheet images")
    p.add_argument("--output", required=True, help="Folder to write annotated images + results.csv")
    p.add_argument("--weights", default="weights/LeafPriority.pt", help="Path to LeafPriority.pt (or other PCD weights)")
    p.add_argument("--conf", type=float, default=0.1, help="Detection confidence threshold (default 0.1)")
    p.add_argument("--imgsz", type=int, default=1280, help="Inference resolution (match training: 1280)")
    p.add_argument("--device", default="", help="'cpu', 'cuda:0', 'mps', or '' to auto-select")
    p.add_argument("--include-buds", action="store_true",
                   help="Also count 'bud' detections toward the flowering decision "
                        "and draw them (default: open flowers only)")
    p.add_argument("--no-annotate", action="store_true", help="Skip saving annotated images (faster, CSV only)")
    return p.parse_args()


def load_model(weights_path, conf, device):
    # Pinned to a stable tag rather than "master": yolov5's master branch has
    # since been refactored to depend on the separate `ultralytics` package
    # for some internals, which changes behavior out from under us over time.
    # v7.0 matches the generation of yolov5 code LeafMachine2 vendored.
    model = torch.hub.load("ultralytics/yolov5:v7.0", "custom", path=weights_path, device=device or None)
    model.conf = conf
    return model


def draw_flower_boxes(image_path, flower_dets, out_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return False

    # Fixed pixel sizes (thickness, font scale) look fine on a ~1500px image
    # but vanish on an 8000px scan and overwhelm a tiny one. Scale relative
    # to image size instead, calibrated so scale=1.0 at ~1500px on the long
    # side (roughly what a modest herbarium sheet scan looks like).
    h, w = img.shape[:2]
    scale = max(h, w) / 1500
    box_thickness = max(round(4 * scale), 2)
    font_scale = max(1.0 * scale, 0.5)
    text_thickness = max(round(2 * scale), 1)

    for _, row in flower_dets.iterrows():
        x1, y1, x2, y2 = int(row.xmin), int(row.ymin), int(row.xmax), int(row.ymax)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), box_thickness)
        label = f"{row['name']} {row.confidence:.2f}"
        cv2.putText(img, label, (x1, max(y1 - int(10 * scale), 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), text_thickness, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return True


def main():
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = output_dir / "annotated"

    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in IMG_EXTENSIONS
    )
    if not image_paths:
        print(f"No images found in {input_dir} (looked for {sorted(IMG_EXTENSIONS)})")
        return

    print(f"Loading model from {args.weights} (this downloads the yolov5 codebase via "
          f"torch.hub on first run, which needs internet access once)...")
    model = load_model(args.weights, args.conf, args.device)

    active_ids = [5, 6, 7] if args.include_buds else [5, 6]
    print(f"Flowering classes: {[FLOWER_CLASS_NAMES[i] for i in active_ids]} "
          f"(conf >= {args.conf})")

    results_rows = []
    for img_path in image_paths:
        print(f"Processing {img_path.name}...")
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Could not read image, skipping.")
            continue

        results = model(img, size=args.imgsz)
        dets = results.pandas().xyxy[0]

        # `flower_dets` drives the decision and the annotations — it honours
        # --include-buds. `all_flower_dets` (always 5/6/7) is only for CSV
        # reporting, so n_bud and all_flower_boxes record buds regardless.
        flower_dets = dets[dets["class"].isin(active_ids)]
        all_flower_dets = dets[dets["class"].isin(FLOWER_CLASS_NAMES)]

        has_flower = len(flower_dets) > 0
        n_flower_one = int((all_flower_dets["class"] == 5).sum())
        n_flower_many = int((all_flower_dets["class"] == 6).sum())
        n_bud = int((all_flower_dets["class"] == 7).sum())
        max_conf = float(flower_dets["confidence"].max()) if has_flower else 0.0

        out_img_path = annotated_dir / f"{img_path.stem}_annotated.jpg"
        if not args.no_annotate:
            draw_flower_boxes(img_path, flower_dets, out_img_path)

        results_rows.append({
            "filename": img_path.name,
            "has_flower": int(has_flower),
            "n_flower_one": n_flower_one,
            "n_flower_many": n_flower_many,
            "n_bud": n_bud,
            "max_flower_confidence": round(max_conf, 4),
            "all_flower_boxes": json.dumps(
                all_flower_dets[["xmin", "ymin", "xmax", "ymax", "confidence", "name"]]
                .round(2).to_dict(orient="records")
            ),
            "annotated_image": str(out_img_path) if not args.no_annotate else "",
        })

        status = "FLOWERING" if has_flower else "not flowering"
        print(f"  -> {status} ({len(flower_dets)} flower box(es), max conf {max_conf:.2f})")

    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results_rows[0].keys()))
        writer.writeheader()
        writer.writerows(results_rows)

    n_flowering = sum(r["has_flower"] for r in results_rows)
    print(f"\nDone. {n_flowering}/{len(results_rows)} images classified as flowering.")
    print(f"Results written to {csv_path}")
    if not args.no_annotate:
        print(f"Annotated images written to {annotated_dir}")


if __name__ == "__main__":
    main()