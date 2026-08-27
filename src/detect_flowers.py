"""
detect_flowers.py

Minimal flower-presence detector for herbarium specimen sheets.

Uses the Plant Component Detector (PCD) weights from LeafMachine2
(Weaver & Smith, 2023, Applications in Plant Sciences), which is a
YOLOv5x6 object detector trained on 494,766+ annotations across 288
herbaria to locate leaves, flowers, fruit, buds, roots, and wood on
herbarium sheets.

We only care about two of its eleven classes:
    5: flower_one    (a single, isolated flower)
    6: flower_many    (a cluster/inflorescence of flowers)

A specimen is labeled "flowering" if at least one detection of either
class clears the confidence threshold.

Usage:
    python src/detect_flowers.py \
        --input sample_images \
        --output sample_outputs \
        --weights weights/LeafPriority.pt \
        --conf 0.2 \
        --imgsz 1280 \
        --device cpu
"""

import argparse
import csv
import json
import warnings
from pathlib import Path

import cv2
import torch

# yolov5 v7.0 is 2022-era code running against a modern torch/CUDA stack, so
# it triggers a stream of FutureWarning/UserWarning noise (pkg_resources,
# torch.load default, torch.cuda.amp.autocast) that has nothing to do with
# whether detection is working correctly. Silencing them here, not upstream,
# since we don't control that vendored code.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Classes that count toward "has flower". Including `bud` is a methodology
# choice: a bud is an unopened flower, not a flower in bloom. Counting it
# here answers "does this specimen show reproductive/flowering structures",
# a slightly broader question than "is there an open flower visible". Worth
# stating explicitly in your writeup, since it changes what "flowering"
# means. Drop `7: "bud"` below to go back to open-flowers-only.
FLOWER_CLASS_IDS = {5: "flower_one", 6: "flower_many", 7: "bud"}
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser(description="Detect flower presence on herbarium sheets")
    p.add_argument("--input", required=True, help="Folder of herbarium sheet images")
    p.add_argument("--output", required=True, help="Folder to write annotated images + results.csv")
    p.add_argument("--weights", required=True, help="Path to LeafPriority.pt (or other PCD weights)")
    p.add_argument("--conf", type=float, default=0.2, help="Confidence threshold (LM2 default for phenology use)")
    p.add_argument("--imgsz", type=int, default=1280, help="Inference resolution (match training: 1280)")
    p.add_argument("--device", default="", help="'cpu', 'cuda:0', or '' to auto-select")
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

    results_rows = []
    for img_path in image_paths:
        print(f"Processing {img_path.name}...")
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Could not read image, skipping.")
            continue

        results = model(img, size=args.imgsz)
        dets = results.pandas().xyxy[0]
        flower_dets = dets[dets["class"].isin(FLOWER_CLASS_IDS)]

        has_flower = len(flower_dets) > 0
        n_flower_one = len(flower_dets[flower_dets["class"] == 5])
        n_flower_many = len(flower_dets[flower_dets["class"] == 6])
        n_bud = len(flower_dets[flower_dets["class"] == 7])
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
                flower_dets[["xmin", "ymin", "xmax", "ymax", "confidence", "name"]]
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