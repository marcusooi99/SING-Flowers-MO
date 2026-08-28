# Herbarium Sheet Flower Detector

A minimal working tool that detects whether a herbarium sheet image contains
flowers (open flowers and buds) and draws bounding boxes around them.

It reuses the pretrained **Plant Component Detector (PCD)** from
[LeafMachine2](https://github.com/Gene-Weaver/LeafMachine2) (Weaver & Smith,
2023) — a YOLOv5x6 detector trained on ~494k reviewed annotations across 288
herbaria — running just that one checkpoint via `torch.hub`, without installing
LeafMachine2 or its heavy dependency stack (Detectron2, vit-pytorch,
pycocotools).

## Setup

Requires **Python 3.10** (3.9–3.11 also work). Use either venv or conda.

```bash
# Option A — venv
python3.10 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

```bash
# Option B — conda
conda create -n flower-detector python=3.10 -y
conda activate flower-detector
pip install -r requirements.txt
```

Then fetch the model weights (~100–300 MB, needs internet once; skips if already
present):

```bash
python scripts/fetch_weights.py
```

## Usage

```bash
python src/detect_flowers.py \
    --input sample_images \
    --output sample_outputs \
    --weights weights/LeafPriority.pt \
    --conf 0.1 \
    --imgsz 1280 \
    --device cpu          # cuda:0 for an NVIDIA GPU, mps for Apple Silicon
```

Flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `--conf` | `0.1` | Detection confidence threshold. |
| `--no-bud` | off | Exclude `bud` detections from the flowering decision and the annotated images (by default buds count as flowering structures). `n_bud` and `all_flower_boxes` in the CSV still record buds regardless. |
| `--no-annotate` | off | Skip writing annotated images (CSV only, faster). |
| `--device` | auto | `cpu`, `cuda:0`, or `mps`. |

Output:

- `sample_outputs/results.csv` — one row per image: `has_flower` (0/1), per-class
  counts (`n_flower_one`, `n_flower_many`, `n_bud`), max confidence, and
  `all_flower_boxes` (every flower/bud box as JSON). The per-class counts and
  `all_flower_boxes` always cover classes 5/6/7, even under `--no-bud`.
- `sample_outputs/annotated/*.jpg` — input images with the counted flower boxes
  drawn (buds omitted when `--no-bud` is set).

The first run clones the yolov5 v7.0 codebase into `~/.cache/torch/hub` (needs
internet once); afterwards it runs offline.

## Citation

Weaver, W. N., and S. A. Smith. 2023. From leaves to labels: Building modular
machine learning networks for rapid herbarium specimen analysis with
LeafMachine2. *Applications in Plant Sciences* 11(5): e11548.
https://doi.org/10.1002/aps3.11548

The PCD checkpoint is distributed by LeafMachine2 under GPL-3.0.
