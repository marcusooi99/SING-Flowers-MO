# Herbarium Sheet Flower Detector

A minimal working tool that detects whether a herbarium sheet image contains
open flowers and draws bounding boxes around them.

It reuses the pretrained **Plant Component Detector (PCD)** from
[LeafMachine2](https://github.com/Gene-Weaver/LeafMachine2) (Weaver & Smith,
2023) — a YOLOv5x6 detector trained on ~494k reviewed annotations across 288
herbaria — running just that one checkpoint via `torch.hub`, without installing
LeafMachine2 or its heavy dependency stack (Detectron2, vit-pytorch,
pycocotools).

## Setup

Requires **Python 3.10** (3.9–3.11 also work). Use either venv or conda.

```bash
cd <your-working-directory>
git clone <repo-url>
cd SING-Flowers-MO
```

Run every command below — `pip install`, `fetch_weights.py`, `detect_flowers.py` —
from this repo root.

```bash
# Option A — venv
python3.10 -m venv SING-Flowers-MO
source SING-Flowers-MO/bin/activate      # Windows: SING-Flowers-MO\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

```bash
# Option B — conda
conda create -n SING-Flowers-MO python=3.10 -y
conda activate SING-Flowers-MO
pip install -r requirements.txt
```

Then fetch the model weights (~100–300 MB, needs internet once; skips if already
present):

```bash
python scripts/fetch_weights.py
```

## Usage

From the repo root. `--input` and `--output` are the only required arguments:

```bash
python src/detect_flowers.py --input images/demo_images --output outputs/demo_outputs
```

Optional flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `--weights` | `weights/LeafPriority.pt` | Path to the PCD checkpoint. |
| `--conf` | `0.1` | Detection confidence threshold. |
| `--imgsz` | `1280` | Inference resolution (the PCD was trained at 1280). |
| `--include-buds` | off | Also count `bud` detections toward the flowering decision and draw them. By default only open flowers (`flower_one`, `flower_many`) count — LM2's `bud` class mixes flower buds with vegetative buds. `n_bud` and `all_flower_boxes` in the CSV record buds either way. |
| `--no-annotate` | off | Skip writing annotated images (CSV only, faster). |
| `--device` | auto | `cpu`, `cuda:0` (NVIDIA GPU), or `mps` (Apple Silicon). |

Output:

- `demo_outputs/results.csv` — one row per image: `has_flower` (0/1), per-class
  counts (`n_flower_one`, `n_flower_many`, `n_bud`), max confidence, and
  `all_flower_boxes` (every flower/bud box as JSON). The per-class counts and
  `all_flower_boxes` always cover classes 5/6/7, even without `--include-buds`.
- `demo_outputs/annotated/*.jpg` — input images with the counted flower boxes
  drawn (buds included only under `--include-buds`).

The first run clones the yolov5 v7.0 codebase into `~/.cache/torch/hub` (needs
internet once); afterwards it runs offline.

## Citation

Weaver, W. N., and S. A. Smith. 2023. From leaves to labels: Building modular
machine learning networks for rapid herbarium specimen analysis with
LeafMachine2. *Applications in Plant Sciences* 11(5): e11548.
https://doi.org/10.1002/aps3.11548

The PCD checkpoint is distributed by LeafMachine2 under GPL-3.0.
