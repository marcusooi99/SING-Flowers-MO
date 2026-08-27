# Herbarium Flower Detector

A minimal tool that detects whether a herbarium specimen sheet contains
flowers, and draws bounding boxes around them.

It does this by re-using the pretrained **Plant Component Detector (PCD)**
from [LeafMachine2](https://github.com/Gene-Weaver/LeafMachine2)
(Weaver & Smith, 2023, *Applications in Plant Sciences*) — a YOLOv5x6 object
detector trained on 494,766+ manually reviewed annotations across 5,648+
herbarium images from 288 institutions. Rather than installing all of
LeafMachine2 (which also does leaf segmentation, landmarking, ruler OCR, and
label transcription, and pulls in Detectron2 + a ViT-based label binarizer
as dependencies), this repo extracts just the one YOLOv5 checkpoint
responsible for locating flowers and runs it standalone.

## Approach & Methodology

1. **Detector**: LeafMachine2's PCD is a single YOLOv5x6 model trained to
   locate 11 classes on a full specimen sheet: `leaf_whole`, `leaf_partial`,
   `leaflet`, `seed_fruit_one`, `seed_fruit_many`, `flower_one`,
   `flower_many`, `bud`, `specimen`, `roots`, `wood`.
2. **Binary decision rule**: a specimen is called *flowering* if at least
   one `flower_one` or `flower_many` detection clears the confidence
   threshold (default 0.2, matching LeafMachine2's own default for
   phenology scoring — see `DetectPhenology.yaml` in the source repo).
3. **Localization**: the same detection gives us the bounding box(es) for
   free — no separate model needed.
4. **No training required** to get a working baseline. Finetuning is only
   needed if zero-shot accuracy on your own images turns out to be poor
   (see Limitations).

This intentionally does *not* use LeafMachine2's Detectron2-based leaf
segmentation, ruler/OCR pipeline, or landmarking — none of those are
relevant to a flower-presence classifier, and skipping them removes the
majority of the dependency and VRAM footprint.

## Tools, Models & Libraries

- **Model**: LeafMachine2 Plant Component Detector, `LeafPriority.pt`
  (YOLOv5x6, ~[see file size after download]MB), by Weaver & Smith (2023),
  University of Michigan. GPL-3.0 licensed.
- **Inference**: PyTorch + `torch.hub.load('ultralytics/yolov5', ...)`
- **Image I/O / drawing**: OpenCV
- **Everything else**: standard scientific Python (pandas, numpy)

## Setup

Requires **Python 3.10** (3.9–3.11 should also work; this is the range
validated against `torch>=2.0` and the `ultralytics/yolov5` hub code this
tool depends on). Pick either option below — you only need one.

### Option 1: venv

```bash
git clone <your-repo-url>
cd herbarium-flower-detector

# Use a Python 3.10 interpreter to create the venv. If `python3.10` isn't
# found, install it first (e.g. via pyenv, or your OS package manager) or
# substitute the closest available 3.9-3.11 interpreter on your system.
python3.10 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option 2: conda

```bash
git clone <your-repo-url>
cd herbarium-flower-detector

conda create -n herbarium-flower-detector python=3.10 -y
conda activate herbarium-flower-detector

pip install -r requirements.txt
```

### Then, either way: fetch the model weights

```bash
# Downloads only the PCD weights (~100-300MB) from LeafMachine2's official
# release archive, not the full LeafMachine2 repo.
python scripts/fetch_weights.py
```

If `fetch_weights.py` fails with a 404, LeafMachine2 has likely shipped a
newer release. Check the `VERSION` variable in
[`fetch_data.py`](https://github.com/Gene-Weaver/LeafMachine2/blob/main/leafmachine2/machine/fetch_data.py)
in the upstream repo and update `RELEASE_VERSION` in `scripts/fetch_weights.py`
to match.

## Usage

```bash
python src/detect_flowers.py \
    --input sample_images \
    --output sample_outputs \
    --weights weights/LeafPriority.pt \
    --conf 0.2 \
    --imgsz 1280 \
    --device cpu   # or cuda:0 if you have a GPU
```

Output:
- `sample_outputs/results.csv` — one row per image: `has_flower` (0/1),
  flower box counts, max confidence, raw box coordinates.
- `sample_outputs/annotated/*.jpg` — each input image with red boxes drawn
  around detected flowers.

## Diagnostic Test Set

Because this tool reuses a pretrained model rather than training on our own
data, evaluation images were deliberately chosen to minimize overlap with
LeafMachine2's training distribution:

- Specimens sourced from herbaria/digitization batches added to
  GBIF/iDigBio after LeafMachine2's last documented model update, to
  reduce the chance they were already seen during training.
- A mix of institutions/regions less likely to be among LeafMachine2's
  288 training-source herbaria (which skew North American).
- A few images from common, heavily-digitized North American herbaria
  included deliberately as an in-distribution "positive control", so
  in-distribution vs. out-of-distribution accuracy can be compared
  side-by-side rather than assumed.

See the presentation for the full sourcing rationale and citation list.

## Limitations

- **Domain shift**: the detector was trained on a specific mix of
  institutional imaging conventions (color cards, mounting style, label
  placement). Sheets that look meaningfully different may see reduced
  accuracy — this is exactly what the diagnostic test set above is
  designed to surface.
- **Flower vs. bud vs. spent flower**: the model does not distinguish
  flowering stage; a specimen with only unopened buds may or may not be
  flagged as "flowering" depending on how the detector was trained to
  treat buds vs. flowers.
- **Small/faded/pressed flowers** and heavy occlusion (by tape, labels, or
  overlapping plant material) are known failure modes for YOLO-style
  detectors on herbarium sheets generally.
- **Multi-species sheets** and taxa without conventional "flower"
  morphology (e.g., grasses, sedges) are not well represented in this
  binary framing.
- **No finetuning included** in this baseline. If accuracy on your
  diagnostic set is unsatisfactory, the same `LeafPriority.pt` checkpoint
  can be used as a warm-start for finetuning on a small hand-labeled set
  using the standard YOLOv5/Ultralytics training CLI.

## Citation

Weaver, W. N., and S. A. Smith. 2023. From leaves to labels: Building
modular machine learning networks for rapid herbarium specimen analysis
with LeafMachine2. *Applications in Plant Sciences* 11(5): e11548.
https://doi.org/10.1002/aps3.11548
