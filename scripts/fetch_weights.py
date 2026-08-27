"""
fetch_weights.py

Downloads ONLY the Plant Component Detector (PCD) weights from LeafMachine2's
official release archive, without cloning the full LeafMachine2 repo or
installing any of its heavy dependencies (Detectron2, vit-pytorch, etc.).

How this works:
    LeafMachine2's own `leafmachine2/machine/fetch_data.py` downloads a single
    zip from https://leafmachine.org/LM2/release_<VERSION>.zip containing all
    of its ML models (ruler classifier, ACD, PCD, landmarks, segmentation...).
    We download that same zip but only extract the two files we actually need:
        release_<VERSION>/pcd/LeafPriority.pt   (current default PCD, recommended)
        release_<VERSION>/pcd/best.pt           (older PCD, kept as a fallback)

NOTE: This points at the version pinned in LeafMachine2 as of when this
script was written (v-2-3). If the download 404s, check
https://github.com/Gene-Weaver/LeafMachine2/blob/main/leafmachine2/machine/fetch_data.py
for the current VERSION string and update RELEASE_VERSION below.
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path

RELEASE_VERSION = "v-2-3"
RELEASE_NAME = f"release_{RELEASE_VERSION}"
ZIP_URL = f"https://leafmachine.org/LM2/{RELEASE_NAME}.zip"

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = REPO_ROOT / "weights"
TMP_ZIP = REPO_ROOT / f"{RELEASE_NAME}.zip"

# Files we want out of the archive: (path inside zip, destination filename)
WANTED_FILES = [
    (f"{RELEASE_NAME}/pcd/LeafPriority.pt", "LeafPriority.pt"),
    (f"{RELEASE_NAME}/pcd/best.pt", "PLANT_GroupAB_200_best.pt"),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
    )
}


def download_zip():
    print(f"Downloading {ZIP_URL}")
    print("This is LeafMachine2's full model bundle (all detectors), so it can")
    print("take a while and use a fair amount of disk space temporarily.")
    req = urllib.request.Request(url=ZIP_URL, headers=HEADERS)
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        with open(TMP_ZIP, "wb") as f:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded/1e6:8.1f} MB / {total/1e6:8.1f} MB ({pct:5.1f}%)",
                          end="", flush=True)
    print("\nDownload complete.")


def extract_wanted_files():
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(TMP_ZIP, "r") as zf:
        names = zf.namelist()
        found_any = False
        for inner_path, out_name in WANTED_FILES:
            if inner_path in names:
                print(f"Extracting {inner_path} -> weights/{out_name}")
                with zf.open(inner_path) as src, open(WEIGHTS_DIR / out_name, "wb") as dst:
                    dst.write(src.read())
                found_any = True
            else:
                print(f"  (not found in archive: {inner_path})")
        if not found_any:
            print("\nNone of the expected files were found in the archive.")
            print("The archive layout may have changed. Inspect it with:")
            print(f"  python -c \"import zipfile; print('\\n'.join(zipfile.ZipFile('{TMP_ZIP.name}').namelist()))\"")
            sys.exit(1)


def cleanup():
    if TMP_ZIP.exists():
        print(f"Removing temporary archive {TMP_ZIP.name}")
        TMP_ZIP.unlink()


if __name__ == "__main__":
    if (WEIGHTS_DIR / "LeafPriority.pt").exists():
        print("weights/LeafPriority.pt already exists. Delete it first if you want to re-download.")
        sys.exit(0)

    try:
        download_zip()
        extract_wanted_files()
    finally:
        cleanup()

    print("\nDone. Weights are in ./weights/")
    print("Use weights/LeafPriority.pt with src/detect_flowers.py")
