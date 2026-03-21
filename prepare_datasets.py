#!/usr/bin/env python3
"""
prepare_dataset.py
-------------------
Downloads and prepares training, validation, and test data for
WindowSeat reflection removal training.

What this script does
----------------------
1. Downloads SIR2 dataset (real paired images) and splits into train/val/test
2. Downloads PASCAL VOC 2012 (used as source for synthetic pairs)
3. Generates synthetic (blended, clean) pairs by alpha-compositing two images
4. Saves everything under data/ in the structure expected by train_windowseat.py

Final structure
---------------
data/
  train/
    synthetic/
      blended/   <- synthetically reflected images
      clean/     <- ground truth clean images
    real/
      blended/   <- real reflected images (SIR2 train split)
      clean/     <- real clean images
  val/
    real/
      blended/   <- SIR2 val split
      clean/
    synthetic/
      blended/
      clean/
  test/          <- Never touched during training
    sir2_500/    <- Downloaded by windowseat_reproducibility.py
    nature/
    real20/

Usage
-----
    python prepare_dataset.py --output-dir data/
    python prepare_dataset.py --output-dir data/ --n-synthetic 8000
"""

import argparse
import os
import random
import shutil
import zipfile
from pathlib import Path

import cv2
import gdown
import numpy as np
import requests
from PIL import Image
from tqdm import tqdm


# ── Dataset links  ────────────────────────────────────────────────────────────

SIR2_URL    = "https://www.dropbox.com/scl/fi/qgg1whla1jb3a9cgis18l/SIR2.zip?rlkey=kmhrc2uk63be2s9hzr43gc3hm&e=2&st=cfsh8sol&dl=1"
PASCAL_URL  = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"

# SIR2 split ratios (of training data — test is the fixed SIR2-500 benchmark)
VAL_RATIO   = 0.15

ARGS = None


# ── Helpers  ──────────────────────────────────────────────────────────────────

def download_file(url: str, dest: str, desc: str = ""):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return
    print(f"  Downloading {desc} ...")
    r = requests.get(url, stream=True)
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                      desc=desc) as bar:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            bar.update(len(chunk))


def unzip(zip_path: str, out_dir: str):
    if os.path.exists(out_dir) and os.listdir(out_dir):
        print(f"  Already extracted: {out_dir}")
        return
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def copy_pairs(blended_srcs, clean_srcs, blended_dst, clean_dst):
    os.makedirs(blended_dst, exist_ok=True)
    os.makedirs(clean_dst,   exist_ok=True)
    for b, c in zip(blended_srcs, clean_srcs):
        shutil.copy2(b, os.path.join(blended_dst, os.path.basename(b)))
        shutil.copy2(c, os.path.join(clean_dst,   os.path.basename(c)))


# ── SIR2 preparation  ─────────────────────────────────────────────────────────

def prepare_sir2(data_root: str):
    """
    Downloads SIR2 and splits SolidObject + Postcard + Wildscene into
    train/val subsets.

    SIR2 structure after extraction:
        SIR2/
          SolidObjectDataset/  -> m.jpg (blended), g.jpg (clean)
          Postcard Dataset/    -> *-m-*.png (blended), *-g-*.png (clean)
          Wildscene/           -> m.jpg (blended), g.jpg (clean)

    The fixed SIR2-500 benchmark (used for testing) is downloaded separately
    by windowseat_reproducibility.py and lives in data/test/sir2_500/.
    """
    raw_dir = os.path.join(data_root, "raw", "sir2")
    zip_path = os.path.join(raw_dir, "SIR2.zip")

    download_file(SIR2_URL, zip_path, "SIR2 dataset")
    unzip(zip_path, os.path.join(raw_dir, "extracted"))

    # Collect all (blended, clean) pairs across subsets
    all_pairs = []
    sir2_root = os.path.join(raw_dir, "extracted", "SIR2")

    # SolidObjectDataset: pairs named m.jpg / g.jpg inside numbered dirs
    solid_root = os.path.join(sir2_root, "SolidObjectDataset", "SolidObjectDataset")
    if os.path.isdir(solid_root):
        for root, _, files in os.walk(solid_root):
            if "m.jpg" in files and "g.jpg" in files:
                rel   = os.path.relpath(root, solid_root).replace(os.sep, "_")
                all_pairs.append((
                    os.path.join(root, "m.jpg"),
                    os.path.join(root, "g.jpg"),
                    f"solid_{rel}",
                ))

    # Postcard: pairs named *-m-*.png / *-g-*.png
    postcard_root = os.path.join(sir2_root, "Postcard Dataset", "Postcard Dataset")
    if os.path.isdir(postcard_root):
        for root, _, files in os.walk(postcard_root):
            m_files = [f for f in files if "-m-" in f and f.endswith(".png")]
            for mf in m_files:
                gf = mf.replace("-m-", "-g-")
                if os.path.exists(os.path.join(root, gf)):
                    rel = os.path.relpath(root, postcard_root).replace(os.sep, "_")
                    all_pairs.append((
                        os.path.join(root, mf),
                        os.path.join(root, gf),
                        f"postcard_{rel}_{mf[:-4]}",
                    ))

    # Wildscene: pairs named m.jpg / g.jpg
    wild_root = os.path.join(sir2_root, "Wildscene")
    if os.path.isdir(wild_root):
        for root, _, files in os.walk(wild_root):
            if "m.jpg" in files and "g.jpg" in files:
                rel = os.path.relpath(root, wild_root).replace(os.sep, "_")
                all_pairs.append((
                    os.path.join(root, "m.jpg"),
                    os.path.join(root, "g.jpg"),
                    f"wild_{rel}",
                ))

    print(f"SIR2: {len(all_pairs)} pairs total")

    random.shuffle(all_pairs)
    n_val   = max(1, int(len(all_pairs) * VAL_RATIO))
    val_p   = all_pairs[:n_val]
    train_p = all_pairs[n_val:]

    print(f"  train={len(train_p)}  val={len(n_val)}")

    for split, pairs in [("train", train_p), ("val", val_p)]:
        blended_dst = os.path.join(data_root, split, "real", "blended")
        clean_dst   = os.path.join(data_root, split, "real", "clean")
        os.makedirs(blended_dst, exist_ok=True)
        os.makedirs(clean_dst,   exist_ok=True)

        for src_b, src_c, stem in tqdm(pairs, desc=f"Copying SIR2 {split}"):
            ext = os.path.splitext(src_b)[1]
            shutil.copy2(src_b, os.path.join(blended_dst, f"{stem}{ext}"))
            shutil.copy2(src_c, os.path.join(clean_dst,   f"{stem}{ext}"))


# ── Synthetic pair generation  ────────────────────────────────────────────────

def blend_images(trans: np.ndarray, refl: np.ndarray) -> np.ndarray:
    """
    Synthetic reflection blending:
      I = alpha * T + (1-alpha) * blur(R)
    alpha ~ U[0.7, 0.95]
    """
    alpha = random.uniform(0.7, 0.95)
    if random.random() < 0.7:
        ksize = random.choice([3, 5, 7])
        refl  = cv2.GaussianBlur(refl, (ksize, ksize), 0)
    blended = alpha * trans.astype(np.float32) + \
              (1.0 - alpha) * refl.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def generate_synthetic_pairs(data_root: str, n_train: int, n_val: int):
    """
    Generate synthetic (blended, clean) pairs from PASCAL VOC 2012 images.

    PASCAL VOC is used as the source of natural images. We randomly select
    two images: one becomes the transmission (clean) and one becomes the
    reflection. Blending simulates looking through glass.

    If PASCAL is not available, falls back to using any images found under
    data/raw/natural_images/.
    """
    # Try to find PASCAL VOC images
    pascal_dir = os.path.join(data_root, "raw", "pascal_voc",
                              "VOCdevkit", "VOC2012", "JPEGImages")
    if not os.path.isdir(pascal_dir):
        print("PASCAL VOC not found. Trying to download...")
        print("NOTE: PASCAL VOC is large (~2GB). You can also manually place")
        print("      images in data/raw/natural_images/ as an alternative.")
        # Try alternate location
        alt = os.path.join(data_root, "raw", "natural_images")
        if os.path.isdir(alt):
            pascal_dir = alt
        else:
            print("No image source found. Skipping synthetic generation.")
            print("To use PASCAL VOC, download from:")
            print("  http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar")
            print("and extract to data/raw/pascal_voc/")
            return

    all_imgs = sorted([
        os.path.join(pascal_dir, f)
        for f in os.listdir(pascal_dir)
        if f.lower().endswith((".jpg", ".png"))
    ])
    print(f"Found {len(all_imgs)} source images for synthetic generation")

    if len(all_imgs) < 2:
        print("Not enough images for synthetic generation.")
        return

    for split, n in [("train", n_train), ("val", n_val)]:
        blended_dst = os.path.join(data_root, split, "synthetic", "blended")
        clean_dst   = os.path.join(data_root, split, "synthetic", "clean")
        os.makedirs(blended_dst, exist_ok=True)
        os.makedirs(clean_dst,   exist_ok=True)

        n_existing = len(os.listdir(blended_dst))
        if n_existing >= n:
            print(f"  {split} synthetic: {n_existing} pairs already exist, skipping")
            continue

        print(f"  Generating {n} synthetic pairs for {split} split ...")
        for i in tqdm(range(n_existing, n), desc=f"Synthetic {split}"):
            # Sample two random distinct images
            idx_t = random.randrange(len(all_imgs))
            idx_r = random.randrange(len(all_imgs))
            while idx_r == idx_t:
                idx_r = random.randrange(len(all_imgs))

            trans_img = np.array(Image.open(all_imgs[idx_t]).convert("RGB"))
            refl_img  = np.array(Image.open(all_imgs[idx_r]).convert("RGB"))

            # Resize reflection to match transmission
            H, W = trans_img.shape[:2]
            refl_img = cv2.resize(refl_img, (W, H))

            blended = blend_images(trans_img, refl_img)

            stem = f"syn_{split}_{i:06d}"
            Image.fromarray(blended).save(
                os.path.join(blended_dst, f"{stem}.jpg"), quality=95
            )
            Image.fromarray(trans_img).save(
                os.path.join(clean_dst, f"{stem}.jpg"), quality=95
            )


# ── Main  ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="Prepare reflection removal training dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir",    default="data/",   help="Root output directory")
    p.add_argument("--n-synthetic",   type=int, default=8000,
                   help="Number of synthetic training pairs to generate")
    p.add_argument("--n-synthetic-val", type=int, default=500)
    p.add_argument("--skip-sir2",     action="store_true")
    p.add_argument("--skip-synthetic",action="store_true")
    p.add_argument("--seed",          type=int, default=42)
    return p.parse_args()


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Preparing dataset under: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.skip_sir2:
        print("\n── SIR2 real pairs ──")
        prepare_sir2(args.output_dir)

    if not args.skip_synthetic:
        print("\n── Synthetic pairs ──")
        generate_synthetic_pairs(
            args.output_dir,
            n_train=args.n_synthetic,
            n_val=args.n_synthetic_val,
        )

    # Print summary
    print("\n── Dataset summary ──")
    for split in ("train", "val"):
        for subset in ("synthetic", "real"):
            d = os.path.join(args.output_dir, split, subset, "blended")
            n = len(os.listdir(d)) if os.path.isdir(d) else 0
            print(f"  {split}/{subset}: {n} pairs")

    print("\nDone. Next steps:")
    print("  1. Run windowseat_reproducibility.py to download test sets")
    print("     (Nature, Real20, SIR2-500) into data/test/")
    print("  2. Run train_windowseat.py --data-root data/")


if __name__ == "__main__":
    main()