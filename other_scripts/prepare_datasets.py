#!/usr/bin/env python3
"""
prepare_dataset.py
-------------------
Downloads and prepares training, validation, and test data.

SIR2 structure (what is actually in the zip):
    extracted/
        Postcard Dataset.zip
        SolidObjectDataset.zip
        Wildscene.zip

Each nested zip is extracted here before walking for image pairs.

PASCAL VOC 2012 is downloaded from the official Oxford mirror.

Usage
-----
    python prepare_dataset.py --output-dir data/
    python prepare_dataset.py --output-dir data/ --n-synthetic 8000 --skip-sir2
"""

import argparse
import os
import random
import shutil
import tarfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image
from tqdm import tqdm


# ── Download links  ───────────────────────────────────────────────────────────

SIR2_URL   = (
    "https://www.dropbox.com/scl/fi/qgg1whla1jb3a9cgis18l/SIR2.zip"
    "?rlkey=kmhrc2uk63be2s9hzr43gc3hm&e=2&st=cfsh8sol&dl=1"
)
PASCAL_URL = (
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/"
    "VOCtrainval_11-May-2012.tar"
)

VAL_RATIO  = 0.15


# ── Helpers  ──────────────────────────────────────────────────────────────────

def download_file(url: str, dest: str, desc: str = ""):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  Already exists: {dest}")
        return True
    print(f"  Downloading {desc} ...")
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=desc
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  ERROR downloading {desc}: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def extract_zip(zip_path: str, out_dir: str, desc: str = ""):
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Extracting {desc or os.path.basename(zip_path)} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def extract_tar(tar_path: str, out_dir: str, desc: str = ""):
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Extracting {desc or os.path.basename(tar_path)} ...")
    with tarfile.open(tar_path) as tf:
        tf.extractall(out_dir)


# ── SIR2  ─────────────────────────────────────────────────────────────────────

def extract_sir2_nested_zips(extracted_root: str) -> str:
    """
    The top-level SIR2.zip contains three nested zips:
        Postcard Dataset.zip
        SolidObjectDataset.zip
        Wildscene.zip

    Extract each of them into extracted_root/subsets/ and return that path.
    """
    subsets_dir = os.path.join(extracted_root, "subsets")
    os.makedirs(subsets_dir, exist_ok=True)

    for fname in os.listdir(extracted_root):
        if not fname.endswith(".zip"):
            continue
        nested_zip  = os.path.join(extracted_root, fname)
        subset_name = fname.replace(".zip", "").replace(" ", "_")
        out_subdir  = os.path.join(subsets_dir, subset_name)

        if os.path.isdir(out_subdir) and os.listdir(out_subdir):
            print(f"  Already extracted: {subset_name}")
            continue

        print(f"  Extracting nested: {fname}")
        extract_zip(nested_zip, out_subdir)

    return subsets_dir


def collect_sir2_pairs(subsets_dir: str) -> list:
    """
    Walk the extracted subsets and collect (blended_path, clean_path, stem) tuples.

    SolidObjectDataset and Wildscene: m.jpg / g.jpg pairs inside numbered dirs
    Postcard_Dataset: *-m-*.png / *-g-*.png naming convention
    """
    pairs = []

    for subset_dir in sorted(os.listdir(subsets_dir)):
        full_path = os.path.join(subsets_dir, subset_dir)
        if not os.path.isdir(full_path):
            continue

        print(f"  Scanning subset: {subset_dir}")

        if "Postcard" in subset_dir:
            # Postcard: look for -m- / -g- pairs
            for root, _, files in os.walk(full_path):
                m_files = [f for f in files if "-m-" in f and
                           f.lower().endswith((".png", ".jpg"))]
                for mf in m_files:
                    gf = mf.replace("-m-", "-g-")
                    gpath = os.path.join(root, gf)
                    if os.path.exists(gpath):
                        rel  = os.path.relpath(root, full_path).replace(os.sep, "_")
                        stem = f"postcard_{rel}_{mf[:-4]}"
                        pairs.append((os.path.join(root, mf), gpath, stem))

        else:
            # SolidObject / Wildscene: look for m.jpg + g.jpg in same dir
            for root, _, files in os.walk(full_path):
                has_m = any(f.lower() in ("m.jpg", "m.png") for f in files)
                has_g = any(f.lower() in ("g.jpg", "g.png") for f in files)
                if has_m and has_g:
                    mf = next(f for f in files if f.lower() in ("m.jpg", "m.png"))
                    gf = next(f for f in files if f.lower() in ("g.jpg", "g.png"))
                    rel  = os.path.relpath(root, full_path).replace(os.sep, "_")
                    prefix = "solid" if "Solid" in subset_dir else "wild"
                    stem = f"{prefix}_{rel}"
                    pairs.append((
                        os.path.join(root, mf),
                        os.path.join(root, gf),
                        stem,
                    ))

    return pairs


def prepare_sir2(data_root: str):
    raw_dir  = os.path.join(data_root, "raw", "sir2")
    zip_path = os.path.join(raw_dir, "SIR2.zip")
    ext_dir  = os.path.join(raw_dir, "extracted")

    # Download
    ok = download_file(SIR2_URL, zip_path, "SIR2")
    if not ok:
        print("  SIR2 download failed. Skipping.")
        return

    # Extract top-level zip if not done
    if not os.path.exists(ext_dir) or not os.listdir(ext_dir):
        extract_zip(zip_path, ext_dir, "SIR2 top-level")

    # Extract nested zips
    subsets_dir = extract_sir2_nested_zips(ext_dir)

    # Collect pairs
    pairs = collect_sir2_pairs(subsets_dir)
    print(f"SIR2: {len(pairs)} pairs total")

    if not pairs:
        print("  WARNING: No pairs found. Check directory structure with:")
        print(f"  find {subsets_dir} -type f | head -30")
        return

    # Split
    random.shuffle(pairs)
    n_val   = max(1, int(len(pairs) * VAL_RATIO))
    val_p   = pairs[:n_val]
    train_p = pairs[n_val:]
    print(f"  train={len(train_p)}  val={len(val_p)}")

    # Copy into split directories
    for split, split_pairs in [("train", train_p), ("val", val_p)]:
        blended_dst = os.path.join(data_root, split, "real", "blended")
        clean_dst   = os.path.join(data_root, split, "real", "clean")
        os.makedirs(blended_dst, exist_ok=True)
        os.makedirs(clean_dst,   exist_ok=True)

        for src_b, src_c, stem in tqdm(split_pairs, desc=f"Copying SIR2 {split}"):
            ext = os.path.splitext(src_b)[1]
            dst_b = os.path.join(blended_dst, f"{stem}{ext}")
            dst_c = os.path.join(clean_dst,   f"{stem}{ext}")
            if not os.path.exists(dst_b):
                shutil.copy2(src_b, dst_b)
            if not os.path.exists(dst_c):
                shutil.copy2(src_c, dst_c)


# ── PASCAL VOC  ───────────────────────────────────────────────────────────────

def prepare_pascal(data_root: str) -> str | None:
    """
    Downloads and extracts PASCAL VOC 2012.
    Returns path to JPEGImages directory, or None if unavailable.
    """
    pascal_root  = os.path.join(data_root, "raw", "pascal_voc")
    jpeg_dir     = os.path.join(pascal_root, "VOCdevkit", "VOC2012", "JPEGImages")

    if os.path.isdir(jpeg_dir) and len(os.listdir(jpeg_dir)) > 1000:
        print(f"  PASCAL VOC already available: {len(os.listdir(jpeg_dir))} images")
        return jpeg_dir

    tar_path = os.path.join(pascal_root, "VOCtrainval_11-May-2012.tar")
    ok = download_file(PASCAL_URL, tar_path, "PASCAL VOC 2012 (~2GB)")

    if not ok:
        print("  PASCAL VOC download failed.")
        print("  Alternative: place any natural images in data/raw/natural_images/")
        print("  and re-run with --skip-pascal-download")
        return None

    extract_tar(tar_path, pascal_root, "PASCAL VOC")

    if os.path.isdir(jpeg_dir):
        print(f"  PASCAL VOC ready: {len(os.listdir(jpeg_dir))} images")
        return jpeg_dir

    print("  ERROR: could not find JPEGImages after extraction.")
    return None


# ── Synthetic pair generation  ────────────────────────────────────────────────

def blend_images(trans: np.ndarray, refl: np.ndarray) -> np.ndarray:
    """
    I = alpha * T + (1-alpha) * blur(R),  alpha ~ U[0.70, 0.95]
    """
    alpha = random.uniform(0.70, 0.95)
    if random.random() < 0.7:
        ksize = random.choice([3, 5, 7])
        refl  = cv2.GaussianBlur(refl, (ksize, ksize), 0)
    blended = alpha * trans.astype(np.float32) + \
              (1.0 - alpha) * refl.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def generate_synthetic_pairs(
    data_root  : str,
    image_dir  : str,
    n_train    : int,
    n_val      : int,
):
    all_imgs = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])
    print(f"  Source images: {len(all_imgs)}")

    if len(all_imgs) < 2:
        print("  Not enough images for synthetic generation.")
        return

    for split, n in [("train", n_train), ("val", n_val)]:
        blended_dst = os.path.join(data_root, split, "synthetic", "blended")
        clean_dst   = os.path.join(data_root, split, "synthetic", "clean")
        os.makedirs(blended_dst, exist_ok=True)
        os.makedirs(clean_dst,   exist_ok=True)

        n_existing = len([f for f in os.listdir(blended_dst)
                          if f.endswith((".jpg", ".png"))])
        if n_existing >= n:
            print(f"  {split} synthetic: {n_existing} already exist, skipping")
            continue

        print(f"  Generating {n - n_existing} synthetic pairs for {split} ...")
        for i in tqdm(range(n_existing, n), desc=f"Synthetic {split}"):
            idx_t = random.randrange(len(all_imgs))
            idx_r = random.randrange(len(all_imgs))
            while idx_r == idx_t:
                idx_r = random.randrange(len(all_imgs))

            try:
                trans_img = np.array(Image.open(all_imgs[idx_t]).convert("RGB"))
                refl_img  = np.array(Image.open(all_imgs[idx_r]).convert("RGB"))
            except Exception:
                continue

            H, W = trans_img.shape[:2]
            refl_img = cv2.resize(refl_img, (W, H))
            blended  = blend_images(trans_img, refl_img)

            stem = f"syn_{i:07d}"
            Image.fromarray(blended).save(
                os.path.join(blended_dst, f"{stem}.jpg"), quality=95
            )
            Image.fromarray(trans_img).save(
                os.path.join(clean_dst, f"{stem}.jpg"), quality=95
            )


# ── Summary  ──────────────────────────────────────────────────────────────────

def print_summary(data_root: str):
    print("\n── Dataset summary ──")
    total = 0
    for split in ("train", "val"):
        for subset in ("synthetic", "real"):
            d = os.path.join(data_root, split, subset, "blended")
            n = len([f for f in os.listdir(d) if not f.startswith(".")] ) \
                if os.path.isdir(d) else 0
            total += n
            print(f"  {split:6s}/{subset:12s}: {n:5d} pairs")
    print(f"  Total train+val pairs: {total}")


# ── Args  ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--output-dir",          default="data/")
    p.add_argument("--n-synthetic",         type=int, default=8000)
    p.add_argument("--n-synthetic-val",     type=int, default=500)
    p.add_argument("--skip-sir2",           action="store_true")
    p.add_argument("--skip-synthetic",      action="store_true")
    p.add_argument("--skip-pascal-download",action="store_true",
                   help="Skip PASCAL download (use existing or natural_images/)")
    p.add_argument("--seed",                type=int, default=42)
    return p.parse_args()


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Output directory: {args.output_dir}\n")

    # ── SIR2 real pairs ───────────────────────────────────────────────────────
    if not args.skip_sir2:
        print("── SIR2 real pairs ──")
        prepare_sir2(args.output_dir)

    # ── Synthetic pairs ───────────────────────────────────────────────────────
    if not args.skip_synthetic:
        print("\n── Synthetic pairs ──")

        image_dir = None

        if not args.skip_pascal_download:
            image_dir = prepare_pascal(args.output_dir)

        # Fallback: user-placed images
        if image_dir is None:
            alt = os.path.join(args.output_dir, "raw", "natural_images")
            if os.path.isdir(alt) and len(os.listdir(alt)) > 10:
                print(f"  Using natural_images fallback: {alt}")
                image_dir = alt

        if image_dir is not None:
            generate_synthetic_pairs(
                args.output_dir, image_dir,
                n_train=args.n_synthetic,
                n_val=args.n_synthetic_val,
            )
        else:
            print("  No image source available for synthetic generation.")
            print("  Options:")
            print("    1. Wait for PASCAL VOC download (re-run without --skip-pascal-download)")
            print("    2. Place images in data/raw/natural_images/ and re-run")

    print_summary(args.output_dir)

    print("\nNext steps:")
    print("  1. Run: python windowseat_reproducibility.py")
    print("     to download Nature, Real20, SIR2-500 test sets")
    print("  2. Run: python train_windowseat.py --data-root data/")


if __name__ == "__main__":
    main()