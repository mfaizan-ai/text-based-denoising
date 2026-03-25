#!/usr/bin/env python3
"""
test.py
--------
Evaluation and visualisation script for WindowSeat Phase 2.

What it does
------------
  1. Loads the LoRA checkpoint onto the frozen Qwen VAE + DiT backbone.
  2. Runs the full test set through the model and reports:
       - Overall PSNR / SSIM
       - Per-task PSNR / SSIM breakdown
  3. Randomly samples `--num-vis` images per task and saves side-by-side
     subplots (input | predicted | ground-truth) to --output-dir/visuals/.

Usage
-----
  python test.py \
      --checkpoint  runs/trainingv1.0/checkpoint_epoch020.pt \
      --data-root   dataset/ \
      --meta-dir    dataset_metadata/ \
      --embed-dir   text/text_embeddings/ \
      --output-dir  runs/trainingv1.0

  # or use the latest checkpoint symlink
  python test.py \
      --checkpoint  runs/trainingv1.0/checkpoint_latest.pt \
      --data-root   dataset/ \
      --meta-dir    dataset_metadata/ \
      --embed-dir   text/text_embeddings/ \
      --output-dir  runs/trainingv1.0
"""

import argparse
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display needed on cluster
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from diffusers import (
    AutoencoderKLQwenImage,
    QwenImageEditPipeline,
    QwenImageTransformer2DModel,
)
from peft import LoraConfig, get_peft_model
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ── Constants (must match train.py exactly) ────────────────────────────────────
BASE_MODEL_URI  = "Qwen/Qwen-Image-Edit-2509"
FIXED_TIMESTEP  = 499
MAX_SEQ_LEN     = 256

TASK_ID_TO_NAME = {
    0: "blur",
    1: "raindrop",
    2: "rainstreak",
    3: "rainstreak_raindrop",
    4: "reflection",
}

TASK_DISPLAY = {
    "blur":                 "Blur",
    "raindrop":             "Raindrop",
    "rainstreak":           "Rain streak",
    "rainstreak_raindrop":  "Rain streak + drop",
    "reflection":           "Reflection",
}


# ── Args ──────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(
        description="Evaluate and visualise WindowSeat LoRA on the test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",   required=True,
                   help="Path to checkpoint_epochXXX.pt or checkpoint_latest.pt")
    p.add_argument("--data-root",    required=True)
    p.add_argument("--meta-dir",     required=True,
                   help="Directory containing test_metadata.json")
    p.add_argument("--embed-dir",    required=True)
    p.add_argument("--output-dir",   required=True,
                   help="Metrics CSV + visual subplots saved here")
    p.add_argument("--base-model",   default=BASE_MODEL_URI)
    p.add_argument("--resolution",   type=int, default=512)
    p.add_argument("--batch-size",   type=int, default=8)
    p.add_argument("--num-workers",  type=int, default=8)
    p.add_argument("--num-vis",      type=int, default=4,
                   help="Number of visual samples to save per task")
    p.add_argument("--lora-rank",    type=int, default=64)
    p.add_argument("--lora-alpha",   type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


# ── Dataset (same as train.py, test split, no augmentation) ───────────────────
class TestRestorationDataset(Dataset):
    EMBED_POOL_SIZE = 20

    def __init__(self, data_root, json_path, embed_root, resolution=512):
        self.data_root  = Path(data_root)
        self.embed_root = Path(embed_root)
        self.resolution = resolution

        with open(json_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        print(f"  [test] {len(self.samples)} pairs loaded")

    def __len__(self):
        return len(self.samples)
    
    def _ensure_min_size(self, a, b):
        R = self.resolution
        W, H = a.size
        if min(W, H) < R:
            scale = R / min(W, H)
            new_w = math.ceil(W * scale)
            new_h = math.ceil(H * scale)
            a = a.resize((new_w, new_h), Image.LANCZOS)
            b = b.resize((new_w, new_h), Image.LANCZOS)
        return a, b

    def _centre_crop(self, blended, clean):
        R = self.resolution
        blended, clean = self._ensure_min_size(blended, clean)
        blended = TF.center_crop(blended, (R, R))
        clean   = TF.center_crop(clean,   (R, R))
        return blended, clean

    def __getitem__(self, idx):
        item    = self.samples[idx]
        blended = Image.open(self.data_root / item["input"]).convert("RGB")
        clean   = Image.open(self.data_root / item["target"]).convert("RGB")

        # deterministic centre crop — same as val in train.py
        blended, clean = self._centre_crop(blended, clean)

        blended_t = TF.to_tensor(blended) * 2.0 - 1.0
        clean_t   = TF.to_tensor(clean)   * 2.0 - 1.0

        # use embed index 0 deterministically at test time
        task_id   = item.get("task_id", 4)
        task_name = TASK_ID_TO_NAME.get(task_id, "reflection")
        embed     = torch.load(
            self.embed_root / task_name / "0.pt", weights_only=True
        ).squeeze(0)                            # (seq_len, D)

        # pad / truncate to MAX_SEQ_LEN
        seq_len, D = embed.shape
        if seq_len < MAX_SEQ_LEN:
            embed = torch.cat([embed, embed.new_zeros(MAX_SEQ_LEN - seq_len, D)], 0)
        else:
            embed = embed[:MAX_SEQ_LEN]

        return {
            "blended":       blended_t,
            "clean":         clean_t,
            "prompt_embeds": embed,
            "task_id":       task_id,
            "task_name":     task_name,
            "input_path":    str(item["input"]),
        }


# ── Model loading (mirrors train.py exactly) ──────────────────────────────────
def load_vae(uri, device):
    vae = AutoencoderKLQwenImage.from_pretrained(
        uri, subfolder="vae",
        torch_dtype=torch.bfloat16, use_safetensors=True,
    )
    vae.to(device, dtype=torch.bfloat16)
    vae.requires_grad_(False)
    vae.eval()
    return vae


def load_transformer_with_lora(uri, device, lora_rank, lora_alpha, lora_dropout):
    transformer = QwenImageTransformer2DModel.from_pretrained(
        uri, subfolder="transformer",
        torch_dtype=torch.bfloat16, device_map=device,
    )
    lora_cfg = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=["to_q", "to_k", "to_v", "to_out.0",
                        "ff.net.0.proj", "ff.net.2"],
        bias="none",
    )
    return get_peft_model(transformer, lora_cfg)


def load_checkpoint(path, transformer):
    print(f"Loading checkpoint: {path}")
    ckpt    = torch.load(path, map_location="cpu")
    missing, unexpected = transformer.load_state_dict(
        ckpt["lora_state_dict"], strict=False
    )
    if missing or unexpected:
        print(f"  Missing: {len(missing)}  Unexpected: {len(unexpected)}")
    epoch = ckpt.get("epoch", "?")
    m     = ckpt.get("metrics", {})
    print(f"  Epoch {epoch}  |  val PSNR={m.get('val/psnr', 'n/a')}  "
          f"SSIM={m.get('val/ssim', 'n/a')}")
    return epoch


# ── VAE encode / decode (identical to train.py) ───────────────────────────────
@torch.no_grad()
def encode(images, vae):
    images = images.to(device=vae.device, dtype=vae.dtype)
    out    = vae.encode(images.unsqueeze(2)).latent_dist.sample()
    mean   = torch.tensor(vae.config.latents_mean,
                          device=out.device, dtype=out.dtype)
    std    = torch.tensor(vae.config.latents_std,
                          device=out.device, dtype=out.dtype)
    mean   = mean.view(1, vae.config.z_dim, 1, 1, 1)
    std    = (1.0 / std).view(1, vae.config.z_dim, 1, 1, 1)
    return (out - mean) * std


@torch.no_grad()
def decode(latents, vae):
    mean = torch.tensor(vae.config.latents_mean,
                        device=latents.device, dtype=latents.dtype)
    std  = torch.tensor(vae.config.latents_std,
                        device=latents.device, dtype=latents.dtype)
    mean = mean.view(1, vae.config.z_dim, 1, 1, 1)
    std  = (1.0 / std).view(1, vae.config.z_dim, 1, 1, 1)
    latents = latents / std + mean
    return vae.decode(latents).sample[:, :, 0]   # (B, 3, H, W)


# ── Single forward pass (identical to train.py) ───────────────────────────────
@torch.no_grad()
def run_inference(latent_input, transformer, vae, prompt_embeds, prompt_mask):
    lat4d = latent_input[:, :, 0] if latent_input.ndim == 5 else latent_input
    B, C, H, W = lat4d.shape
    device = next(transformer.parameters()).device

    prompt_embeds = prompt_embeds.to(device=device, dtype=torch.bfloat16)
    prompt_mask   = prompt_mask.to(device=device)

    packed = QwenImageEditPipeline._pack_latents(
        lat4d, batch_size=B, num_channels_latents=C, height=H, width=W,
    ).to(torch.bfloat16)

    timestep     = torch.full((B,), float(FIXED_TIMESTEP) / 1000.0,
                              device=device, dtype=torch.bfloat16)
    img_shapes   = [[(1, H // 2, W // 2)]] * B
    txt_seq_lens = prompt_mask.sum(dim=1).tolist()

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model_pred = transformer(
            hidden_states=packed,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_mask,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            guidance=None,
            return_dict=False,
        )[0]

    td = vae.config.get("temperal_downsample", None)
    sf = 2 ** len(td) if td is not None else 8

    model_pred = QwenImageEditPipeline._unpack_latents(
        model_pred, height=H * sf, width=W * sf, vae_scale_factor=sf,
    )
    return model_pred   # (B, C, 1, H, W)


# ── Helpers ───────────────────────────────────────────────────────────────────
def tensor_to_uint8(t):
    """(3, H, W) float in [-1, 1]  ->  HWC uint8 numpy."""
    return ((t.float().clamp(-1, 1) + 1.0) / 2.0 * 255).byte() \
             .permute(1, 2, 0).cpu().numpy()


def tensor_to_float(t):
    """(3, H, W) float in [-1, 1]  ->  HWC float32 [0, 1] numpy."""
    return ((t.float().clamp(-1, 1) + 1.0) / 2.0).permute(1, 2, 0).cpu().numpy()


# ── Visualisation ─────────────────────────────────────────────────────────────
def save_comparison_grid(samples, task_name, out_path):
    """
    samples : list of dicts with keys
                'input'  (HWC float32 [0,1])
                'pred'   (HWC float32 [0,1])
                'target' (HWC float32 [0,1])
                'psnr'   float
                'ssim'   float
    """
    n  = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n), dpi=120)
    if n == 1:
        axes = axes[None, :]   # ensure 2-D indexing

    col_titles = ["Input (degraded)", "Predicted (restored)", "Ground truth"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=13, fontweight="bold", pad=8)

    for row, s in enumerate(samples):
        for col, key in enumerate(["input", "pred", "target"]):
            axes[row, col].imshow(s[key])
            axes[row, col].axis("off")

        # annotate PSNR / SSIM on the prediction column
        axes[row, 1].set_xlabel(
            f"PSNR {s['psnr']:.2f} dB  |  SSIM {s['ssim']:.4f}",
            fontsize=10, labelpad=4,
        )
        axes[row, 1].xaxis.set_label_position("bottom")

    fig.suptitle(
        f"Task: {TASK_DISPLAY.get(task_name, task_name)}",
        fontsize=15, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved visual -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    vis_dir = os.path.join(args.output_dir, "visuals")
    os.makedirs(vis_dir, exist_ok=True)

    # ── Dataset & loader ──────────────────────────────────────────────────────
    print("Loading test set ...")
    test_ds = TestRestorationDataset(
        data_root=args.data_root,
        json_path=os.path.join(args.meta_dir, "test_metadata.json"),
        embed_root=args.embed_dir,
        resolution=args.resolution,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ── Models ────────────────────────────────────────────────────────────────
    print(f"\nLoading base model: {args.base_model}")
    vae         = load_vae(args.base_model, device)
    transformer = load_transformer_with_lora(
        args.base_model, device,
        args.lora_rank, args.lora_alpha, args.lora_dropout,
    )
    load_checkpoint(args.checkpoint, transformer)
    transformer.to(device)
    transformer.eval()

    # ── Inference + metrics ───────────────────────────────────────────────────
    print("\nRunning inference on test set ...")
    all_psnr  = []
    all_ssim  = []
    task_psnr = defaultdict(list)
    task_ssim = defaultdict(list)

    # reservoir sampler: keep up to num_vis examples per task for visualisation
    vis_reservoir = defaultdict(list)   # task_name -> list of sample dicts

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            blended       = batch["blended"].to(device, dtype=torch.bfloat16)
            clean         = batch["clean"].to(device)
            prompt_embeds = batch["prompt_embeds"].to(device, dtype=torch.bfloat16)
            B             = blended.shape[0]
            prompt_mask   = torch.ones((B, MAX_SEQ_LEN), dtype=torch.bool, device=device)

            z_degraded    = encode(blended, vae)
            pred_velocity = run_inference(
                z_degraded, transformer, vae, prompt_embeds, prompt_mask
            )

            z_pred   = z_degraded.float() - pred_velocity.float()
            img_pred = decode(z_pred.to(vae.dtype), vae)   # (B, 3, H, W)

            pred_np  = ((img_pred.float().clamp(-1, 1) + 1) / 2).cpu().numpy()
            clean_np = ((clean.float().clamp(-1, 1)    + 1) / 2).cpu().numpy()

            for b in range(B):
                p  = pred_np[b].transpose(1, 2, 0)
                c  = clean_np[b].transpose(1, 2, 0)

                psnr = compute_psnr(c, p, data_range=1.0)
                try:
                    ssim = compute_ssim(c, p, data_range=1.0, channel_axis=-1)
                except TypeError:
                    ssim = compute_ssim(c, p, data_range=1.0, multichannel=True)

                all_psnr.append(psnr)
                all_ssim.append(ssim)

                tname = batch["task_name"][b]
                task_psnr[tname].append(psnr)
                task_ssim[tname].append(ssim)

                # reservoir sampling for visuals (keep num_vis per task)
                res = vis_reservoir[tname]
                if len(res) < args.num_vis:
                    res.append({
                        "input":  tensor_to_float(blended[b].float()),
                        "pred":   p,
                        "target": c,
                        "psnr":   psnr,
                        "ssim":   ssim,
                    })
                else:
                    j = random.randint(0, len(task_psnr[tname]) - 1)
                    if j < args.num_vis:
                        res[j] = {
                            "input":  tensor_to_float(blended[b].float()),
                            "pred":   p,
                            "target": c,
                            "psnr":   psnr,
                            "ssim":   ssim,
                        }

    # ── Print metrics ─────────────────────────────────────────────────────────
    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  {'Task':<26}  {'PSNR (dB)':>9}  {'SSIM':>8}")
    print(sep)
    for tid in sorted(TASK_ID_TO_NAME.keys()):
        tname = TASK_ID_TO_NAME[tid]
        if tname not in task_psnr:
            continue
        p = np.mean(task_psnr[tname])
        s = np.mean(task_ssim[tname])
        n = len(task_psnr[tname])
        print(f"  {TASK_DISPLAY[tname]:<26}  {p:>9.2f}  {s:>8.4f}  (n={n})")
    print(sep)
    overall_psnr = float(np.mean(all_psnr))
    overall_ssim = float(np.mean(all_ssim))
    print(f"  {'Overall':<26}  {overall_psnr:>9.2f}  {overall_ssim:>8.4f}  "
          f"(n={len(all_psnr)})")
    print(sep)

    # ── Save metrics CSV ──────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "test_metrics.csv")
    with open(csv_path, "w") as f:
        f.write("task,psnr,ssim,n\n")
        for tid in sorted(TASK_ID_TO_NAME.keys()):
            tname = TASK_ID_TO_NAME[tid]
            if tname not in task_psnr:
                continue
            f.write(
                f"{tname},"
                f"{np.mean(task_psnr[tname]):.4f},"
                f"{np.mean(task_ssim[tname]):.4f},"
                f"{len(task_psnr[tname])}\n"
            )
        f.write(f"overall,{overall_psnr:.4f},{overall_ssim:.4f},{len(all_psnr)}\n")
    print(f"\n  Metrics saved -> {csv_path}")

    # ── Save visual subplots ──────────────────────────────────────────────────
    print("\nSaving visual comparisons ...")
    for tname, samples in vis_reservoir.items():
        out_path = os.path.join(vis_dir, f"{tname}.png")
        save_comparison_grid(samples, tname, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()