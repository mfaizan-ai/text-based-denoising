#!/usr/bin/env python3
"""
train_windowseat_baseline.py
------------------------------
Baseline training script for WindowSeat multi-task image restoration.
Fully aligned with the paper (Section 3.3) implementation details.

Paper-aligned fixes vs the previous version
---------------------------------------------
  1. LOSS: PSNR (MSE) + SSIM in PIXEL SPACE, not latent MSE + LPIPS.
           λ_PSNR=0.1, λ_SSIM=20 (paper values).
  2. FLOW SIGN: z_edit = z_B + v  (paper eq.).
           velocity_target = z_clean - z_degraded.
           Previous version subtracted velocity — wrong direction.
  3. LORA RANK: 128 with Gaussian initialisation (paper: rank=128).
  4. LR SCHEDULE: 1e-5 warmup 100 steps to 1e-4, then linear decay
           5e-6 per 1000 steps (matches paper exactly).
  5. RESOLUTION: 608×608 (paper training resolution).
  6. AUGMENTATION: random crop + color jitter (brightness, contrast,
           saturation, hue) as stated in paper.
  7. BATCH SIZE: default=2 (paper uses batch size 2).
  8. GRADIENT CHECKPOINTING: enabled for VAE and DiT (paper states this).
  9. STEP-BASED training loop matching the paper's 11k total steps.

Dataset
--------
  Driven by JSON metadata files (train/val/test_metadata.json).
  JSON format:
    [
      {
        "input":     "dataset_full/blur/real/blended/002500.png",
        "target":    "dataset_full/blur/real/clean/002500.png",
        "task_name": "blur",
        "task_id":   0,
        "data_type": "real",
        "width":     1280,
        "height":    720
      }, ...
    ]

  task_id mapping:
    0 - blur
    1 - raindrop
    2 - rainstreak
    3 - rainstreak_raindrop
    4 - reflection

Text embeddings
---------------
  embed_dir/
    blur/                0.pt ... 19.pt   shape (1, 256, hidden_dim)
    raindrop/            0.pt ... 19.pt
    rainstreak/          0.pt ... 19.pt
    rainstreak_raindrop/ 0.pt ... 19.pt
    reflection/          0.pt ... 19.pt

Usage
-----
  # Single GPU
  python train_windowseat_baseline.py \\
      --data-root . \\
      --meta-dir  dataset_metadata \\
      --embed-dir text/text_embeddings \\
      --output-dir runs/baseline

  # Multi-GPU
  torchrun --nproc_per_node=4 train_windowseat_baseline.py \\
      --data-root . \\
      --meta-dir  dataset_metadata \\
      --embed-dir text/text_embeddings \\
      --output-dir runs/baseline

  # Resume
  python train_windowseat_baseline.py ... \\
      --resume runs/baseline/checkpoint_latest.pt
"""

import argparse
import csv
import shutil
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter
import wandb
from diffusers import (
    AutoencoderKLQwenImage,
    QwenImageEditPipeline,
    QwenImageTransformer2DModel,
)
from peft import LoraConfig, get_peft_model
from PIL import Image
from pytorch_msssim import SSIM as SSIMLoss
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm


# ── Constants ──────────────────────────────────────────────────────────────────

BASE_MODEL_URI  = "Qwen/Qwen-Image-Edit-2509"
FIXED_TIMESTEP  = 499       # single fixed denoising timestep (WindowSeat)
MAX_SEQ_LEN     = 256       # Qwen sequence length (not CLIP's 77)
EMBED_POOL_SIZE = 20        # number of .pt embedding files per task

TASK_ID_TO_NAME = {
    0: "blur",
    1: "raindrop",
    2: "rainstreak",
    3: "rainstreak_raindrop",
    4: "reflection",
}


# ── Args ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="WindowSeat baseline training — paper-aligned",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    p.add_argument("--data-root",   required=True,
                   help="Project root; JSON input/target paths are joined here")
    p.add_argument("--meta-dir",    required=True,
                   help="Folder containing train_metadata.json and val_metadata.json")
    p.add_argument("--embed-dir",   required=True,
                   help="Folder containing per-task .pt embedding pools")
    p.add_argument("--output-dir",  required=True,
                   help="Where to save checkpoints and logs")
    p.add_argument("--resume",      default=None,
                   help="Path to checkpoint (.pt) to resume from")

    # Model
    p.add_argument("--base-model",    default=BASE_MODEL_URI)
    p.add_argument("--lora-rank",     type=int,   default=128,
                   help="LoRA rank — paper uses 128")
    p.add_argument("--lora-alpha",    type=int,   default=128)
    p.add_argument("--lora-dropout",  type=float, default=0.0,
                   help="LoRA dropout — paper does not mention nonzero dropout")

    # Training — defaults match paper exactly
    p.add_argument("--total-steps",   type=int,   default=11_000,
                   help="Total optimiser steps (paper: 11k)")
    p.add_argument("--batch-size",    type=int,   default=2,
                   help="Per-GPU batch size (paper: 2)")
    p.add_argument("--grad-accum",    type=int,   default=1,
                   help="Gradient accumulation steps")
    p.add_argument("--lr-start",      type=float, default=1e-5,
                   help="Initial LR before warmup (paper: 1e-5)")
    p.add_argument("--lr-peak",       type=float, default=1e-4,
                   help="Peak LR after warmup (paper: 1e-4)")
    p.add_argument("--lr-decay",      type=float, default=5e-6,
                   help="Linear LR decay per 1000 steps (paper: 5e-6/1k)")
    p.add_argument("--warmup-steps",  type=int,   default=100,
                   help="Warmup steps (paper: 100)")
    p.add_argument("--max-grad-norm", type=float, default=1.0,
                   help="Global gradient norm clipping")
    p.add_argument("--resolution",    type=int,   default=608,
                   help="Square crop resolution (paper: 608)")

    # Loss weights — paper: λ_PSNR=0.1, λ_SSIM=20
    p.add_argument("--lambda-psnr",   type=float, default=0.1)
    p.add_argument("--lambda-ssim",   type=float, default=20.0)

    # Logging / saving
    p.add_argument("--log-interval",  type=int,   default=50,
                   help="Log every N optimiser steps")
    p.add_argument("--val-interval",  type=int,   default=500,
                   help="Validate every N optimiser steps")
    p.add_argument("--save-interval", type=int,   default=1_000,
                   help="Save checkpoint every N optimiser steps")
    p.add_argument("--wandb-project", default="windowseat_baseline")
    p.add_argument("--run-name",      default=None)

    # System
    p.add_argument("--num-workers",   type=int,   default=4)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--local-rank",    type=int,   default=-1,
                   help="Set automatically by torchrun")

    return p.parse_args()


# ── Distributed helpers ────────────────────────────────────────────────────────

def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_distributed(local_rank: int):
    if local_rank == -1:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu"), 0, 1
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}"), dist.get_rank(), dist.get_world_size()


# ── Dataset ────────────────────────────────────────────────────────────────────

class JsonRestorationDataset(Dataset):
    """
    Multi-task restoration dataset driven by a JSON metadata file.

    Resolution strategy (paper-aligned):
      Training  : random 608×608 crop + color jitter (brightness, contrast,
                  saturation, hue) applied identically to blended and clean.
      Validation: deterministic centre crop, no augmentation.

    Upscaling: 400px reflection images and small syn images are upscaled
    with LANCZOS to --resolution before cropping so no black borders appear.

    Each sample also loads a randomly chosen text embedding from a pool
    of 20 .pt files per task (matching task_name from the JSON).
    """

    def __init__(
        self,
        data_root:  str,
        json_path:  str,
        embed_root: str,
        resolution: int  = 608,
        is_train:   bool = True,
    ):
        self.data_root  = Path(data_root)
        self.embed_root = Path(embed_root)
        self.resolution = resolution
        self.is_train   = is_train

        with open(json_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        if is_main():
            tag     = "train" if is_train else "val"
            missing = sum(
                1 for s in self.samples
                if not (self.data_root / s["input"]).exists()
            )
            print(f"  [{tag}] {len(self.samples)} pairs loaded "
                  f"({missing} input files not found on disk)")
            if missing > 0:
                print(f"  [WARN] {missing} files missing — check --data-root")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        item    = self.samples[idx]
        blended = Image.open(self.data_root / item["input"]).convert("RGB")
        clean   = Image.open(self.data_root / item["target"]).convert("RGB")

        if self.is_train:
            blended, clean = self._augment(blended, clean)
        else:
            blended, clean = self._val_transform(blended, clean)

        # Normalise to [-1, 1] — required by the Qwen VAE
        blended_t = TF.to_tensor(blended) * 2.0 - 1.0   # (3, R, R)
        clean_t   = TF.to_tensor(clean)   * 2.0 - 1.0   # (3, R, R)

        prompt_embeds = self._load_embed(item.get("task_id", 4))

        return {
            "blended":       blended_t,         # (3, R, R) float32 in [-1, 1]
            "clean":         clean_t,            # (3, R, R) float32 in [-1, 1]
            "prompt_embeds": prompt_embeds,      # (MAX_SEQ_LEN, D) float32
            "task_id":       item["task_id"],
            "task_name":     item["task_name"],
            "data_type":     item.get("data_type", "unknown"),
        }

    # ── Augmentation ──────────────────────────────────────────────────────────

    def _ensure_min_size(self, a: Image.Image, b: Image.Image) -> tuple:
        """Upscale both images identically if shorter side < resolution."""
        R    = self.resolution
        W, H = a.size
        if min(W, H) < R:
            scale = R / min(W, H)
            nw    = math.ceil(W * scale)
            nh    = math.ceil(H * scale)
            a = a.resize((nw, nh), Image.LANCZOS)
            b = b.resize((nw, nh), Image.LANCZOS)
        return a, b

    def _augment(self, blended: Image.Image, clean: Image.Image) -> tuple:
        """
        Paper augmentations applied to BOTH images with the SAME parameters:
          1. Color jitter (brightness, contrast, saturation, hue)
          2. Random 608×608 crop
          3. Random horizontal flip
        Applying the same transform to both preserves the blended→clean mapping.
        """
        R = self.resolution
        blended, clean = self._ensure_min_size(blended, clean)

        # Sample color jitter parameters once, apply to both
        fn_idx, b_f, c_f, s_f, h_f = ColorJitter.get_params(
            brightness=[0.8, 1.2],
            contrast=[0.8, 1.2],
            saturation=[0.8, 1.2],
            hue=[-0.1, 0.1],
        )
        jitter_ops = {
            0: lambda img: TF.adjust_brightness(img, b_f),
            1: lambda img: TF.adjust_contrast(img, c_f),
            2: lambda img: TF.adjust_saturation(img, s_f),
            3: lambda img: TF.adjust_hue(img, h_f),
        }
        for fn in fn_idx:
            blended = jitter_ops[int(fn)](blended)
            clean   = jitter_ops[int(fn)](clean)

        # Random crop — same for both
        W, H = blended.size
        i    = random.randint(0, H - R)
        j    = random.randint(0, W - R)
        blended = TF.crop(blended, i, j, R, R)
        clean   = TF.crop(clean,   i, j, R, R)

        # Random horizontal flip — same for both
        if random.random() < 0.5:
            blended = TF.hflip(blended)
            clean   = TF.hflip(clean)

        return blended, clean

    def _val_transform(self, blended: Image.Image, clean: Image.Image) -> tuple:
        """Deterministic centre crop, no augmentation."""
        R = self.resolution
        blended, clean = self._ensure_min_size(blended, clean)
        blended = TF.center_crop(blended, (R, R))
        clean   = TF.center_crop(clean,   (R, R))
        return blended, clean

    # ── Embedding loader ───────────────────────────────────────────────────────

    def _load_embed(self, task_id: int) -> torch.Tensor:
        task_name = TASK_ID_TO_NAME.get(task_id, "reflection")
        idx       = random.randint(0, EMBED_POOL_SIZE - 1)
        path      = self.embed_root / task_name / f"{idx}.pt"
        embed     = torch.load(path, weights_only=True)   # (1, seq_len, D)
        embed     = embed.squeeze(0)                       # (seq_len, D)

        seq_len, D = embed.shape
        if seq_len < MAX_SEQ_LEN:
            pad   = embed.new_zeros(MAX_SEQ_LEN - seq_len, D)
            embed = torch.cat([embed, pad], dim=0)
        else:
            embed = embed[:MAX_SEQ_LEN]

        return embed    # (MAX_SEQ_LEN, D)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_vae(uri: str, device: torch.device) -> AutoencoderKLQwenImage:
    """Load and freeze the VAE. Enable gradient checkpointing (paper)."""
    vae = AutoencoderKLQwenImage.from_pretrained(
        uri, subfolder="vae",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )
    vae.to(device, dtype=torch.bfloat16)
    vae.requires_grad_(False)
    vae.eval()
    # VAE is fully frozen -- gradient checkpointing not needed and not
    # supported by AutoencoderKLQwenImage. Only the transformer needs it.
    return vae


def load_transformer_with_lora(
    uri:          str,
    device:       torch.device,
    lora_rank:    int,
    lora_alpha:   int,
    lora_dropout: float,
) -> QwenImageTransformer2DModel:
    """Load DiT backbone and attach LoRA adapters (paper: rank=128, Gaussian)."""
    transformer = QwenImageTransformer2DModel.from_pretrained(
        uri, subfolder="transformer",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    # PAPER: rank=128, Gaussian initialisation
    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",
            "ff.net.0.proj", "ff.net.2",
        ],
        bias="none",
        init_lora_weights="gaussian",
    )
    transformer = get_peft_model(transformer, lora_cfg)

    if is_main():
        transformer.print_trainable_parameters()

    return transformer


# ── VAE encode / decode ────────────────────────────────────────────────────────

@torch.no_grad()
def encode(images: torch.Tensor, vae: AutoencoderKLQwenImage) -> torch.Tensor:
    """(B, 3, H, W) in [-1, 1]  →  normalised latents (B, C, 1, H/8, W/8)."""
    images = images.to(device=vae.device, dtype=vae.dtype)
    out    = vae.encode(images.unsqueeze(2)).latent_dist.sample()
    mean   = torch.tensor(vae.config.latents_mean, device=out.device, dtype=out.dtype)
    std    = torch.tensor(vae.config.latents_std,  device=out.device, dtype=out.dtype)
    mean   = mean.view(1, vae.config.z_dim, 1, 1, 1)
    std    = (1.0 / std).view(1, vae.config.z_dim, 1, 1, 1)
    return (out - mean) * std


def decode(latents: torch.Tensor, vae: AutoencoderKLQwenImage) -> torch.Tensor:
    """Normalised latents (B, C, 1, H/8, W/8)  →  (B, 3, H, W) in [-1, 1]."""
    mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype)
    std  = torch.tensor(vae.config.latents_std,  device=latents.device, dtype=latents.dtype)
    mean = mean.view(1, vae.config.z_dim, 1, 1, 1)
    std  = (1.0 / std).view(1, vae.config.z_dim, 1, 1, 1)
    latents = latents / std + mean
    return vae.decode(latents).sample[:, :, 0]   # (B, 3, H, W)


# ── Transformer forward ────────────────────────────────────────────────────────

def forward_transformer(
    latent_input:  torch.Tensor,
    transformer:   QwenImageTransformer2DModel,
    vae:           AutoencoderKLQwenImage,
    prompt_embeds: torch.Tensor,
    prompt_mask:   torch.Tensor,
) -> torch.Tensor:
    """Single forward pass. Returns predicted velocity (same shape as input)."""
    lat4d = latent_input[:, :, 0] if latent_input.ndim == 5 else latent_input
    B, C, H, W = lat4d.shape

    device        = next(transformer.parameters()).device
    prompt_embeds = prompt_embeds.to(device=device, dtype=torch.bfloat16)
    prompt_mask   = prompt_mask.to(device=device)

    packed = QwenImageEditPipeline._pack_latents(
        lat4d, batch_size=B,
        num_channels_latents=C, height=H, width=W,
    ).to(torch.bfloat16)

    timestep     = torch.full(
        (B,), float(FIXED_TIMESTEP) / 1000.0,
        device=device, dtype=torch.bfloat16
    )
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
        model_pred,
        height=H * sf, width=W * sf,
        vae_scale_factor=sf,
    )
    return model_pred   # (B, C, 1, H, W)


# ── Loss (paper-aligned) ───────────────────────────────────────────────────────

class WindowSeatLoss(torch.nn.Module):
    """
    Paper loss (Section 3.3):
      L = λ_PSNR * L_PSNR(Ŷ, T) + λ_SSIM * L_SSIM(Ŷ, T)
      λ_PSNR = 0.1,  λ_SSIM = 20

    Both terms computed in PIXEL SPACE on decoded predictions.
    L_PSNR is MSE  (minimising MSE maximises PSNR).
    L_SSIM = 1 - SSIM  (minimising this maximises SSIM).

    Flow sign (paper eq.):
      z_edit = z_B + v_θ(z_B; p)       ← ADD velocity
      velocity target = z_clean - z_degraded
    """

    def __init__(
        self,
        lambda_psnr: float = 0.1,
        lambda_ssim: float = 20.0,
        device:      torch.device = None,
    ):
        super().__init__()
        self.lambda_psnr = lambda_psnr
        self.lambda_ssim = lambda_ssim
        self.ssim_fn     = SSIMLoss(
            data_range=1.0, size_average=True, channel=3
        ).to(device)

    def forward(
        self,
        pred_image:   torch.Tensor,   # (B, 3, H, W) in [0, 1]
        target_image: torch.Tensor,   # (B, 3, H, W) in [0, 1]
    ):
        loss_psnr = F.mse_loss(pred_image, target_image)
        loss_ssim = 1.0 - self.ssim_fn(pred_image, target_image)
        total     = self.lambda_psnr * loss_psnr + self.lambda_ssim * loss_ssim
        return total, loss_psnr, loss_ssim


# ── LR scheduler (paper-aligned) ──────────────────────────────────────────────

def get_scheduler(
    optimizer:    torch.optim.Optimizer,
    warmup_steps: int,
    lr_start:     float,
    lr_peak:      float,
    lr_decay:     float,
):
    """
    Paper schedule:
      Linear warmup from lr_start (1e-5) to lr_peak (1e-4) over warmup_steps.
      Then linear decay of lr_decay (5e-6) every 1000 steps.
    LambdaLR multiplies lr_peak by the returned scalar.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            t  = step / max(1, warmup_steps)
            lr = lr_start + t * (lr_peak - lr_start)
            return lr / lr_peak
        steps_post_warmup = step - warmup_steps
        decay             = lr_decay * (steps_post_warmup // 1000)
        lr                = max(lr_peak - decay, 1e-7)
        return lr / lr_peak

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Validation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    vae:         AutoencoderKLQwenImage,
    transformer: QwenImageTransformer2DModel,
    val_loader:  DataLoader,
    device:      torch.device,
    global_step: int,
) -> dict:
    """
    Compute validation PSNR and SSIM overall and per task.
    Per-task breakdown lets you see which tasks are struggling
    before adding your improvements.
    """
    transformer.eval()
    torch.cuda.empty_cache()

    overall_psnr = []
    overall_ssim = []
    task_psnr    = {name: [] for name in TASK_ID_TO_NAME.values()}
    task_ssim    = {name: [] for name in TASK_ID_TO_NAME.values()}

    for batch in tqdm(val_loader, desc=f"Val @step {global_step}",
                      leave=False, disable=not is_main()):
        blended       = batch["blended"].to(device, dtype=torch.bfloat16)
        clean         = batch["clean"].to(device,   dtype=torch.bfloat16)
        prompt_embeds = batch["prompt_embeds"].to(device, dtype=torch.bfloat16)
        B             = blended.shape[0]
        prompt_mask   = torch.ones((B, MAX_SEQ_LEN), dtype=torch.bool, device=device)

        z_degraded    = encode(blended, vae)

        # PAPER: z_edit = z_B + v  (add velocity)
        pred_velocity = forward_transformer(
            z_degraded, transformer, vae, prompt_embeds, prompt_mask
        )
        z_pred   = z_degraded.float() + pred_velocity.float()
        img_pred = decode(z_pred.to(vae.dtype), vae)

        pred_np  = ((img_pred.float().clamp(-1, 1) + 1) / 2).cpu().numpy()
        clean_np = ((clean.float().clamp(-1, 1)   + 1) / 2).cpu().numpy()

        for b in range(B):
            p    = pred_np[b].transpose(1, 2, 0)
            c    = clean_np[b].transpose(1, 2, 0)
            psnr = compute_psnr(c, p, data_range=1.0)
            try:
                ssim = compute_ssim(c, p, data_range=1.0, channel_axis=-1)
            except TypeError:
                ssim = compute_ssim(c, p, data_range=1.0, multichannel=True)

            overall_psnr.append(psnr)
            overall_ssim.append(ssim)

            tname = batch["task_name"][b]
            if tname in task_psnr:
                task_psnr[tname].append(psnr)
                task_ssim[tname].append(ssim)

    transformer.train()

    metrics = {
        "val/psnr": float(np.mean(overall_psnr)),
        "val/ssim": float(np.mean(overall_ssim)),
    }
    for tname in TASK_ID_TO_NAME.values():
        if task_psnr[tname]:
            metrics[f"val/psnr_{tname}"] = float(np.mean(task_psnr[tname]))
            metrics[f"val/ssim_{tname}"] = float(np.mean(task_ssim[tname]))

    return metrics


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_checkpoint(
    output_dir:  str,
    step:        int,
    transformer: QwenImageTransformer2DModel,
    optimizer:   torch.optim.Optimizer,
    scheduler,
    metrics:     dict,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    model = transformer.module if hasattr(transformer, "module") else transformer
    ckpt  = {
        "step":            step,
        "lora_state_dict": {k: v for k, v in model.state_dict().items()
                            if "lora" in k},
        "optimizer":       optimizer.state_dict(),
        "scheduler":       scheduler.state_dict(),
        "metrics":         metrics,
    }
    path   = os.path.join(output_dir, f"checkpoint_step{step:06d}.pt")
    latest = os.path.join(output_dir, "checkpoint_latest.pt")
    torch.save(ckpt, path)
    if os.path.lexists(latest):
        os.remove(latest)
    try:
        os.symlink(os.path.basename(path), latest)
    except OSError:
        import shutil
        shutil.copy2(path, latest)
    if is_main():
        print(f"  Checkpoint saved → {path}")
    return path


def load_checkpoint(
    path:        str,
    transformer: QwenImageTransformer2DModel,
    optimizer:   torch.optim.Optimizer,
    scheduler,
) -> int:
    if is_main():
        print(f"Resuming from {path}")
    ckpt  = torch.load(path, map_location="cpu")
    model = transformer.module if hasattr(transformer, "module") else transformer
    missing, unexpected = model.load_state_dict(
        ckpt["lora_state_dict"], strict=False
    )
    if is_main() and (missing or unexpected):
        print(f"  Missing: {len(missing)}  Unexpected: {len(unexpected)}")
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    step = ckpt.get("step", 0)
    if is_main():
        m = ckpt.get("metrics", {})
        print(f"  Resumed at step {step}  |  last val PSNR={m.get('val/psnr','n/a')}")
    return step


# ── Metrics logger (CSV) ──────────────────────────────────────────────────────

class MetricsLogger:
    """
    Writes training and validation metrics to separate CSV files so you
    can plot them after training without needing W&B or any other tool.

    Files created in output_dir:
      train_metrics.csv  -- one row per log_interval steps
                           columns: step, loss, loss_psnr, loss_ssim,
                                    train_psnr, train_ssim, lr
      val_metrics.csv    -- one row per val_interval steps
                           columns: step, val_psnr, val_ssim,
                                    val_psnr_<task>, val_ssim_<task> ...
    """

    TRAIN_FIELDS = [
        "step", "epoch", "loss", "loss_psnr", "loss_ssim",
        "train_psnr", "train_ssim", "lr",
    ]

    def __init__(self, output_dir: str):
        self.output_dir  = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.train_path  = os.path.join(output_dir, "train_metrics.csv")
        self.val_path    = os.path.join(output_dir, "val_metrics.csv")
        self._val_fields = None   # determined on first val write

        # Write headers (or append if resuming)
        self._init_csv(self.train_path, self.TRAIN_FIELDS)

    def _init_csv(self, path: str, fields: list):
        """Write header row only if the file does not already exist."""
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    def log_train(
        self,
        step:       int,
        epoch:      float,
        loss:       float,
        loss_psnr:  float,
        loss_ssim:  float,
        train_psnr: float,
        train_ssim: float,
        lr:         float,
    ):
        row = {
            "step":       step,
            "epoch":      round(epoch, 3),   # fractional epoch e.g. 1.45
            "loss":       round(loss,       6),
            "loss_psnr":  round(loss_psnr,  6),
            "loss_ssim":  round(loss_ssim,  6),
            "train_psnr": round(train_psnr, 4),
            "train_ssim": round(train_ssim, 4),
            "lr":         f"{lr:.2e}",
        }
        with open(self.train_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.TRAIN_FIELDS).writerow(row)

    def log_val(self, step: int, metrics: dict):
        """
        metrics keys: "val/psnr", "val/ssim",
                      "val/psnr_blur", "val/ssim_blur", etc.
        We flatten to CSV-friendly names (replace / with _).
        """
        flat = {"step": step}
        for k, v in metrics.items():
            flat[k.replace("/", "_")] = round(float(v), 4)

        # Initialise val CSV with discovered field names on first call
        if self._val_fields is None:
            self._val_fields = list(flat.keys())
            self._init_csv(self.val_path, self._val_fields)

        with open(self.val_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=self._val_fields, extrasaction="ignore"
            )
            writer.writerow(flat)


def save_best_checkpoint(
    output_dir:   str,
    step:         int,
    transformer:  QwenImageTransformer2DModel,
    optimizer:    torch.optim.Optimizer,
    scheduler,
    metrics:      dict,
    best_psnr:    float,
) -> float:
    """
    Save a checkpoint named checkpoint_best.pt whenever val PSNR improves.
    Returns the new best PSNR (unchanged if no improvement).
    """
    current_psnr = metrics.get("val/psnr", -float("inf"))
    if current_psnr <= best_psnr:
        return best_psnr

    # New best — save
    model = transformer.module if hasattr(transformer, "module") else transformer
    ckpt  = {
        "step":            step,
        "lora_state_dict": {k: v for k, v in model.state_dict().items()
                            if "lora" in k},
        "optimizer":       optimizer.state_dict(),
        "scheduler":       scheduler.state_dict(),
        "metrics":         metrics,
    }
    best_path = os.path.join(output_dir, "checkpoint_best.pt")
    torch.save(ckpt, best_path)
    print(
        f"  ★ New best checkpoint  PSNR {best_psnr:.4f} → {current_psnr:.4f} dB"
        f"  saved → {best_path}"
    )
    return current_psnr


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    device, rank, world_size = setup_distributed(args.local_rank)

    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── W&B ───────────────────────────────────────────────────────────────────
    if is_main() and args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name or f"baseline_r{args.lora_rank}",
            config=vars(args),
        )

    # ── Datasets ──────────────────────────────────────────────────────────────
    if is_main():
        print(f"\nLoading datasets from: {args.meta_dir}")

    train_ds = JsonRestorationDataset(
        data_root=args.data_root,
        json_path=os.path.join(args.meta_dir, "train_metadata.json"),
        embed_root=args.embed_dir,
        resolution=args.resolution,
        is_train=True,
    )
    val_ds = JsonRestorationDataset(
        data_root=args.data_root,
        json_path=os.path.join(args.meta_dir, "val_metadata.json"),
        embed_root=args.embed_dir,
        resolution=args.resolution,
        is_train=False,
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True)  if world_size > 1 else None
    val_sampler   = DistributedSampler(val_ds,   shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    # ── Models ────────────────────────────────────────────────────────────────
    if is_main():
        print(f"\nLoading base model: {args.base_model}")

    vae         = load_vae(args.base_model, device)
    transformer = load_transformer_with_lora(
        args.base_model, device,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    transformer.train()

    if world_size > 1:
        transformer = DDP(transformer, device_ids=[args.local_rank])

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = WindowSeatLoss(
        lambda_psnr=args.lambda_psnr,
        lambda_ssim=args.lambda_ssim,
        device=device,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    lora_params = [p for n, p in transformer.named_parameters() if "lora" in n]
    if is_main():
        print(f"Trainable LoRA parameters: {sum(p.numel() for p in lora_params):,}")

    # PAPER: AdamW, initialised at lr_start (1e-5); LambdaLR handles schedule
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=args.lr_peak,     # base lr — LambdaLR scales it
        weight_decay=0.01,
    )
    scheduler = get_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        lr_start=args.lr_start,
        lr_peak=args.lr_peak,
        lr_decay=args.lr_decay,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(
            args.resume, transformer, optimizer, scheduler
        )

    # ── Step-based training loop ───────────────────────────────────────────────
    # Step-based (not epoch-based) to match the paper's 11k step report
    # and to keep val/save intervals consistent across experiments.

    global_step = start_step
    scaler      = torch.cuda.amp.GradScaler()
    metrics     = {}
    best_psnr   = -float("inf")   # track best val PSNR for checkpoint_best.pt

    # CSV loggers — written only by main process
    if is_main():
        csv_logger = MetricsLogger(args.output_dir)

    # Running accumulators for train PSNR/SSIM between log intervals
    # We compute these on decoded predictions in the training loop itself
    # so we get a true train metric (not just loss).
    train_psnr_acc = []
    train_ssim_acc = []

    optimizer.zero_grad()

    # steps_per_epoch: how many optimiser steps fit in one full pass
    # through the training set. Used to convert step → fractional epoch.
    steps_per_epoch = max(1, len(train_ds) // args.batch_size // args.grad_accum)

    def infinite_loader(loader, sampler):
        epoch = 0
        while True:
            if sampler is not None:
                sampler.set_epoch(epoch)
            yield from loader
            epoch += 1

    train_iter = infinite_loader(train_loader, train_sampler)
    pbar       = tqdm(
        total=args.total_steps - start_step,
        desc="Training",
        disable=not is_main(),
    )

    if is_main():
        print(f"\nTraining from step {start_step} → {args.total_steps}")

    grad_accum_count = 0

    while global_step < args.total_steps:
        batch = next(train_iter)

        blended       = batch["blended"].to(device, dtype=torch.bfloat16)
        clean         = batch["clean"].to(device,   dtype=torch.bfloat16)
        prompt_embeds = batch["prompt_embeds"].to(device, dtype=torch.bfloat16)
        B             = blended.shape[0]
        prompt_mask   = torch.ones((B, MAX_SEQ_LEN), dtype=torch.bool, device=device)

        # Encode — no grad needed (VAE frozen)
        with torch.no_grad():
            z_degraded = encode(blended, vae)
            z_clean    = encode(clean,   vae)   # encoded but not used in loss
                                                 # kept for reference / debugging

        # Forward
        pred_velocity = forward_transformer(
            z_degraded, transformer, vae, prompt_embeds, prompt_mask
        )

        # PAPER FLOW: z_edit = z_B + v  (add velocity)
        # Decode to pixel space — loss computed there, not in latent space
        z_pred   = z_degraded.float() + pred_velocity.float()
        img_pred = decode(z_pred.to(vae.dtype), vae)

        # [0, 1] for loss and metrics
        img_pred_01   = (img_pred.float().clamp(-1, 1) + 1) / 2
        img_target_01 = (clean.float().clamp(-1, 1)   + 1) / 2

        loss, loss_psnr, loss_ssim = criterion(img_pred_01, img_target_01)

        # Accumulate train PSNR/SSIM on decoded predictions (same as val)
        # Done with no_grad and detach so it doesn't affect the backward pass
        with torch.no_grad():
            pred_np_tr  = img_pred_01.detach().cpu().numpy()
            clean_np_tr = img_target_01.detach().cpu().numpy()
            for b in range(pred_np_tr.shape[0]):
                p_tr = pred_np_tr[b].transpose(1, 2, 0)
                c_tr = clean_np_tr[b].transpose(1, 2, 0)
                train_psnr_acc.append(float(compute_psnr(c_tr, p_tr, data_range=1.0)))
                try:
                    train_ssim_acc.append(float(
                        compute_ssim(c_tr, p_tr, data_range=1.0, channel_axis=-1)
                    ))
                except TypeError:
                    train_ssim_acc.append(float(
                        compute_ssim(c_tr, p_tr, data_range=1.0, multichannel=True)
                    ))

        scaler.scale(loss / args.grad_accum).backward()
        grad_accum_count += 1

        if grad_accum_count % args.grad_accum == 0:
            # PAPER: global gradient norm clipping before each optimizer step
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            pbar.update(1)

            lr_now = scheduler.get_last_lr()[0]

            # Logging every log_interval steps
            if is_main() and global_step % args.log_interval == 0:
                t_psnr = float(np.mean(train_psnr_acc)) if train_psnr_acc else 0.0
                t_ssim = float(np.mean(train_ssim_acc)) if train_ssim_acc else 0.0
                train_psnr_acc.clear()
                train_ssim_acc.clear()

                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    tr_psnr=f"{t_psnr:.2f}",
                    tr_ssim=f"{t_ssim:.4f}",
                    lr=f"{lr_now:.2e}",
                )

                # Write to CSV
                # Fractional epoch: how many full passes through train data
                current_epoch = global_step / steps_per_epoch

                csv_logger.log_train(
                    step=global_step,
                    epoch=current_epoch,
                    loss=loss.item(),
                    loss_psnr=loss_psnr.item(),
                    loss_ssim=loss_ssim.item(),
                    train_psnr=t_psnr,
                    train_ssim=t_ssim,
                    lr=lr_now,
                )

                if args.wandb_project:
                    wandb.log({
                        "train/loss":      loss.item(),
                        "train/loss_psnr": loss_psnr.item(),
                        "train/loss_ssim": loss_ssim.item(),
                        "train/psnr":      t_psnr,
                        "train/ssim":      t_ssim,
                        "train/lr":        lr_now,
                    }, step=global_step)

            # Validation
            if global_step % args.val_interval == 0:
                unwrapped = transformer.module if world_size > 1 else transformer
                metrics   = validate(
                    vae, unwrapped, val_loader, device, global_step
                )
                if is_main():
                    print(
                        f"\nStep {global_step:6d}  "
                        f"PSNR={metrics['val/psnr']:.2f} dB  "
                        f"SSIM={metrics['val/ssim']:.4f}"
                    )
                    for tname in TASK_ID_TO_NAME.values():
                        pk = f"val/psnr_{tname}"
                        if pk in metrics:
                            print(
                                f"  {tname:<25}  "
                                f"PSNR={metrics[pk]:.2f}  "
                                f"SSIM={metrics[f'val/ssim_{tname}']:.4f}"
                            )

                    # Write val metrics to CSV (include epoch for easy plotting)
                    metrics["epoch"] = round(global_step / steps_per_epoch, 3)
                    csv_logger.log_val(global_step, metrics)

                    if args.wandb_project:
                        wandb.log(metrics, step=global_step)

                    # Save best checkpoint if val PSNR improved
                    best_psnr = save_best_checkpoint(
                        args.output_dir, global_step,
                        unwrapped, optimizer, scheduler,
                        metrics, best_psnr,
                    )

            # Save checkpoint
            if is_main() and global_step % args.save_interval == 0:
                unwrapped = transformer.module if world_size > 1 else transformer
                save_checkpoint(
                    args.output_dir, global_step,
                    unwrapped, optimizer, scheduler, metrics,
                )

    pbar.close()

    # Final checkpoint and validation
    if is_main():
        print(f"\nTraining complete at step {global_step}.")
        unwrapped     = transformer.module if world_size > 1 else transformer
        final_metrics = validate(
            vae, unwrapped, val_loader, device, global_step
        )
        print(
            f"Final  PSNR={final_metrics['val/psnr']:.2f} dB  "
            f"SSIM={final_metrics['val/ssim']:.4f}"
        )

        # Save final regular checkpoint
        save_checkpoint(
            args.output_dir, global_step,
            unwrapped, optimizer, scheduler, final_metrics,
        )

        # Log final val metrics to CSV
        final_metrics["epoch"] = round(global_step / steps_per_epoch, 3)
        csv_logger.log_val(global_step, final_metrics)

        # Check if final is also the best
        save_best_checkpoint(
            args.output_dir, global_step,
            unwrapped, optimizer, scheduler,
            final_metrics, best_psnr,
        )

        print(f"\nOutput files in {args.output_dir}:")
        print("  checkpoint_best.pt   ← best val PSNR checkpoint (use for inference)")
        print("  checkpoint_latest.pt ← most recent checkpoint")
        print("  train_metrics.csv    ← plot training curves")
        print("  val_metrics.csv      ← plot validation curves per task")

        if args.wandb_project:
            wandb.log(final_metrics, step=global_step)
            wandb.finish()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()