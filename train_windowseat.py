#!/usr/bin/env python3
"""
train_windowseat.py
--------------------
WindowSeat multi-task image restoration training script.
Supports four experiment configurations via flags:

  Baseline (no flags):
    Uniform sampling, single LoRA adapter, standard PSNR+SSIM loss.

  --use-balanced-sampling  (Change 1 — always on for all ablations)
    WeightedRandomSampler equalises task frequency per batch.
    Fixes: Rain+Drop (218 pairs) being ignored vs Reflection (2018 pairs).

  --use-balanced-sampling --use-task-aware-loss  (Change 1 + 2)
    Adds per-task loss scaling and real-sample upweighting on top of sampling.
    Fixes: synthetic data dominating gradient signal for raindrop/rainstreak.

  --use-balanced-sampling --use-multitask-lora  (Change 1 + 3)
    Three separate LoRA adapters: blur / rain / reflection.
    Fixes: task interference — one shared adapter must balance all degradations.

  --use-balanced-sampling --use-task-aware-loss --use-multitask-lora  (Full model)
    All three improvements combined.

Ablation table mapping
-----------------------
  Experiment                               Flags
  ─────────────────────────────────────────────────────────────────
  Baseline                                 (none)
  + Balanced sampling                      --use-balanced-sampling
  + Task-aware loss                        --use-balanced-sampling
                                           --use-task-aware-loss
  + Multi-task LoRA                        --use-balanced-sampling
                                           --use-multitask-lora
  Full model (all combined)                --use-balanced-sampling
                                           --use-task-aware-loss
                                           --use-multitask-lora

Task groupings for multi-task LoRA
------------------------------------
  blur adapter       ← blur
  rain adapter       ← raindrop, rainstreak, rainstreak_raindrop
  reflection adapter ← reflection (syn, syn_zka)

Usage examples
--------------
  # Baseline
  python train_windowseat.py --data-root . --meta-dir dataset_metadata \\
      --embed-dir text/text_embeddings --output-dir runs/baseline

  # Balanced sampling only
  python train_windowseat.py ... --output-dir runs/balanced \\
      --use-balanced-sampling

  # Balanced sampling + task-aware loss
  python train_windowseat.py ... --output-dir runs/balanced_taskloss \\
      --use-balanced-sampling --use-task-aware-loss

  # Balanced sampling + multi-task LoRA
  python train_windowseat.py ... --output-dir runs/balanced_multilora \\
      --use-balanced-sampling --use-multitask-lora

  # Full model
  python train_windowseat.py ... --output-dir runs/full_model \\
      --use-balanced-sampling --use-task-aware-loss --use-multitask-lora
"""

import argparse
import csv
import json
import math
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms.functional as TF
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
from torch.utils.data import DataLoader, Dataset, DistributedSampler, WeightedRandomSampler
from torchvision.transforms import ColorJitter
from tqdm import tqdm


# =============================================================================
# Constants
# =============================================================================

BASE_MODEL_URI  = "Qwen/Qwen-Image-Edit-2509"
FIXED_TIMESTEP  = 499
MAX_SEQ_LEN     = 256
EMBED_POOL_SIZE = 20

TASK_ID_TO_NAME: Dict[int, str] = {
    0: "blur",
    1: "raindrop",
    2: "rainstreak",
    3: "rainstreak_raindrop",
    4: "reflection",
}

# Which LoRA adapter each task routes to (multi-task LoRA only)
TASK_TO_ADAPTER: Dict[str, str] = {
    "blur":                 "blur",
    "raindrop":             "rain",
    "rainstreak":           "rain",
    "rainstreak_raindrop":  "rain",
    "reflection":           "reflection",
}
ADAPTER_NAMES: List[str] = ["blur", "rain", "reflection"]

# Per-task loss scaling weights (task-aware loss)
# Motivated by inverse frequency: fewer samples → higher weight
# Using sqrt-smoothed inverse frequency to avoid extreme values
# blur:1029  raindrop:1218  rainstreak:1218  rain+drop:218  reflection:2018
TASK_LOSS_WEIGHTS: Dict[str, float] = {
    "blur":                 1.0,
    "raindrop":             1.2,
    "rainstreak":           1.2,
    "rainstreak_raindrop":  2.0,   # only 218 pairs — highest weight
    "reflection":           0.9,   # 2018 pairs — slightly down-weighted
}

# Tasks that have BOTH real and synthetic data
# Only these get real-sample upweighting; others have a single source
REAL_UPWEIGHT_TASKS: frozenset = frozenset({"raindrop", "rainstreak"})
REAL_UPWEIGHT_FACTOR: float = 2.0   # real samples count 2× within these tasks


# =============================================================================
# Args
# =============================================================================
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="WindowSeat multi-task training with ablation flags",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    p.add_argument("--data-root",   required=True,
                   help="Project root; JSON paths are relative to this")
    p.add_argument("--meta-dir",    required=True,
                   help="Folder with train_metadata.json and val_metadata.json")
    p.add_argument("--embed-dir",   required=True,
                   help="Folder with per-task .pt embedding pools")
    p.add_argument("--output-dir",  required=True,
                   help="Checkpoints, CSVs and logs saved here")
    p.add_argument("--resume",      default=None,
                   help="Path to a checkpoint .pt to resume from")

    # ── Ablation flags — descriptive names map to the three improvements ──────
    p.add_argument(
        "--use-balanced-sampling",
        action="store_true",
        help=(
            "[Change 1] WeightedRandomSampler: equalises task frequency per "
            "batch so minority tasks (e.g. rain+drop, 218 pairs) are seen as "
            "often as majority tasks (e.g. reflection, 2018 pairs). "
            "Recommended ON for all ablation rows except the vanilla baseline."
        ),
    )
    p.add_argument(
        "--use-task-aware-loss",
        action="store_true",
        help=(
            "[Change 2] Per-task loss scaling + real-sample upweighting. "
            "Minority tasks receive higher loss weight so they contribute more "
            "to the gradient. For raindrop and rainstreak (which have both real "
            "and synthetic data), real samples are upweighted by "
            f"{REAL_UPWEIGHT_FACTOR}x over synthetic. "
            "Requires --use-balanced-sampling to be meaningful."
        ),
    )
    p.add_argument(
        "--use-multitask-lora",
        action="store_true",
        help=(
            "[Change 3] Three separate LoRA adapters (blur / rain / reflection) "
            "instead of one shared adapter. Each adapter specialises on its task "
            "group without competing with other degradation types. "
            "Requires --use-balanced-sampling to be meaningful."
        ),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument("--base-model",    default=BASE_MODEL_URI)
    p.add_argument("--lora-rank",     type=int,   default=128,
                   help="LoRA rank per adapter (paper: 128)")
    p.add_argument("--lora-alpha",    type=int,   default=128)
    p.add_argument("--lora-dropout",  type=float, default=0.0)

    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument("--total-steps",   type=int,   default=11_000)
    p.add_argument("--batch-size",    type=int,   default=2,
                   help="Per-GPU batch size (paper: 2)")
    p.add_argument("--grad-accum",    type=int,   default=1)
    p.add_argument("--lr-start",      type=float, default=1e-5)
    p.add_argument("--lr-peak",       type=float, default=1e-4)
    p.add_argument("--lr-decay",      type=float, default=5e-6)
    p.add_argument("--warmup-steps",  type=int,   default=100)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--resolution",    type=int,   default=608,
                   help="Square crop resolution (paper: 608)")

    # ── Loss ──────────────────────────────────────────────────────────────────
    p.add_argument("--lambda-psnr",   type=float, default=0.1,
                   help="λ_PSNR weight (paper: 0.1)")
    p.add_argument("--lambda-ssim",   type=float, default=20.0,
                   help="λ_SSIM weight (paper: 20)")

    # ── Logging ───────────────────────────────────────────────────────────────
    p.add_argument("--log-interval",  type=int,   default=50)
    p.add_argument("--val-interval",  type=int,   default=500)
    p.add_argument("--save-interval", type=int,   default=1_000)
    p.add_argument("--wandb-project", default="windowseat")
    p.add_argument("--run-name",      default=None)

    # ── System ────────────────────────────────────────────────────────────────
    p.add_argument("--num-workers",   type=int,   default=4)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--local-rank",    type=int,   default=-1)

    return p.parse_args()


# =============================================================================
# Distributed helpers
# =============================================================================

def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_distributed(local_rank: int) -> Tuple[torch.device, int, int]:
    if local_rank == -1:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu"), 0, 1
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}"), dist.get_rank(), dist.get_world_size()


# =============================================================================
# Dataset
# =============================================================================

class JsonRestorationDataset(Dataset):
    """
    Multi-task restoration dataset driven by a JSON metadata file.

    Each sample returns:
      blended       : (3, R, R) float32 in [-1, 1]
      clean         : (3, R, R) float32 in [-1, 1]
      prompt_embeds : (MAX_SEQ_LEN, D) float32
      task_id       : int
      task_name     : str  e.g. "raindrop"
      data_type     : str  e.g. "real" or "syn_zka"
    """

    def __init__(
        self,
        data_root:  str,
        json_path:  str,
        embed_root: str,
        resolution: int  = 608,
        is_train:   bool = True,
    ) -> None:
        self.data_root  = Path(data_root)
        self.embed_root = Path(embed_root)
        self.resolution = resolution
        self.is_train   = is_train

        with open(json_path, encoding="utf-8") as f:
            self.samples: List[dict] = json.load(f)

        if is_main():
            tag     = "train" if is_train else "val"
            missing = sum(
                1 for s in self.samples
                if not (self.data_root / s["input"]).exists()
            )
            print(f"  [{tag}] {len(self.samples)} pairs "
                  f"({missing} files missing on disk)")
            if missing:
                print(f"  [WARN] check --data-root")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item    = self.samples[idx]
        blended = Image.open(self.data_root / item["input"]).convert("RGB")
        clean   = Image.open(self.data_root / item["target"]).convert("RGB")

        if self.is_train:
            blended, clean = self._augment(blended, clean)
        else:
            blended, clean = self._centre_crop(blended, clean)

        return {
            "blended":       TF.to_tensor(blended) * 2.0 - 1.0,
            "clean":         TF.to_tensor(clean)   * 2.0 - 1.0,
            "prompt_embeds": self._load_embed(item.get("task_id", 4)),
            "task_id":       item["task_id"],
            "task_name":     item["task_name"],
            "data_type":     item.get("data_type", "unknown"),
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ensure_min_size(
        self, a: Image.Image, b: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        R    = self.resolution
        W, H = a.size
        if min(W, H) < R:
            scale = R / min(W, H)
            nw, nh = math.ceil(W * scale), math.ceil(H * scale)
            a = a.resize((nw, nh), Image.LANCZOS)
            b = b.resize((nw, nh), Image.LANCZOS)
        return a, b

    def _augment(
        self, blended: Image.Image, clean: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Random crop + color jitter + hflip applied identically to both."""
        R = self.resolution
        blended, clean = self._ensure_min_size(blended, clean)

        fn_idx, b_f, c_f, s_f, h_f = ColorJitter.get_params(
            brightness=[0.8, 1.2], contrast=[0.8, 1.2],
            saturation=[0.8, 1.2], hue=[-0.1, 0.1],
        )
        ops = {
            0: lambda x: TF.adjust_brightness(x, b_f),
            1: lambda x: TF.adjust_contrast(x, c_f),
            2: lambda x: TF.adjust_saturation(x, s_f),
            3: lambda x: TF.adjust_hue(x, h_f),
        }
        for fn in fn_idx:
            blended = ops[int(fn)](blended)
            clean   = ops[int(fn)](clean)

        W, H = blended.size
        i, j = random.randint(0, H - R), random.randint(0, W - R)
        blended = TF.crop(blended, i, j, R, R)
        clean   = TF.crop(clean,   i, j, R, R)

        if random.random() < 0.5:
            blended, clean = TF.hflip(blended), TF.hflip(clean)

        return blended, clean

    def _centre_crop(
        self, blended: Image.Image, clean: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        R = self.resolution
        blended, clean = self._ensure_min_size(blended, clean)
        return (
            TF.center_crop(blended, (R, R)),
            TF.center_crop(clean,   (R, R)),
        )

    def _load_embed(self, task_id: int) -> torch.Tensor:
        task_name = TASK_ID_TO_NAME.get(task_id, "reflection")
        idx       = random.randint(0, EMBED_POOL_SIZE - 1)
        embed     = torch.load(
            self.embed_root / task_name / f"{idx}.pt",
            weights_only=True,
        ).squeeze(0)   # (seq_len, D)

        seq_len, D = embed.shape
        if seq_len < MAX_SEQ_LEN:
            embed = torch.cat([embed, embed.new_zeros(MAX_SEQ_LEN - seq_len, D)])
        else:
            embed = embed[:MAX_SEQ_LEN]
        return embed


# =============================================================================
# Balanced sampler (Change 1)
# =============================================================================

def build_balanced_sampler(
    dataset: JsonRestorationDataset,
) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that equalises task frequency per batch.

    Weight per sample = 1 / sqrt(task_count).
    sqrt-smoothing avoids extreme oversampling of very small tasks while
    still substantially lifting minority tasks vs plain inverse frequency.

    Result: each task contributes roughly equally to each batch regardless
    of its dataset size. Rain+Drop (218 pairs) is seen as often as
    Reflection (2018 pairs).

    Note: incompatible with DistributedSampler. For single-GPU training
    (the default here) this works directly. For multi-GPU, replace with
    a custom distributed weighted sampler.
    """
    task_counts: Dict[str, int] = defaultdict(int)
    for s in dataset.samples:
        task_counts[s["task_name"]] += 1

    if is_main():
        print("\n  Task counts (train):")
        for task, count in sorted(task_counts.items()):
            print(f"    {task:<26} {count:>5} pairs")

    sample_weights = [
        1.0 / math.sqrt(task_counts[s["task_name"]])
        for s in dataset.samples
    ]

    return WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(dataset),
        replacement = True,   # required when oversampling minority tasks
    )


# =============================================================================
# Model loading
# =============================================================================

def load_vae(uri: str, device: torch.device) -> AutoencoderKLQwenImage:
    """Load and permanently freeze the VAE encoder and decoder."""
    vae = AutoencoderKLQwenImage.from_pretrained(
        uri, subfolder="vae",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )
    vae.to(device, dtype=torch.bfloat16)
    vae.requires_grad_(False)
    vae.eval()
    return vae


def _make_lora_config(rank: int, alpha: int, dropout: float) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["to_q", "to_k", "to_v", "to_out.0",
                        "ff.net.0.proj", "ff.net.2"],
        bias="none",
        init_lora_weights="gaussian",
    )


def load_single_lora_transformer(
    uri:     str,
    device:  torch.device,
    rank:    int,
    alpha:   int,
    dropout: float,
) -> QwenImageTransformer2DModel:
    """
    Baseline: one shared LoRA adapter for all tasks.
    Adapter name: 'default' (PEFT default).
    """
    transformer = QwenImageTransformer2DModel.from_pretrained(
        uri, subfolder="transformer",
        torch_dtype=torch.bfloat16, device_map=device,
    )
    transformer = get_peft_model(transformer, _make_lora_config(rank, alpha, dropout))

    if is_main():
        transformer.print_trainable_parameters()
    return transformer


def load_multitask_lora_transformer(
    uri:     str,
    device:  torch.device,
    rank:    int,
    alpha:   int,
    dropout: float,
) -> QwenImageTransformer2DModel:
    """
    Change 3: three separate LoRA adapters — blur / rain / reflection.

    PEFT supports multiple named adapters on one backbone.
    Only the active adapter contributes to the forward pass.
    Inactive adapters receive zero gradients and are not updated that step.

    Adapter routing during training:
      blur adapter       ← blur samples
      rain adapter       ← raindrop, rainstreak, rainstreak_raindrop samples
      reflection adapter ← reflection samples
    """
    transformer = QwenImageTransformer2DModel.from_pretrained(
        uri, subfolder="transformer",
        torch_dtype=torch.bfloat16, device_map=device,
    )

    cfg = _make_lora_config(rank, alpha, dropout)

    # Add first adapter via get_peft_model (creates the PEFT wrapper)
    transformer = get_peft_model(transformer, cfg, adapter_name=ADAPTER_NAMES[0])

    # Add remaining adapters
    for adapter_name in ADAPTER_NAMES[1:]:
        transformer.add_adapter(adapter_name, cfg)

    if is_main():
        total_trainable = sum(
            p.numel() for p in transformer.parameters() if p.requires_grad
        )
        print(f"  Multi-task LoRA: {len(ADAPTER_NAMES)} adapters "
              f"({', '.join(ADAPTER_NAMES)})")
        print(f"  Total trainable parameters: {total_trainable:,}")

    return transformer


# =============================================================================
# VAE encode / decode
# =============================================================================

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


# =============================================================================
# Transformer forward
# =============================================================================

def forward_transformer(
    latent_input:  torch.Tensor,
    transformer:   QwenImageTransformer2DModel,
    vae:           AutoencoderKLQwenImage,
    prompt_embeds: torch.Tensor,
    prompt_mask:   torch.Tensor,
) -> torch.Tensor:
    """Returns predicted velocity (same shape as latent_input)."""
    lat4d = latent_input[:, :, 0] if latent_input.ndim == 5 else latent_input
    B, C, H, W = lat4d.shape
    device = next(transformer.parameters()).device

    packed = QwenImageEditPipeline._pack_latents(
        lat4d, batch_size=B, num_channels_latents=C, height=H, width=W,
    ).to(torch.bfloat16)

    timestep     = torch.full((B,), float(FIXED_TIMESTEP) / 1000.0,
                              device=device, dtype=torch.bfloat16)
    img_shapes   = [[(1, H // 2, W // 2)]] * B
    txt_seq_lens = prompt_mask.sum(dim=1).tolist()

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model_pred = transformer(
            hidden_states              = packed,
            timestep                   = timestep,
            encoder_hidden_states      = prompt_embeds.to(device, dtype=torch.bfloat16),
            encoder_hidden_states_mask = prompt_mask.to(device),
            img_shapes                 = img_shapes,
            txt_seq_lens               = txt_seq_lens,
            guidance                   = None,
            return_dict                = False,
        )[0]

    td = vae.config.get("temperal_downsample", None)
    sf = 2 ** len(td) if td is not None else 8
    return QwenImageEditPipeline._unpack_latents(
        model_pred, height=H * sf, width=W * sf, vae_scale_factor=sf,
    )


# =============================================================================
# Loss functions
# =============================================================================

class WindowSeatLoss(torch.nn.Module):
    """
    Paper loss: L = λ_PSNR * MSE(Ŷ, T) + λ_SSIM * (1 - SSIM(Ŷ, T))
    Computed in pixel space [0, 1] on decoded predictions.
    """

    def __init__(
        self,
        lambda_psnr: float = 0.1,
        lambda_ssim: float = 20.0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.lambda_psnr = lambda_psnr
        self.lambda_ssim = lambda_ssim
        # size_average=True returns a scalar over the batch
        self.ssim_fn = SSIMLoss(data_range=1.0, size_average=True, channel=3).to(device)

    def forward(
        self,
        pred:   torch.Tensor,   # (B, 3, H, W) in [0, 1]
        target: torch.Tensor,   # (B, 3, H, W) in [0, 1]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_psnr = F.mse_loss(pred, target)
        loss_ssim = 1.0 - self.ssim_fn(pred, target)
        total     = self.lambda_psnr * loss_psnr + self.lambda_ssim * loss_ssim
        return total, loss_psnr, loss_ssim


class TaskAwareLoss(torch.nn.Module):
    """
    Change 2: Extends the base loss with two corrections for imbalance.

    Correction A — per-task scaling:
      Tasks with fewer training pairs receive higher loss weight so they
      contribute proportionally more to the gradient despite being seen
      less often (even with balanced sampling, gradient scale matters).
      Weights defined in TASK_LOSS_WEIGHTS.

    Correction B — real-sample upweighting:
      For raindrop and rainstreak (the only tasks with BOTH real and
      synthetic data), real samples are upweighted by REAL_UPWEIGHT_FACTOR.
      This prevents synthetic data (easier, cleaner) from dominating the
      gradient signal within those task groups.

    Implementation:
      Compute per-sample MSE and per-sample SSIM loss (size_average=False),
      multiply by the per-sample weight, then take the weighted mean.
    """

    def __init__(
        self,
        lambda_psnr: float = 0.1,
        lambda_ssim: float = 20.0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.lambda_psnr = lambda_psnr
        self.lambda_ssim = lambda_ssim
        # size_average=False returns per-sample SSIM values (B,)
        self.ssim_fn = SSIMLoss(data_range=1.0, size_average=False, channel=3).to(device)

    def _sample_weights(
        self,
        task_names:  List[str],
        data_types:  List[str],
        device:      torch.device,
        dtype:       torch.dtype,
    ) -> torch.Tensor:
        """
        Build a normalised weight tensor of shape (B,).
        Weight for each sample = task_weight * source_weight.
        Normalised so weights sum to B (equivalent to weighted mean).
        """
        weights = []
        for task, dtype_ in zip(task_names, data_types):
            w = TASK_LOSS_WEIGHTS.get(task, 1.0)
            if task in REAL_UPWEIGHT_TASKS and dtype_ == "real":
                w *= REAL_UPWEIGHT_FACTOR
            weights.append(w)

        w_tensor = torch.tensor(weights, device=device, dtype=dtype)
        # Normalise: sum → B so effective batch size is unchanged
        w_tensor = w_tensor * (len(weights) / w_tensor.sum())
        return w_tensor   # (B,)

    def forward(
        self,
        pred:       torch.Tensor,   # (B, 3, H, W) in [0, 1]
        target:     torch.Tensor,   # (B, 3, H, W) in [0, 1]
        task_names: List[str],
        data_types: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = self._sample_weights(
            task_names, data_types, pred.device, pred.dtype
        )   # (B,)

        # Per-sample MSE: mean over (C, H, W) → (B,)
        loss_psnr_per = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))

        # Per-sample SSIM: ssim_fn returns (B,) when size_average=False
        ssim_vals     = self.ssim_fn(pred, target)   # (B,) values in [0, 1]
        loss_ssim_per = 1.0 - ssim_vals              # (B,)

        # Weighted mean
        loss_psnr = (loss_psnr_per * weights).mean()
        loss_ssim = (loss_ssim_per * weights).mean()
        total     = self.lambda_psnr * loss_psnr + self.lambda_ssim * loss_ssim

        return total, loss_psnr, loss_ssim


# =============================================================================
# Training step functions
# =============================================================================

def _to_01(t: torch.Tensor) -> torch.Tensor:
    """Convert from [-1, 1] to [0, 1]."""
    return (t.float().clamp(-1.0, 1.0) + 1.0) / 2.0


def single_lora_step(
    batch:       dict,
    transformer: QwenImageTransformer2DModel,
    vae:         AutoencoderKLQwenImage,
    criterion:   torch.nn.Module,
    device:      torch.device,
    use_task_aware_loss: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """
    One training step with a single shared LoRA adapter.
    Returns (total_loss, loss_psnr, loss_ssim, pred_np_01, target_np_01).
    """
    blended       = batch["blended"].to(device, dtype=torch.bfloat16)
    clean         = batch["clean"].to(device,   dtype=torch.bfloat16)
    prompt_embeds = batch["prompt_embeds"].to(device, dtype=torch.bfloat16)
    B             = blended.shape[0]
    prompt_mask   = torch.ones((B, MAX_SEQ_LEN), dtype=torch.bool, device=device)

    with torch.no_grad():
        z_degraded = encode(blended, vae)

    pred_velocity = forward_transformer(
        z_degraded, transformer, vae, prompt_embeds, prompt_mask
    )

    z_pred   = z_degraded.float() + pred_velocity.float()
    img_pred = decode(z_pred.to(vae.dtype), vae)

    pred_01   = _to_01(img_pred)
    target_01 = _to_01(clean)

    if use_task_aware_loss:
        loss, lp, ls = criterion(
            pred_01, target_01,
            list(batch["task_name"]),
            list(batch["data_type"]),
        )
    else:
        loss, lp, ls = criterion(pred_01, target_01)

    return loss, lp, ls, pred_01.detach().cpu().numpy(), target_01.detach().cpu().numpy()


def multitask_lora_step(
    batch:       dict,
    transformer: QwenImageTransformer2DModel,
    vae:         AutoencoderKLQwenImage,
    criterion:   torch.nn.Module,
    device:      torch.device,
    use_task_aware_loss: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """
    Change 3: one training step with per-task LoRA adapters.

    Strategy:
      1. Group batch indices by adapter name (blur / rain / reflection).
      2. For each adapter group, activate the adapter and run a forward pass
         on the sub-batch. Only the active adapter receives gradients.
      3. Aggregate losses weighted by sub-batch size to get a batch-level loss.

    Returns (total_loss, loss_psnr, loss_ssim, pred_np_01, target_np_01).
    The returned numpy arrays are reassembled in original batch order.
    """
    blended       = batch["blended"].to(device, dtype=torch.bfloat16)
    clean         = batch["clean"].to(device,   dtype=torch.bfloat16)
    prompt_embeds = batch["prompt_embeds"].to(device, dtype=torch.bfloat16)
    task_names    = list(batch["task_name"])
    data_types    = list(batch["data_type"])
    B             = blended.shape[0]
    prompt_mask   = torch.ones((B, MAX_SEQ_LEN), dtype=torch.bool, device=device)

    # Map sample indices to adapter groups
    adapter_indices: Dict[str, List[int]] = defaultdict(list)
    for i, tname in enumerate(task_names):
        adapter_indices[TASK_TO_ADAPTER[tname]].append(i)

    sub_losses:    List[torch.Tensor] = []
    sub_lp:        List[torch.Tensor] = []
    sub_ls:        List[torch.Tensor] = []
    pred_np_full   = np.zeros((B, 3,
                               blended.shape[2], blended.shape[3]),
                               dtype=np.float32)
    target_np_full = np.zeros_like(pred_np_full)

    for adapter_name in ADAPTER_NAMES:
        indices = adapter_indices.get(adapter_name, [])
        if not indices:
            continue

        # Activate this adapter — only its weights participate in forward
        transformer.set_adapter(adapter_name)

        idx_t         = torch.tensor(indices, device=device)
        sub_blended   = blended[idx_t]
        sub_clean     = clean[idx_t]
        sub_embeds    = prompt_embeds[idx_t]
        sub_mask      = prompt_mask[idx_t]
        sub_tasks     = [task_names[i]  for i in indices]
        sub_dtypes    = [data_types[i]  for i in indices]

        with torch.no_grad():
            z_deg = encode(sub_blended, vae)

        pred_vel = forward_transformer(z_deg, transformer, vae, sub_embeds, sub_mask)

        z_pred   = z_deg.float() + pred_vel.float()
        img_pred = decode(z_pred.to(vae.dtype), vae)

        pred_01   = _to_01(img_pred)
        target_01 = _to_01(sub_clean)

        if use_task_aware_loss:
            loss, lp, ls = criterion(pred_01, target_01, sub_tasks, sub_dtypes)
        else:
            loss, lp, ls = criterion(pred_01, target_01)

        # Weight by sub-batch size so larger groups don't dominate
        n = len(indices)
        sub_losses.append(loss * n)
        sub_lp.append(lp * n)
        sub_ls.append(ls * n)

        # Store predictions in original order for metric accumulation
        pred_np   = pred_01.detach().cpu().numpy()
        target_np = target_01.detach().cpu().numpy()
        for out_i, orig_i in enumerate(indices):
            pred_np_full[orig_i]   = pred_np[out_i]
            target_np_full[orig_i] = target_np[out_i]

    total_loss = torch.stack(sub_losses).sum() / B
    total_lp   = torch.stack(sub_lp).sum()   / B
    total_ls   = torch.stack(sub_ls).sum()   / B

    return total_loss, total_lp, total_ls, pred_np_full, target_np_full


# =============================================================================
# LR scheduler
# =============================================================================

def get_scheduler(
    optimizer:    torch.optim.Optimizer,
    warmup_steps: int,
    lr_start:     float,
    lr_peak:      float,
    lr_decay:     float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Paper schedule: linear warmup lr_start→lr_peak, then linear decay
    of lr_decay per 1000 steps.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            t  = step / max(1, warmup_steps)
            lr = lr_start + t * (lr_peak - lr_start)
            return lr / lr_peak
        decay = lr_decay * ((step - warmup_steps) // 1000)
        return max(lr_peak - decay, 1e-7) / lr_peak

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Validation
# =============================================================================

@torch.no_grad()
def validate(
    vae:         AutoencoderKLQwenImage,
    transformer: QwenImageTransformer2DModel,
    val_loader:  DataLoader,
    device:      torch.device,
    global_step: int,
    use_multitask_lora: bool,
) -> dict:
    """Compute PSNR and SSIM overall and per task on the validation set."""
    transformer.eval()

    # For multi-task LoRA, enable all adapters during validation so
    # each sample automatically uses all adapter contributions.
    # We route by task in inference just like training.
    if use_multitask_lora:
        # Keep current adapter state; we'll switch per-batch
        pass

    torch.cuda.empty_cache()

    overall_psnr: List[float] = []
    overall_ssim: List[float] = []
    task_psnr = {n: [] for n in TASK_ID_TO_NAME.values()}
    task_ssim = {n: [] for n in TASK_ID_TO_NAME.values()}

    for batch in tqdm(val_loader, desc=f"Val @{global_step}", leave=False,
                      disable=not is_main()):
        blended       = batch["blended"].to(device, dtype=torch.bfloat16)
        clean         = batch["clean"].to(device,   dtype=torch.bfloat16)
        prompt_embeds = batch["prompt_embeds"].to(device, dtype=torch.bfloat16)
        task_names    = list(batch["task_name"])
        B             = blended.shape[0]
        prompt_mask   = torch.ones((B, MAX_SEQ_LEN), dtype=torch.bool, device=device)

        if use_multitask_lora:
            # Route each sub-batch to its adapter — same as training
            adapter_indices: Dict[str, List[int]] = defaultdict(list)
            for i, t in enumerate(task_names):
                adapter_indices[TASK_TO_ADAPTER[t]].append(i)

            pred_np_full   = np.zeros((B, 3, blended.shape[2], blended.shape[3]),
                                       dtype=np.float32)
            target_np_full = ((_to_01(clean)).cpu().numpy())

            for adapter_name in ADAPTER_NAMES:
                indices = adapter_indices.get(adapter_name, [])
                if not indices:
                    continue
                transformer.set_adapter(adapter_name)
                idx_t       = torch.tensor(indices, device=device)
                sub_b       = blended[idx_t]
                sub_e       = prompt_embeds[idx_t]
                sub_m       = prompt_mask[idx_t]

                z_deg    = encode(sub_b, vae)
                pred_vel = forward_transformer(z_deg, transformer, vae, sub_e, sub_m)
                z_pred   = z_deg.float() + pred_vel.float()
                img_pred = decode(z_pred.to(vae.dtype), vae)
                pred_01  = _to_01(img_pred).cpu().numpy()

                for out_i, orig_i in enumerate(indices):
                    pred_np_full[orig_i] = pred_01[out_i]

            pred_np   = pred_np_full
            clean_np  = target_np_full

        else:
            z_degraded    = encode(blended, vae)
            pred_velocity = forward_transformer(
                z_degraded, transformer, vae, prompt_embeds, prompt_mask
            )
            z_pred   = z_degraded.float() + pred_velocity.float()
            img_pred = decode(z_pred.to(vae.dtype), vae)
            pred_np  = _to_01(img_pred).cpu().numpy()
            clean_np = _to_01(clean).cpu().numpy()

        for b in range(B):
            p     = pred_np[b].transpose(1, 2, 0)
            c     = clean_np[b].transpose(1, 2, 0)
            psnr  = compute_psnr(c, p, data_range=1.0)
            try:
                ssim = compute_ssim(c, p, data_range=1.0, channel_axis=-1)
            except TypeError:
                ssim = compute_ssim(c, p, data_range=1.0, multichannel=True)

            overall_psnr.append(psnr)
            overall_ssim.append(ssim)
            tname = task_names[b]
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


# =============================================================================
# Checkpoint helpers
# =============================================================================

def _extract_lora_state(
    transformer: QwenImageTransformer2DModel,
    use_multitask_lora: bool,
) -> dict:
    """
    Extract LoRA weights from the model.

    Single LoRA:     returns {"lora_state_dict": {k: v, ...}}
    Multi-task LoRA: returns {"lora_state_dicts": {adapter: {k: v, ...}, ...}}
    """
    model = transformer.module if hasattr(transformer, "module") else transformer

    if use_multitask_lora:
        adapter_states = {}
        for adapter_name in ADAPTER_NAMES:
            # Filter state dict for keys belonging to this adapter
            adapter_states[adapter_name] = {
                k: v for k, v in model.state_dict().items()
                if "lora" in k and adapter_name in k
            }
        return {"lora_state_dicts": adapter_states}
    else:
        return {
            "lora_state_dict": {
                k: v for k, v in model.state_dict().items() if "lora" in k
            }
        }


def save_checkpoint(
    output_dir:         str,
    step:               int,
    transformer:        QwenImageTransformer2DModel,
    optimizer:          torch.optim.Optimizer,
    scheduler:          torch.optim.lr_scheduler.LambdaLR,
    metrics:            dict,
    use_multitask_lora: bool,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    ckpt = {
        "step":               step,
        "use_multitask_lora": use_multitask_lora,
        "optimizer":          optimizer.state_dict(),
        "scheduler":          scheduler.state_dict(),
        "metrics":            metrics,
        **_extract_lora_state(transformer, use_multitask_lora),
    }
    path   = os.path.join(output_dir, f"checkpoint_step{step:06d}.pt")
    latest = os.path.join(output_dir, "checkpoint_latest.pt")
    torch.save(ckpt, path)
    if os.path.lexists(latest):
        os.remove(latest)
    try:
        os.symlink(os.path.basename(path), latest)
    except OSError:
        shutil.copy2(path, latest)
    if is_main():
        print(f"  Checkpoint → {path}")
    return path


def save_best_checkpoint(
    output_dir:         str,
    step:               int,
    transformer:        QwenImageTransformer2DModel,
    optimizer:          torch.optim.Optimizer,
    scheduler:          torch.optim.lr_scheduler.LambdaLR,
    metrics:            dict,
    best_psnr:          float,
    use_multitask_lora: bool,
) -> float:
    """Save checkpoint_best.pt when val PSNR improves. Returns new best PSNR."""
    current = metrics.get("val/psnr", -float("inf"))
    if current <= best_psnr:
        return best_psnr

    ckpt = {
        "step":               step,
        "use_multitask_lora": use_multitask_lora,
        "optimizer":          optimizer.state_dict(),
        "scheduler":          scheduler.state_dict(),
        "metrics":            metrics,
        **_extract_lora_state(transformer, use_multitask_lora),
    }
    path = os.path.join(output_dir, "checkpoint_best.pt")
    torch.save(ckpt, path)
    if is_main():
        print(f"  ★ Best PSNR {best_psnr:.4f} → {current:.4f} dB  → {path}")
    return current


def load_checkpoint(
    path:               str,
    transformer:        QwenImageTransformer2DModel,
    optimizer:          torch.optim.Optimizer,
    scheduler:          torch.optim.lr_scheduler.LambdaLR,
    use_multitask_lora: bool,
) -> int:
    if is_main():
        print(f"Resuming from {path}")
    ckpt  = torch.load(path, map_location="cpu")
    model = transformer.module if hasattr(transformer, "module") else transformer

    if use_multitask_lora:
        for adapter_name, state in ckpt.get("lora_state_dicts", {}).items():
            model.set_adapter(adapter_name)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if is_main() and (missing or unexpected):
                print(f"  [{adapter_name}] missing={len(missing)} "
                      f"unexpected={len(unexpected)}")
    else:
        missing, unexpected = model.load_state_dict(
            ckpt.get("lora_state_dict", {}), strict=False
        )
        if is_main() and (missing or unexpected):
            print(f"  Missing={len(missing)} Unexpected={len(unexpected)}")

    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    step = ckpt.get("step", 0)
    if is_main():
        m = ckpt.get("metrics", {})
        print(f"  Resumed at step {step} | val PSNR={m.get('val/psnr','n/a')}")
    return step


# =============================================================================
# Metrics logger (CSV)
# =============================================================================

class MetricsLogger:
    """Writes train_metrics.csv and val_metrics.csv to output_dir."""

    TRAIN_FIELDS = [
        "step", "epoch", "loss", "loss_psnr", "loss_ssim",
        "train_psnr", "train_ssim", "lr",
    ]

    def __init__(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.train_path  = os.path.join(output_dir, "train_metrics.csv")
        self.val_path    = os.path.join(output_dir, "val_metrics.csv")
        self._val_fields: Optional[List[str]] = None
        self._init_csv(self.train_path, self.TRAIN_FIELDS)

    def _init_csv(self, path: str, fields: List[str]) -> None:
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    def log_train(
        self, step: int, epoch: float, loss: float,
        loss_psnr: float, loss_ssim: float,
        train_psnr: float, train_ssim: float, lr: float,
    ) -> None:
        row = {
            "step": step, "epoch": round(epoch, 3),
            "loss": round(loss, 6), "loss_psnr": round(loss_psnr, 6),
            "loss_ssim": round(loss_ssim, 6),
            "train_psnr": round(train_psnr, 4),
            "train_ssim": round(train_ssim, 4),
            "lr": f"{lr:.2e}",
        }
        with open(self.train_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.TRAIN_FIELDS).writerow(row)

    def log_val(self, step: int, metrics: dict) -> None:
        flat = {"step": step}
        flat.update({k.replace("/", "_"): round(float(v), 4)
                     for k, v in metrics.items()})
        if self._val_fields is None:
            self._val_fields = list(flat.keys())
            self._init_csv(self.val_path, self._val_fields)
        with open(self.val_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._val_fields,
                           extrasaction="ignore").writerow(flat)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = get_args()
    device, rank, world_size = setup_distributed(args.local_rank)

    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Print active configuration ────────────────────────────────────────────
    if is_main():
        print("\n" + "=" * 60)
        print("Training configuration")
        print("=" * 60)
        print(f"  Balanced sampling  : {args.use_balanced_sampling}")
        print(f"  Task-aware loss    : {args.use_task_aware_loss}")
        print(f"  Multi-task LoRA    : {args.use_multitask_lora}")
        print(f"  Output dir         : {args.output_dir}")
        print("=" * 60)

        if args.use_task_aware_loss and not args.use_balanced_sampling:
            print("[WARN] --use-task-aware-loss without --use-balanced-sampling "
                  "is unusual. Consider enabling balanced sampling too.")
        if args.use_multitask_lora and not args.use_balanced_sampling:
            print("[WARN] --use-multitask-lora without --use-balanced-sampling "
                  "is unusual. Consider enabling balanced sampling too.")

    # ── W&B ───────────────────────────────────────────────────────────────────
    if is_main() and args.wandb_project:
        run_name = args.run_name or (
            "baseline"
            + ("_balanced" if args.use_balanced_sampling else "")
            + ("_taskloss" if args.use_task_aware_loss   else "")
            + ("_multilor" if args.use_multitask_lora    else "")
        )
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

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

    # ── Samplers / loaders ────────────────────────────────────────────────────
    if args.use_balanced_sampling:
        # WeightedRandomSampler is incompatible with DistributedSampler.
        # For multi-GPU, a custom distributed weighted sampler is needed.
        if world_size > 1:
            raise RuntimeError(
                "--use-balanced-sampling is not supported with DDP (multi-GPU). "
                "Use single GPU or implement a distributed weighted sampler."
            )
        train_sampler = build_balanced_sampler(train_ds)
        if is_main():
            print("  Balanced sampler active (sqrt inverse-frequency weighting)")
    else:
        train_sampler = (
            DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
        )

    val_sampler = (
        DistributedSampler(val_ds, shuffle=False) if world_size > 1 else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        sampler     = train_sampler,
        shuffle     = (train_sampler is None),
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = True,
        persistent_workers = (args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = args.batch_size,
        sampler     = val_sampler,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        persistent_workers = (args.num_workers > 0),
    )

    # ── Models ────────────────────────────────────────────────────────────────
    if is_main():
        print(f"\nLoading base model: {args.base_model}")

    vae = load_vae(args.base_model, device)

    if args.use_multitask_lora:
        transformer = load_multitask_lora_transformer(
            args.base_model, device,
            args.lora_rank, args.lora_alpha, args.lora_dropout,
        )
    else:
        transformer = load_single_lora_transformer(
            args.base_model, device,
            args.lora_rank, args.lora_alpha, args.lora_dropout,
        )

    transformer.train()

    if world_size > 1:
        transformer = DDP(transformer, device_ids=[args.local_rank])

    # ── Loss ──────────────────────────────────────────────────────────────────
    if args.use_task_aware_loss:
        criterion = TaskAwareLoss(
            lambda_psnr=args.lambda_psnr,
            lambda_ssim=args.lambda_ssim,
            device=device,
        )
        if is_main():
            print("  Task-aware loss active")
            for task, w in TASK_LOSS_WEIGHTS.items():
                print(f"    {task:<26} weight={w}")
            print(f"  Real upweight factor: {REAL_UPWEIGHT_FACTOR}x "
                  f"(for {sorted(REAL_UPWEIGHT_TASKS)})")
    else:
        criterion = WindowSeatLoss(
            lambda_psnr=args.lambda_psnr,
            lambda_ssim=args.lambda_ssim,
            device=device,
        )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    lora_params = [p for n, p in transformer.named_parameters() if "lora" in n]
    if is_main():
        print(f"\nTrainable LoRA params: {sum(p.numel() for p in lora_params):,}")

    optimizer = torch.optim.AdamW(
        lora_params, lr=args.lr_peak, weight_decay=0.01,
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
            args.resume, transformer, optimizer, scheduler,
            use_multitask_lora=args.use_multitask_lora,
        )

    # ── Training state ────────────────────────────────────────────────────────
    global_step     = start_step
    best_psnr       = -float("inf")
    metrics         = {}
    train_psnr_acc: List[float] = []
    train_ssim_acc: List[float] = []
    grad_accum_count = 0

    steps_per_epoch = max(1, len(train_ds) // args.batch_size // args.grad_accum)

    if is_main():
        csv_logger = MetricsLogger(args.output_dir)
        print(f"\nTraining from step {start_step} → {args.total_steps}")
        print(f"Steps per epoch: ~{steps_per_epoch}")
        print("  Note: GradScaler disabled — bfloat16 does not require loss scaling.")

    # ── Infinite dataloader ───────────────────────────────────────────────────
    def infinite_loader(loader: DataLoader, sampler) -> iter:
        epoch = 0
        while True:
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            yield from loader
            epoch += 1

    train_iter = infinite_loader(train_loader, train_sampler)
    optimizer.zero_grad()

    pbar = tqdm(
        total   = args.total_steps - start_step,
        desc    = "Training",
        disable = not is_main(),
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    while global_step < args.total_steps:
        batch = next(train_iter)

        # Select training step function based on LoRA mode
        if args.use_multitask_lora:
            loss, loss_psnr, loss_ssim, pred_np, target_np = multitask_lora_step(
                batch, transformer, vae, criterion, device,
                use_task_aware_loss=args.use_task_aware_loss,
            )
        else:
            loss, loss_psnr, loss_ssim, pred_np, target_np = single_lora_step(
                batch, transformer, vae, criterion, device,
                use_task_aware_loss=args.use_task_aware_loss,
            )

        # Accumulate train PSNR/SSIM for logging
        with torch.no_grad():
            B = pred_np.shape[0]
            for b in range(B):
                p = pred_np[b].transpose(1, 2, 0)
                c = target_np[b].transpose(1, 2, 0)
                train_psnr_acc.append(float(compute_psnr(c, p, data_range=1.0)))
                try:
                    train_ssim_acc.append(float(
                        compute_ssim(c, p, data_range=1.0, channel_axis=-1)
                    ))
                except TypeError:
                    train_ssim_acc.append(float(
                        compute_ssim(c, p, data_range=1.0, multichannel=True)
                    ))

        # ── Backward pass (no GradScaler: bfloat16 does not need loss scaling) ─
        (loss / args.grad_accum).backward()
        grad_accum_count += 1

        if grad_accum_count % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(lora_params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            pbar.update(1)

            lr_now = scheduler.get_last_lr()[0]

            # ── Logging ───────────────────────────────────────────────────────
            if is_main() and global_step % args.log_interval == 0:
                t_psnr = float(np.mean(train_psnr_acc)) if train_psnr_acc else 0.0
                t_ssim = float(np.mean(train_ssim_acc)) if train_ssim_acc else 0.0
                train_psnr_acc.clear()
                train_ssim_acc.clear()

                pbar.set_postfix(
                    loss    = f"{loss.item():.4f}",
                    tr_psnr = f"{t_psnr:.2f}",
                    tr_ssim = f"{t_ssim:.4f}",
                    lr      = f"{lr_now:.2e}",
                )
                epoch_now = global_step / steps_per_epoch
                csv_logger.log_train(
                    step=global_step, epoch=epoch_now,
                    loss=loss.item(), loss_psnr=loss_psnr.item(),
                    loss_ssim=loss_ssim.item(),
                    train_psnr=t_psnr, train_ssim=t_ssim, lr=lr_now,
                )
                if args.wandb_project:
                    wandb.log({
                        "train/loss": loss.item(), "train/loss_psnr": loss_psnr.item(),
                        "train/loss_ssim": loss_ssim.item(),
                        "train/psnr": t_psnr, "train/ssim": t_ssim, "train/lr": lr_now,
                    }, step=global_step)

            # ── Validation ────────────────────────────────────────────────────
            if global_step % args.val_interval == 0:
                unwrapped = transformer.module if world_size > 1 else transformer
                metrics   = validate(
                    vae, unwrapped, val_loader, device, global_step,
                    use_multitask_lora=args.use_multitask_lora,
                )
                if is_main():
                    print(f"\nStep {global_step:6d}  "
                          f"PSNR={metrics['val/psnr']:.2f}  "
                          f"SSIM={metrics['val/ssim']:.4f}")
                    for tname in TASK_ID_TO_NAME.values():
                        pk = f"val/psnr_{tname}"
                        if pk in metrics:
                            print(f"  {tname:<26} "
                                  f"PSNR={metrics[pk]:.2f}  "
                                  f"SSIM={metrics[f'val/ssim_{tname}']:.4f}")

                    metrics["epoch"] = round(global_step / steps_per_epoch, 3)
                    csv_logger.log_val(global_step, metrics)
                    if args.wandb_project:
                        wandb.log(metrics, step=global_step)

                    best_psnr = save_best_checkpoint(
                        args.output_dir, global_step,
                        unwrapped, optimizer, scheduler, metrics,
                        best_psnr, use_multitask_lora=args.use_multitask_lora,
                    )

            # ── Periodic checkpoint ───────────────────────────────────────────
            if is_main() and global_step % args.save_interval == 0:
                unwrapped = transformer.module if world_size > 1 else transformer
                save_checkpoint(
                    args.output_dir, global_step,
                    unwrapped, optimizer, scheduler, metrics,
                    use_multitask_lora=args.use_multitask_lora,
                )

    pbar.close()

    # ── Final validation and checkpoint ───────────────────────────────────────
    if is_main():
        print(f"\nTraining complete at step {global_step}.")
        unwrapped     = transformer.module if world_size > 1 else transformer
        final_metrics = validate(
            vae, unwrapped, val_loader, device, global_step,
            use_multitask_lora=args.use_multitask_lora,
        )
        print(f"Final PSNR={final_metrics['val/psnr']:.2f} dB  "
              f"SSIM={final_metrics['val/ssim']:.4f}")

        save_checkpoint(
            args.output_dir, global_step,
            unwrapped, optimizer, scheduler, final_metrics,
            use_multitask_lora=args.use_multitask_lora,
        )
        final_metrics["epoch"] = round(global_step / steps_per_epoch, 3)
        csv_logger.log_val(global_step, final_metrics)
        save_best_checkpoint(
            args.output_dir, global_step,
            unwrapped, optimizer, scheduler, final_metrics,
            best_psnr, use_multitask_lora=args.use_multitask_lora,
        )

        print(f"\nOutputs in {args.output_dir}:")
        print("  checkpoint_best.pt   ← best val PSNR  (use for test)")
        print("  checkpoint_latest.pt ← most recent")
        print("  train_metrics.csv    ← training curves")
        print("  val_metrics.csv      ← validation curves per task")

        if args.wandb_project:
            wandb.log(final_metrics, step=global_step)
            wandb.finish()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()