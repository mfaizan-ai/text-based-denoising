#!/usr/bin/env python3
"""
train_windowseat.py
--------------------
Phase 2 training script for WindowSeat multi-task image restoration.
Fine-tunes LoRA adapters on the Qwen-Image-Edit-2509 DiT backbone
using flow matching loss in latent space.

Changes from Phase 1
---------------------
  - TaskEmbedding (learnable fixed embed) is REMOVED entirely.
    Real pre-computed text embeddings are loaded from disk instead,
    one pool of 20 .pt files per task, sampled randomly per sample.
  - Dataset is now driven by JSON metadata files (train/val/test_metadata.json)
    instead of a hardcoded directory structure.
  - MAX_SEQ_LEN = 256 everywhere (was incorrectly 77 — CLIP legacy constant
    that has nothing to do with Qwen's architecture).
  - Resolution default lowered to 512 to handle 400px reflection images
    without excessive upscaling, while still covering 1280x720 blur images
    with a random crop.a

Supported tasks (from JSON task_id field)
------------------------------------------
  0 - blur                (real,  1280x720)
  1 - raindrop            (syn,   1296x728)
  2 - rainstreak          (syn,   1296x728)
  3 - rainstreak_raindrop (syn,   1296x728)
  4 - reflection          (syn,   ~400x400)

Dataset JSON format
--------------------
  [
    {
      "input":     "dataset_full/blur/real/blended/000614.png",
      "target":    "dataset_full/blur/real/clean/000614.png",
      "task_name": "blur",
      "task_id":   0,
      "data_type": "real",
      "width":     1280,
      "height":    720
    },
    ...
  ]

Embed directory layout
-----------------------
  embed_dir/
    blur/                   0.pt ... 19.pt   # shape (1, 256, hidden_dim)
    raindrop/               0.pt ... 19.pt
    rainstreak/             0.pt ... 19.pt
    rainstreak_raindrop/    0.pt ... 19.pt
    reflection/             0.pt ... 19.pt

Usage
-----
  # Single GPU
  python train_windowseat.py \\
      --data-root /path/to/project \\
      --meta-dir  dataset_metadata \\
      --embed-dir text/text_embeddings \\
      --output-dir runs/phase2

  # Multi-GPU
  torchrun --nproc_per_node=4 train_windowseat.py \\
      --data-root /path/to/project \\
      --meta-dir  dataset_metadata \\
      --embed-dir text/text_embeddings \\
      --output-dir runs/phase2 --batch-size 4

  # Resume
  python train_windowseat.py ... --resume runs/phase2/checkpoint_epoch005.pt
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

import lpips as lpips_lib
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
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm


# ── Constants ─────────────────────────────────────────────────────────────────

BASE_MODEL_URI = "Qwen/Qwen-Image-Edit-2509"
FIXED_TIMESTEP = 499    # single fixed denoising timestep used by WindowSeat

# CHANGED: was 77 (CLIP legacy constant — wrong for Qwen).
# Must match the max_sequence_length used in TextEmbeddingProcessor.
MAX_SEQ_LEN = 256

TASK_ID_TO_NAME = {
    0: "blur",
    1: "raindrop",
    2: "rainstreak",
    3: "rainstreak_raindrop",
    4: "reflection",
}


# ── Args ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="Train WindowSeat LoRA for multi-task image restoration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    # CHANGED: replaced --data-root alone with three explicit path args.
    # --data-root  : project root; JSON paths are relative to this.
    # --meta-dir   : folder holding train/val/test_metadata.json.
    # --embed-dir  : folder holding per-task embedding pools.
    p.add_argument("--data-root",    required=True,
                   help="Project root; JSON input/target paths are joined here")
    p.add_argument("--meta-dir",     required=True,
                   help="Directory containing train_metadata.json and val_metadata.json")
    p.add_argument("--embed-dir",    required=True,
                   help="Directory containing per-task .pt embedding pools")
    p.add_argument("--output-dir",   required=True,
                   help="Where to save checkpoints and logs")
    p.add_argument("--resume",       default=None,
                   help="Path to checkpoint to resume from")

    # Model
    p.add_argument("--base-model",    default=BASE_MODEL_URI)
    p.add_argument("--lora-rank",     type=int,   default=64)
    p.add_argument("--lora-alpha",    type=int,   default=64)
    p.add_argument("--lora-dropout",  type=float, default=0.05)

    # Training
    p.add_argument("--epochs",        type=int,   default=20)
    p.add_argument("--batch-size",    type=int,   default=4,
                   help="Per-GPU batch size")
    p.add_argument("--grad-accum",    type=int,   default=4,
                   help="Gradient accumulation steps")
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight-decay",  type=float, default=0.01)
    p.add_argument("--warmup-steps",  type=int,   default=500)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    # CHANGED: default 768 -> 512.
    # 512 handles 400px reflection images with only mild upscaling (~28%)
    # while still providing a good random crop from 1280x720 blur images.
    p.add_argument("--resolution",    type=int,   default=512,
                   help="Square crop size. 512 is recommended for this dataset mix.")

    # Loss
    p.add_argument("--lambda-lpips",  type=float, default=0.1,
                   help="Weight for LPIPS perceptual loss (0 = disabled)")

    # Logging / saving
    p.add_argument("--log-interval",  type=int,   default=50)
    p.add_argument("--val-interval",  type=int,   default=1,
                   help="Validate every N epochs")
    p.add_argument("--save-interval", type=int,   default=1,
                   help="Save checkpoint every N epochs")
    p.add_argument("--wandb-project", default="windowseat")
    p.add_argument("--run-name",      default=None)

    # System
    p.add_argument("--num-workers",   type=int,   default=8)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--local-rank",    type=int,   default=-1,
                   help="Set automatically by torchrun")

    return p.parse_args()


# ── Distributed helpers ───────────────────────────────────────────────────────
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_distributed(local_rank: int):
    if local_rank == -1:
        return torch.device("cuda"), 0, 1
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}"), dist.get_rank(), dist.get_world_size()


# ── Dataset ───────────────────────────────────────────────────────────────────
# CHANGED: ReflectionPairDataset (hardcoded folder structure, no text embeds)
# is replaced by JsonRestorationDataset which:
#   - reads pairs from a JSON metadata file
#   - supports all 5 tasks in one loader
#   - loads a randomly sampled text embedding for each sample
#   - handles mixed resolutions (400px-1280px) via _ensure_min_size

class JsonRestorationDataset(Dataset):
    """
    Multi-task restoration dataset driven by a JSON metadata file.

    Resolution strategy
    -------------------
    Training  : random crop of `resolution x resolution`.
                If the shorter side < resolution the image is first
                upscaled with LANCZOS so no black borders are introduced.
    Validation: deterministic centre crop of the same size, giving
                stable and reproducible PSNR/SSIM numbers across runs.

    For this dataset:
      - 1280x720 and 1296x728 images: short edge (720/728) >= 512,
        so they are directly cropped with no upscaling at all.
      - 400x400 reflection images: short edge < 512, so they are scaled
        to 512x512 first (~28% upscale, negligible quality impact).
    """

    EMBED_POOL_SIZE = 20

    def __init__(
        self,
        data_root:  str,
        json_path:  str,
        embed_root: str,
        resolution: int  = 512,
        is_train:   bool = True,
    ):
        self.data_root  = Path(data_root)
        self.embed_root = Path(embed_root)
        self.resolution = resolution
        self.is_train   = is_train

        with open(json_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        if is_main_process():
            missing = sum(
                1 for s in self.samples
                if not (self.data_root / s["input"]).exists()
            )
            tag = "train" if is_train else "val"
            print(
                f"  [{tag}] {len(self.samples)} pairs loaded "
                f"({missing} input paths not found on disk)"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        item    = self.samples[idx]
        blended = Image.open(self.data_root / item["input"]).convert("RGB")
        clean   = Image.open(self.data_root / item["target"]).convert("RGB")

        if self.is_train:
            blended, clean = self._random_crop(blended, clean)
        else:
            blended, clean = self._centre_crop(blended, clean)

        # Normalise to [-1, 1] — required by the Qwen VAE
        blended = TF.to_tensor(blended) * 2.0 - 1.0   # (3, R, R)
        clean   = TF.to_tensor(clean)   * 2.0 - 1.0   # (3, R, R)

        prompt_embeds = self._load_embed(item.get("task_id", 4))

        return {
            "blended":       blended,        # (3, R, R)         float32 in [-1, 1]
            "clean":         clean,          # (3, R, R)         float32 in [-1, 1]
            "prompt_embeds": prompt_embeds,  # (MAX_SEQ_LEN, D)  float32
        }

    # ── Augmentation ─────────────────────────────────────────────────────────

    def _ensure_min_size(
        self, a: Image.Image, b: Image.Image
    ) -> tuple:
        """Upscale both images identically if shorter edge < resolution."""
        R    = self.resolution
        W, H = a.size
        if min(W, H) < R:
            scale = R / min(W, H)
            new_w = math.ceil(W * scale)
            new_h = math.ceil(H * scale)
            a = a.resize((new_w, new_h), Image.LANCZOS)
            b = b.resize((new_w, new_h), Image.LANCZOS)
        return a, b

    def _random_crop(self, blended: Image.Image, clean: Image.Image) -> tuple:
        R = self.resolution
        blended, clean = self._ensure_min_size(blended, clean)
        W, H = blended.size
        i = random.randint(0, H - R)
        j = random.randint(0, W - R)
        blended = TF.crop(blended, i, j, R, R)
        clean   = TF.crop(clean,   i, j, R, R)
        if random.random() < 0.5:
            blended = TF.hflip(blended)
            clean   = TF.hflip(clean)
        return blended, clean

    def _centre_crop(self, blended: Image.Image, clean: Image.Image) -> tuple:
        R = self.resolution
        blended, clean = self._ensure_min_size(blended, clean)
        blended = TF.center_crop(blended, (R, R))
        clean   = TF.center_crop(clean,   (R, R))
        return blended, clean

    # ── Embedding loader ──────────────────────────────────────────────────────
    def _load_embed(self, task_id: int) -> torch.Tensor:
        task_name = TASK_ID_TO_NAME.get(task_id, "reflection")
        idx       = random.randint(0, self.EMBED_POOL_SIZE - 1)
        path      = self.embed_root / task_name / f"{idx}.pt"
        embed     = torch.load(path, weights_only=True)   # (1, seq_len, D)
        embed     = embed.squeeze(0)                       # (seq_len, D)

        seq_len, D = embed.shape
        if seq_len < MAX_SEQ_LEN:
            pad   = embed.new_zeros(MAX_SEQ_LEN - seq_len, D)
            embed = torch.cat([embed, pad], dim=0)         # pad to MAX_SEQ_LEN
        else:
            embed = embed[:MAX_SEQ_LEN]                    # truncate if over

        return embed                                       # (MAX_SEQ_LEN, D)


# ── Model loading ─────────────────────────────────────────────────────────────
def load_vae(uri: str, device: torch.device) -> AutoencoderKLQwenImage:
    """Load and permanently freeze the VAE."""
    vae = AutoencoderKLQwenImage.from_pretrained(
        uri,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )
    vae.to(device, dtype=torch.bfloat16)
    vae.requires_grad_(False)
    vae.eval()
    return vae


def load_transformer_with_lora(
    uri:          str,
    device:       torch.device,
    lora_rank:    int,
    lora_alpha:   int,
    lora_dropout: float,
) -> QwenImageTransformer2DModel:
    """Load the DiT backbone in bf16 and attach LoRA adapters."""
    transformer = QwenImageTransformer2DModel.from_pretrained(
        uri,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",
            "ff.net.0.proj", "ff.net.2",
        ],
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_cfg)

    if is_main_process():
        transformer.print_trainable_parameters()

    return transformer


# ── VAE encode / decode ───────────────────────────────────────────────────────

@torch.no_grad()
def encode(images: torch.Tensor, vae: AutoencoderKLQwenImage) -> torch.Tensor:
    """(B, 3, H, W) in [-1, 1]  ->  normalised latents (B, C, 1, H/8, W/8)."""
    images = images.to(device=vae.device, dtype=vae.dtype)
    out    = vae.encode(images.unsqueeze(2)).latent_dist.sample()
    mean   = torch.tensor(vae.config.latents_mean, device=out.device, dtype=out.dtype)
    std    = torch.tensor(vae.config.latents_std,  device=out.device, dtype=out.dtype)
    mean   = mean.view(1, vae.config.z_dim, 1, 1, 1)
    std    = (1.0 / std).view(1, vae.config.z_dim, 1, 1, 1)
    return (out - mean) * std


def decode(latents: torch.Tensor, vae: AutoencoderKLQwenImage) -> torch.Tensor:
    """Normalised latents (B, C, 1, H/8, W/8)  ->  (B, 3, H, W) in [-1, 1]."""
    mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype)
    std  = torch.tensor(vae.config.latents_std,  device=latents.device, dtype=latents.dtype)
    mean = mean.view(1, vae.config.z_dim, 1, 1, 1)
    std  = (1.0 / std).view(1, vae.config.z_dim, 1, 1, 1)
    latents = latents / std + mean
    return vae.decode(latents).sample[:, :, 0]   # (B, 3, H, W)


# ── Transformer forward pass ──────────────────────────────────────────────────
def flow_step_train(
    latent_input:  torch.Tensor,                  # (B, C, 1, H, W)
    transformer:   QwenImageTransformer2DModel,
    vae:           AutoencoderKLQwenImage,
    prompt_embeds: torch.Tensor,                  # (B, MAX_SEQ_LEN, D)
    prompt_mask:   torch.Tensor,                  # (B, MAX_SEQ_LEN)
) -> torch.Tensor:
    """
    Single-step flow-matching forward pass with gradients enabled for
    LoRA parameter updates.  Returns predicted velocity (same shape as input).
    """
    lat4d = latent_input[:, :, 0] if latent_input.ndim == 5 else latent_input
    B, C, H, W = lat4d.shape

    device = next(transformer.parameters()).device
    prompt_embeds = prompt_embeds.to(device=device, dtype=torch.bfloat16)
    prompt_mask   = prompt_mask.to(device=device)

    packed = QwenImageEditPipeline._pack_latents(
        lat4d, batch_size=B,
        num_channels_latents=C, height=H, width=W,
    ).to(torch.bfloat16)

    timestep     = torch.full((B,), float(FIXED_TIMESTEP) / 1000.0,
                              device=device, dtype=torch.bfloat16)
    img_shapes   = [[(1, H // 2, W // 2)]] * B
    # txt_seq_lens reads actual length from mask — works for any MAX_SEQ_LEN
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


# ── Loss ──────────────────────────────────────────────────────────────────────
class ReflectionRemovalLoss(torch.nn.Module):
    """
    Flow-matching MSE loss in latent space, plus optional LPIPS perceptual loss.

    velocity_target = z_degraded - z_clean
    loss_flow       = MSE(pred_velocity, velocity_target)
    loss_lpips      = LPIPS(decode(z_degraded - pred_velocity), img_clean)
    total           = loss_flow + lambda_lpips * loss_lpips
    """

    def __init__(self, lambda_lpips: float = 0.1, device: torch.device = None):
        super().__init__()
        self.lambda_lpips = lambda_lpips
        if lambda_lpips > 0.0:
            self.lpips_net = lpips_lib.LPIPS(net="alex").to(device)
            self.lpips_net.requires_grad_(False)
            self.lpips_net.eval()
        else:
            self.lpips_net = None

    def forward(
        self,
        pred_velocity: torch.Tensor,   # (B, C, 1, H, W)
        z_degraded:    torch.Tensor,   # (B, C, 1, H, W)
        z_clean:       torch.Tensor,   # (B, C, 1, H, W)
        vae:           AutoencoderKLQwenImage,
        clean_images:  torch.Tensor,   # (B, 3, H, W) in [-1, 1]
    ):
        velocity_target = z_degraded - z_clean
        loss_flow = F.mse_loss(pred_velocity.float(), velocity_target.float())

        loss_lpips = pred_velocity.new_zeros(1).squeeze()
        if self.lpips_net is not None and self.lambda_lpips > 0.0:
            with torch.no_grad():
                z_pred   = z_degraded.float() - pred_velocity.float()
                img_pred = decode(z_pred.to(vae.dtype), vae)

            img_pred   = img_pred.float().clamp(-1.0, 1.0)
            clean_f    = clean_images.float().to(img_pred.device)
            loss_lpips = self.lpips_net(img_pred, clean_f).mean()

        total = loss_flow + self.lambda_lpips * loss_lpips
        return total, loss_flow, loss_lpips


# ── LR scheduler (cosine with linear warmup) ─────────────────────────────────

def get_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Validation ────────────────────────────────────────────────────────────────
# CHANGED: task_embed parameter removed entirely — embeddings come from the
# DataLoader batch just like they do during training.
@torch.no_grad()
def validate(
    vae:         AutoencoderKLQwenImage,
    transformer: QwenImageTransformer2DModel,
    val_loader:  DataLoader,
    device:      torch.device,
    epoch:       int,
) -> dict:
    """Compute mean PSNR and SSIM on the validation set."""
    transformer.eval()
    psnr_vals, ssim_vals = [], []
    torch.cuda.empty_cache()

    for batch in tqdm(val_loader, desc=f"Val epoch {epoch}",
                      leave=False, disable=not is_main_process()):
        blended       = batch["blended"].to(device, dtype=torch.bfloat16)
        clean         = batch["clean"].to(device)
        # CHANGED: load real embeddings from batch instead of calling task_embed(B)
        prompt_embeds = batch["prompt_embeds"].to(device, dtype=torch.bfloat16)
        B             = blended.shape[0]
        # CHANGED: mask size is MAX_SEQ_LEN (256), not hardcoded 77
        prompt_mask   = torch.ones((B, MAX_SEQ_LEN), dtype=torch.bool, device=device)

        z_degraded    = encode(blended, vae)
        pred_velocity = flow_step_train(
            z_degraded, transformer, vae, prompt_embeds, prompt_mask
        )

        z_pred   = z_degraded.float() - pred_velocity.float()
        img_pred = decode(z_pred.to(vae.dtype), vae)

        pred_np  = ((img_pred.float().clamp(-1.0, 1.0) + 1.0) / 2.0).cpu().numpy()
        clean_np = ((clean.float().clamp(-1.0, 1.0)   + 1.0) / 2.0).cpu().numpy()

        for b in range(B):
            p = pred_np[b].transpose(1, 2, 0)
            c = clean_np[b].transpose(1, 2, 0)
            psnr_vals.append(compute_psnr(c, p, data_range=1.0))
            try:
                ssim_vals.append(
                    compute_ssim(c, p, data_range=1.0, channel_axis=-1)
                )
            except TypeError:
                ssim_vals.append(
                    compute_ssim(c, p, data_range=1.0, multichannel=True)
                )

    transformer.train()
    return {
        "val/psnr": float(np.mean(psnr_vals)),
        "val/ssim": float(np.mean(ssim_vals)),
    }


# ── Checkpont helpers ────────────────────────────────────────────────────────
# CHANGED: task_embed removed from both save and load — checkpoints only
# contain LoRA weights + optimizer/scheduler state.
def save_checkpoint(
    output_dir:  str,
    epoch:       int,
    transformer: QwenImageTransformer2DModel,
    optimizer:   torch.optim.Optimizer,
    scheduler,
    metrics:     dict,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    model = transformer.module if hasattr(transformer, "module") else transformer

    ckpt = {
        "epoch":           epoch,
        "lora_state_dict": {
            k: v for k, v in model.state_dict().items() if "lora" in k
        },
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict(),
        "metrics":    metrics,
    }

    path   = os.path.join(output_dir, f"checkpoint_epoch{epoch:03d}.pt")
    latest = os.path.join(output_dir, "checkpoint_latest.pt")
    torch.save(ckpt, path)

    if os.path.lexists(latest):
        os.remove(latest)
    try:
        os.symlink(os.path.basename(path), latest)
    except OSError:
        import shutil
        shutil.copy2(path, latest)

    return path


def load_checkpoint(
    path:        str,
    transformer: QwenImageTransformer2DModel,
    optimizer:   torch.optim.Optimizer,
    scheduler,
) -> int:
    if is_main_process():
        print(f"Resuming from {path}")

    ckpt  = torch.load(path, map_location="cpu")
    model = transformer.module if hasattr(transformer, "module") else transformer

    missing, unexpected = model.load_state_dict(
        ckpt["lora_state_dict"], strict=False
    )
    if is_main_process() and (missing or unexpected):
        print(f"  Missing: {len(missing)}  Unexpected: {len(unexpected)}")

    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])

    if is_main_process():
        m    = ckpt.get("metrics", {})
        psnr = m.get("val/psnr", "n/a")
        print(f"  Resumed at epoch {ckpt['epoch']}  |  last val PSNR={psnr}")

    return ckpt["epoch"]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = get_args()
    device, rank, world_size = setup_distributed(args.local_rank)

    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── W&B ───────────────────────────────────────────────────────────────────
    if is_main_process() and args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args),
        )

    # ── Datasets ──────────────────────────────────────────────────────────────
    # CHANGED: JsonRestorationDataset replaces ReflectionPairDataset
    if is_main_process():
        print(f"Loading datasets  (resolution={args.resolution}px) ...")

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
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ── Models ────────────────────────────────────────────────────────────────
    if is_main_process():
        print(f"Loading base model: {args.base_model}")

    vae         = load_vae(args.base_model, device)
    transformer = load_transformer_with_lora(
        args.base_model, device,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    transformer.train()

    # CHANGED: TaskEmbedding is gone — no learnable embed module needed
    if world_size > 1:
        transformer = DDP(transformer, device_ids=[args.local_rank])

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = ReflectionRemovalLoss(
        lambda_lpips=args.lambda_lpips, device=device
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # CHANGED: single param group — no separate lr_embed needed
    lora_params = [
        p for n, p in transformer.named_parameters() if "lora" in n
    ]

    if is_main_process():
        print(f"Trainable LoRA parameters: {sum(p.numel() for p in lora_params):,}")

    optimizer = torch.optim.AdamW(
        lora_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) // args.grad_accum * args.epochs
    scheduler   = get_scheduler(optimizer, args.warmup_steps, total_steps)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        # CHANGED: no task_embed arg
        start_epoch = load_checkpoint(args.resume, transformer, optimizer, scheduler)

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step = 0
    scaler      = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        transformer.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            disable=not is_main_process(),
        )

        for step, batch in enumerate(pbar):
            blended = batch["blended"].to(device, dtype=torch.bfloat16)
            clean   = batch["clean"].to(device,   dtype=torch.bfloat16)
            # CHANGED: load real text embeddings from batch
            prompt_embeds = batch["prompt_embeds"].to(device, dtype=torch.bfloat16)
            B             = blended.shape[0]
            # CHANGED: mask is MAX_SEQ_LEN wide, not 77
            prompt_mask   = torch.ones((B, MAX_SEQ_LEN), dtype=torch.bool, device=device)

            with torch.no_grad():
                z_degraded = encode(blended, vae)
                z_clean    = encode(clean,   vae)

            pred_velocity = flow_step_train(
                z_degraded, transformer, vae, prompt_embeds, prompt_mask
            )

            loss, loss_flow, loss_lpips = criterion(
                pred_velocity, z_degraded, z_clean, vae,
                clean_images=batch["clean"].to(device),
            )

            scaled_loss = loss / args.grad_accum
            scaler.scale(scaled_loss).backward()
            epoch_loss += loss.item()

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(lora_params, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_main_process() and global_step % args.log_interval == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    pbar.set_postfix(
                        loss=f"{loss_flow.item():.4f}",
                        lpips=f"{loss_lpips.item():.4f}",
                        lr=f"{lr_now:.2e}",
                    )
                    if args.wandb_project:
                        wandb.log(
                            {
                                "train/loss":       loss_flow.item(),
                                "train/loss_lpips": loss_lpips.item(),
                                "train/lr":         lr_now,
                            },
                            step=global_step,
                        )

        avg_loss = epoch_loss / len(train_loader)
        if is_main_process():
            print(f"Epoch {epoch + 1:3d}  avg_loss={avg_loss:.4f}")

        metrics = {}

        if (epoch + 1) % args.val_interval == 0:
            unwrapped = transformer.module if world_size > 1 else transformer
            # CHANGED: no task_embed arg
            metrics   = validate(vae, unwrapped, val_loader, device, epoch + 1)
            if is_main_process():
                print(
                    f"  -> PSNR={metrics['val/psnr']:.2f} dB  "
                    f"SSIM={metrics['val/ssim']:.4f}"
                )
                if args.wandb_project:
                    wandb.log(metrics, step=global_step)

        if is_main_process() and (epoch + 1) % args.save_interval == 0:
            unwrapped = transformer.module if world_size > 1 else transformer
            # CHANGED: no task_embed arg
            ckpt_path = save_checkpoint(
                args.output_dir, epoch + 1,
                unwrapped, optimizer, scheduler, metrics,
            )
            print(f"  Checkpoint -> {ckpt_path}")

    if is_main_process():
        print("Training complete.")
        if args.wandb_project:
            wandb.finish()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
