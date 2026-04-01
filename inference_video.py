#!/usr/bin/env python3
"""
inference_video.py
------------------
Run WindowSeat restoration inference on a single video, frame by frame.

Each frame is resized to the training resolution (608x608), passed through
the LoRA-fine-tuned model with the embedding for the given task label, and
the restored frames are re-encoded into an output video at the original FPS.

Supported task labels (--task)
-------------------------------
  blur
  raindrop
  rainstreak
  rainstreak_raindrop
  reflection

Usage
-----
  python inference_video.py \\
      --video       path/to/input.mp4 \\
      --task        raindrop \\
      --embed-dir   text/text_embeddings \\
      --output      path/to/output.mp4 \\
      --checkpoint  runs/baseline/checkpoint_best.pt

  # Optionally keep original aspect ratio with letterboxing instead of cropping:
  python inference_video.py ... --resize-mode letterbox

  # Optional overrides (must match training config):
  python inference_video.py ... --lora-rank 128 --resolution 608
"""

import argparse
import math
import shutil
import subprocess
from pathlib import Path

import cv2
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
from tqdm import tqdm


# ── Constants — must match train.py exactly ────────────────────────────────────
BASE_MODEL_URI = "Qwen/Qwen-Image-Edit-2509"
FIXED_TIMESTEP = 499
MAX_SEQ_LEN    = 256

ADAPTER_NAMES = ["blur", "rain", "reflection"]
TASK_TO_ADAPTER = {
    "blur":                "blur",
    "raindrop":            "rain",
    "rainstreak":          "rain",
    "rainstreak_raindrop": "rain",
    "reflection":          "reflection",
}

VALID_TASKS = set(TASK_TO_ADAPTER.keys())

SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


# ── Frame resize ───────────────────────────────────────────────────────────────
def resize_frame_crop(frame_bgr: np.ndarray, resolution: int) -> np.ndarray:
    """
    Resize a BGR frame so the short side == resolution, then centre-crop to
    (resolution x resolution).  Matches the training pipeline exactly.
    """
    H, W = frame_bgr.shape[:2]
    if min(H, W) != resolution:
        scale = resolution / min(H, W)
        nW    = math.ceil(W * scale)
        nH    = math.ceil(H * scale)
        frame_bgr = cv2.resize(frame_bgr, (nW, nH), interpolation=cv2.INTER_LANCZOS4)

    H, W = frame_bgr.shape[:2]
    y0   = (H - resolution) // 2
    x0   = (W - resolution) // 2
    return frame_bgr[y0 : y0 + resolution, x0 : x0 + resolution]


def resize_frame_letterbox(frame_bgr: np.ndarray, resolution: int) -> np.ndarray:
    """
    Fit the frame into (resolution x resolution) preserving aspect ratio,
    padding with black on the shorter axis.
    """
    H, W  = frame_bgr.shape[:2]
    scale = resolution / max(H, W)
    nW    = math.ceil(W * scale)
    nH    = math.ceil(H * scale)
    resized = cv2.resize(frame_bgr, (nW, nH), interpolation=cv2.INTER_LANCZOS4)
    canvas  = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    y0      = (resolution - nH) // 2
    x0      = (resolution - nW) // 2
    canvas[y0 : y0 + nH, x0 : x0 + nW] = resized
    return canvas


def bgr_to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 HWC  ->  RGB float32 CHW in [-1, 1]."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t   = TF.to_tensor(Image.fromarray(rgb))   # (3, H, W) in [0, 1]
    return t * 2.0 - 1.0


def tensor_to_bgr(tensor: torch.Tensor) -> np.ndarray:
    """RGB float32 CHW in [-1, 1]  ->  BGR uint8 HWC."""
    rgb = ((tensor.float().clamp(-1, 1) + 1.0) / 2.0 * 255.0) \
          .byte().permute(1, 2, 0).cpu().numpy()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ── Embedding loading ──────────────────────────────────────────────────────────
def load_embedding(embed_root: Path, task_name: str) -> torch.Tensor:
    """Load embedding index 0 for the task, padded/truncated to MAX_SEQ_LEN."""
    embed_path = embed_root / task_name / "0.pt"
    if not embed_path.exists():
        raise FileNotFoundError(
            f"Embedding not found for task '{task_name}': {embed_path}\n"
            f"--embed-dir must contain a sub-folder named '{task_name}'."
        )
    embed   = torch.load(embed_path, weights_only=True).squeeze(0)  # (seq, D)
    seq, D  = embed.shape
    if seq < MAX_SEQ_LEN:
        embed = torch.cat([embed, embed.new_zeros(MAX_SEQ_LEN - seq, D)], dim=0)
    else:
        embed = embed[:MAX_SEQ_LEN]
    return embed   # (MAX_SEQ_LEN, D)


# ── Model construction ─────────────────────────────────────────────────────────
def _make_lora_config(rank, alpha, dropout):
    return LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=["to_q", "to_k", "to_v", "to_out.0",
                        "ff.net.0.proj", "ff.net.2"],
        bias="none",
        init_lora_weights="gaussian",
    )


def load_vae(uri: str, device: torch.device) -> AutoencoderKLQwenImage:
    vae = AutoencoderKLQwenImage.from_pretrained(
        uri, subfolder="vae",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )
    vae.to(device, dtype=torch.bfloat16).eval()
    vae.requires_grad_(False)
    return vae

def load_transformer_with_lora(uri, device, rank, alpha, dropout,
                                use_multitask=False):
    transformer = QwenImageTransformer2DModel.from_pretrained(
        uri, subfolder="transformer",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    cfg = _make_lora_config(rank, alpha, dropout)
    if use_multitask:
        transformer = get_peft_model(transformer, cfg, adapter_name=ADAPTER_NAMES[0])
        for name in ADAPTER_NAMES[1:]:
            transformer.add_adapter(name, cfg)
    else:
        transformer = get_peft_model(transformer, cfg)
    return transformer


def load_checkpoint(path: str, transformer, use_multitask=False):
    print(f"\nLoading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu")
    if "lora_state_dict" in ckpt:
        miss, unexp = transformer.load_state_dict(ckpt["lora_state_dict"], strict=False)
        if miss or unexp:
            print(f"  Missing: {len(miss)}  Unexpected: {len(unexp)}")
    elif "lora_state_dicts" in ckpt:
        for adapter_name, state in ckpt["lora_state_dicts"].items():
            transformer.set_adapter(adapter_name)
            miss, unexp = transformer.load_state_dict(state, strict=False)
            if miss or unexp:
                print(f"  [{adapter_name}] Missing: {len(miss)}  Unexpected: {len(unexp)}")
    else:
        raise KeyError("Checkpoint missing 'lora_state_dict' or 'lora_state_dicts'.")
    step = ckpt.get("step", "?")
    m    = ckpt.get("metrics", {})
    print(f"  Step {step}  |  "
          f"val PSNR={m.get('val/psnr', 'n/a')}  "
          f"val SSIM={m.get('val/ssim', 'n/a')}")


# ── VAE encode / decode ────────────────────────────────────────────────────────
@torch.no_grad()
def encode(images: torch.Tensor, vae) -> torch.Tensor:
    images = images.to(device=vae.device, dtype=vae.dtype)
    out    = vae.encode(images.unsqueeze(2)).latent_dist.sample()
    mean   = torch.tensor(vae.config.latents_mean, device=out.device, dtype=out.dtype)
    std    = torch.tensor(vae.config.latents_std,  device=out.device, dtype=out.dtype)
    mean   = mean.view(1, vae.config.z_dim, 1, 1, 1)
    std    = (1.0 / std).view(1, vae.config.z_dim, 1, 1, 1)
    return (out - mean) * std


@torch.no_grad()
def decode(latents: torch.Tensor, vae) -> torch.Tensor:
    mean    = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype)
    std     = torch.tensor(vae.config.latents_std,  device=latents.device, dtype=latents.dtype)
    mean    = mean.view(1, vae.config.z_dim, 1, 1, 1)
    std     = (1.0 / std).view(1, vae.config.z_dim, 1, 1, 1)
    latents = latents / std + mean
    return vae.decode(latents).sample[:, :, 0]   # (B, 3, H, W)

# ── Transformer forward ────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(latent_input: torch.Tensor, transformer, vae,
                  prompt_embeds: torch.Tensor,
                  prompt_mask: torch.Tensor) -> torch.Tensor:
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
    return QwenImageEditPipeline._unpack_latents(
        model_pred, height=H * sf, width=W * sf, vae_scale_factor=sf,
    )   # (B, C, 1, H, W)

# ── Single frame restoration ───────────────────────────────────────────────────
def restore_frame(frame_tensor: torch.Tensor, transformer, vae,
                  prompt_embeds: torch.Tensor, prompt_mask: torch.Tensor,
                  device: torch.device) -> torch.Tensor:
    """
    frame_tensor : (1, 3, H, W) bfloat16 on device, values in [-1, 1]
    returns      : (1, 3, H, W) bfloat16 on device, values in [-1, 1]
    """
    z_in     = encode(frame_tensor, vae)
    velocity = run_inference(z_in, transformer, vae, prompt_embeds, prompt_mask)
    z_pred   = z_in.float() + velocity.float()
    return decode(z_pred.to(vae.dtype), vae)   # (1, 3, H, W)


# ── Args ───────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(
        description="WindowSeat video inference — frame-by-frame restoration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--video", required=True,
        help="Path to the input video file (mp4, mov, avi, mkv, ...)",
    )
    p.add_argument(
        "--task", required=True,
        choices=sorted(VALID_TASKS),
        metavar="TASK",
        help=(f"Restoration task label. Choices: {sorted(VALID_TASKS)}"),
    )
    p.add_argument(
        "--embed-dir", required=True,
        help="Folder containing per-task embedding sub-folders",
    )
    p.add_argument(
        "--output", required=True,
        help="Path for the output restored video (e.g. output/restored.mp4)",
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="Path to checkpoint_best.pt or checkpoint_latest.pt",
    )
    p.add_argument("--base-model",   default=BASE_MODEL_URI)
    p.add_argument("--resolution",   type=int, default=608,
                   help="Frame resolution — must match training (default: 608)")
    p.add_argument("--lora-rank",    type=int, default=128,
                   help="LoRA rank — must match training (default: 128)")
    p.add_argument("--lora-alpha",   type=int, default=128)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument(
        "--resize-mode", choices=["crop", "letterbox"], default="crop",
        help=("How to fit frames into resolution x resolution.\n"
              "  crop      — resize short side, centre-crop (matches training, default)\n"
              "  letterbox — fit whole frame, pad with black bars"),
    )
    p.add_argument("--use-multitask-lora", action="store_true",
                   help="Load adapters for multi-task LoRA checkpoints")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    video_path  = Path(args.video)
    output_path = Path(args.output)
    embed_root  = Path(args.embed_dir)

    # Validate inputs
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if video_path.suffix.lower() not in SUPPORTED_VIDEO_EXTS:
        raise ValueError(
            f"Unsupported video format '{video_path.suffix}'. "
            f"Supported: {sorted(SUPPORTED_VIDEO_EXTS)}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resize_fn = resize_frame_crop if args.resize_mode == "crop" else resize_frame_letterbox

    # ── 1. Open input video — read metadata ──────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nInput video  : {video_path}")
    print(f"  Resolution : {orig_W}x{orig_H}  ->  {args.resolution}x{args.resolution}  "
          f"(resize-mode: {args.resize_mode})")
    print(f"  FPS        : {fps:.3f}")
    print(f"  Frames     : {total_frames}")
    print(f"Task         : {args.task}")
    print(f"Output video : {output_path}")

    # ── 2. Load models ────────────────────────────────────────────────────────
    print(f"\nLoading base model: {args.base_model}")
    vae = load_vae(args.base_model, device)

    ckpt_meta     = torch.load(args.checkpoint, map_location="cpu")
    use_multitask = (
        args.use_multitask_lora
        or "lora_state_dicts" in ckpt_meta
        or ckpt_meta.get("use_multitask_lora", False)
    )
    transformer = load_transformer_with_lora(
        args.base_model, device,
        args.lora_rank, args.lora_alpha, args.lora_dropout,
        use_multitask=use_multitask,
    )
    load_checkpoint(args.checkpoint, transformer, use_multitask)
    transformer.to(device).eval()

    if use_multitask:
        adapter = TASK_TO_ADAPTER.get(args.task, "blur")
        transformer.set_adapter(adapter)
        print(f"  Using adapter: {adapter}")

    # ── 3. Load text embedding for the task (once, reused every frame) ────────
    print(f"\nLoading embedding for task: '{args.task}'")
    embed         = load_embedding(embed_root, args.task)
    prompt_embeds = embed.unsqueeze(0).to(device, dtype=torch.bfloat16)
    prompt_mask   = torch.ones((1, MAX_SEQ_LEN), dtype=torch.bool, device=device)

    # ── 4. Check ffmpeg is available ──────────────────────────────────────────
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it with:\n"
            "  conda install -c conda-forge ffmpeg   # or   module load ffmpeg"
        )

    R = args.resolution

    # ── 5. Open ffmpeg pipe — receives raw RGB24 frames, writes H.264 mp4 ────
    #
    #  -f rawvideo -pix_fmt rgb24  : tell ffmpeg what we are piping in
    #  -r {fps}                    : input frame rate
    #  -s {R}x{R}                  : frame size
    #  -i pipe:0                   : read from stdin
    #  -vcodec libx264             : encode H.264 (available everywhere)
    #  -pix_fmt yuv420p            : broadest player compatibility
    #  -crf 18                     : near-lossless quality (0=lossless, 51=worst)
    #  -preset fast                : speed/compression trade-off
    #  -y                          : overwrite output without asking
    ffmpeg_cmd = [
        "ffmpeg",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-s", f"{R}x{R}",
        "-i", "pipe:0",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "fast",
        "-y",
        str(output_path),
    ]
    print(f"\nffmpeg command: {chr(32).join(ffmpeg_cmd)}")
    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,   # capture ffmpeg log so it doesn't clutter tqdm
    )

    # ── 6. Frame-by-frame inference ───────────────────────────────────────────
    print(f"\nProcessing {total_frames} frame(s) ...")

    pbar = tqdm(
        total=total_frames,
        desc=f"[{args.task}]  {video_path.name}",
        unit="frame",
        dynamic_ncols=True,
        colour="cyan",
    )

    frames_written = 0
    try:
        with torch.no_grad():
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                # Resize frame to model resolution
                frame_resized = resize_fn(frame_bgr, args.resolution)

                # BGR -> tensor (1, 3, H, W) in [-1, 1]
                frame_t = bgr_to_tensor(frame_resized) \
                          .unsqueeze(0).to(device, dtype=torch.bfloat16)

                # Restore
                restored = restore_frame(
                    frame_t, transformer, vae,
                    prompt_embeds, prompt_mask, device,
                )

                # Tensor -> RGB numpy -> pipe to ffmpeg
                rgb_frame = ((restored[0].float().clamp(-1, 1) + 1.0) / 2.0 * 255.0) \
                            .byte().permute(1, 2, 0).cpu().numpy()  # (H, W, 3) RGB uint8
                ffmpeg_proc.stdin.write(rgb_frame.tobytes())

                frames_written += 1
                pbar.update(1)

    finally:
        pbar.close()
        cap.release()
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()

    if ffmpeg_proc.returncode != 0:
        err = ffmpeg_proc.stderr.read().decode(errors="replace")
        raise RuntimeError(f"ffmpeg failed (exit {ffmpeg_proc.returncode}):\n{err}")

    # ── 7. Summary ────────────────────────────────────────────────────────────
    size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"\nDone.")
    print(f"  Frames processed : {frames_written}")
    print(f"  Output           : {output_path}  ({size_mb:.1f} MB)")
    print(f"  Resolution       : {R}x{R}  @  {fps:.2f} fps")
    print(f"  Task             : {args.task}")

if __name__ == "__main__":
    main()