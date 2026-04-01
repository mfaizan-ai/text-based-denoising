#!/usr/bin/env python3
"""
inference.py
------------
Run WindowSeat restoration inference on a folder of images without ground-truth labels.

Reads a plain-text prompt file that maps image filenames to free-text restoration
descriptions, e.g.:

    1.png
    Remove raindrops and reflection.

    2.png
    Remove raindrops and remove blur.

    3.png
    Remove raindrop.

Each description is parsed into an ORDERED list of restoration tasks based on
keyword positions in the text.  When a description mentions more than one
cross-category task the model runs them as SEQUENTIAL passes:

    1.png  →  pass 1 (raindrop embedding)   →  intermediate saved
           →  pass 2 (reflection embedding)  →  final output saved

The special case "rainstreak + raindrop together, nothing else" maps to the
combined rainstreak_raindrop embedding and is done in a SINGLE pass.

Intermediates are saved to  <output-dir>/intermediates/
Final outputs are saved to  <output-dir>/

Supported task embedding folders
---------------------------------
  blur                 -> blur/
  raindrop             -> raindrop/
  rainstreak           -> rainstreak/
  rainstreak_raindrop  -> rainstreak_raindrop/
  reflection           -> reflection/

Usage
-----
  # Input folder layout (text_description.txt is auto-discovered):
  #   inference_images/
  #   ├── 1.png
  #   ├── 2.png
  #   ├── 3.png
  #   └── text_description.txt

  python inference.py \\
      --input-dir   inference_images/ \\
      --embed-dir   text/text_embeddings \\
      --output-dir  path/to/restored \\
      --checkpoint  runs/baseline/checkpoint_best.pt

  # Output layout:
  #   output_dir/
  #   ├── 1_restored.png
  #   ├── 2_restored.png
  #   ├── 3_restored.png
  #   └── intermediates/
  #       ├── 1_step1_raindrop.png
  #       └── 3_step1_raindrop.png

  # Skip saving intermediate images:
  python inference.py ... --no-intermediates

  # Optional overrides (must match training config):
  python inference.py ... --lora-rank 128 --resolution 608
"""

import argparse
import math
import re
from pathlib import Path

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


# ── Constants — must match train.py / test_windowseat.py exactly ───────────────
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

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

# (task_name, regex_pattern) — checked in order; first match wins per task family.
# Position of match in the description determines execution order.
_TASK_PATTERNS: list[tuple[str, str]] = [
    ("rainstreak", r"rain\s*streak"),
    ("raindrop",   r"rain\s*drop|raindrop"),
    ("rainstreak", r"streak"),
    ("raindrop",   r"\bdrop\b"),
    ("rainstreak", r"\brain\b"),     # bare "rain" with no "drop" -> streak
    ("blur",       r"blur|haze|defocus|fog"),
    ("reflection", r"reflect|glare|glass|window"),
]


# ── Task parsing ───────────────────────────────────────────────────────────────
def description_to_tasks(description: str) -> list:
    """
    Parse a free-text description into an ORDERED list of task names.

    Algorithm
    ---------
    1. For each task family scan the (lowercased) description for keyword matches
       and record the earliest character position of any hit.
    2. Sort detected tasks by their position in the text -> preserves the order
       the user stated them in (e.g. "raindrops AND reflection" -> raindrop first).
    3. Special case: if the ONLY detected tasks are rainstreak + raindrop (and
       nothing else), merge into the single combined rainstreak_raindrop embedding
       (one pass is sufficient and more accurate).
    4. Deduplicate while preserving order.

    Examples
    --------
    "Remove raindrops and reflection."        -> ["raindrop", "reflection"]
    "Remove raindrops and remove blur."       -> ["raindrop", "blur"]
    "Remove rain streaks and raindrops."      -> ["rainstreak_raindrop"]  (merged)
    "Remove blur, raindrops and reflection."  -> ["blur", "raindrop", "reflection"]
    "Remove raindrop."                        -> ["raindrop"]
    """
    d = description.lower()

    # task_name -> earliest match position
    task_pos: dict = {}
    for task_name, pattern in _TASK_PATTERNS:
        if task_name in task_pos:
            continue                    # already have an earlier hit for this task
        m = re.search(pattern, d)
        if m:
            task_pos[task_name] = m.start()

    if not task_pos:
        print(f"  [warn] No task keywords found in '{description.strip()}' "
              f"-- defaulting to 'raindrop'.")
        return ["raindrop"]

    # Sort by position in the description text
    ordered = [t for t, _ in sorted(task_pos.items(), key=lambda x: x[1])]

    # Merge rain-only combo into the dedicated combined embedding (single pass)
    if set(ordered) == {"rainstreak", "raindrop"}:
        return ["rainstreak_raindrop"]

    return ordered


# ── Prompt file parser ─────────────────────────────────────────────────────────
def parse_prompt_file(prompt_file: str) -> dict:
    """
    Parse a prompt file with alternating filename / description blocks
    separated by blank lines.

    Returns: {filename: description_text}

    Format:
        1.png
        Remove raindrops and reflection.

        2.png
        Remove blur.
    """
    path = Path(prompt_file)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    text   = path.read_text(encoding="utf-8")
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]

    mapping: dict = {}
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 2:
            print(f"  [warn] Skipping incomplete block: {block!r}")
            continue
        filename    = lines[0].strip()
        description = " ".join(ln.strip() for ln in lines[1:])

        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_IMAGE_EXTS:
            print(f"  [skip] '{filename}' -- unsupported format '{ext}'. "
                  f"Supported: {sorted(SUPPORTED_IMAGE_EXTS)}")
            continue
        mapping[filename] = description

    if not mapping:
        raise ValueError("No valid image entries found in prompt file.")

    print(f"  Parsed {len(mapping)} image entries from prompt file.")
    return mapping


# ── Image preprocessing ────────────────────────────────────────────────────────
def load_and_preprocess(image_path: Path, resolution: int, pbar=None) -> torch.Tensor:
    """
    Load an image and reshape to exactly (resolution x resolution).

    Steps
    -----
    1. If already (resolution x resolution) use as-is.
    2. Otherwise:
       a. Scale short side up to resolution (LANCZOS, preserves aspect ratio).
       b. Centre-crop to (resolution x resolution).

    Prints a notice whenever the input needs reshaping.
    """
    img            = Image.open(image_path).convert("RGB")
    orig_W, orig_H = img.size

    if (orig_W, orig_H) != (resolution, resolution):
        msg = (f"  [reshape] {image_path.name}  "
               f"{orig_W}x{orig_H} -> {resolution}x{resolution}")
        pbar.write(msg) if pbar is not None else print(msg)

        if min(orig_W, orig_H) != resolution:
            scale = resolution / min(orig_W, orig_H)
            img   = img.resize(
                (math.ceil(orig_W * scale), math.ceil(orig_H * scale)),
                Image.LANCZOS,
            )
        img = TF.center_crop(img, (resolution, resolution))

    return TF.to_tensor(img) * 2.0 - 1.0   # (3, H, W) in [-1, 1]


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """(3, H, W) float in [-1, 1]  ->  PIL RGB Image."""
    arr = ((tensor.float().clamp(-1, 1) + 1.0) / 2.0 * 255.0) \
          .byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


# ── Embedding loading ──────────────────────────────────────────────────────────
def load_embedding(embed_root: Path, task_name: str) -> torch.Tensor:
    """
    Load embedding index 0 for the task (same deterministic choice as test script).
    Pads or truncates to MAX_SEQ_LEN.
    """
    embed_path = embed_root / task_name / "0.pt"
    if not embed_path.exists():
        raise FileNotFoundError(
            f"Embedding not found for task '{task_name}': {embed_path}\n"
            f"--embed-dir must contain sub-folders named after each task."
        )
    embed = torch.load(embed_path, weights_only=True).squeeze(0)   # (seq, D)
    seq_len, D = embed.shape
    if seq_len < MAX_SEQ_LEN:
        embed = torch.cat([embed, embed.new_zeros(MAX_SEQ_LEN - seq_len, D)], dim=0)
    else:
        embed = embed[:MAX_SEQ_LEN]
    return embed   # (MAX_SEQ_LEN, D)


# ── Model construction ─────────────────────────────────────────────────────────
def _make_lora_config(lora_rank, lora_alpha, lora_dropout):
    return LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
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
    vae.to(device, dtype=torch.bfloat16)
    vae.requires_grad_(False)
    vae.eval()
    return vae


def load_transformer_with_lora(uri, device, lora_rank, lora_alpha, lora_dropout,
                                use_multitask_lora=False):
    transformer = QwenImageTransformer2DModel.from_pretrained(
        uri, subfolder="transformer",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    cfg = _make_lora_config(lora_rank, lora_alpha, lora_dropout)
    if use_multitask_lora:
        transformer = get_peft_model(transformer, cfg, adapter_name=ADAPTER_NAMES[0])
        for name in ADAPTER_NAMES[1:]:
            transformer.add_adapter(name, cfg)
    else:
        transformer = get_peft_model(transformer, cfg)
    return transformer


def load_checkpoint(path: str, transformer, use_multitask_lora=False):
    print(f"\nLoading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu")

    if "lora_state_dict" in ckpt:
        missing, unexpected = transformer.load_state_dict(
            ckpt["lora_state_dict"], strict=False)
        if missing or unexpected:
            print(f"  Missing: {len(missing)}  Unexpected: {len(unexpected)}")
    elif "lora_state_dicts" in ckpt:
        for adapter_name, state in ckpt["lora_state_dicts"].items():
            transformer.set_adapter(adapter_name)
            missing, unexpected = transformer.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(f"  [{adapter_name}] Missing: {len(missing)}  "
                      f"Unexpected: {len(unexpected)}")
    else:
        raise KeyError(
            "Checkpoint missing 'lora_state_dict' or 'lora_state_dicts'.")

    step = ckpt.get("step", "?")
    m    = ckpt.get("metrics", {})
    print(f"  Step {step}  |  "
          f"val PSNR={m.get('val/psnr', 'n/a')}  "
          f"val SSIM={m.get('val/ssim', 'n/a')}")


# ── VAE encode / decode (identical to train.py) ────────────────────────────────
@torch.no_grad()
def encode(images: torch.Tensor, vae) -> torch.Tensor:
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
def decode(latents: torch.Tensor, vae) -> torch.Tensor:
    mean = torch.tensor(vae.config.latents_mean,
                        device=latents.device, dtype=latents.dtype)
    std  = torch.tensor(vae.config.latents_std,
                        device=latents.device, dtype=latents.dtype)
    mean = mean.view(1, vae.config.z_dim, 1, 1, 1)
    std  = (1.0 / std).view(1, vae.config.z_dim, 1, 1, 1)
    latents = latents / std + mean
    return vae.decode(latents).sample[:, :, 0]   # (B, 3, H, W)


# ── Transformer forward pass ───────────────────────────────────────────────────
@torch.no_grad()
def run_inference(latent_input: torch.Tensor, transformer, vae,
                  prompt_embeds: torch.Tensor,
                  prompt_mask: torch.Tensor) -> torch.Tensor:
    """Returns predicted velocity tensor (B, C, 1, H, W)."""
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


# ── One full restoration pass (encode -> infer -> decode) ─────────────────────
def restore_one_pass(img_tensor: torch.Tensor, task_name: str,
                     embed_root: Path, transformer, vae,
                     device: torch.device, use_multitask: bool) -> torch.Tensor:
    """
    Single encode -> infer -> decode cycle for one task.

    Parameters
    ----------
    img_tensor : (1, 3, H, W) bfloat16 tensor on device, values in [-1, 1]

    Returns
    -------
    (1, 3, H, W) bfloat16 tensor on device, values in [-1, 1]
    """
    embed         = load_embedding(embed_root, task_name)
    prompt_embeds = embed.unsqueeze(0).to(device, dtype=torch.bfloat16)
    prompt_mask   = torch.ones((1, MAX_SEQ_LEN), dtype=torch.bool, device=device)

    if use_multitask:
        transformer.set_adapter(TASK_TO_ADAPTER.get(task_name, "blur"))

    z_in     = encode(img_tensor, vae)
    velocity = run_inference(z_in, transformer, vae, prompt_embeds, prompt_mask)
    z_pred   = z_in.float() + velocity.float()
    return decode(z_pred.to(vae.dtype), vae)   # (1, 3, H, W)


# ── Args ───────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(
        description="WindowSeat inference -- sequential multi-task restoration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir",   required=True,
                   help=("Folder containing the degraded input images AND\n"
                         "text_description.txt — auto-discovered, no separate flag needed."))
    p.add_argument("--embed-dir",   required=True,
                   help="Folder with per-task embedding sub-folders")
    p.add_argument("--output-dir",  required=True,
                   help="Where to save final restored images")
    p.add_argument("--checkpoint",  required=True,
                   help="Path to checkpoint_best.pt or checkpoint_latest.pt")
    p.add_argument("--base-model",      default=BASE_MODEL_URI)
    p.add_argument("--resolution",      type=int, default=608,
                   help="Spatial resolution -- must match training")
    p.add_argument("--lora-rank",       type=int, default=128,
                   help="LoRA rank -- must match training")
    p.add_argument("--lora-alpha",      type=int, default=128)
    p.add_argument("--lora-dropout",    type=float, default=0.0)
    p.add_argument("--use-multitask-lora", action="store_true",
                   help="Load adapters for multi-task LoRA checkpoints")
    p.add_argument("--output-suffix",   default="_restored",
                   help="Suffix appended to final output filenames")
    p.add_argument("--no-intermediates", action="store_true",
                   help="Do NOT save intermediate images for multi-step tasks")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    input_dir  = Path(args.input_dir)
    embed_root = Path(args.embed_dir)
    output_dir = Path(args.output_dir)
    inter_dir  = output_dir / "intermediates"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_intermediates = not args.no_intermediates
    if save_intermediates:
        inter_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Auto-discover and parse text_description.txt ─────────────────────
    desc_file = input_dir / "text_description.txt"
    if not desc_file.exists():
        raise FileNotFoundError(
            f"Expected text_description.txt inside --input-dir but not found:\n"
            f"  {desc_file}\n"
            f"Make sure the file is named exactly 'text_description.txt' "
            f"and lives in the same folder as your images."
        )
    print(f"\nFound description file: {desc_file}")
    prompt_map = parse_prompt_file(str(desc_file))

    # ── 2. Build and print restoration plan ──────────────────────────────────
    print("\nRestoration plan:")
    print(f"  {'File':<20}  {'Pipeline':<38}  Description")
    print("  " + "-" * 90)

    entries = []   # (image_path: Path, tasks: list[str], description: str)
    for filename, description in prompt_map.items():
        image_path = input_dir / filename
        if not image_path.exists():
            print(f"  [warn] Not found, skipping: {image_path}")
            continue
        tasks    = description_to_tasks(description)
        pipeline = " -> ".join(tasks)
        print(f"  {filename:<20}  {pipeline:<38}  {description[:48]}")
        entries.append((image_path, tasks, description))

    if not entries:
        print("\nNo valid images to process. Exiting.")
        return

    # ── 3. Load models ────────────────────────────────────────────────────────
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
        use_multitask_lora=use_multitask,
    )
    load_checkpoint(args.checkpoint, transformer, use_multitask)
    transformer.to(device)
    transformer.eval()

    # ── 4. Sequential multi-step inference ───────────────────────────────────
    total_passes = sum(len(t) for _, t, _ in entries)
    print(f"\nRunning inference -- {len(entries)} image(s), "
          f"{total_passes} total pass(es) ...\n")

    summary = []   # (orig_name, pipeline_str, out_name)

    outer_pbar = tqdm(
        entries,
        desc="Images",
        unit="img",
        dynamic_ncols=True,
        colour="cyan",
        position=0,
    )

    with torch.no_grad():
        for image_path, tasks, description in outer_pbar:
            stem    = image_path.stem
            suffix  = image_path.suffix
            n_steps = len(tasks)

            outer_pbar.set_description(
                f"{'->'.join(tasks)}  |  {image_path.name}"
            )

            # Load + reshape input image once
            current = load_and_preprocess(
                image_path, args.resolution, pbar=outer_pbar
            )
            current = current.unsqueeze(0).to(device, dtype=torch.bfloat16)
            # shape: (1, 3, 608, 608)

            # Inner progress bar — one tick per restoration pass
            step_pbar = tqdm(
                enumerate(tasks),
                total=n_steps,
                desc="  passes",
                unit="pass",
                dynamic_ncols=True,
                colour="green",
                position=1,
                leave=False,
            )

            for step_idx, task in step_pbar:
                is_last = (step_idx == n_steps - 1)
                step_pbar.set_description(
                    f"  pass {step_idx + 1}/{n_steps}  [{task}]"
                )

                # ── encode -> infer -> decode ─────────────────────────────────
                current = restore_one_pass(
                    current, task, embed_root,
                    transformer, vae, device, use_multitask,
                )
                # shape stays (1, 3, 608, 608)

                # Save intermediate (all passes except the last)
                if not is_last and save_intermediates:
                    inter_name = f"{stem}_step{step_idx + 1}_{task}{suffix}"
                    inter_path = inter_dir / inter_name
                    tensor_to_pil(current[0]).save(inter_path)
                    outer_pbar.write(
                        f"  [intermediate]  {inter_path.relative_to(output_dir)}"
                    )

            step_pbar.close()

            # Save final output
            out_name = f"{stem}{args.output_suffix}{suffix}"
            out_path = output_dir / out_name
            tensor_to_pil(current[0]).save(out_path)
            summary.append((image_path.name, " -> ".join(tasks), out_name))

    outer_pbar.close()

    # ── 5. Summary table ──────────────────────────────────────────────────────
    sep = "-" * 74
    print(f"\n{sep}")
    print(f"  Done.  {len(summary)} image(s) saved to: {output_dir}")
    print(sep)
    print(f"  {'Input':<20}  {'Pipeline':<32}  Output")
    print(sep)
    for orig, pipeline, out in summary:
        print(f"  {orig:<20}  {pipeline:<32}  {out}")
    print(sep)
    if save_intermediates:
        print(f"  Intermediates: {inter_dir}")
    print()


if __name__ == "__main__":
    main()