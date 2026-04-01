#!/usr/bin/env python3
"""
app.py
------
Gradio demo for WindowSeat image restoration.

Upload a degraded image, type a restoration instruction, and get the
restored image back — all in a browser UI.

The model is loaded ONCE at startup from the checkpoint and stays in
GPU memory for the lifetime of the app.

Usage
-----
  python app.py \
      --checkpoint  runs/baseline_full/checkpoint_best.pt \
      --embed-dir   text/text_embeddings

  # Public shareable link (tunnelled via Gradio):
  python app.py ... --share

  # Custom host/port for running on a server:
  python app.py ... --host 0.0.0.0 --port 7860
"""

import argparse
import math
import re
from pathlib import Path

import gradio as gr
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

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_MODEL_URI = "Qwen/Qwen-Image-Edit-2509"
FIXED_TIMESTEP = 499
MAX_SEQ_LEN    = 256
RESOLUTION     = 608

ADAPTER_NAMES  = ["blur", "rain", "reflection"]
TASK_TO_ADAPTER = {
    "blur":                "blur",
    "raindrop":            "rain",
    "rainstreak":          "rain",
    "rainstreak_raindrop": "rain",
    "reflection":          "reflection",
}

TASK_DISPLAY = {
    "blur":                "🌫️  Blur removal",
    "raindrop":            "💧 Raindrop removal",
    "rainstreak":          "🌧️  Rain streak removal",
    "rainstreak_raindrop": "🌧️💧 Rain streak + raindrop removal",
    "reflection":          "🪟 Reflection removal",
}

_TASK_PATTERNS: list[tuple[str, str]] = [
    ("rainstreak", r"rain\s*streak"),
    ("raindrop",   r"rain\s*drop|raindrop"),
    ("rainstreak", r"streak"),
    ("raindrop",   r"\bdrop\b"),
    ("rainstreak", r"\brain\b"),
    ("blur",       r"blur|haze|defocus|fog"),
    ("reflection", r"reflect|glare|glass|window"),
]

# ── Global model state (loaded once at startup) ────────────────────────────────
_vae         = None
_transformer = None
_embed_root  = None
_device      = None
_use_multitask = False


# ── Task parsing (same logic as inference.py) ──────────────────────────────────
def description_to_tasks(description: str) -> list[str]:
    d        = description.lower()
    task_pos = {}
    for task_name, pattern in _TASK_PATTERNS:
        if task_name in task_pos:
            continue
        m = re.search(pattern, d)
        if m:
            task_pos[task_name] = m.start()

    if not task_pos:
        return ["raindrop"]

    ordered = [t for t, _ in sorted(task_pos.items(), key=lambda x: x[1])]

    # Merge rain-only combo into the dedicated combined embedding
    if set(ordered) == {"rainstreak", "raindrop"}:
        return ["rainstreak_raindrop"]

    return ordered


# ── Embedding ──────────────────────────────────────────────────────────────────
def load_embedding(task_name: str) -> torch.Tensor:
    embed_path = _embed_root / task_name / "0.pt"
    if not embed_path.exists():
        raise FileNotFoundError(
            f"No embedding found for task '{task_name}' at {embed_path}.\n"
            f"Make sure --embed-dir contains a sub-folder named '{task_name}'."
        )
    embed   = torch.load(embed_path, weights_only=True).squeeze(0)
    seq, D  = embed.shape
    if seq < MAX_SEQ_LEN:
        embed = torch.cat([embed, embed.new_zeros(MAX_SEQ_LEN - seq, D)], dim=0)
    else:
        embed = embed[:MAX_SEQ_LEN]
    return embed


# ── Image pre/post processing ──────────────────────────────────────────────────
def preprocess(pil_image: Image.Image) -> torch.Tensor:
    """
    PIL RGB → (1, 3, 608, 608) bfloat16 tensor in [-1, 1].
    Resizes short side to 608, centre-crops to 608×608.
    """
    img    = pil_image.convert("RGB")
    W, H   = img.size
    if (W, H) != (RESOLUTION, RESOLUTION):
        if min(W, H) != RESOLUTION:
            scale = RESOLUTION / min(W, H)
            img   = img.resize(
                (math.ceil(W * scale), math.ceil(H * scale)), Image.LANCZOS
            )
        img = TF.center_crop(img, (RESOLUTION, RESOLUTION))
    t = TF.to_tensor(img) * 2.0 - 1.0   # (3, H, W) in [-1, 1]
    return t.unsqueeze(0)                 # (1, 3, H, W)


def postprocess(tensor: torch.Tensor) -> Image.Image:
    """(1, 3, H, W) or (3, H, W) float in [-1, 1] → PIL RGB Image."""
    if tensor.ndim == 4:
        tensor = tensor[0]
    arr = ((tensor.float().clamp(-1, 1) + 1.0) / 2.0 * 255.0) \
          .byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


# ── VAE encode / decode ────────────────────────────────────────────────────────
@torch.no_grad()
def encode(images: torch.Tensor) -> torch.Tensor:
    images = images.to(device=_vae.device, dtype=_vae.dtype)
    out    = _vae.encode(images.unsqueeze(2)).latent_dist.sample()
    mean   = torch.tensor(_vae.config.latents_mean, device=out.device, dtype=out.dtype)
    std    = torch.tensor(_vae.config.latents_std,  device=out.device, dtype=out.dtype)
    mean   = mean.view(1, _vae.config.z_dim, 1, 1, 1)
    std    = (1.0 / std).view(1, _vae.config.z_dim, 1, 1, 1)
    return (out - mean) * std


@torch.no_grad()
def decode(latents: torch.Tensor) -> torch.Tensor:
    mean    = torch.tensor(_vae.config.latents_mean, device=latents.device, dtype=latents.dtype)
    std     = torch.tensor(_vae.config.latents_std,  device=latents.device, dtype=latents.dtype)
    mean    = mean.view(1, _vae.config.z_dim, 1, 1, 1)
    std     = (1.0 / std).view(1, _vae.config.z_dim, 1, 1, 1)
    latents = latents / std + mean
    return _vae.decode(latents).sample[:, :, 0]   # (B, 3, H, W)


# ── Transformer forward ────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(latent_input, prompt_embeds, prompt_mask):
    lat4d = latent_input[:, :, 0] if latent_input.ndim == 5 else latent_input
    B, C, H, W = lat4d.shape
    device = next(_transformer.parameters()).device

    packed = QwenImageEditPipeline._pack_latents(
        lat4d, batch_size=B, num_channels_latents=C, height=H, width=W,
    ).to(torch.bfloat16)

    timestep     = torch.full((B,), float(FIXED_TIMESTEP) / 1000.0,
                              device=device, dtype=torch.bfloat16)
    img_shapes   = [[(1, H // 2, W // 2)]] * B
    txt_seq_lens = prompt_mask.sum(dim=1).tolist()

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model_pred = _transformer(
            hidden_states=packed,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds.to(device, dtype=torch.bfloat16),
            encoder_hidden_states_mask=prompt_mask.to(device),
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            guidance=None,
            return_dict=False,
        )[0]

    td = _vae.config.get("temperal_downsample", None)
    sf = 2 ** len(td) if td is not None else 8
    return QwenImageEditPipeline._unpack_latents(
        model_pred, height=H * sf, width=W * sf, vae_scale_factor=sf,
    )


# ── Core restore function — called by Gradio ───────────────────────────────────
def restore(pil_image: Image.Image, instruction: str):
    """
    Main Gradio callback.
    Returns (restored_PIL_image, status_string).
    """
    if pil_image is None:
        return None, "⚠️  Please upload an image first."
    if not instruction.strip():
        return None, "⚠️  Please enter a restoration instruction."

    # Parse instruction → ordered task list
    tasks = description_to_tasks(instruction)
    pipeline_str = " → ".join(TASK_DISPLAY.get(t, t) for t in tasks)
    print(f"\nInstruction : {instruction}")
    print(f"Pipeline    : {pipeline_str}")

    # Preprocess input image
    img_tensor = preprocess(pil_image).to(_device, dtype=torch.bfloat16)
    current    = img_tensor

    # Sequential passes
    with torch.no_grad():
        for step_idx, task in enumerate(tasks):
            print(f"  Pass {step_idx + 1}/{len(tasks)}: [{task}]")

            embed         = load_embedding(task)
            prompt_embeds = embed.unsqueeze(0).to(_device, dtype=torch.bfloat16)
            prompt_mask   = torch.ones((1, MAX_SEQ_LEN), dtype=torch.bool, device=_device)

            if _use_multitask:
                _transformer.set_adapter(TASK_TO_ADAPTER.get(task, "blur"))

            z_in     = encode(current)
            velocity = run_inference(z_in, prompt_embeds, prompt_mask)
            z_pred   = z_in.float() + velocity.float()
            current  = decode(z_pred.to(_vae.dtype))

    restored_pil = postprocess(current)
    status       = f"✅  Done — {pipeline_str}"
    print(f"  {status}")
    return restored_pil, status


# ── Model loading ──────────────────────────────────────────────────────────────
def _make_lora_config(rank, alpha, dropout):
    return LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=["to_q", "to_k", "to_v", "to_out.0",
                        "ff.net.0.proj", "ff.net.2"],
        bias="none",
        init_lora_weights="gaussian",
    )


def load_models(checkpoint_path: str, base_model: str,
                lora_rank: int, lora_alpha: int, lora_dropout: float):
    global _vae, _transformer, _use_multitask, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {_device}")
    print(f"Loading base model: {base_model}")

    _vae = AutoencoderKLQwenImage.from_pretrained(
        base_model, subfolder="vae",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )
    _vae.to(_device, dtype=torch.bfloat16).eval()
    _vae.requires_grad_(False)

    ckpt_meta     = torch.load(checkpoint_path, map_location="cpu")
    _use_multitask = (
        "lora_state_dicts" in ckpt_meta
        or ckpt_meta.get("use_multitask_lora", False)
    )

    transformer = QwenImageTransformer2DModel.from_pretrained(
        base_model, subfolder="transformer",
        torch_dtype=torch.bfloat16,
        device_map=_device,
    )
    cfg = _make_lora_config(lora_rank, lora_alpha, lora_dropout)
    if _use_multitask:
        transformer = get_peft_model(transformer, cfg, adapter_name=ADAPTER_NAMES[0])
        for name in ADAPTER_NAMES[1:]:
            transformer.add_adapter(name, cfg)
    else:
        transformer = get_peft_model(transformer, cfg)

    print(f"Loading checkpoint: {checkpoint_path}")
    if "lora_state_dict" in ckpt_meta:
        miss, unexp = transformer.load_state_dict(ckpt_meta["lora_state_dict"], strict=False)
        if miss or unexp:
            print(f"  Missing: {len(miss)}  Unexpected: {len(unexp)}")
    elif "lora_state_dicts" in ckpt_meta:
        for adapter_name, state in ckpt_meta["lora_state_dicts"].items():
            transformer.set_adapter(adapter_name)
            miss, unexp = transformer.load_state_dict(state, strict=False)
            if miss or unexp:
                print(f"  [{adapter_name}] Missing: {len(miss)}  Unexpected: {len(unexp)}")

    step = ckpt_meta.get("step", "?")
    m    = ckpt_meta.get("metrics", {})
    print(f"  Step {step}  |  "
          f"val PSNR={m.get('val/psnr', 'n/a')}  "
          f"val SSIM={m.get('val/ssim', 'n/a')}")

    _transformer = transformer
    _transformer.to(_device).eval()
    print("Models ready.\n")


# ── Gradio UI ──────────────────────────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    examples = [
        ["Remove raindrops."],
        ["Remove raindrops and reflection."],
        ["Remove rain streaks and raindrops."],
        ["Remove blur."],
        ["Remove reflection."],
        ["Remove raindrops and remove blur."],
    ]

    with gr.Blocks(
        title="WindowSeat — Image Restoration Demo",
    ) as demo:

        gr.Markdown(
            """
            # 🪟 WindowSeat — Image Restoration Demo
            Upload a degraded image (rain, blur, or reflection through a window),
            describe what to remove, and the model restores it.

            **Multi-step restoration is supported** — e.g. *"Remove raindrops and reflection"*
            runs two sequential passes automatically.
            """
        )

        with gr.Row():
            # ── Left column: inputs ───────────────────────────────────────────
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Input image (degraded)",
                    type="pil",
                    height=400,
                )
                instruction = gr.Textbox(
                    label="Restoration instruction",
                    placeholder=(
                        "e.g.  Remove raindrops.\n"
                        "      Remove raindrops and reflection.\n"
                        "      Remove blur and raindrops."
                    ),
                    lines=3,
                )
                run_btn = gr.Button("✨  Restore image", variant="primary", size="lg")

                gr.Markdown("### Example instructions")
                gr.Examples(
                    examples=examples,
                    inputs=instruction,
                    label="",
                )

            # ── Right column: output ──────────────────────────────────────────
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Restored image",
                    type="pil",
                    height=400,
                    interactive=False,
                )
                status_box = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=1,
                )

        run_btn.click(
            fn=restore,
            inputs=[input_image, instruction],
            outputs=[output_image, status_box],
        )

        # Also trigger on pressing Enter in the instruction box
        instruction.submit(
            fn=restore,
            inputs=[input_image, instruction],
            outputs=[output_image, status_box],
        )

        gr.Markdown(
            """
            ---
            **Supported degradation types**

            | Instruction keyword | Task |
            |---|---|
            | rain, raindrop, drop | Raindrop removal |
            | streak, rain streak | Rain streak removal |
            | blur, haze, defocus, fog | Blur removal |
            | reflect, glare, glass, window | Reflection removal |

            Multi-keyword instructions run as sequential passes in the order you write them.
            """
        )

    return demo


# ── Args ───────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(
        description="WindowSeat Gradio demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",  required=True,
                   help="Path to checkpoint_best.pt")
    p.add_argument("--embed-dir",   required=True,
                   help="Folder containing per-task embedding sub-folders")
    p.add_argument("--base-model",  default=BASE_MODEL_URI)
    p.add_argument("--lora-rank",   type=int, default=128)
    p.add_argument("--lora-alpha",  type=int, default=128)
    p.add_argument("--lora-dropout",type=float, default=0.0)
    p.add_argument("--share",       action="store_true",
                   help="Create a public Gradio share link (tunnelled)")
    p.add_argument("--host",        default="127.0.0.1",
                   help="Host to bind to (use 0.0.0.0 for all interfaces on a server)")
    p.add_argument("--port",        type=int, default=7860)
    return p.parse_args()


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = get_args()

    # Store embed root globally so restore() can access it
    _embed_root = Path(args.embed_dir)

    # Load models once before the UI starts
    load_models(
        checkpoint_path=args.checkpoint,
        base_model=args.base_model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
    )