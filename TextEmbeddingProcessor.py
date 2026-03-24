import torch
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from diffusers import QwenImageEditPipeline

MAX_SEQ_LEN = 256  # must match train_windowseat.py

# Smallest valid dummy image — just needs to exist, content doesn't matter
# since we only want the text embeddings from the output
DUMMY_IMAGE = Image.new("RGB", (64, 64), color=0)


class TextEmbeddingProcessor:
    def __init__(
        self,
        model_path: str = "Qwen/Qwen-Image-Edit-2509",
        device: str = "cuda",
        dtype=torch.bfloat16,
    ):
        self.dtype = dtype
        print(f"Loading pipeline from {model_path} ...")
        self.pipe = QwenImageEditPipeline.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device,
        )
        print("Pipeline loaded.")

    def run(self, prompt_dict: dict, output_root: str = "text/text_embeddings"):
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        for task_name, prompts in prompt_dict.items():
            task_dir = output_root / task_name
            task_dir.mkdir(exist_ok=True)

            print(f"\nTask: {task_name}  ({len(prompts)} prompts)")
            for i, text in enumerate(tqdm(prompts, desc=f"  Encoding {task_name}")):
                embeds = self._encode(text)              # (1, MAX_SEQ_LEN, hidden_dim)
                torch.save(embeds.cpu(), task_dir / f"{i}.pt")

        print(f"\nAll embeddings saved to: {output_root}")

    @torch.no_grad()
    def _encode(self, text: str) -> torch.Tensor:
        device = self.pipe.text_encoder.device

        result = self.pipe.encode_prompt(
            prompt=text,
            image=DUMMY_IMAGE,      # pipeline always needs an image internally
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=MAX_SEQ_LEN,
        )

        # Returns (prompt_embeds, prompt_embeds_mask)
        prompt_embeds = result[0] if isinstance(result, tuple) else result
        return prompt_embeds   # (1, MAX_SEQ_LEN, hidden_dim)


if __name__ == "__main__":
    PROMPTS_JSON = "text/text_prompts.json"

    with open(PROMPTS_JSON, "r", encoding="utf-8") as f:
        task_prompts = json.load(f)

    processor = TextEmbeddingProcessor()
    processor.run(task_prompts)