import torch
import json
import os
from pathlib import Path
from tqdm import tqdm
from diffusers import QwenImageEditPipeline


class TextEmbeddingProcessor:
    """
    Handles the pre-computation of text embeddings using the Qwen Image Edit model.
    Decoupled from file management to allow flexible execution on GPU environments.
    """

    def __init__(self, model_path="Qwen/Qwen-Image-Edit-2509", device="cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        print(f"Loading Text Encoder from {model_path}...")

        # We only need the tokenizer and text_encoder, but loading from pipeline is safer
        # to ensure all internal configurations match.
        pipe = QwenImageEditPipeline.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            device_map=self.device
        )
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.text_encoder.eval()  # Set to evaluation mode

        print("Text Encoder loaded successfully.")

    def run(self, prompt_dict, output_root="text/text_embeddings"):
        """
        Processes a dictionary of prompts and saves them as .pt files.
        :param prompt_dict: Dictionary { "task_name": ["prompt1", "prompt2", ...] }
        :param output_root: Target directory for .pt files
        """
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        for task_name, prompts in prompt_dict.items():
            task_dir = output_root / task_name
            task_dir.mkdir(exist_ok=True)

            print(f"Processing task: {task_name} ({len(prompts)} prompts)")

            for i, text in enumerate(tqdm(prompts, desc=f"Encoding {task_name}")):
                embeds = self._encode_text(text)

                # Save as PyTorch binary (Shape: [1, 77, 4096])
                save_path = task_dir / f"{i}.pt"
                torch.save(embeds.cpu(), save_path)

        print(f"\nAll embeddings saved to: {output_root}")

    @torch.no_grad()
    def _encode_text(self, text):
        """Internal helper to convert string to tensor."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True
        ).to(self.device)

        # Get hidden states from the last layer
        outputs = self.text_encoder(
            inputs.input_ids,
            attention_mask=inputs.attention_mask
        )
        return outputs.last_hidden_state  # [1, 77, 4096]



if __name__ == "__main__":
    PROMPTS_JSON = "text/text_prompts.json"
    with open(PROMPTS_JSON, 'r', encoding='utf-8') as f:
        task_prompts = json.load(f)

    processor = TextEmbeddingProcessor()
    processor.run(task_prompts)