"""Preserve released text and image placement; expose no helper/map metadata."""
from PIL import Image
import torch
from qwen_vl_utils import process_vision_info

from .data import image_path
from .model import LATENT_TOKENS

ASSISTANT = "<|im_start|>assistant"


def encode(processor, root, split, row, training=True, latents=LATENT_TOKENS):
    text = row["text_input"]
    if text.count("<image>") != 1 or any(t in text for t in LATENT_TOKENS):
        raise ValueError("Expected one released image placeholder and no latent tokens")
    before, after = text.split("<image>")
    with Image.open(image_path(root, split, row)) as source:
        picture = source.convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "text", "text": before}, {"type": "image", "image": picture},
        {"type": "text", "text": after}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt += "".join(latents) + ASSISTANT
    images, _ = process_vision_info(messages)
    batch = processor(text=[prompt], images=images, padding=False, return_tensors="pt")
    # Tokenize the continuation separately to make the prefix boundary identical
    # for training and inference (no BPE merge across the supervised boundary).
    if training:
        answer = processor.tokenizer(row["text_output"] + "<|im_end|>", add_special_tokens=False, return_tensors="pt")["input_ids"]
        prefix_length = batch["input_ids"].shape[-1]
        batch["input_ids"] = torch.cat([batch["input_ids"], answer], dim=-1)
        batch["attention_mask"] = torch.ones_like(batch["input_ids"])
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][:, :prefix_length] = -100
    return dict(batch)
