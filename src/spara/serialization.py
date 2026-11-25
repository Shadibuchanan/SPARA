"""Preserve released text and image placement; expose no helper/map metadata."""
from PIL import Image
import torch
from qwen_vl_utils import process_vision_info

from .data import image_path, image_paths
from .model import LATENT_TOKENS

ASSISTANT = "<|im_start|>assistant"


def latent_kwargs(model):
    """`encode`'s latent argument for `model`: region latents by grid size, or patch latents by image, when
    installed (spara.region_latents); otherwise none, so `encode` inserts LIVR's four, as every earlier run did."""
    return {"latents": model.config.livr_latent_tokens} if getattr(model.config, "livr_region_latents", None) else {}


def pictures(root, split, row):
    """The row's images in placeholder order: the released map, or each of `image_inputs` (several images,
    e.g. a perception question), as RGB."""
    paths = image_paths(root, split, row) if "image_inputs" in row else [image_path(root, split, row)]
    out = []
    for path in paths:
        with Image.open(path) as source:
            out.append(source.convert("RGB"))
    return out


def encode(processor, root, split, row, training=True, latents=LATENT_TOKENS, generation_prompt=False):
    """`latents`: a list of latent tokens; {grid size (as a string): tokens} for region latents; or
    {"globals": tokens, "per_image": [tokens per image]} for patch latents, which carry one set per image.
    `generation_prompt`: the chat template's own assistant header, as the released model was trained;
    for a model without latent tokens (spara.reasoning_probe)."""
    if generation_prompt and latents:
        raise ValueError("The chat template's assistant header carries no latent tokens")
    text = row["text_input"]
    images = pictures(root, split, row)
    if text.count("<image>") != len(images) or "<|livr_" in text:
        raise ValueError("Expected one image placeholder per image and no latent tokens")
    if row.get("blank_image"):  # the text-only ceiling (docs/PERCEPTION_CEILING_SPEC.md): same size, no map
        images = [Image.new("RGB", picture.size, "white") for picture in images]
    grid_size = None
    if isinstance(latents, dict) and "per_image" in latents:  # patch latents: the mask needs only the images
        if len(images) > len(latents["per_image"]):
            raise ValueError("More images than the patch latents cover")
        latents = latents["globals"] + [t for tokens in latents["per_image"][:len(images)] for t in tokens]
    elif isinstance(latents, dict):  # region latents: the model also needs the grid size for its block mask
        grid_size = len(row["map_desc"])
        latents = latents[str(grid_size)]
    pieces = text.split("<image>")
    content = [{"type": "text", "text": pieces[0]}]
    for picture, piece in zip(images, pieces[1:]):
        content += [{"type": "image", "image": picture}, {"type": "text", "text": piece}]
    messages = [{"role": "user", "content": content}]
    if generation_prompt:
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
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
    batch = dict(batch)
    if grid_size is not None:
        batch["livr_grid_size"] = torch.tensor([grid_size])
    if row.get("livr_hide_globals"):  # spara.region_latents.hidden: the answer may not see the global latents
        batch["livr_hide_globals"] = torch.tensor([True])
    return batch
