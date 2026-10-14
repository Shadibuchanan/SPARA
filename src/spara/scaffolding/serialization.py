"""Released prompts and answers, plus training-only helper inputs."""
from PIL import Image
from qwen_vl_utils import process_vision_info

from ..data import image_path
from .helper import helper_image

ASSISTANT = "<|im_start|>assistant"


def prompt(processor, root, split, row):
    """The released user turn with its image, then the assistant header without a
    trailing newline, matching the LIVR baseline's Mirage-derived serialization."""
    text = row["text_input"]
    if text.count("<image>") != 1 or "<|scaffold_" in text:
        raise ValueError("Expected one released image placeholder and no scaffold tokens")
    before, after = text.split("<image>")
    with Image.open(image_path(root, split, row)) as source:
        picture = source.convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "text", "text": before}, {"type": "image", "image": picture},
        {"type": "text", "text": after}]}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) + ASSISTANT
    images, _ = process_vision_info(messages)
    return dict(processor(text=[chat], images=images, padding=False, return_tensors="pt"))


def answer(processor, row):
    return processor.tokenizer(row["text_output"] + "<|im_end|>", add_special_tokens=False,
                               return_tensors="pt")["input_ids"]


def helper(processor, root, row, config):
    """Helper pixels for the scaffolding encoder only, preprocessed like input images."""
    messages = [{"role": "user", "content": [{"type": "image", "image": helper_image(root, row, config)}]}]
    images, _ = process_vision_info(messages)
    batch = processor.image_processor(images=images, return_tensors="pt")
    return {"pixel_values": batch["pixel_values"], "image_grid_thw": batch["image_grid_thw"]}


def training_sample(processor, root, row, config, with_helper=False):
    sample = {"prompt": prompt(processor, root, "train", row), "answer": answer(processor, row)}
    if with_helper:
        sample["helper"] = helper(processor, root, row, config)
    return sample
