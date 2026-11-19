"""Released prompts and answers, plus training-only helper inputs."""
from qwen_vl_utils import process_vision_info

from ..serialization import pictures
from .helper import helper_image

ASSISTANT = "<|im_start|>assistant"


def prompt(processor, root, split, row):
    """The released user turn with its images (one, or several for perception questions), then the
    assistant header without a trailing newline, matching the LIVR baseline's Mirage-derived serialization."""
    text = row["text_input"]
    images = pictures(root, split, row)
    if text.count("<image>") != len(images) or "<|scaffold_" in text:
        raise ValueError("Expected one image placeholder per image and no scaffold tokens")
    pieces = text.split("<image>")
    content = [{"type": "text", "text": pieces[0]}]
    for picture, piece in zip(images, pieces[1:]):
        content += [{"type": "image", "image": picture}, {"type": "text", "text": piece}]
    messages = [{"role": "user", "content": content}]
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


def training_sample(processor, root, row, config, with_helper=False, split="train"):
    sample = {"prompt": prompt(processor, root, split, row), "answer": answer(processor, row)}
    if with_helper:
        sample["helper"] = helper(processor, root, row, config)
    return sample
