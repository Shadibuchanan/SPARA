import os
from pathlib import Path
import pytest
import torch
from transformers import AutoProcessor
from spara.data import records
from spara.model import LATENT_TOKENS, initialize_latents
from spara.runtime import read_config
from spara.serialization import encode
from spara.training import epoch_order


@pytest.mark.integration
def test_sft_prefix_is_livr_prefix_minus_latents(tiny_model):
    root = Path(os.environ.get("SPARA_TEST_DATA", "data/vsp"))
    processor_path = Path(os.environ.get("SPARA_TEST_PROCESSOR", ".cache/processor"))
    if not processor_path.is_dir() or not (root / "manifest.json").is_file():
        pytest.skip("Prepare reference data and processor")
    processor = AutoProcessor.from_pretrained(processor_path)
    # The SFT run never adds latent tokens to its tokenizer.
    pristine = AutoProcessor.from_pretrained(processor_path)
    ids = initialize_latents(tiny_model, processor)
    assert ids == [151665, 151666, 151667, 151668]
    row = records(root, "train")[0]
    livr, sft = encode(processor, root, "train", row), encode(pristine, root, "train", row, latents=[])
    livr_prefix = encode(processor, root, "train", row, training=False)["input_ids"][0]
    sft_prefix = encode(pristine, root, "train", row, training=False, latents=[])["input_ids"][0]
    assert livr_prefix.shape[-1] - sft_prefix.shape[-1] == 4
    kept = ~torch.isin(livr_prefix, torch.tensor(ids))
    assert kept.sum().item() == sft_prefix.shape[-1]
    assert torch.equal(livr_prefix[kept], sft_prefix)
    assert torch.equal(livr["input_ids"][0][~torch.isin(livr["input_ids"][0], torch.tensor(ids))], sft["input_ids"][0])
    assert sft_prefix.eq(151655).sum().item() == 121
    assert sft["image_grid_thw"].tolist() == livr["image_grid_thw"].tolist() == [[1, 22, 22]]
    assert torch.equal(sft["pixel_values"], livr["pixel_values"])
    length = sft_prefix.shape[-1]
    assert sft["labels"][:, :length].eq(-100).all() and sft["labels"][:, length:].ne(-100).all()
    assert torch.equal(sft["labels"][:, length:], livr["labels"][:, length + 4:])
    assert pristine.tokenizer.decode(sft["labels"][0, length:]) == row["text_output"] + "<|im_end|>"
    decoded = pristine.tokenizer.decode(sft_prefix)
    assert decoded.endswith("<|im_start|>assistant") and not any(t in decoded for t in LATENT_TOKENS)


def test_order_cycle_replays_livr_permutations():
    livr = read_config("configs/livr.json")
    sft = read_config("configs/sft.json")
    staged = read_config("configs/sft-staged.json")
    livr_orders = [epoch_order(livr, epoch, 1000) for epoch in range(livr["epochs_per_stage"])]
    # LIVR's effective seeding is unchanged: seed + within-stage epoch.
    for epoch, order in enumerate(livr_orders):
        expected = torch.randperm(1000, generator=torch.Generator().manual_seed(42 + epoch)).tolist()
        assert order == expected
    assert [epoch_order(sft, epoch, 1000) for epoch in range(20)] == livr_orders * 2
    assert [epoch_order(staged, epoch, 1000) for epoch in range(10)] == livr_orders
