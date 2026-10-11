import os
from pathlib import Path
import pytest
import torch
from transformers import AutoProcessor
from spara.data import records
from spara.model import LATENT_TOKENS, initialize_latents
from spara.serialization import encode


@pytest.mark.integration
def test_real_tokenizer_input_boundary_and_no_metadata(tiny_model, tmp_path):
    root = Path(os.environ.get("SPARA_TEST_DATA", "data/vsp"))
    processor_path = Path(os.environ.get("SPARA_TEST_PROCESSOR", ".cache/processor"))
    if not processor_path.is_dir() or not (root / "manifest.json").is_file():
        pytest.skip("Prepare reference data and processor")
    processor = AutoProcessor.from_pretrained(processor_path)
    ids = initialize_latents(tiny_model, processor)
    assert ids == [151665, 151666, 151667, 151668]
    row = records(root, "train")[0]
    altered = dict(row, image_output="does-not-exist.png", map_desc="MUST_NOT_APPEAR", map_id="MUST_NOT_APPEAR")
    prefix = encode(processor, root, "train", altered, training=False)
    training = encode(processor, root, "train", altered)
    length = prefix["input_ids"].shape[-1]
    assert torch.equal(training["input_ids"][:, :length], prefix["input_ids"])
    assert training["labels"][:, :length].eq(-100).all()
    assert training["labels"][:, length:].ne(-100).all()
    assert prefix["input_ids"][0, -1].item() not in ids
    decoded = processor.tokenizer.decode(prefix["input_ids"][0])
    assert decoded.endswith("".join(LATENT_TOKENS) + "<|im_start|>assistant")
    assert "MUST_NOT_APPEAR" not in decoded
    assert processor.tokenizer.decode(training["labels"][0, length:]) == row["text_output"] + "<|im_end|>"
    assert prefix["input_ids"].eq(151655).sum().item() == 121
    assert training["image_grid_thw"].tolist() == [[1, 22, 22]]
    tiny_model.save_pretrained(tmp_path)
    processor.save_pretrained(tmp_path)
    restored = AutoProcessor.from_pretrained(tmp_path)
    assert restored.tokenizer.convert_tokens_to_ids(LATENT_TOKENS) == ids
    assert torch.equal(encode(restored, root, "train", row)["input_ids"], training["input_ids"])
