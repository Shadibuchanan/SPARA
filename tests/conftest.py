import torch
import pytest
from transformers import Qwen2_5_VLConfig
from spara.model import LivrQwen, freeze_vision


@pytest.fixture
def tiny_model():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    config = Qwen2_5_VLConfig(
        vocab_size=40, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        image_token_id=3, video_token_id=4, vision_start_token_id=2,
        vision_end_token_id=5, pad_token_id=0, bos_token_id=1, eos_token_id=6,
        rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                       "out_hidden_size": 32, "num_heads": 4, "patch_size": 14,
                       "temporal_patch_size": 2, "spatial_merge_size": 2,
                       "window_size": 28, "fullatt_block_indexes": [0]},
    )
    config._attn_implementation = "eager"
    config.livr_latent_ids = [10, 11, 12, 13]
    config.livr_stage = 1
    model = LivrQwen(config)
    freeze_vision(model)
    return model


@pytest.fixture
def tiny_batch():
    # One visual token from 2x2 spatial patches; four latents then two header tokens.
    ids = torch.tensor([[1, 2, 3, 5, 7, 10, 11, 12, 13, 8, 9, 14, 6]])
    labels = ids.clone(); labels[:, :11] = -100
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels,
            "pixel_values": torch.randn(4, 3 * 2 * 14 * 14), "image_grid_thw": torch.tensor([[1, 2, 2]])}
