import numpy as np
from PIL import Image

import torch

from spara.vstar_attention import attention_rows, band_means, cells, heat, overlay


def test_cells_cover_the_box_and_stay_in_the_grid():
    assert cells([0, 0, 28, 28], 1, 1, 10, 10) == (0, 0, 1, 1)
    assert cells([30, 60, 30, 1], 1, 1, 10, 10) == (1, 2, 3, 3)
    assert cells([560, 560, 100, 100], 0.5, 0.5, 10, 10) == (9, 9, 10, 10)  # scaled: 280..330 px is cell 10, clipped
    assert cells([5000, 5000, 10, 10], 1, 1, 10, 10) == (9, 9, 10, 10)


def test_band_means():
    assert band_means([1, 1, 2, 2, 3, 3]) == [1, 2, 3, 2]


def test_heat_and_overlay():
    assert heat(np.array([[0.0, 1.0]])).tolist() == [[[0, 0, 0], [1, 1, 1]]]
    picture = Image.new("RGB", (56, 28), (10, 200, 30))
    panel = overlay(picture, np.array([[0.0, 1.0]]), (112, 56), boxes=[[0, 0, 10, 10]])
    assert panel.size == (112, 56)
    assert overlay(picture, None, (56, 28)).getpixel((30, 20)) == (10, 200, 30)


def test_attention_rows_match_the_eager_implementation():
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention
    torch.manual_seed(0)
    config = Qwen2_5_VLConfig(hidden_size=32, num_attention_heads=4, num_key_value_heads=2,
                              rope_scaling={"type": "default", "mrope_section": [2, 1, 1]})
    module = Qwen2_5_VLAttention(config, layer_idx=0).eval()
    length, rows = 9, 3
    hidden = torch.randn(1, length, 32)
    angles = torch.randn(3, 1, length, 8)
    embeddings = (angles.cos(), angles.sin())
    causal = torch.full((length, length), float("-inf")).triu(1)[None, None]
    blocked = causal.clone()
    blocked[..., -1, 2] = float("-inf")  # a mask of our own: the last position may not see key 2
    for mask in (causal, blocked):
        _, expected, _ = module(hidden_states=hidden, attention_mask=mask, position_embeddings=embeddings,
                                output_attentions=True)
        ours = attention_rows(module, hidden, mask, embeddings, rows)
        assert torch.allclose(ours, expected[:, :, -rows:], atol=1e-6)
    assert torch.allclose(attention_rows(module, hidden, None, embeddings, rows),
                          attention_rows(module, hidden, causal, embeddings, rows), atol=1e-6)
