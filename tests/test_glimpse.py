from types import SimpleNamespace

import torch

from spara.glimpse import restrict_glimpses, with_glimpses
from spara.model import LATENT_TOKENS, allowed_mask

IMAGE, LATENTS, COMMA = 3, [10, 11, 12, 13], 30
DIGITS = list(range(20, 30))
CONFIG = SimpleNamespace(image_token_id=IMAGE, livr_latent_ids=LATENTS,
                         livr_glimpse={"digits": DIGITS, "comma": COMMA},
                         vision_config=SimpleNamespace(spatial_merge_size=2, patch_size=14))


def cell(r, c):
    return [DIGITS[r], COMMA, DIGITS[c]] + LATENTS


def seen(allowed, ids, q):
    """The image tokens position `q` may attend to, as (row, column) on the 7 x 7 token grid."""
    image = (ids[0] == IMAGE).nonzero().squeeze(-1)
    return {(int(i) // 7, int(i) % 7) for i in (allowed[0, q, image]).nonzero().squeeze(-1)}


def test_each_written_latent_sees_only_the_neighbouring_cell_in_its_direction():
    # A 3 x 3 map on 7 x 7 image tokens (196 px at 64 px per cell): the cells own token rows 0-1, 2-4 and 5-6.
    prompt = [1, 2] + [IMAGE] * 49 + [5, 7] + LATENTS + [8, 9]
    ids = torch.tensor([prompt + cell(1, 1) + [14, 15] + cell(0, 0) + [14] + LATENTS])
    padding = torch.ones_like(ids)
    start = len(prompt)
    band = {0: (0, 1), 1: (2, 3, 4), 2: (5, 6)}
    tokens = lambda r, c: {(i, j) for i in band[r] for j in band[c]}
    for stage_one in (True, False):
        allowed = restrict_glimpses(allowed_mask(ids, padding, LATENTS, IMAGE, image_block=stage_one), ids, 0, CONFIG)
        up, down, left, right = (start + 3 + k for k in range(4))
        assert seen(allowed, ids, up) == tokens(0, 1) and seen(allowed, ids, down) == tokens(2, 1)
        assert seen(allowed, ids, left) == tokens(1, 0) and seen(allowed, ids, right) == tokens(1, 2)
        corner = start + 7 + 2 + 3  # the latents after "0,0": up and left are off the grid
        assert seen(allowed, ids, corner) == seen(allowed, ids, corner + 2) == set()
        assert seen(allowed, ids, corner + 1) == tokens(1, 0) and seen(allowed, ids, corner + 3) == tokens(0, 1)
        assert all(seen(allowed, ids, q) == set() for q in range(ids.shape[1] - 4, ids.shape[1]))  # no cell named
        assert allowed[0, up, :2].all() and allowed[0, up, start:up + 1].all()  # the text before it, and itself
        assert len(seen(allowed, ids, 2 + 49 + 2)) == 49  # a prompt latent still sees the whole image
        assert len(seen(allowed, ids, start + 7)) == (0 if stage_one else 49)  # the text of the answer, as before
    one_step = restrict_glimpses(allowed_mask(ids, padding, LATENTS, IMAGE, query_start=up, image_block=False),
                                 ids, up, CONFIG)  # cached decoding reads the mask from its query position on
    assert torch.equal(one_step, allowed[:, up:])


def test_the_model_runs_with_glimpses_in_both_stages(tiny_model, tiny_batch):
    tiny_model.config.livr_glimpse = {"digits": DIGITS, "comma": COMMA}
    ids = torch.cat([tiny_batch["input_ids"][:, :-1], torch.tensor([cell(1, 1) + [6]])], dim=-1)
    batch = tiny_batch | {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}
    for stage in (1, 2):  # one image token is not a map: the written latents then see no image token
        tiny_model.config.livr_stage = stage
        assert torch.isfinite(tiny_model(**batch).loss)
    prompt = {k: v for k, v in batch.items() if k != "labels"}
    assert tiny_model.generate(**prompt, max_new_tokens=3, do_sample=False).shape[1] > ids.shape[1]


def test_glimpses_follow_each_cell_and_leave_the_text_as_the_hole_check():
    text = "0,0 hole - > D\n1,0 hole R > D\n\\boxed{DOWN, DOWN}"
    out = with_glimpses({"text_output": text})["text_output"]
    four = "".join(LATENT_TOKENS)
    assert out == f"0,0{four} hole - > D\n1,0{four} hole R > D\n\\boxed{{DOWN, DOWN}}"
    assert out.replace(four, "") == text
