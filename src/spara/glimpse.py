"""Glimpse latents (docs/HOLE_CHECK_SPEC.md, "Glimpse latents: G3"): per-move latents for the hole check.

After each line's cell the answer carries LIVR's four latent tokens, one per direction (U, D, L, R). Of the
image, each may attend only to the cell next to the named cell in its direction, so the lookup the lines
failed to learn is done by the mask and the latent only has to say whether that one cell is a hole. The
grid size comes from the image itself: the maps are drawn at `CELL_PIXELS` per cell.
"""
import math
import re

import torch

from .hole_check import MOVES
from .latent_battery import cell_weights
from .model import LATENT_TOKENS, latent_queries

CELL_PIXELS = 64
CELL = re.compile(r"^(\d+,\d+) hole", re.MULTILINE)


def install(model, processor):
    """Turn the glimpse mask on: the model then needs the token IDs that spell a cell."""
    ids = [processor.tokenizer(text, add_special_tokens=False)["input_ids"] for text in [*map(str, range(10)), ","]]
    if any(len(found) != 1 for found in ids):
        raise ValueError("A cell must tokenize as digit, comma, digit")
    model.config.livr_glimpse = {"digits": [found[0] for found in ids[:10]], "comma": ids[10][0]}


def with_glimpses(row):
    """A hole-check row with the four latents after each line's cell."""
    return row | {"text_output": CELL.sub(lambda m: m[1] + "".join(LATENT_TOKENS) + " hole", row["text_output"])}


def cell_tokens(image_tokens, config):
    """(grid size, [cells, image tokens] boolean): each image token belongs to the cell that covers most of
    it. The grid size is 0 for an image that is not a square map."""
    side = math.isqrt(image_tokens)
    merge = config.vision_config.spatial_merge_size
    n = round(side * merge * config.vision_config.patch_size / CELL_PIXELS)
    if side * side != image_tokens or n < 1:
        return 0, None
    owner = cell_weights(n, (1, side * merge, side * merge), merge).argmax(0)
    return n, owner[None, :] == torch.arange(n * n)[:, None]


def restrict_glimpses(allowed, full_ids, query_start, config):
    """In the boolean mask [B, Q, K], give each latent written in the answer its glimpse: of the image tokens,
    only those of the cell next to the cell named before the four latents, in the latent's direction. A
    latent with no such cell (off the grid, or no cell named) sees no image token, and with `blind` set none
    does (the ablation). The latents' prompt occurrences are untouched."""
    glimpse, latents = config.livr_glimpse, config.livr_latent_ids[:4]
    digit = {token: value for value, token in enumerate(glimpse["digits"])}
    allowed = allowed.clone()
    for b in range(full_ids.shape[0]):
        ids = full_ids[b]
        image = (ids == config.image_token_id).nonzero().squeeze(-1)
        n, cells = cell_tokens(len(image), config)
        written = ~latent_queries(ids[None], latents)[0]
        for k, (token, (dr, dc)) in enumerate(zip(latents, MOVES.values())):
            for q in ((ids == token) & written).nonzero().squeeze(-1).tolist():
                if q < query_start:
                    continue
                see = torch.zeros(len(image), dtype=torch.bool, device=ids.device)
                at = q - k  # the first of the four latents; the cell is the three tokens before it
                if n and at >= 3 and int(ids[at - 2]) == glimpse["comma"] and not glimpse.get("blind"):
                    r, c = digit.get(int(ids[at - 3])), digit.get(int(ids[at - 1]))
                    if r is not None and c is not None and 0 <= r + dr < n and 0 <= c + dc < n:
                        see = cells[(r + dr) * n + c + dc].to(ids.device)
                allowed[b, q - query_start, image] = see
    return allowed
