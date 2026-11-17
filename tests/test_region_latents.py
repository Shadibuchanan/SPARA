import json
import os
import zlib
from pathlib import Path

import pytest
import torch

from spara import detour_scaling, expert_iteration, expert_round2, perception_jepa, region_latents, training
from spara.latent_battery import latent_positions
from spara.model import LATENT_TOKENS, bottleneck_mask, initialize_latents
from spara.perception_jepa import HEADS_FILE, map_key
from spara.model import allowed_mask
from spara.region_latents import (BLOCK_TOKENS, READOUT_FILE, REGION_TOKENS, Readout, RegionLoss, block_probe,
                                  block_targets, blocks, compare_region, extend_latents, restrict_regions, ridge_fit,
                                  run_region_arm, token_similarity, tokens_by_size)
from spara.runtime import read_config
from spara.serialization import encode, latent_kwargs
from test_detour_scaling import Evaluator, SavingProcessor
from test_expert_iteration import tiny
from test_expert_round2 import SETTINGS as ROUND2, data, route_rows, synthetic_states, write_jsonl  # noqa: F401
from test_planning_probe import SUPPRESS

TINY_IDS = [10, 11, 12, 13] + list(range(21, 33))  # 16 latent ids inside the stand-in model's 40-token vocabulary


# Blocks and tokens ----------------------------------------------------------------------------

def test_blocks_cover_every_cell_once_in_its_place():
    for n, count in zip(range(3, 9), (4, 4, 9, 9, 16, 16)):
        found = blocks(n)
        assert len(found) == count and len(tokens_by_size()[str(n)]) == count
        cells = sorted(c for b in found for c, _ in b)
        assert cells == list(range(n * n))
    three = blocks(3)
    assert three[1] == [(2, 0), (5, 2)]  # the right edge block: its cells keep their left-column places
    assert three[3] == [(8, 0)]
    assert tokens_by_size()["4"] == LATENT_TOKENS and tokens_by_size()["8"] == REGION_TOKENS


def test_block_targets_mask_missing_cells():
    targets = torch.arange(9 * 2, dtype=torch.float32).view(9, 2)
    out, mask = block_targets(targets, 3)
    assert out.shape == (4, 4, 2) and mask.sum() == 9
    assert mask[1].tolist() == [True, False, True, False] and torch.equal(out[1, 2], targets[5])
    assert mask[3].tolist() == [True, False, False, False]


def test_ridge_fit_recovers_a_linear_map():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(400, 6, generator=g)
    w = torch.randn(6, 3, generator=g)
    y = x @ w + 2.0 + 0.01 * torch.randn(400, 3, generator=g)
    weight, bias, _ = ridge_fit(x, y)
    torch.testing.assert_close(weight, w, atol=0.05, rtol=0.05)
    torch.testing.assert_close(bias, torch.full((3,), 2.0), atol=0.05, rtol=0.05)


# The loss on hand-built prompts -----------------------------------------------------------------

def region_batch(tiny_batch, k, training=True):
    """The stand-in prompt with `k` latent tokens: bos, vision start, one image token, vision end, a
    text token, the latents, the assistant header; then the answer when training."""
    ids = [1, 2, 3, 5, 7] + TINY_IDS[:k] + [8, 9] + ([14, 6] if training else [])
    batch = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
             "pixel_values": tiny_batch["pixel_values"].clone(), "image_grid_thw": tiny_batch["image_grid_thw"].clone()}
    if training:
        labels = batch["input_ids"].clone()
        labels[:, :-2] = -100
        batch["labels"] = labels
    return batch


def region_model(model):
    model.config.livr_latent_ids = TINY_IDS
    model.config.livr_latent_tokens = tokens_by_size()
    model.config.livr_region_latents = {"block": 2, "tokens": REGION_TOKENS}
    return model


def region_loss(model, sizes_targets, settings=None):
    head = Readout(model.config.hidden_size, target_dim=3)
    s = {"context_layer": 2, "target_layer": 1, "weight": 1.0, "head_learning_rate": 1e-3, "seed": 0} | (settings or {})
    return RegionLoss(s, head, sizes_targets, model.config.livr_latent_ids)


def test_the_mask_and_positions_handle_sixteen_latents(tiny_batch):
    batch = region_batch(tiny_batch, 16)
    ids = batch["input_ids"]
    assert latent_positions(ids, TINY_IDS).tolist() == list(range(5, 21))
    mask = bottleneck_mask(ids, batch["attention_mask"], TINY_IDS, 3, torch.float32)[0, 0]
    image = 2
    assert (mask[5:21, image] == 0).all()  # every latent sees the image
    assert (mask[21:, image] < 0).all()  # the header and answer do not


def test_the_readout_loss_reaches_the_model_through_the_latents(tiny_model, tiny_batch):
    model = region_model(tiny_model.train())
    model.config.livr_stage = 2
    key = ("fresh/any", "img/0.png")
    aux = region_loss(model, {key: torch.randn(64, 3)})
    aux.bind([("fresh/any", {"image_input": "img/0.png", "map_desc": [[0] * 8] * 8})])
    aux.prepare(None, [{"image_input": "img/0.png", "map_desc": [[0] * 8] * 8}])
    batch = region_batch(tiny_batch, 16)
    out = model(**batch, use_cache=False, output_hidden_states=True)
    out.hidden_states[2].retain_grad()
    total, metrics = aux.loss(out.hidden_states, 0, batch)
    total.backward()
    grad = out.hidden_states[2].grad[0]
    assert grad[5:21].abs().sum(-1).gt(0).all()  # every one of the 16 latents
    assert grad[:5].abs().sum() == 0 and grad[21:].abs().sum() == 0  # nothing else, at the readout's layer
    assert metrics["shuffled_gap"] is None and metrics["region_unexplained"] > 0
    with pytest.raises(ValueError, match="carries 9 latents"):
        aux.loss(model(**region_batch(tiny_batch, 9), use_cache=False, output_hidden_states=True).hidden_states, 0,
                 region_batch(tiny_batch, 9))


# The count-matched probe ------------------------------------------------------------------------

def test_block_probe_reads_a_block_from_its_own_token():
    import random
    from spara.fresh_maps import draw_detour_layouts
    rng, g = random.Random(0), torch.Generator().manual_seed(0)
    items, states = [], []
    for population, count in (("pool78", 120), ("heldout78", 60)):
        layouts, _ = draw_detour_layouts(count, rng, set(), [7], [1])
        for i, desc in enumerate(layouts):
            holes = torch.tensor([float(v == -1) for row in desc for v in row])
            tokens = torch.zeros(16, 8)
            for b, cells in enumerate(blocks(7)):  # each token holds only its own block's holes
                for cell, slot in cells:
                    tokens[b, slot] = holes[cell]
            states.append((tokens + 0.05 * torch.randn(16, 8, generator=g))[None])
            items.append({"population": population, "split": "s", "row_index": i, "map_desc": desc})
    saved = {"layers": [12], "states": torch.stack(states), "items": items}
    own = expert_round2.probe_score(block_probe(saved, 12, lambda n, b: b))["hole_balanced_accuracy"]
    first = expert_round2.probe_score(block_probe(saved, 12, lambda n, b: 0))["hole_balanced_accuracy"]
    assert own > 0.95 and first < 0.7


# The arm end to end on the stand-in model -------------------------------------------------------

class RegionTokenizer:
    """The stand-in tokenizer, with LIVR's four latents and room for the region tokens."""
    pad_token_id = 0

    def __init__(self, base):
        self.base = base
        self.vocab = dict(zip(LATENT_TOKENS, [10, 11, 12, 13]))

    def decode(self, ids, skip_special_tokens=False):
        return self.base.decode(ids, skip_special_tokens)

    def get_vocab(self):
        return dict(self.vocab)

    def add_tokens(self, tokens, special_tokens=False):
        for t in tokens:
            self.vocab[t] = 21 + len(self.vocab) - 4  # new tokens take 21, 22, ... inside the 40-token vocabulary

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab[t] for t in tokens]

    def __len__(self):
        return 40


class RegionProcessor(SavingProcessor):
    def __init__(self):
        self.tokenizer = RegionTokenizer(SavingProcessor.tokenizer)


@pytest.fixture
def stand_in(data, tiny_batch, monkeypatch):
    """Every module's `encode` builds the stand-in prompt with the latents it is asked for, and an image
    that differs by map."""
    def encode(processor, root, split, row, training=True, latents=LATENT_TOKENS, **kwargs):
        size = None
        if isinstance(latents, dict):
            size = len(row["map_desc"])
            latents = latents[str(size)]
        batch = region_batch(tiny_batch, len(latents), training)
        batch["input_ids"][0, 5:5 + len(latents)] = torch.tensor(processor.tokenizer.convert_tokens_to_ids(latents))
        if size is not None:
            batch["livr_grid_size"] = torch.tensor([size])
        if row.get("livr_hide_globals"):
            batch["livr_hide_globals"] = torch.tensor([True])
        g = torch.Generator().manual_seed(zlib.crc32(f"{split}/{row['image_input']}".encode()))
        batch["pixel_values"] = batch["pixel_values"] + torch.randn(batch["pixel_values"].shape, generator=g)
        return batch
    for module in (perception_jepa, expert_round2, detour_scaling, expert_iteration, training, region_latents):
        monkeypatch.setattr(module, "encode", encode)
        if hasattr(module, "records"):
            monkeypatch.setattr(module, "records", lambda root, split: data[split])
    monkeypatch.setattr(region_latents, "run_benchmark", lambda model, processor, config, root, output, manifest:
                        output.mkdir())
    return data


def gate_dir(path, data, hidden, dim=3, drop=None):
    path.mkdir()
    g = torch.Generator().manual_seed(0)
    keys = {map_key(split, row): len(row["map_desc"]) for split, rows in data.items() for row in rows}
    targets = {k: torch.randn(n * n, dim, generator=g).half() for k, n in keys.items() if k != drop}
    torch.save({"settings": {"target_layer": 1, "context_layer": 2}, "targets": targets,
                "centres": {str(n): torch.zeros(n * n, hidden) for n in range(3, 9)},
                "projection": torch.randn(dim, hidden, generator=g)}, path / HEADS_FILE)
    (path / "summary.json").write_text(json.dumps({"valid": True, "reading": "gates passed: run EJF"}))


SETTINGS = {"context_layer": 2, "target_layer": 1, "head_learning_rate": 1e-3, "head_weight_decay": 0.01,
            "shuffle_pool": 8, "monitor_maps": 2, "pilot_rows": 2, "beta_bounds": [0.0, 1e9], "check_rows": 2,
            "identity_tolerance": 1e-4, "warm_maps": {"large": None, "middle": 4, "small": 4}, "seed": 0}


def runs_for(tmp_path, data):
    round1 = write_jsonl(tmp_path / "round1.jsonl", route_rows(data["fresh/p"], {0, 1}))
    search = tmp_path / "search"
    search.mkdir()
    write_jsonl(search / "routes.jsonl", route_rows(data["fresh/p78"], {1, 3}))
    write_jsonl(search / "frontier.jsonl", [])
    return round1, search


def test_the_region_arm_end_to_end(stand_in, tiny_model, tmp_path):
    model = tiny(tiny_model)
    model.generation_config.suppress_tokens = SUPPRESS + TINY_IDS[4:]
    config = read_config("configs/livr.json") | {"gradient_checkpointing": False}
    round1, search = runs_for(tmp_path, stand_in)
    dropped = map_key("train", stand_in["train"][0])
    gate_dir(tmp_path / "gate", stand_in, model.config.hidden_size, drop=dropped)
    summary = run_region_arm(model, RegionProcessor(), Evaluator(), config, "unused", tmp_path / "ejr", SETTINGS,
                             ROUND2, tmp_path / "gate", round1, search, {})
    assert summary["reading"] == "complete", summary["reading"]
    check = summary["start_check"]
    assert check["matched"] and check["small_rows"] > 0 and check["max_relative_state_difference"] <= 1e-4
    assert check["small_loss_change"] is not None and check["insertion_disturbance"] is not None
    assert summary["targets_added"] == 1 and summary["weight"]["beta"] > 0
    assert model.config.livr_latent_ids == TINY_IDS and model.config.livr_region_latents["tokens"] == REGION_TOKENS
    assert set(summary["training"]["jepa_error"]["start"]) == {"validation", "training"}
    steps = [json.loads(line) for line in (tmp_path / "ejr/phase1/steps.jsonl").read_text().splitlines()]
    assert steps[0]["region_loss"] is not None and steps[0]["head_grad_norm"] is not None
    states = torch.load(tmp_path / "ejr/states.pt")
    assert states["states"].shape[2] == 9  # the 5 x 5 and 6 x 6 stand-in maps carry nine latents
    saved = torch.load(tmp_path / "ejr/phase2" / READOUT_FILE, weights_only=False)
    assert saved["settings"]["weight"] == pytest.approx(summary["weight"]["beta"])
    assert saved["state_dict"]["mean"].abs().sum() > 0  # the standardisation was set at the start


def test_a_changed_small_map_withholds_the_arm(stand_in, tiny_model, tmp_path, monkeypatch):
    model = tiny(tiny_model)
    config = read_config("configs/livr.json") | {"gradient_checkpointing": False}
    round1, search = runs_for(tmp_path, stand_in)
    gate_dir(tmp_path / "gate", stand_in, model.config.hidden_size)
    real = region_latents.extend_latents

    def disturbing(model, processor, *args):
        ids = real(model, processor, *args)
        with torch.no_grad():
            model.get_input_embeddings().weight[ids[0]] += 1.0  # breaks LIVR's own token
        return ids
    monkeypatch.setattr(region_latents, "extend_latents", disturbing)
    summary = run_region_arm(model, RegionProcessor(), Evaluator(), config, "unused", tmp_path / "ejr", SETTINGS,
                             ROUND2, tmp_path / "gate", round1, search, {})
    assert summary["reading"].startswith("withheld") and not summary["start_check"]["matched"]
    assert not (tmp_path / "ejr/phase1").exists()


# The real tokenizer: extension, encoding and a saved checkpoint ---------------------------------

@pytest.mark.integration
def test_region_latents_with_the_real_tokenizer_round_trip(tiny_model, tmp_path):
    from transformers import AutoProcessor
    from spara.data import records
    from spara.runtime import load_model
    root = Path(os.environ.get("SPARA_TEST_DATA", "data/vsp"))
    processor_path = Path(os.environ.get("SPARA_TEST_PROCESSOR", ".cache/processor"))
    if not processor_path.is_dir() or not (root / "manifest.json").is_file():
        pytest.skip("Prepare reference data and processor")
    processor = AutoProcessor.from_pretrained(processor_path)
    base = initialize_latents(tiny_model, processor)
    tiny_model.config.livr_stage = 2
    before = tiny_model.get_input_embeddings().weight[base].clone()
    ids = extend_latents(tiny_model, processor)
    assert ids[:4] == base and len(set(ids)) == 16
    weight = tiny_model.get_input_embeddings().weight
    torch.testing.assert_close(weight[base], before)  # LIVR's own tokens are untouched
    assert (weight[ids[4]] - weight[ids[0]]).abs().max() < 0.2
    rows = {n: next(r for r in records(root, split) if len(r["map_desc"]) == n)
            for n, split in ((3, "train"), (8, "fresh/detour-heldout-78"))}
    split_of = {3: "train", 8: "fresh/detour-heldout-78"}
    for n, count in ((3, 4), (8, 16)):
        prompt = encode(processor, root, split_of[n], rows[n], training=False, **latent_kwargs(tiny_model))
        assert len(latent_positions(prompt["input_ids"], ids)) == count
    assert torch.equal(encode(processor, root, "train", rows[3], training=False)["input_ids"],
                       encode(processor, root, "train", rows[3], training=False,
                              **latent_kwargs(tiny_model))["input_ids"])  # small maps are unchanged
    tiny_model.save_pretrained(tmp_path)
    processor.save_pretrained(tmp_path)
    (tmp_path / "complete.json").write_text(json.dumps({"progress": {}}))
    config = read_config("configs/livr.json") | {"attention_backend": "eager"}
    model, restored = load_model(config, tmp_path)
    assert model.config.livr_latent_ids == ids and model.config.livr_latent_tokens == tokens_by_size()
    prompt = encode(restored, root, split_of[8], rows[8], training=False, **latent_kwargs(model))
    assert len(latent_positions(prompt["input_ids"], model.config.livr_latent_ids)) == 16


# Global + region latents with the block mask (EGR) ------------------------------------------------

BLOCK_IDS = list(range(21, 37))
GLOBAL_IDS = [10, 11, 12, 13]


def masked_model(model):
    model.config.livr_latent_ids = GLOBAL_IDS + BLOCK_IDS
    model.config.livr_latent_tokens = tokens_by_size(layout="global+region")
    model.config.livr_region_latents = {"block": 2, "tokens": LATENT_TOKENS + BLOCK_TOKENS, "layout": "global+region",
                                        "globals": 4, "mask": True}
    return model


def quadrant_batch(tiny_batch, training=False, blocks_present=4):
    """A 4 x 4 map's prompt with a 4 x 4 patch grid: four merged image tokens, one per 2 x 2 block."""
    ids = [1, 2, 3, 3, 3, 3, 5, 7] + GLOBAL_IDS + BLOCK_IDS[:blocks_present] + [8, 9] + ([14, 6] if training else [])
    g = torch.Generator().manual_seed(0)
    batch = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
             "pixel_values": torch.randn(16, tiny_batch["pixel_values"].shape[1], generator=g),
             "image_grid_thw": torch.tensor([[1, 4, 4]]), "livr_grid_size": torch.tensor([4])}
    if training:
        labels = batch["input_ids"].clone()
        labels[:, :-2] = -100
        batch["labels"] = labels
    return batch


def test_each_block_latent_sees_only_its_block(tiny_model, tiny_batch):
    model = masked_model(tiny_model)
    batch = quadrant_batch(tiny_batch)
    ids = batch["input_ids"]
    image = [2, 3, 4, 5]
    for stage_one in (False, True):
        allowed = allowed_mask(ids, batch["attention_mask"], model.config.livr_latent_ids, 3, 0, image_block=stage_one)
        allowed = restrict_regions(allowed, ids, 0, model.config, batch["livr_grid_size"], batch["image_grid_thw"])[0]
        for k in range(4):
            q = 12 + k  # the block latents follow the four globals at 8 to 11
            assert allowed[q].nonzero().squeeze(-1).tolist() == [0, 1, image[k], q]
        for q in range(8, 12):
            assert allowed[q, image].all()  # the globals see the whole image
        assert bool(allowed[16, image].any()) != stage_one  # the header sees the image only in Stage 2


def test_the_block_mask_reaches_the_attention(tiny_model, tiny_batch):
    model = masked_model(tiny_model.eval())
    for stage in (1, 2):
        model.config.livr_stage = stage
        out = model(**quadrant_batch(tiny_batch), output_attentions=True, use_cache=False)
        for layer in out.attentions:
            weights = layer[0].sum(0)  # heads summed: [queries, keys]
            for k in range(4):
                others = [2 + j for j in range(4) if j != k]
                assert weights[12 + k, others].abs().max() == 0 and weights[12 + k, 2 + k] > 0


def test_the_mask_path_without_block_latents_matches_the_default(tiny_model, tiny_batch):
    plain = tiny_model.eval()
    plain.config.livr_stage = 2
    batch = quadrant_batch(tiny_batch, blocks_present=0)
    expected = plain(**{k: v for k, v in batch.items() if k != "livr_grid_size"}, use_cache=False).logits
    masked = masked_model(plain)
    torch.testing.assert_close(masked(**batch, use_cache=False).logits, expected, atol=1e-5, rtol=1e-4)


def test_generation_carries_the_grid_size_in_both_stages(tiny_model, tiny_batch):
    model = masked_model(tiny(tiny_model))
    model.generation_config.suppress_tokens = SUPPRESS + BLOCK_IDS
    for stage in (1, 2):
        model.config.livr_stage = stage
        out = model.generate(**quadrant_batch(tiny_batch), do_sample=False, max_new_tokens=3, use_cache=True)
        assert out.shape[1] == quadrant_batch(tiny_batch)["input_ids"].shape[1] + 3
    with pytest.raises(ValueError, match="grid size"):
        model(**{k: v for k, v in quadrant_batch(tiny_batch).items() if k != "livr_grid_size"}, use_cache=False)


def test_a_monitor_only_readout_sends_nothing_into_the_model(tiny_model, tiny_batch):
    model = masked_model(tiny_model.train())
    model.config.livr_stage = 2
    key = ("fresh/any", "img/0.png")
    row = {"image_input": "img/0.png", "map_desc": [[0] * 4] * 4}
    head = Readout(model.config.hidden_size, target_dim=3)
    aux = RegionLoss({"context_layer": 2, "target_layer": 1, "weight": 1.0, "head_learning_rate": 1e-3, "globals": 4,
                      "readout_only": True}, head, {key: torch.randn(16, 3)}, model.config.livr_latent_ids)
    aux.bind([("fresh/any", row)])
    aux.prepare(None, [row])
    out = model(**quadrant_batch(tiny_batch), use_cache=False, output_hidden_states=True)
    total, _ = aux.loss(out.hidden_states, 0, quadrant_batch(tiny_batch))
    total.backward()
    assert all(p.grad is None for p in model.parameters()) and head.linear.weight.grad.abs().sum() > 0


def test_token_similarity_tells_separate_latents_from_copies():
    import random
    from spara.fresh_maps import draw_detour_layouts
    rng, g = random.Random(1), torch.Generator().manual_seed(1)
    items, own, copies = [], [], []
    for population, count in (("pool78", 120), ("heldout78", 60)):
        for i, desc in enumerate(draw_detour_layouts(count, rng, set(), [7], [1])[0]):
            holes = torch.tensor([float(v == -1) for row in desc for v in row])
            tokens = torch.zeros(20, 8)
            for b, cells in enumerate(blocks(7)):
                for cell, slot in cells:
                    tokens[4 + b, slot] = holes[cell]
            noise = 0.05 * torch.randn(20, 8, generator=g)
            own.append((tokens + noise)[None])
            copy = tokens.clone()
            copy[4:, :4] = holes[:4]  # every block latent holds the same four cells
            copies.append((copy + noise)[None])
            items.append({"population": population, "split": "s", "row_index": i, "map_desc": desc})
    separate = token_similarity({"layers": [12], "states": torch.stack(own), "items": items}, 12, offset=4)["7"]
    same = token_similarity({"layers": [12], "states": torch.stack(copies), "items": items}, 12, offset=4)["7"]
    assert separate["similarity"] < 0.3 and same["similarity"] > 0.9
    assert separate["own_block"] > separate["other_blocks"] + 0.5


def test_the_global_region_arm_end_to_end(stand_in, tiny_model, tmp_path):
    model = tiny(tiny_model)
    model.generation_config.suppress_tokens = SUPPRESS + BLOCK_IDS
    config = read_config("configs/livr.json") | {"gradient_checkpointing": False}
    round1, search = runs_for(tmp_path, stand_in)
    gate_dir(tmp_path / "gate", stand_in, model.config.hidden_size)
    settings = SETTINGS | {"arm": "egr", "layout": "global+region", "mask": True, "readout_only": True,
                           "benchmark": False}
    summary = run_region_arm(model, RegionProcessor(), Evaluator(), config, "unused", tmp_path / "egr", settings,
                             ROUND2, tmp_path / "gate", round1, search, {})
    assert summary["reading"] == "complete", summary["reading"]
    assert summary["arm"] == "egr" and summary["weight"] is None
    check = summary["start_check"]
    assert check["matched"] and check["rows_checked"] == check["small_rows"] + check["large_rows"]
    assert model.config.livr_latent_ids == GLOBAL_IDS + BLOCK_IDS and model.config.livr_region_latents["mask"]
    states = torch.load(tmp_path / "egr/states.pt")
    assert states["states"].shape[2] == 4 + 9  # four globals and nine block latents at 5 x 5 and 6 x 6
    assert not (tmp_path / "egr/evaluation").exists()  # no benchmark
    steps = [json.loads(line) for line in (tmp_path / "egr/phase1/steps.jsonl").read_text().splitlines()]
    assert steps[0]["region_loss"] is not None


def block_states(noise, seed=0):
    """6 x 6 maps with four global latents and nine block latents, each holding its own block's holes."""
    import random
    from spara.fresh_maps import draw_detour_layouts
    rng, g = random.Random(seed), torch.Generator().manual_seed(seed)
    items, states = [], []
    for population, count in (("pool78", 60), ("heldout78", 30)):
        for i, desc in enumerate(draw_detour_layouts(count, rng, set(), [6], [1])[0]):
            holes = torch.tensor([float(v == -1) for row in desc for v in row])
            tokens = noise * torch.randn(13, 16, generator=g)
            for b, cells in enumerate(blocks(6)):
                for cell, slot in cells:
                    tokens[4 + b, slot] += holes[cell]
            states.append(torch.stack([tokens, tokens]))
            items.append({"population": population, "split": "s", "row_index": i, "map_desc": desc})
    return {"layers": [1, 2], "states": torch.stack(states).half(), "items": items}


def test_compare_region_gates_on_separation_and_reads_masked_greedy(data, tmp_path, monkeypatch):
    from test_expert_round2 import write_run
    from test_perception_jepa import arm_files
    monkeypatch.setattr(perception_jepa, "records", lambda root, split: data[split])
    monkeypatch.setattr(perception_jepa, "paired", lambda a, b, keys: detour_scaling.paired(a, b, keys) | {"p": 0.0})
    monkeypatch.setattr(region_latents, "paired", lambda a, b, keys: detour_scaling.paired(a, b, keys) | {"p": 0.0})
    noisy = synthetic_states(30.0)
    write_run(tmp_path / "e", data, [0, 0, 0, 0], [0, 0, 0, 0], states=noisy)
    write_run(tmp_path / "e2", data, [1, 0, 0, 0], [0, 0, 0, 0], states=noisy, benchmark=[1, 1, 0])
    write_run(tmp_path / "eb2", data, [1, 0, 0, 0], [0, 0, 0, 0], states=noisy, benchmark=[1, 0, 0])
    write_run(tmp_path / "egr", data, [1, 1, 1, 0], [1, 1, 1, 0], states=block_states(0.05), benchmark=[1, 0, 1])
    arm_files(tmp_path / "eb2", 0.1)
    arm_files(tmp_path / "egr", 0.3, jepa=True)
    summary = json.loads((tmp_path / "egr/summary.json").read_text())
    (tmp_path / "egr/summary.json").write_text(json.dumps(summary | {"start_check": {"matched": True}}))
    (tmp_path / "gate").mkdir()
    (tmp_path / "gate/summary.json").write_text(json.dumps({"valid": True}))
    livr = write_jsonl(tmp_path / "livr.jsonl", [{"row_index": i, "success": True} for i in range(3)])
    runs = {name: tmp_path / name for name in ("e", "e2", "eb2", "egr")}
    result = compare_region("unused", ROUND2, runs, tmp_path / "gate", livr, arm="egr", offset=4, separation=0.6)
    assert result["manipulation"]["separation"]["separated"] and result["manipulation"]["took"]
    assert result["discriminating_masked"]["reading"] == "EGR solves more with the image path cut"
    assert "probe_block_latents" in result["descriptive"]
    assert result["descriptive"]["benchmark_vs_livr"]["egr"] is not None
    import shutil
    shutil.rmtree(tmp_path / "egr/evaluation")  # EGR skips the benchmark
    skipped = compare_region("unused", ROUND2, runs, tmp_path / "gate", livr, arm="egr", offset=4, separation=0.6)
    assert skipped["descriptive"]["benchmark_vs_livr"]["egr"] is None and skipped["manipulation"]["took"]
    strict = compare_region("unused", ROUND2, runs, tmp_path / "gate", livr, arm="egr", offset=4, separation=-1.0)
    assert not strict["manipulation"]["took"] and strict["discriminating_masked"]["reading"] == "did not take; no reading"


@pytest.mark.integration
def test_masked_global_region_latents_with_the_real_tokenizer(tiny_model, tmp_path):
    from transformers import AutoProcessor
    from spara.data import records
    from spara.region_latents import block_image_tokens
    from spara.runtime import load_model
    root = Path(os.environ.get("SPARA_TEST_DATA", "data/vsp"))
    processor_path = Path(os.environ.get("SPARA_TEST_PROCESSOR", ".cache/processor"))
    if not processor_path.is_dir() or not (root / "manifest.json").is_file():
        pytest.skip("Prepare reference data and processor")
    processor = AutoProcessor.from_pretrained(processor_path)
    base = initialize_latents(tiny_model, processor)
    tiny_model.config.livr_stage = 2
    vocab = processor.tokenizer.convert_tokens_to_ids(["<|image_pad|>", "<|video_pad|>", "<|vision_start|>",
                                                       "<|vision_end|>"])
    for name, value in zip(("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"), vocab):
        setattr(tiny_model.config, name, value)  # the stand-in model reads real prompts
    ids = extend_latents(tiny_model, processor, layout="global+region", mask=True)
    assert ids[:4] == base and len(ids) == 20
    rows = {3: ("train", next(r for r in records(root, "train") if len(r["map_desc"]) == 3)),
            8: ("fresh/detour-heldout-78", next(r for r in records(root, "fresh/detour-heldout-78")
                                               if len(r["map_desc"]) == 8))}
    for n, count in ((3, 8), (8, 20)):
        split, row = rows[n]
        prompt = encode(processor, root, split, row, training=False, **latent_kwargs(tiny_model))
        assert len(latent_positions(prompt["input_ids"], ids)) == count and int(prompt["livr_grid_size"]) == n
        # E's prompt is a prefix of this one, through the four global latents.
        plain = encode(processor, root, split, row, training=False)["input_ids"][0]
        end = int(latent_positions(plain[None], base)[-1]) + 1
        assert torch.equal(prompt["input_ids"][0, :end], plain[:end])
    split, row = rows[8]
    prompt = encode(processor, root, split, row, training=False, **latent_kwargs(tiny_model))
    visible = block_image_tokens(8, prompt["image_grid_thw"][0], tiny_model.config.vision_config.spatial_merge_size)
    images = int(prompt["input_ids"].eq(tiny_model.config.image_token_id).sum())
    assert visible.shape == (16, images) and visible.any(0).all()  # every image token belongs to some block
    assert visible.sum(1).max() < images  # and no block sees the whole image
    tiny_model.eval()
    for stage in (1, 2):
        tiny_model.config.livr_stage = stage
        with torch.no_grad():
            assert torch.isfinite(tiny_model(**prompt, use_cache=False).logits).all()
    tiny_model.config.livr_stage = 2
    tiny_model.save_pretrained(tmp_path)
    processor.save_pretrained(tmp_path)
    (tmp_path / "complete.json").write_text(json.dumps({"progress": {}}))
    model, restored = load_model(read_config("configs/livr.json") | {"attention_backend": "eager"}, tmp_path)
    assert model.config.livr_latent_ids == ids and model.config.livr_region_latents["mask"]
    again = encode(restored, root, split, row, training=False, **latent_kwargs(model))
    assert torch.equal(again["input_ids"], prompt["input_ids"])


def test_the_swap_test_on_a_masked_model(stand_in, tiny_model, tmp_path):
    from spara.region_latents import run_latent_swap
    model = tiny(tiny_model)
    model.generation_config.suppress_tokens = SUPPRESS + BLOCK_IDS
    processor = RegionProcessor()
    extend_latents(model, processor, 0, "global+region", True)
    settings = {"sizes": [5, 6], "maps_per_size": 3, "reproduce": 0.95, "seed": 0}
    first = run_latent_swap(model, processor, Evaluator(), "unused", tmp_path / "swap", settings, ROUND2, {})
    rows = [json.loads(line) for line in (tmp_path / "swap/swaps.jsonl").read_text().splitlines()]
    assert first["maps"] == len(rows) > 0 and all(first["identity"])
    assert set(first["conditions"]) == {"own", "block_swap", "block_mean", "global_swap"}
    assert all("block_mean" in r and "own" in r for r in rows)
    assert all(r["donor"] is None or r["donor_route"] == rows[r["donor"]]["own"]["route"] for r in rows)
    assert first["reading"] in ("the answer ignores the block latents", "mixed",
                                "the answer reads the block latents, but their content does not shape its moves")
    recorded = write_jsonl(tmp_path / "masked.jsonl", [{"population": "heldout78", "row_index": r["row_index"],
                                                         "route": r["own"]["route"]} for r in rows])
    again = run_latent_swap(model, processor, Evaluator(), "unused", tmp_path / "again", settings, ROUND2, {},
                            recorded)
    assert again["reproduced"]["share"] == 1.0 and again["conditions"] == first["conditions"]
    with pytest.raises(ValueError, match="EGR"):
        plain = tiny(copy_model(tiny_model))
        run_latent_swap(plain, RegionProcessor(), Evaluator(), "unused", tmp_path / "no", settings, ROUND2, {})


def copy_model(model):
    import copy
    fresh = copy.deepcopy(model)
    fresh.config.livr_region_latents = None
    return fresh



# Global dropout from EGR (EGD) ----------------------------------------------------------------------

def test_hidden_globals_are_invisible_to_the_answer(tiny_model, tiny_batch):
    from spara.region_latents import hide_globals
    model = masked_model(tiny_model)
    batch = quadrant_batch(tiny_batch)
    ids = batch["input_ids"]
    allowed = allowed_mask(ids, batch["attention_mask"], model.config.livr_latent_ids, 3, 0, image_block=True)
    allowed = restrict_regions(allowed, ids, 0, model.config, batch["livr_grid_size"], batch["image_grid_thw"])
    shown = allowed[0].clone()
    allowed = hide_globals(allowed, ids, 0, model.config, torch.tensor([True]))[0]
    globals_ = list(range(8, 12))
    assert not allowed[16:, globals_].any() and shown[16:, globals_].all()  # the header and answer lose them
    assert torch.equal(allowed[:16], shown[:16])  # the latents and everything before are unchanged
    assert allowed[16:, 12:16].all()  # the block latents stay visible
    unflagged = hide_globals(shown[None].clone(), ids, 0, model.config, torch.tensor([False]))[0]
    assert torch.equal(unflagged, shown)


def test_hidden_globals_in_generation_and_encoding(tiny_model, tiny_batch):
    model = masked_model(tiny(tiny_model))
    model.generation_config.suppress_tokens = SUPPRESS + BLOCK_IDS
    batch = quadrant_batch(tiny_batch) | {"livr_hide_globals": torch.tensor([True])}
    for stage in (1, 2):
        model.config.livr_stage = stage
        out = model(**batch, output_attentions=True, use_cache=False)
        for layer in out.attentions:
            assert layer[0, :, 16:, 8:12].abs().max() == 0  # no attention from the header to the globals
        assert model.generate(**batch, do_sample=False, max_new_tokens=3, use_cache=True).shape[1] == 18 + 3


def test_the_dropout_arm_end_to_end(stand_in, tiny_model, tmp_path):
    from spara.region_latents import compare_dropout, run_global_dropout
    model = tiny(tiny_model)
    model.generation_config.suppress_tokens = SUPPRESS + BLOCK_IDS
    processor = RegionProcessor()
    extend_latents(model, processor, 0, "global+region", True)
    config = read_config("configs/livr.json") | {"gradient_checkpointing": False}
    round1, search = runs_for(tmp_path, stand_in)
    settings = {"swap": {"sizes": [5, 6], "maps_per_size": 2, "reproduce": 0.95, "seed": 0},
                "uses_change": 0.3, "uses_donor": 0.3, "reproduce": 0.95}
    summary = run_global_dropout(model, processor, Evaluator(), config, "unused", tmp_path / "egd", settings, ROUND2,
                                 round1, search, {})
    assert set(summary["evaluation"]) == {"masked_hidden", "masked", "greedy"}
    assert all(n == len(stand_in["fresh/h78"]) for _, n in summary["evaluation"].values())
    assert set(summary["swap"]["conditions"]) == {"own", "block_swap", "block_mean"}
    assert summary["swap"]["reproduced"]["share"] == 1.0 and all(summary["swap"]["identity"])
    assert (tmp_path / "egd/states.pt").exists() and (tmp_path / "egd/swap/swaps.jsonl").exists()
    steps = [json.loads(line) for line in (tmp_path / "egd/phase1/steps.jsonl").read_text().splitlines()]
    assert steps and all(r["stage"] == 1 for r in steps)
    # The readings, with the same files standing in for the other arms.
    for name in ("egr", "eb2", "e2"):
        (tmp_path / name).mkdir()
        for file in ("greedy.jsonl", "masked.jsonl"):
            (tmp_path / name / file).write_text((tmp_path / "egd" / file).read_text())
    result = compare_dropout({"egd": tmp_path / "egd", "egr": tmp_path / "egr", "eb2": tmp_path / "eb2",
                              "e2": tmp_path / "e2"}, settings)
    assert result["valid"] and set(result["tests"]) == {"masked", "greedy"}
    separate = compare_dropout({"egd": tmp_path / "egd", "egr": tmp_path / "egr", "eb2": tmp_path / "eb2",
                                "e2": tmp_path / "e2"}, settings, swap_run=tmp_path / "egd/swap")
    assert separate["reading"] == result["reading"] and separate["valid"]  # the same readings from a separate job
    assert result["reading"] in ("hiding the globals did not force reliance on the block latents",
                                 "reading the map from the latents does not fix the policy; the limit is the decision process")


# The planning probe's state probe on a block-latent model --------------------------------------------

GRID4 = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]]
OPEN4 = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]]


def test_tracking_runs_in_both_answer_modes(tiny_model, tiny_batch, tmp_path, monkeypatch):
    from spara import planning_probe
    from test_planning_probe import Evaluator as RouteEvaluator, Processor
    model = masked_model(tiny(tiny_model))
    model.generation_config.suppress_tokens = SUPPRESS + BLOCK_IDS
    data = {"held": [{"map_desc": GRID4}, {"map_desc": OPEN4}, {"map_desc": GRID4}],
            "pool": [{"map_desc": grid, "text_output": answer}
                     for grid, answer in [(GRID4, "\\boxed{RIGHT, RIGHT, RIGHT, DOWN, DOWN, DOWN}"),
                                          (OPEN4, "\\boxed{DOWN, DOWN, DOWN, RIGHT, RIGHT, RIGHT}")] * 3]}
    seen = []

    def encode(processor, root, split, row, training=True, latents=None):
        seen.append((split, latents is not None, bool(row.get("livr_hide_globals")), model.config.livr_stage))
        batch = quadrant_batch(tiny_batch)
        if row.get("livr_hide_globals"):
            batch["livr_hide_globals"] = torch.tensor([True])
        return batch
    monkeypatch.setattr(planning_probe, "records", lambda root, split: data[split])
    monkeypatch.setattr(planning_probe, "encode", encode)
    monkeypatch.setattr(planning_probe, "ReferenceEvaluator", RouteEvaluator)
    summary = planning_probe.tracking(model, Processor(), "unused", tmp_path / "run", {}, 0, "held", "pool",
                                      max_new_tokens=4)
    assert list(summary["modes"]) == ["greedy", "masked_hidden"]
    assert all(latents for _, latents, _, _ in seen)  # every prompt carries the block latents
    assert {(hide, stage) for _, _, hide, stage in seen} == {(False, 2), (True, 1)}  # each mode's mask and flag
    assert model.config.livr_stage == 2  # restored after the masked mode
    shown = torch.load(tmp_path / "run/greedy/features.pt")
    hidden_ = torch.load(tmp_path / "run/masked_hidden/features.pt")
    assert shown["train"]["states"].shape == hidden_["train"]["states"].shape == (36, 1, 32)
    assert not torch.equal(shown["train"]["states"], hidden_["train"]["states"])  # the modes read different states
    for mode in ("greedy", "masked_hidden"):
        result = json.loads((tmp_path / f"run/{mode}/summary.json").read_text())
        assert result["maps"] == 3 and result["reproduced"] is None and result["reading"]
    # The reproduction check against a run's recorded routes in each mode.
    record = tmp_path / "recorded"
    record.mkdir()
    for mode in ("greedy", "masked_hidden"):
        routes = [json.loads(line) for line in (tmp_path / f"run/{mode}/routes.jsonl").read_text().splitlines()]
        write_jsonl(record / f"{mode}.jsonl", [{"row_index": r["row_index"], "route": r["route"]}
                                               for r in routes if r["split"] == "held"])
    again = planning_probe.tracking(model, Processor(), "unused", tmp_path / "again", {}, 0, "held", "pool",
                                    recorded={m: record / f"{m}.jsonl" for m in ("greedy", "masked_hidden")},
                                    max_new_tokens=4)
    assert all(m["reproduced"]["share"] == 1.0 for m in again["modes"].values())
    with pytest.raises(ValueError, match="Unknown modes"):
        planning_probe.tracking(model, Processor(), "unused", tmp_path / "bad", {}, 0, "held", "pool", modes=("x",))


def test_the_tracking_reading_needs_reproduced_routes_and_a_usable_probe():
    from spara.planning_probe import tracking_reading
    report = {"splits": {"test": {"solved_step_accuracy": 0.7, "S1": {"reading": "tracking contributes"}}}}
    assert tracking_reading(report, None, 0.95, 0.5) == "tracking contributes"
    assert tracking_reading(report, {"share": 0.9}, 0.95, 0.5).startswith("invalid")
    weak = {"splits": {"test": {"solved_step_accuracy": 0.4, "S1": {"reading": "tracking intact"}}}}
    assert tracking_reading(weak, {"share": 1.0}, 0.95, 0.5).startswith("inconclusive")


# Global + patch latents: tiles of each image, for images without a map grid ---------------------------

def test_patch_tiles_cut_any_image_grid():
    from spara.region_latents import patch_tiles
    square = patch_tiles([1, 16, 16], 2, 4).view(8, 8)  # an 8 x 8 merged grid: 2 x 2 tokens per tile
    assert torch.equal(square[::2, ::2], torch.arange(16).view(4, 4)) and torch.bincount(square.flatten()).eq(4).all()
    wide = patch_tiles([1, 10, 14], 2, 4)  # 5 x 7 merged tokens
    counts = torch.bincount(wide, minlength=16)
    assert len(wide) == 35 and counts.min() >= 1 and counts.max() <= 4
    small = patch_tiles([1, 4, 6], 2, 4)  # 2 x 3 merged tokens: some tiles stay empty
    assert small.tolist() == [0, 1, 2, 8, 9, 10]
    with pytest.raises(ValueError, match="still images"):
        patch_tiles([2, 4, 4], 2, 4)


def test_image_runs_split_the_images():
    from spara.region_latents import image_runs
    ids = torch.tensor([1, 2, 3, 3, 5, 7, 2, 3, 3, 3, 5, 9])
    assert [r.tolist() for r in image_runs(ids, 3)] == [[2, 3], [7, 8, 9]] and image_runs(ids[:2], 3) == []


PATCH_IDS = list(range(21, 37))


def patch_model(model, grid=2, max_images=2, text="all"):
    """The stand-in model with 4 globals, then grid x grid patch latents per image for `max_images` images."""
    per = grid * grid
    model.config.livr_latent_ids = GLOBAL_IDS + PATCH_IDS[:per * max_images]
    model.config.livr_region_latents = {"tokens": [], "layout": "global+patch", "globals": 4, "mask": True,
                                        "grid": grid, "max_images": max_images, "text": text}
    return model


def two_image_batch(images=2, grid=2):
    """Two 2 x 2-token images (each 4 x 4 patches), then a text token, the globals, grid x grid patch
    latents per image and the header. Image m's tokens are at 2 + 6m to 5 + 6m."""
    ids = [1]
    for _ in range(images):
        ids += [2, 3, 3, 3, 3, 5]
    ids += [7] + GLOBAL_IDS + PATCH_IDS[:grid * grid * images] + [8, 9]
    g = torch.Generator().manual_seed(1)
    return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            "pixel_values": torch.randn(16 * images, 3 * 2 * 14 * 14, generator=g),
            "image_grid_thw": torch.tensor([[1, 4, 4]] * images)}


def patch_allowed(model, batch, stage_one):
    from spara.region_latents import restrict_patches
    ids = batch["input_ids"]
    allowed = allowed_mask(ids, batch["attention_mask"], model.config.livr_latent_ids, model.config.image_token_id, 0,
                           image_block=stage_one)
    return restrict_patches(allowed, ids, 0, model.config, batch["image_grid_thw"])[0]


def test_each_patch_latent_sees_only_its_tile_and_the_text(tiny_model):
    model = patch_model(tiny_model)
    batch = two_image_batch()
    text = [0, 1, 6, 7, 12, 13]  # bos, the vision markers and the text token
    images = [[2, 3, 4, 5], [8, 9, 10, 11]]
    for stage_one in (False, True):
        allowed = patch_allowed(model, batch, stage_one)
        for m in range(2):
            for t in range(4):
                q = 18 + m * 4 + t  # the patch latents follow the globals at 14 to 17
                assert allowed[q].nonzero().squeeze(-1).tolist() == sorted(text + [images[m][t], q])
        for q in range(14, 18):
            assert allowed[q, images[0] + images[1]].all()  # the globals see both images
        assert bool(allowed[26, images[0]].any()) != stage_one  # the header sees the images only in Stage 2
    prefix = patch_allowed(patch_model(tiny_model, text="prefix"), batch, False)
    assert prefix[18].nonzero().squeeze(-1).tolist() == [0, 1, 2, 18]  # only the text before the first image


def test_empty_tiles_leave_a_patch_latent_only_text(tiny_model):
    model = patch_model(tiny_model, grid=4, max_images=1)
    batch = two_image_batch(images=1, grid=4)  # a 2 x 2-token image under a 4 x 4 tile grid
    allowed = patch_allowed(model, batch, True)
    first = 12  # bos, one image (1 to 6), the text token, the globals at 8 to 11
    seen = {t: allowed[first + t, 2:6].nonzero().squeeze(-1).tolist() for t in range(16)}
    assert {t: v for t, v in seen.items() if v} == {0: [0], 2: [1], 8: [2], 10: [3]}
    assert allowed[first + 1].nonzero().squeeze(-1).tolist() == [0, 1, 6, 7, first + 1]


def test_the_patch_mask_reaches_attention_and_generation(tiny_model):
    from spara.region_latents import hide_globals
    model = patch_model(tiny(tiny_model))
    model.generation_config.suppress_tokens = SUPPRESS + PATCH_IDS
    batch = two_image_batch()
    images = [[2, 3, 4, 5], [8, 9, 10, 11]]
    for stage in (1, 2):
        model.config.livr_stage = stage
        out = model(**batch, output_attentions=True, use_cache=False)  # no grid size is needed
        for layer in out.attentions:
            weights = layer[0].sum(0)
            for m in range(2):
                for t in range(4):
                    q = 18 + m * 4 + t
                    others = [i for i in images[0] + images[1] if i != images[m][t]]
                    assert weights[q, others].abs().max() == 0 and weights[q, images[m][t]] > 0
        assert model.generate(**batch, do_sample=False, max_new_tokens=3, use_cache=True).shape[1] == 28 + 3
        hidden_ = model(**batch | {"livr_hide_globals": torch.tensor([True])}, output_attentions=True, use_cache=False)
        for layer in hidden_.attentions:
            assert layer[0, :, 26:, 14:18].abs().max() == 0  # the header loses the globals
    with pytest.raises(ValueError, match="More images"):
        patch_allowed(patch_model(tiny_model, max_images=1), batch, False)


def test_masked_global_patch_latents_with_the_real_processor(tiny_model, tmp_path):
    """Two FrozenLake images of different sizes stand in for a two-image perception question."""
    import shutil
    from transformers import AutoProcessor
    from spara.data import image_path, records
    from spara.region_latents import PATCH_TOKENS, image_runs, patch_latents, patch_tiles
    from spara.runtime import load_model
    root = Path(os.environ.get("SPARA_TEST_DATA", "data/vsp"))
    processor_path = Path(os.environ.get("SPARA_TEST_PROCESSOR", ".cache/processor"))
    if not processor_path.is_dir() or not (root / "manifest.json").is_file():
        pytest.skip("Prepare reference data and processor")
    processor = AutoProcessor.from_pretrained(processor_path)
    base = initialize_latents(tiny_model, processor)
    tiny_model.config.livr_stage = 2
    vocab = processor.tokenizer.convert_tokens_to_ids(["<|image_pad|>", "<|video_pad|>", "<|vision_start|>",
                                                       "<|vision_end|>"])
    for name, value in zip(("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"), vocab):
        setattr(tiny_model.config, name, value)
    ids = extend_latents(tiny_model, processor, layout="global+patch", mask=True)
    assert ids[:4] == base and len(ids) == 4 + len(PATCH_TOKENS) == 68
    assert tiny_model.config.livr_latent_tokens == patch_latents()
    small = ("train", next(r for r in records(root, "train") if len(r["map_desc"]) == 3))
    large = ("fresh/detour-heldout-78", next(r for r in records(root, "fresh/detour-heldout-78")
                                             if len(r["map_desc"]) == 8))
    folder = tmp_path / "data" / "pairs"
    folder.mkdir(parents=True)
    for name, (split, row) in zip(("a.png", "b.png"), (small, large)):
        shutil.copy(image_path(root, split, row), folder / name)
    pair = {"text_input": "Image 1: <image>\nImage 2: <image>\nWhich map has more holes?",
            "image_inputs": ["a.png", "b.png"], "text_output": "\\boxed{B}"}
    prompt = encode(processor, tmp_path / "data", "pairs", pair, training=False, **latent_kwargs(tiny_model))
    assert "livr_grid_size" not in prompt
    text = processor.tokenizer.decode(prompt["input_ids"][0])
    assert text.count("<|vision_start|>") == 2 and "Which map has more holes?" in text
    assert text.index("Which map") < text.index("<|livr_0|>")  # the question precedes the latents
    sequence = prompt["input_ids"][0]
    assert len(latent_positions(prompt["input_ids"], ids)) == 4 + 2 * 16  # the globals and two images' patches
    runs = image_runs(sequence, tiny_model.config.image_token_id)
    assert len(runs) == 2 and len(runs[0]) != len(runs[1])
    allowed = patch_allowed(tiny_model, prompt, True)
    for m, run in enumerate(runs):
        tiles = patch_tiles(prompt["image_grid_thw"][m], 2, 4)
        covered = torch.zeros(len(run), dtype=torch.bool)
        for t in range(16):
            q = int((sequence == ids[4 + m * 16 + t]).nonzero()[0])
            seen = allowed[q].nonzero().squeeze(-1)
            image_seen = seen[torch.isin(seen, torch.cat(runs))]
            assert torch.isin(image_seen, run[tiles == t]).all() and len(image_seen) == int((tiles == t).sum())
            covered |= torch.isin(run, image_seen)
        assert covered.all()  # every image token belongs to one tile
    first_global = int((sequence == ids[0]).nonzero()[0])
    assert allowed[first_global, torch.cat(runs)].all()
    header = len(sequence) - 1  # the assistant header's last token, in Stage 1: no image
    assert not allowed[header, torch.cat(runs)].any()
    tiny_model.eval()
    for stage in (1, 2):
        tiny_model.config.livr_stage = stage
        with torch.no_grad():
            assert torch.isfinite(tiny_model(**prompt, use_cache=False).logits).all()
    single = encode(processor, root, *large, training=False, **latent_kwargs(tiny_model))
    assert len(latent_positions(single["input_ids"], ids)) == 20  # a FrozenLake map: one image, 16 tiles
    plain = encode(processor, root, *large, training=False)["input_ids"][0]
    end = int(latent_positions(plain[None], base)[-1]) + 1
    assert torch.equal(single["input_ids"][0, :end], plain[:end])  # E's prompt through the globals
    tiny_model.config.livr_stage = 2
    tiny_model.save_pretrained(tmp_path / "ckpt")
    processor.save_pretrained(tmp_path / "ckpt")
    (tmp_path / "ckpt/complete.json").write_text(json.dumps({"progress": {}}))
    model, restored = load_model(read_config("configs/livr.json") | {"attention_backend": "eager"}, tmp_path / "ckpt")
    assert model.config.livr_latent_ids == ids and model.config.livr_region_latents["layout"] == "global+patch"
    again = encode(restored, tmp_path / "data", "pairs", pair, training=False, **latent_kwargs(model))
    assert torch.equal(again["input_ids"], prompt["input_ids"])


def test_extend_latents_checks_the_patch_settings(tiny_model):
    class Fake:
        tokenizer = RegionTokenizer(SavingProcessor.tokenizer)
    with pytest.raises(ValueError, match="layout must be"):
        extend_latents(tiny_model, Fake(), layout="tiles")
    with pytest.raises(ValueError, match="text must be"):
        extend_latents(tiny_model, Fake(), layout="global+patch", mask=True, text="none")
    with pytest.raises(ValueError, match="block mask"):
        extend_latents(tiny_model, Fake(), layout="region", mask=True)
