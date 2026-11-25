import copy
import json
import zlib

import pytest
import torch

from spara import detour_scaling, expert_iteration, expert_round2, perception_jepa, training
from spara.grounding import StepGrounding
from spara.perception_jepa import (FEATURES_FILE, HEADS_FILE, JepaHeads, LatentJepa, Queue, SIGReg, cell_states,
                                   centres_of, compare, dependence, first_mistake, fixed_projection, from_saved,
                                   heads_error, hole_share, holes_read, layer_probe, load_heads, objective,
                                   participation, position_features, require_round2_checkpoint, run_features,
                                   run_fixed_gate, run_gate, run_jepa_arm, warm_start, within_position_share)
from spara.runtime import read_config
from test_detour_scaling import Evaluator, SavingProcessor
from test_expert_iteration import tiny
from test_expert_round2 import SETTINGS as ROUND2, data, route_rows, synthetic_states, write_jsonl, write_run  # noqa: F401
from test_planning_probe import SUPPRESS, Processor, prompt_of

JEPA = {"context_layer": 2, "target_layer": 2, "target_layers": [1, 2], "separates": 0.0, "dim": 8, "width": 16,
        "predictor_width": 16, "dropout": 0.0, "frequencies": 2, "sigreg_weight": 0.02, "directions": 16,
        "queue_context": 32, "queue_target": 64, "shuffle_pool": 8, "head_learning_rate": 1e-3,
        "head_weight_decay": 0.01, "warm_start": {"epochs": 3, "maps_per_batch": 4, "learning_rate": 1e-3},
        "validation_maps": 2, "monitor_maps": 2, "pilot_rows": 2, "beta_bounds": [0.0, 1e9],
        "swap_maps_per_size": 2, "swap_follows": 0.0, "max_new_tokens": 6, "seed": 0, "weight": 1.0}


class FullEvaluator(Evaluator):
    """Adds the reference evaluator's `answer` field, which the latent battery's `answer` reads."""
    def score(self, row, text):
        return super().score(row, text) | {"answer": text.split("<|im_start|>assistant")[-1]}


def aux_for(model, settings=None, centres=None):
    settings = JEPA | (settings or {})
    heads = JepaHeads(model.config.hidden_size, dim=8, width=16, predictor_width=16, dropout=0.0, frequencies=2)
    centres = centres or {n: torch.zeros(n * n, model.config.hidden_size) for n in range(3, 9)}
    return LatentJepa(settings, heads, centres, model.config.num_hidden_layers,
                      model.config.vision_config.spatial_merge_size, model.config.image_token_id,
                      model.config.livr_latent_ids)


# SIGReg and the queue ------------------------------------------------------------------------

def test_sigreg_prefers_a_standard_gaussian():
    sigreg, draw = SIGReg(64), torch.Generator().manual_seed(0)
    gaussian = torch.randn(1024, 16, generator=draw)
    score = lambda z: float(sigreg(z, torch.Generator().manual_seed(1)))
    assert score(gaussian) < 3.0  # scaled by N, the statistic stays of order one under the null
    assert score(gaussian * 5) > 10 * score(gaussian)
    assert score(torch.zeros(1024, 16) + 0.3) > 10 * score(gaussian)
    assert score(gaussian + 3) > 10 * score(gaussian)
    assert score(gaussian) == score(gaussian)  # seeded directions


def test_sigreg_spreads_a_collapsed_sample():
    z = (0.01 * torch.randn(512, 8, generator=torch.Generator().manual_seed(0))).requires_grad_()
    sigreg, optimizer = SIGReg(32), torch.optim.Adam([z], lr=0.05)
    generator = torch.Generator().manual_seed(2)
    for _ in range(200):
        optimizer.zero_grad()
        sigreg(z, generator).backward()
        optimizer.step()
    assert 0.5 < float(z.detach().std(0).mean()) < 1.5


def test_queue_keeps_the_newest_rows_detached():
    queue = Queue(3)
    first = torch.ones(2, 4, requires_grad=True)
    assert queue.with_current(first) is first
    queue.push(first)
    queue.push(torch.full((2, 4), 2.0))
    assert queue.items.shape == (3, 4) and not queue.items.requires_grad
    assert queue.items[:, 0].tolist() == [2.0, 2.0, 1.0]
    assert queue.with_current(torch.zeros(1, 4))[0, 0] == 0  # current rows come first


def test_position_features_tell_cells_apart():
    for n in (3, 8):
        features = position_features(n, 4)
        assert features.shape == (n * n, 17) and torch.unique(features, dim=0).shape[0] == n * n


# Targets and gradients ------------------------------------------------------------------------

def test_cell_states_pool_the_image_tokens(tiny_model, tiny_batch):
    model = tiny(tiny_model)
    prompt = prompt_of(tiny_batch)
    with torch.no_grad():
        hidden = model(**prompt, output_hidden_states=True, use_cache=False).hidden_states
    cells = cell_states(hidden, 2, prompt["input_ids"], 3, prompt["image_grid_thw"][0], 2, 3)
    image = hidden[2][0, 2]  # the tiny prompt's single merged image token
    assert cells.shape == (9, 32)
    torch.testing.assert_close(cells, image.expand(9, -1).float())
    with pytest.raises(ValueError):
        cell_states(hidden, 2, prompt["input_ids"], 3, torch.tensor([2, 2, 2]), 2, 3)


def test_the_term_reaches_the_model_only_through_the_latents(tiny_model, tiny_batch):
    model = tiny_model.train()
    model.config.livr_stage = 2
    aux = aux_for(model)
    aux.prepare(None, [{"map_desc": [[0] * 4] * 4}])
    out = model(**tiny_batch, use_cache=False, output_hidden_states=True)
    out.hidden_states[2].retain_grad()
    total, metrics = aux.loss(out.hidden_states, 0, tiny_batch)
    total.backward()
    grad = out.hidden_states[2].grad[0]
    assert grad[2].abs().sum() == 0  # the image token: the target is detached
    assert grad[5:9].abs().sum() > 0  # the four latents
    assert metrics["shuffled_gap"] is None and metrics["jepa_error"] > 0
    out = model(**tiny_batch, use_cache=False, output_hidden_states=True)
    assert aux.loss(out.hidden_states, 0, tiny_batch)[1]["shuffled_gap"] is not None  # a map is now pooled


def rows_for_training():
    return [{"map_id": i, "map_desc": [[1, 0, 0], [0, -1, 0], [0, 0, 2]], "text_output": "x",
             "image_input": f"img/{i}.png"} for i in range(4)]


def train(model, tiny_batch, tmp_path, monkeypatch, name, grounding=None, data=None):
    config = read_config("configs/livr.json") | {"epochs_per_stage": 2, "gradient_accumulation": 2,
                                                 "gradient_checkpointing": False}
    monkeypatch.setattr(training, "records", lambda *_: rows_for_training())
    monkeypatch.setattr(training, "encode", lambda *args, **kwargs: {k: v.clone() for k, v in tiny_batch.items()})
    training.train_stage(model, SavingProcessor(), "unused", tmp_path / name, config, 2, grounding=grounding,
                         data=data)
    return [json.loads(line) for line in (tmp_path / name / "steps.jsonl").read_text().splitlines()]


def test_zero_weight_matches_plain_training(tiny_model, tiny_batch, tmp_path, monkeypatch):
    plain, other = tiny_model, copy.deepcopy(tiny_model)
    plain_log = train(plain, tiny_batch, tmp_path, monkeypatch, "plain")
    jepa_log = train(other, tiny_batch, tmp_path, monkeypatch, "jepa", aux_for(other, {"weight": 0.0}))
    assert [r["loss"] for r in plain_log] == [r["loss"] for r in jepa_log]
    # The heads are clipped on their own, so the language model's clipping norm is the plain run's.
    assert [r["grad_norm"] for r in plain_log] == [r["grad_norm"] for r in jepa_log]
    assert all(r["head_grad_norm"] is not None for r in jepa_log) and "head_grad_norm" not in plain_log[0]
    for key, expected in plain.state_dict().items():
        torch.testing.assert_close(other.state_dict()[key], expected, atol=1e-8, rtol=1e-6)


def test_a_jepa_run_logs_its_terms_and_saves_its_heads(tiny_model, tiny_batch, tmp_path, monkeypatch):
    aux = aux_for(tiny_model)
    log = train(tiny_model, tiny_batch, tmp_path, monkeypatch, "run", aux,
                data=[("fresh/any", row) for row in rows_for_training()])
    assert all(r["jepa_loss"] is not None and r["sigreg_target"] is not None for r in log)
    assert log[-1]["shuffled_gap"] is not None
    heads, centres, saved = load_heads(tmp_path / "run" / HEADS_FILE)
    assert saved["settings"]["weight"] == 1.0 and set(centres) == set(range(3, 9))
    assert saved["queues"]["target"].shape[0] == min(JEPA["queue_target"], 4 * 2 * 9)
    torch.testing.assert_close(heads.state_dict()["predictor.0.weight"], aux.head.predictor[0].weight.detach().cpu())


def test_step_grounding_still_refuses_other_data(tiny_model, tiny_batch, tmp_path, monkeypatch):
    settings = {"layer": 2, "weight": 1.0, "head_learning_rate": 1e-3}
    with pytest.raises(ValueError, match="released training split"):
        train(tiny_model, tiny_batch, tmp_path, monkeypatch, "g", StepGrounding(settings, 32, 3),
              data=[("fresh/any", row) for row in rows_for_training()])


# The gate's pieces --------------------------------------------------------------------------

def test_layer_probe_reads_holes_and_centres_remove_position():
    generator = torch.Generator().manual_seed(0)
    rows, cells = [], []
    for i in range(40):
        desc = [[-1 if torch.rand(1, generator=generator) < 0.3 else 0 for _ in range(4)] for _ in range(4)]
        desc[0][0], desc[3][3] = 1, 2
        holes = torch.tensor([float(v == -1) for line in desc for v in line])
        x = torch.randn(16, 6, generator=generator) * 0.1
        x[:, 0] += 3 * holes
        x[:, 1] += torch.arange(16.0)  # position
        rows.append({"map_desc": desc})
        cells.append(x)
    assert layer_probe(cells, rows, list(range(30)), list(range(30, 40)), "cpu")["4"] > 0.95
    centres = centres_of(cells, rows, range(30))
    centred = torch.stack([cells[i] - centres[4] for i in range(30)])
    assert centred[:, :, 1].std(0).max() < 0.5  # position gone


def test_warm_start_lowers_the_validation_error():
    generator = torch.Generator().manual_seed(0)
    mixing = torch.randn(4 * 6, 16 * 6, generator=generator)

    def maps(count):
        contexts = [torch.randn(4, 6, generator=generator) for _ in range(count)]
        cells = [(c.flatten() @ mixing).view(16, 6) for c in contexts]
        return contexts, cells, [4] * count
    heads = JepaHeads(6, dim=8, width=32, predictor_width=64, dropout=0.0, frequencies=2)
    settings = JEPA | {"warm_start": {"epochs": 40, "maps_per_batch": 16, "learning_rate": 3e-3}}
    validation = maps(16)
    start = heads_error(heads, *validation, "cpu")
    record, queues = warm_start(heads, SIGReg(16), maps(64), validation, settings, "cpu")
    assert start > 0.9 and record["best_validation_error"] < 0.5 * start
    assert heads_error(heads, *validation, "cpu") == pytest.approx(record["best_validation_error"])  # best kept
    assert queues["context"].shape[1] == 8 and len(queues["target"]) <= JEPA["queue_target"]


def test_require_round2_checkpoint(tmp_path):
    config = read_config("configs/livr.json")
    round2 = json.loads(open("configs/expert_round2.json").read())
    tuned = config | {"epochs_per_stage": 1, "learning_rate": 1e-5, "warmup_ratio": 0.05}
    progress = {"stage": 2, "epoch": 1, "next_row": 0, "optimizer_steps": 188, "config": tuned}
    (tmp_path / "complete.json").write_text(json.dumps({"progress": progress}))
    require_round2_checkpoint(tmp_path, config, round2)
    (tmp_path / "complete.json").write_text(json.dumps({"progress": progress | {"optimizer_steps": 187}}))
    with pytest.raises(ValueError):
        require_round2_checkpoint(tmp_path, config, round2)


# End to end on the tiny model ----------------------------------------------------------------

@pytest.fixture
def patched(data, tiny_batch, monkeypatch):
    monkeypatch.setattr(perception_jepa, "records", lambda root, split: data[split])

    def encode(*args, training=True, **kwargs):
        return {k: v.clone() for k, v in tiny_batch.items()} if training else prompt_of(tiny_batch)
    for module in (perception_jepa, expert_round2, detour_scaling, expert_iteration, training):
        monkeypatch.setattr(module, "encode", encode)
    monkeypatch.setattr(perception_jepa, "run_benchmark", lambda model, processor, config, root, output, manifest:
                        output.mkdir())
    return data


def round_runs(tmp_path, data):
    round1 = write_jsonl(tmp_path / "round1.jsonl", route_rows(data["fresh/p"], {0, 1}))
    search = tmp_path / "search"
    search.mkdir()
    write_jsonl(search / "routes.jsonl", route_rows(data["fresh/p78"], {1, 3}))
    write_jsonl(search / "frontier.jsonl", [])
    (search / "manifest.json").write_text(json.dumps({"settings": ROUND2}))  # as a search job writes it
    return round1, search


def test_the_gate_then_the_arm(patched, tiny_model, tmp_path):
    model = tiny(tiny_model)
    model.generation_config.suppress_tokens = SUPPRESS  # the random model must not write image or latent tokens
    config = read_config("configs/livr.json") | {"gradient_checkpointing": False}
    round1, search = round_runs(tmp_path, patched)
    e2 = copy.deepcopy(model)
    gate = run_gate(model, SavingProcessor(), FullEvaluator(), config, "unused", tmp_path / "gate", JEPA, ROUND2, round1,
                    search, lambda: e2, {})
    assert gate["valid"] and gate["reading"] == "gates passed: run EJ2"
    assert gate["target_layer"] == 1 and gate["weight"]["beta"] > 0 and len(gate["weight"]["norms"]) == 2
    assert all(n["answer"] > 0 and n["jepa"] > 0 and n["jepa_heads"] > 0 for n in gate["weight"]["norms"])
    assert 0 <= gate["swap_e"]["first_two_follow_donor"] <= gate["swap_e"]["maps"]
    assert gate["swap_e"]["identity"] == [True] * len(gate["swap_e"]["identity"]) and "swap_e2" in gate
    aux = from_saved(tmp_path / "gate" / HEADS_FILE, model)
    assert aux.weight == pytest.approx(gate["weight"]["beta"]) and aux.settings["target_layer"] == 1
    assert aux.queues["target"].items is not None

    summary = run_jepa_arm(model, SavingProcessor(), Evaluator(), config, "unused", tmp_path / "ej2", ROUND2,
                           tmp_path / "gate", round1, search, {})
    stages = [json.loads(line)["stage"] for p in ("phase1", "phase2")
              for line in (tmp_path / "ej2" / p / "steps.jsonl").read_text().splitlines()]
    assert stages == [1, 2] and summary["training"]["phase1"]["stage"] == 1
    assert set(summary["training"]["jepa_error"]) == {"start", "phase1", "phase2"}
    assert (tmp_path / "ej2/phase2" / HEADS_FILE).exists() and (tmp_path / "ej2/states.pt").exists()
    steps = [json.loads(line) for line in (tmp_path / "ej2/phase1/steps.jsonl").read_text().splitlines()]
    assert steps[0]["jepa_loss"] is not None


def test_a_failed_gate_stops(patched, tiny_model, tmp_path):
    model = tiny(tiny_model)
    config = read_config("configs/livr.json") | {"gradient_checkpointing": False}
    round1, search = round_runs(tmp_path, patched)
    summary = run_gate(model, SavingProcessor(), FullEvaluator(), config, "unused", tmp_path / "gate",
                       JEPA | {"separates": 1.01}, ROUND2, round1, search, lambda: None, {})
    assert not summary["valid"] and summary["reading"].startswith("gate failed")
    with pytest.raises(ValueError, match="did not pass"):
        run_jepa_arm(model, SavingProcessor(), Evaluator(), config, "unused", tmp_path / "ej2", ROUND2,
                     tmp_path / "gate", round1, search, {})


# The retry with fixed targets (EJF) ----------------------------------------------------------

FIXED = {"context_layer": 12, "target_layer": 4, "components": 8, "fallback_components": 4, "separates": 0.95,
         "hole_share": 0.10, "shuffled_gap": 0.02, "holes_from_predictions": 0.55, "dim": 16, "width": 32,
         "predictor_width": 64, "dropout": 0.0, "frequencies": 2, "sigreg_weight": 0.0, "directions": 16,
         "queue_context": 32, "queue_target": 64, "shuffle_pool": 8, "head_learning_rate": 1e-3,
         "head_weight_decay": 0.01, "warm_start": {"epochs": 40, "maps_per_batch": 16, "learning_rate": 3e-3},
         "validation_maps": 20, "monitor_maps": 2, "pilot_rows": 2, "beta_bounds": [0.0, 1e9], "seed": 0}


def synthetic_features(path, latents_see_holes=True, hidden=24, seed=0, like_e=False):
    """A features file whose cells carry position (large), holes and an outlier channel, and whose
    latents carry the map's holes linearly, or nothing. `like_e`: as E's latents, a few huge constant
    channels and a map-dependent part a thousand times smaller."""
    g = torch.Generator().manual_seed(seed)
    offsets = {n: 5 * torch.randn(n * n, hidden, generator=g) for n in (3, 4)}
    hole_dir, mixing = torch.randn(hidden, generator=g), torch.randn(16, 4 * hidden, generator=g)
    keys, descs, cells, contexts = [], [], [], []
    groups = {"training": [], "pool": [], "heldout": [], "validation": []}
    for split, count, n in (("train", 30, 3), ("train", 120, 4), ("p", 60, 4), ("h", 40, 4)):
        for k in range(count):
            holes = (torch.rand(n * n, generator=g) < 0.3).float()
            desc = [[-1 if holes[r * n + c] else 0 for c in range(n)] for r in range(n)]
            cell = offsets[n] + holes[:, None] * hole_dir + 0.3 * torch.randn(n * n, hidden, generator=g)
            cell[:, 0] += 100 * torch.randn(n * n, generator=g)  # one huge channel of noise
            padded = torch.cat([holes, torch.zeros(16 - n * n)])
            context = (padded @ mixing if latents_see_holes else 3 * torch.randn(4 * hidden, generator=g))
            key = (split, len(keys))
            keys.append(key), descs.append(desc), cells.append(cell.half())
            context = (context + 0.3 * torch.randn(4 * hidden, generator=g)).view(4, hidden)
            if like_e:
                context = 1e-3 * context
                context[:, :3] += 300.0
            contexts.append(context.float() if like_e else context.half())
            group = {"train": "training", "p": "pool", "h": "heldout"}[split]
            groups[group].append(key)
            if split == "p" and k < 20:
                groups["validation"].append(key)
    path.mkdir()
    torch.save({"context_layer": 12, "target_layer": 4, "keys": keys, "map_desc": descs, "groups": groups,
                "contexts": torch.stack(contexts), "cells": cells}, path / FEATURES_FILE)
    return keys, descs, cells


def test_fixed_projection_removes_position_and_ignores_channel_scale():
    g = torch.Generator().manual_seed(0)
    cells = torch.randn(2000, 12, generator=g) @ torch.randn(12, 12, generator=g)
    cells = cells - cells.mean(0)
    matrix, spectrum = fixed_projection(cells, 5)
    targets = cells @ matrix.T
    assert matrix.shape == (5, 12) and 0 < spectrum["variance_kept"] < 1
    assert targets.var(0).mean() == pytest.approx(1.0, rel=1e-3)
    assert targets.mean(0).abs().max() < 1e-4  # linear: a zero-mean position keeps zero-mean targets
    scaled = cells * torch.logspace(-2, 3, 12)  # channels on very different scales give the same targets
    torch.testing.assert_close((scaled @ fixed_projection(scaled, 5)[0].T).abs(), targets.abs(), atol=1e-3, rtol=1e-3)


def test_hole_share_within_position_share_and_holes_read():
    g = torch.Generator().manual_seed(0)
    holes = [(torch.rand(16, generator=g) < 0.3).float() for _ in range(60)]
    signal = [h[:, None] * torch.ones(1, 4) + 0.1 * torch.randn(16, 4, generator=g) for h in holes]
    noise = [torch.randn(16, 4, generator=g) for _ in holes]
    assert hole_share(signal, holes) > 0.8 and hole_share(noise, holes) < 0.05
    position = torch.randn(1, 16, 4, generator=g).expand(60, 16, 4)
    assert within_position_share(position + 1e-3 * torch.stack(noise)) < 1e-4
    assert within_position_share(torch.stack(noise)) > 0.9
    rows = [{"map_desc": [[-1 if h[r * 4 + c] else 0 for c in range(4)] for r in range(4)]} for h in holes]
    assert holes_read(signal, rows, range(40), range(40, 60))["4"] > 0.95
    assert holes_read(noise, rows, range(40), range(40, 60))["4"] < 0.7


def test_dependence_measures_use_of_the_latents():
    g = torch.Generator().manual_seed(0)
    latents = [torch.randn(3, generator=g) for _ in range(30)]
    weights = torch.randn(3, 9 * 2, generator=g)
    targets = [(z @ weights).view(9, 2) + 0.1 * torch.randn(9, 2, generator=g) for z in latents]
    uses = dependence(lambda i, j: (latents[j] @ weights).view(9, 2), targets, [3] * 30)
    ignores = dependence(lambda i, j: torch.zeros(9, 2), targets, [3] * 30)
    assert uses["unexplained"] < 0.05 and uses["shuffled_gap"] > 1.0
    assert ignores["unexplained"] == pytest.approx(1.0, abs=0.05) and ignores["shuffled_gap"] == 0
    assert dependence(lambda i, j: torch.zeros(9, 2), targets[:2], [3, 4])["shuffled_gap"] is None  # no donors


def test_objective_without_sigreg_is_the_squared_error():
    heads = JepaHeads(6, dim=4, width=8, predictor_width=8, dropout=0.0, frequencies=2, target="fixed", target_dim=3)
    total, error, reg_c, reg_t = objective(heads, SIGReg(4), torch.randn(4, 4), torch.randn(9, 3), 3, None, None,
                                           None, 0.0)
    assert total is error and reg_c is None and reg_t is None
    assert isinstance(heads.target, torch.nn.Identity) and heads.predictor[-1].out_features == 3
    with pytest.raises(ValueError):
        JepaHeads(6, dim=4, target_dim=3)  # a learned target projector outputs `dim`


def test_the_fixed_gate_passes_when_the_latents_see_the_holes(tmp_path):
    keys, descs, cells = synthetic_features(tmp_path / "features")
    summary = run_fixed_gate(tmp_path / "features", tmp_path / "gate", FIXED)
    assert summary["valid"] and summary["reading"] == "gates passed: run EJF", summary["reading"]
    first = summary["targets"][0]
    assert summary["components"] == 8 and len(summary["targets"]) == 1
    assert first["hole_probe"]["4"] >= 0.95 and first["hole_share"]["4"] >= 0.10
    assert first["between_position_share"]["4"] < 0.1  # position is gone from the targets
    found = summary["latent_dependence"]["4"]
    assert found["shuffled_gap"] >= 0.02 and found["holes_from_predictions"] >= 0.55
    assert summary["linear_ceiling"]["4"]["holes_from_predictions"] > 0.55
    assert summary["warm_start"]["best_validation_error"] < summary["warm_start"]["start_validation_error"]
    heads, centres, saved = load_heads(tmp_path / "gate" / HEADS_FILE)
    assert heads.config["target"] == "fixed" and set(saved["targets"]) == set(keys)
    assert saved["targets"][keys[0]].shape == (9, 8) and saved["gate"]["beta"] is None
    # Fitted maps' targets have zero mean at every position, by construction.
    fitted = torch.stack([saved["targets"][k].float() for k, d in zip(keys, descs)
                          if len(d) == 4 and k[0] in ("train", "p")])
    assert fitted.mean(0).abs().max() < 2e-2


def test_the_fixed_gate_stops_when_the_latents_carry_nothing(tmp_path):
    synthetic_features(tmp_path / "features", latents_see_holes=False)
    summary = run_fixed_gate(tmp_path / "features", tmp_path / "gate", FIXED)
    assert not summary["valid"] and summary["reading"].startswith("gate 3c failed"), summary["reading"]
    assert "linear ceiling fails too" in summary["reading"]
    assert (tmp_path / "gate" / HEADS_FILE).exists()  # kept for diagnosis
    synthetic_features(tmp_path / "f2")
    summary = run_fixed_gate(tmp_path / "f2", tmp_path / "gate2", FIXED | {"hole_share": 0.999})
    assert summary["reading"].startswith("gate 3b failed") and [t["components"] for t in summary["targets"]] == [8, 4]
    with pytest.raises(ValueError, match="no SIGReg"):
        run_fixed_gate(tmp_path / "f2", tmp_path / "gate3", FIXED | {"sigreg_weight": 0.02})


def test_standardising_the_latents_lets_the_heads_see_the_map(tmp_path):
    synthetic_features(tmp_path / "features", like_e=True)
    raw = run_fixed_gate(tmp_path / "features", tmp_path / "raw", FIXED | {"holes_from_predictions": None})
    fixed = run_fixed_gate(tmp_path / "features", tmp_path / "fixed",
                           FIXED | {"holes_from_predictions": None, "standardise_context": True})
    assert fixed["valid"] and fixed["context_standardised_on"] == "training maps", fixed["reading"]
    assert fixed["latent_dependence"]["4"]["shuffled_gap"] > 10 * max(raw["latent_dependence"]["4"]["shuffled_gap"], 1e-3)
    assert (fixed["latent_dependence"]["4"]["hole_direction_share"]
            > raw["latent_dependence"]["4"]["hole_direction_share"] + 0.15)
    assert "hole_direction_share" in fixed["linear_ceiling"]["4"]
    heads, _, saved = load_heads(tmp_path / "fixed" / HEADS_FILE)
    assert heads.config["standardise"] and saved["state_dict"]["context.0.mean"].shape == (4, 24)
    features = torch.load(tmp_path / "features" / FEATURES_FILE, weights_only=False)
    train = [i for i, k in enumerate(features["keys"]) if k in set(features["groups"]["training"])]
    torch.testing.assert_close(heads.context[0].mean, features["contexts"][train].float().mean(0))


def test_passes_3c_without_a_holes_threshold():
    from spara.perception_jepa import passes_3c
    found = {"7": {"shuffled_gap": 0.05, "holes_from_predictions": 0.5}}
    assert passes_3c(found, {"shuffled_gap": 0.02, "holes_from_predictions": None})
    assert not passes_3c(found, {"shuffled_gap": 0.02, "holes_from_predictions": 0.55})
    assert not passes_3c({"7": None}, {"shuffled_gap": 0.02, "holes_from_predictions": None})


def fixed_aux(model, keys, n=3, components=4):
    heads = JepaHeads(model.config.hidden_size, dim=8, width=16, predictor_width=16, dropout=0.0, frequencies=2,
                      target="fixed", target_dim=components)
    g = torch.Generator().manual_seed(0)
    targets = {k: torch.randn(n * n, components, generator=g).half() for k in keys}
    return LatentJepa(JEPA | {"sigreg_weight": 0.0}, heads, {}, model.config.num_hidden_layers,
                      model.config.vision_config.spatial_merge_size, model.config.image_token_id,
                      model.config.livr_latent_ids, targets=targets)


def test_fixed_targets_train_through_the_latents(tiny_model, tiny_batch, tmp_path, monkeypatch):
    data = [("fresh/any", row) for row in rows_for_training()]
    aux = fixed_aux(tiny_model, [(s, r["image_input"]) for s, r in data])
    with pytest.raises(ValueError, match="Bind"):
        aux.prepare(None, [row for _, row in data])
    aux.bind([("fresh/other", row) for _, row in data])
    with pytest.raises(ValueError, match="No fixed targets"):
        aux.prepare(None, [row for _, row in data])
    aux.bind(data)
    log = train(tiny_model, tiny_batch, tmp_path, monkeypatch, "run", aux, data=data)
    assert all(r["jepa_loss"] is not None and r["sigreg_target"] is None for r in log)
    assert all(r["jepa_loss"] == pytest.approx(r["jepa_error"]) for r in log)  # no SIGReg
    heads, _, saved = load_heads(tmp_path / "run" / HEADS_FILE)
    assert set(saved["targets"]) == set(aux.targets) and heads.config["target"] == "fixed"


@pytest.fixture
def varied(patched, tiny_batch, monkeypatch):
    """As `patched`, but each map's prompt pass sees its own image, so its cells differ."""
    def encode(processor, root, split, row, training=True, **kwargs):
        batch = {k: v.clone() for k, v in tiny_batch.items()} if training else prompt_of(tiny_batch)
        g = torch.Generator().manual_seed(zlib.crc32(f"{split}/{row['map_id']}".encode()))
        batch["pixel_values"] = batch["pixel_values"] + torch.randn(batch["pixel_values"].shape, generator=g)
        return batch
    monkeypatch.setattr(perception_jepa, "encode", encode)
    return patched


def test_features_then_the_fixed_gate_then_the_arm(varied, tiny_model, tmp_path):
    model = tiny(tiny_model)
    model.generation_config.suppress_tokens = SUPPRESS
    config = read_config("configs/livr.json") | {"gradient_checkpointing": False}
    round1, search = round_runs(tmp_path, varied)
    first = varied["train"][0]
    other = next(r for r in varied["fresh/p"] if len(r["map_desc"]) != len(first["map_desc"]))
    # A released training map whose ID repeats at another level, as in the data: a different map.
    varied["train"].append(other | {"map_id": first["map_id"], "image_input": "img/other-level/0.png"})
    settings = FIXED | {"context_layer": 2, "target_layer": 1, "components": 4, "fallback_components": 2,
                        "dim": 8, "width": 16, "predictor_width": 16, "validation_maps": 2, "separates": 0.0,
                        "hole_share": 0.0, "shuffled_gap": -1.0, "holes_from_predictions": None,
                        "standardise_context": True,
                        "warm_start": {"epochs": 2, "maps_per_batch": 4, "learning_rate": 1e-3}}
    features = run_features(model, SavingProcessor(), config, "unused", tmp_path / "features", settings, ROUND2,
                            round1, search, {})
    saved = torch.load(tmp_path / "features" / FEATURES_FILE, weights_only=False)
    assert features["maps"] == len(saved["keys"]) == len(saved["cells"]) == saved["contexts"].shape[0]
    assert saved["contexts"].shape[1:] == (4, model.config.hidden_size)
    assert set(saved["groups"]) == {"training", "pool", "heldout", "validation"}
    train_keys = [k for k in saved["keys"] if k[0] == "train"]
    assert len(train_keys) == 3 and len({k for k in train_keys}) == 3  # the repeated ID is a separate map
    gate = run_fixed_gate(tmp_path / "features", tmp_path / "gate", settings)
    assert gate["valid"], gate["reading"]
    # A map the gate lacks gets its target at the start of the arm, from the same pass and projection.
    heads_file = torch.load(tmp_path / "gate" / HEADS_FILE, weights_only=False)
    dropped = ("train", "img/other-level/0.png")
    kept = heads_file["targets"].pop(dropped)
    torch.save(heads_file, tmp_path / "gate" / HEADS_FILE)
    summary = run_jepa_arm(model, SavingProcessor(), Evaluator(), config, "unused", tmp_path / "ejf", ROUND2,
                           tmp_path / "gate", round1, search, {})
    assert summary["targets_added"] == 1
    _, _, after = load_heads(tmp_path / "ejf/phase2" / HEADS_FILE)
    torch.testing.assert_close(after["targets"][dropped].float(), kept.float(), atol=2e-3, rtol=1e-2)
    assert summary["arm"] == "ejf" and summary["weight"]["beta"] > 0 and len(summary["weight"]["norms"]) == 2
    trained, _, _ = load_heads(tmp_path / "ejf/phase2" / HEADS_FILE)
    gate_heads, _, _ = load_heads(tmp_path / "gate" / HEADS_FILE)
    assert trained.config["standardise"]  # the arm keeps the gate's fixed standardisation
    torch.testing.assert_close(trained.context[0].std, gate_heads.context[0].std)
    assert set(summary["training"]["jepa_error"]) == {"start", "phase1", "phase2"}
    assert set(summary["training"]["jepa_error"]["start"]["validation"]) == {"unexplained", "shuffled_gap"}
    steps = [json.loads(line) for line in (tmp_path / "ejf/phase1/steps.jsonl").read_text().splitlines()]
    assert steps[0]["jepa_loss"] is not None and steps[0]["sigreg_context"] is None
    progress = json.loads((tmp_path / "ejf/phase2/summary.json").read_text())
    assert progress["grounding"]["weight"] == pytest.approx(summary["weight"]["beta"])
    assert (tmp_path / "ejf/phase2" / HEADS_FILE).exists() and (tmp_path / "ejf/states.pt").exists()


def test_the_fixed_arm_holds_an_out_of_bounds_weight(varied, tiny_model, tmp_path):
    model = tiny(tiny_model)
    config = read_config("configs/livr.json") | {"gradient_checkpointing": False}
    round1, search = round_runs(tmp_path, varied)
    settings = FIXED | {"context_layer": 2, "target_layer": 1, "components": 4, "fallback_components": 2,
                        "dim": 8, "width": 16, "predictor_width": 16, "validation_maps": 2, "separates": 0.0,
                        "hole_share": 0.0, "shuffled_gap": -1.0, "holes_from_predictions": 0.0,
                        "warm_start": {"epochs": 1, "maps_per_batch": 4, "learning_rate": 1e-3},
                        "beta_bounds": [1e8, 1e9]}
    run_features(model, SavingProcessor(), config, "unused", tmp_path / "features", settings, ROUND2, round1, search, {})
    assert run_fixed_gate(tmp_path / "features", tmp_path / "gate", settings)["valid"]
    summary = run_jepa_arm(model, SavingProcessor(), Evaluator(), config, "unused", tmp_path / "ejf", ROUND2,
                           tmp_path / "gate", round1, search, {})
    assert summary["reading"].startswith("held: the loss weight") and not (tmp_path / "ejf/phase1").exists()


# Readings ------------------------------------------------------------------------------------

def test_first_mistake():
    desc = [[1, 0, 0], [0, -1, 0], [0, 0, 2]]
    assert first_mistake(desc, "RRDD") == (None, False)
    assert first_mistake(desc, "DR")[0] == "hole"
    assert first_mistake(desc, "R") == ("stopped", True)
    assert first_mistake(desc, "U")[0] == "edge"
    assert first_mistake(desc, "RRDDL") == ("overrun", False)


def test_participation_counts_directions():
    states = synthetic_states(1.0)
    ratio = participation(states, 1)
    assert set(ratio) == {"6"} and 1 < ratio["6"] < 30


def arm_files(path, first_loss, jepa=False):
    lines = [json.dumps({"loss": first_loss}), json.dumps({"loss": first_loss / 2})]
    (path / "phase1").mkdir()
    (path / "phase1/steps.jsonl").write_text("\n".join(lines) + "\n")
    if jepa:
        summary = json.loads((path / "summary.json").read_text())
        summary["training"]["jepa_error"] = {"start": {"validation": 1.0}}
        (path / "summary.json").write_text(json.dumps(summary))


def test_compare_takes_a_supplied_manipulation_check_and_start_check(data, tmp_path, monkeypatch):
    monkeypatch.setattr(perception_jepa, "records", lambda root, split: data[split])
    monkeypatch.setattr(perception_jepa, "paired", lambda a, b, keys: detour_scaling.paired(a, b, keys) | {"p": 0.0})
    clean, noisy = synthetic_states(0.0), synthetic_states(30.0)
    write_run(tmp_path / "e", data, [0, 0, 0, 0], [0, 0, 0, 0], states=noisy)
    write_run(tmp_path / "e2", data, [1, 0, 0, 0], [0, 0, 0, 0], states=noisy, benchmark=[1, 1, 0])
    write_run(tmp_path / "eb2", data, [1, 0, 0, 0], [0, 0, 0, 0], states=noisy, benchmark=[1, 0, 0])
    write_run(tmp_path / "ejr", data, [1, 1, 1, 0], [1, 1, 0, 0], states=noisy, benchmark=[1, 0, 1])
    arm_files(tmp_path / "eb2", 0.1)
    arm_files(tmp_path / "ejr", 0.7, jepa=True)  # a different first step: the start check stands in
    (tmp_path / "gate").mkdir()
    (tmp_path / "gate/summary.json").write_text(json.dumps({"valid": True}))
    livr = write_jsonl(tmp_path / "livr.jsonl", [{"row_index": i, "success": True} for i in range(3)])
    runs = {name: tmp_path / name for name in ("e", "e2", "eb2", "ejr")}
    supplied = {"ejr": expert_round2.probe_predictions(clean, 1), "eb2": expert_round2.probe_predictions(noisy, 1)}
    result = compare("unused", ROUND2, runs, tmp_path / "gate", livr, arm="ejr", manipulation_probes=supplied,
                     start_check={"matched": True, "small_rows": 2})
    assert result["validity"]["valid"] and result["validity"]["matched_start"]["small_rows"] == 2
    assert result["manipulation"]["took"] and "all_tokens" in result["manipulation"]
    assert result["manipulation"]["all_tokens"]["ejr"]["hole_balanced_accuracy"] < 0.7  # noisy states, all tokens
    assert result["primary"]["reading"] == "the latent JEPA adds to expert iteration"
    withheld = compare("unused", ROUND2, runs, tmp_path / "gate", livr, arm="ejr", manipulation_probes=supplied,
                       start_check={"matched": False})
    assert withheld["primary"]["reading"].startswith("withheld")


def test_compare_applies_the_fixed_rules(data, tmp_path, monkeypatch):
    monkeypatch.setattr(perception_jepa, "records", lambda root, split: data[split])
    monkeypatch.setattr(perception_jepa, "paired", lambda a, b, keys: detour_scaling.paired(a, b, keys) | {"p": 0.0})
    clean, noisy = synthetic_states(0.0), synthetic_states(30.0)
    write_run(tmp_path / "e", data, [0, 0, 0, 0], [0, 0, 0, 0], states=noisy)
    write_run(tmp_path / "e2", data, [1, 0, 0, 0], [0, 0, 0, 0], states=noisy, benchmark=[1, 1, 0])
    write_run(tmp_path / "eb2", data, [1, 0, 0, 0], [0, 0, 0, 0], states=noisy, benchmark=[1, 0, 0])
    write_run(tmp_path / "ej2", data, [1, 1, 1, 0], [1, 1, 0, 0], states=clean, benchmark=[1, 0, 1])
    arm_files(tmp_path / "eb2", 0.1)
    arm_files(tmp_path / "ej2", 0.1, jepa=True)
    (tmp_path / "gate").mkdir()
    (tmp_path / "gate/summary.json").write_text(json.dumps({"valid": True, "weight": {"beta": 2.0}}))
    livr = write_jsonl(tmp_path / "livr.jsonl", [{"row_index": i, "success": True} for i in range(3)])
    runs = {name: tmp_path / name for name in ("e", "e2", "eb2", "ej2")}
    result = compare("unused", ROUND2, runs, tmp_path / "gate", livr)
    assert result["validity"]["valid"] and result["manipulation"]["took"]
    assert result["primary"]["reading"] == "the latent JEPA adds to expert iteration"
    assert result["secondary"]["reading"] == "EJ2 solves more than E2"
    assert result["descriptive"]["held_out_success"] == {"e": 0, "e2": 1, "eb2": 1, "ej2": 3}
    assert sum(result["descriptive"]["first_mistakes"]["ej2"].values()) == 1
    assert result["gate"]["beta"] == 2.0

    arm_files_mismatch = tmp_path / "ej2/phase1/steps.jsonl"
    arm_files_mismatch.write_text(json.dumps({"loss": 0.5}) + "\n")
    assert compare("unused", ROUND2, runs, tmp_path / "gate", livr)["primary"]["reading"].startswith("withheld")
    arm_files_mismatch.write_text(json.dumps({"loss": 0.1}) + "\n")
    torch.save(noisy, tmp_path / "ej2/states.pt")
    failed = compare("unused", ROUND2, runs, tmp_path / "gate", livr)
    assert failed["primary"]["reading"] == "the latent JEPA did not take; no reading on perception"
    assert failed["secondary"]["reading"] == "EJ2 solves more than E2"  # not gated on the manipulation

    # EJF: the same readings under its own name; its loss weight is in its own summary, not the gate's.
    torch.save(clean, tmp_path / "ej2/states.pt")
    (tmp_path / "ej2").rename(tmp_path / "ejf")
    summary = json.loads((tmp_path / "ejf/summary.json").read_text())
    (tmp_path / "ejf/summary.json").write_text(json.dumps(summary | {"weight": {"beta": 3.0}}))
    (tmp_path / "gate/summary.json").write_text(json.dumps({"valid": True, "latent_dependence": {"7": {}}}))
    runs = {name: tmp_path / name for name in ("e", "e2", "eb2", "ejf")}
    fixed = compare("unused", ROUND2, runs, tmp_path / "gate", livr, arm="ejf")
    assert fixed["secondary"]["reading"] == "EJF solves more than E2" and fixed["manipulation"]["took"]
    assert fixed["gate"]["beta"] == 3.0 and "latent_dependence" in fixed["gate"]
    assert fixed["validity"]["matched_start"]["ejf"] == pytest.approx(0.1)
