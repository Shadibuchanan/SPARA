"""CPU tests of the latent diagnostics (docs/LATENT_DIAGNOSTICS_SPEC.md) on a tiny model."""
import copy
import json
import math
import random

import pytest
import torch
from transformers import Qwen2_5_VLConfig

from spara import expert_iteration, fair_rl
from spara import latent_diagnostics as diagnostics
from spara.fresh_maps import answer_text, draw_detour_layouts, label_route
from spara.latent_rl import LatentAdjuster, add_latent_heads
from spara.model import LivrQwen, freeze_vision
from spara.text_rl import group_advantages, token_logprobs

SUPPRESS = [2, 3, 4, 5, 10, 11, 12, 13]
ALPHABET = sorted(set("\\boxed{}UPDOWNLEFTRIGHT, ?"))
EOS = 6


class CharTokenizer:
    """One token per character (ids from 14), `<|im_end|>` as the end token; enough for the
    move alignment helpers. Unknown ids decode to '?'."""
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        ids, rest = [], text
        while rest:
            if rest.startswith("<|im_end|>"):
                ids.append(EOS); rest = rest[len("<|im_end|>"):]
            else:
                ids.append(14 + ALPHABET.index(rest[0])); rest = rest[1:]
        return {"input_ids": torch.tensor([ids])}

    def decode(self, ids, skip_special_tokens=False):
        out = []
        for i in [int(x) for x in ids]:
            if i == EOS:
                out.append("" if skip_special_tokens else "<|im_end|>")
            elif 14 <= i < 14 + len(ALPHABET):
                out.append(ALPHABET[i - 14])
            elif i >= 14:
                out.append("?")
        return "".join(out)


class Processor:
    tokenizer = CharTokenizer()


class Evaluator:
    """Labels succeed; other answers succeed when their text has an even length."""
    def score(self, row, text):
        answer = text.split("<|im_start|>assistant")[-1]
        moves = "".join(c for c in answer if c in "UDLR")
        success = text.endswith(row.get("text_output", "")) or len(text) % 2 == 0
        return {"answer": answer, "extracted_answer": moves, "success": success}


def tiny():
    torch.manual_seed(42)
    config = Qwen2_5_VLConfig(
        vocab_size=14 + len(ALPHABET) + 4, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256, image_token_id=3,
        video_token_id=4, vision_start_token_id=2, vision_end_token_id=5, pad_token_id=0, bos_token_id=1,
        eos_token_id=EOS, rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64, "out_hidden_size": 32,
                       "num_heads": 4, "patch_size": 14, "temporal_patch_size": 2, "spatial_merge_size": 2,
                       "window_size": 28, "fullatt_block_indexes": [0]})
    config._attn_implementation = "eager"
    config.livr_latent_ids = [10, 11, 12, 13]
    config.livr_stage = 2
    model = LivrQwen(config)
    freeze_vision(model)
    model.generation_config.eos_token_id = EOS
    model.generation_config.pad_token_id = 0
    return model


def prompt(seed=0):
    generator = torch.Generator().manual_seed(seed)
    ids = torch.tensor([[1, 2, 3, 5, 7, 10, 11, 12, 13, 8, 9]])
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "pixel_values": torch.randn(4, 3 * 2 * 14 * 14, generator=generator), "image_grid_thw": torch.tensor([[1, 2, 2]])}


def labelled(desc, i):
    return {"map_id": i, "map_desc": desc, "text_output": answer_text(label_route(desc)),
            "image_input": f"img/{i}.png", "text_input": "<image>"}


# Units ---------------------------------------------------------------------------------------

def test_families_are_norm_matched():
    generator = torch.Generator().manual_seed(0)
    directions = torch.linalg.qr(torch.randn(4, 32, 32, generator=generator)).Q[:, :, :3].contiguous()
    pca = {"directions": directions, "scales": torch.rand(4, 3, generator=generator) + 0.2}
    energy = diagnostics.slot_energy(pca)
    other = diagnostics.matched({"directions": directions, "scales": torch.rand(4, 3, generator=generator)}, energy)
    torch.testing.assert_close(other["weights"].double().square().sum(-1), energy, rtol=1e-5, atol=1e-6)
    iso = diagnostics.isotropic(energy, 32)
    torch.testing.assert_close((iso["std"].double().square() * 32), energy, rtol=1e-5, atol=1e-6)
    joint_dirs = torch.linalg.qr(torch.randn(128, 5, generator=generator)).Q.T.reshape(5, 4, 32)
    fam = diagnostics.joint(joint_dirs, float(energy.sum()))
    assert math.isclose(fam["weight"] ** 2 * 5, float(energy.sum()), rel_tol=1e-6)
    # Empirically: the mean squared size of draws matches c^2 times the energy for every family.
    for family in (diagnostics.matched(pca, energy), other, iso, fam):
        offsets, _ = diagnostics.draw(family, 2.0, 4000, generator, "cpu")
        assert math.isclose(float(offsets.double().square().sum((1, 2)).mean()), 4 * float(energy.sum()), rel_tol=0.06)


def test_ridge_leave_one_out_matches_refits():
    generator = torch.Generator().manual_seed(1)
    X = torch.randn(30, 10, generator=generator, dtype=torch.float64)
    Y = X @ torch.randn(10, 2, generator=generator, dtype=torch.float64) + 0.1 * torch.randn(30, 2, generator=generator, dtype=torch.float64)
    X, Y = X - X.mean(0), Y - Y.mean(0)
    weights, lam = diagnostics.ridge(X, Y, [0.01])
    scale = torch.linalg.eigvalsh(X @ X.T).clamp_min(0).mean()
    direct = torch.linalg.solve(X.T @ X + lam * scale * torch.eye(10, dtype=torch.float64), X.T @ Y)
    torch.testing.assert_close(weights, direct, rtol=1e-6, atol=1e-8)


def test_map_subspace_removes_planted_map_content():
    rng = random.Random(0)
    detours, _ = draw_detour_layouts(60, rng, set(), [4], [1])
    rows = [labelled(d, i) for i, d in enumerate(detours)]
    generator = torch.Generator().manual_seed(2)
    # Map content in a 12-dimensional subspace of 40 dimensions; the rest is unrelated variation.
    mixing = torch.randn(16 * 3, 12, generator=generator, dtype=torch.float64)
    targets = torch.tensor([sum((diagnostics.map_targets(r)[k] for k in ("goal", "start", "holes")), []) for r in rows],
                           dtype=torch.float64)
    rotation = torch.linalg.qr(torch.randn(40, 40, generator=generator, dtype=torch.float64)).Q
    X = torch.cat([targets @ mixing, 0.3 * torch.randn(60, 28, generator=generator, dtype=torch.float64)], 1) @ rotation
    settings = {"probe": {"ridge": [0.01, 1.0], "inlp_iterations": 3, "inlp_dims_per_iteration": 8, "max_map_dims": 20}}
    sizes = [4] * 60
    M, trace = diagnostics.map_subspace(X, rows, sizes, list(range(48)), list(range(48, 60)), settings)
    torch.testing.assert_close(M.T @ M, torch.eye(M.shape[1], dtype=torch.float64), atol=1e-8, rtol=0)
    content = torch.linalg.qr(rotation[:12].T).Q  # the planted map-content directions
    assert float((M.T @ content).square().sum()) / 12 > 0.8
    assert trace[-1]["scores"]["4"]["goal"] <= trace[0]["scores"]["4"]["goal"]
    states = X[:, None].float().repeat(1, 4, 1)
    families, info, _ = diagnostics.layer_bases(states, sizes, rows, list(range(48)), list(range(48, 60)),
                                                {"basis_k": 3} | settings)
    perp = families["pca_perp_map"]["directions"][0].double()
    subspace = diagnostics.map_subspace(X, rows, sizes, list(range(48)), list(range(48, 60)), settings)[0]
    assert float((subspace.T @ perp).abs().max()) < 1e-4  # the perpendicular basis avoids the map subspace
    assert info["pca_energy_inside_map_subspace"][0] > 0.5  # the planted content dominates the PCA basis


def text_gradient(model, adjuster_context, prompt_, completion):
    model.zero_grad(set_to_none=True)
    with adjuster_context:
        token_logprobs(model, prompt_, completion, 1.0).sum().backward()
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def test_offset_replay_at_zero_is_b_prime_and_absolute_replay_cuts():
    model = tiny()
    adjuster = LatentAdjuster(model, 2)
    p, completion = prompt(), torch.tensor([20, 21, EOS])
    with torch.no_grad(), adjuster.capture() as state:
        model(**p, use_cache=False)
    prior = state["priors"][0].float()
    plain = text_gradient(model, torch.enable_grad(), p, completion)
    offset = text_gradient(model, adjuster.offset(torch.zeros_like(prior)), p, completion)
    absolute = text_gradient(model, adjuster.replay(prior), p, completion)
    for name in plain:
        torch.testing.assert_close(offset[name], plain[name], atol=1e-6, rtol=1e-5)
    cut = sum(float((absolute.get(n, torch.zeros_like(g)) - g).square().sum()) for n, g in plain.items())
    assert cut > 1e-12  # replaying the detached state drops the path through the latents' later layers


def test_group_norms_add_up():
    model = tiny()
    text_gradient(model, torch.enable_grad(), prompt(), torch.tensor([20, 21, EOS]))
    norms = diagnostics.group_norms(model)
    assert math.isclose(math.sqrt(sum(v ** 2 for v in norms["groups"].values())), norms["total"], rel_tol=1e-9)
    assert "layers 1-8" in norms["groups"] and "visual" not in norms["groups"]


def test_answer_noise_touches_only_the_last_position():
    model = tiny().eval()
    p = prompt()
    basis = {"directions": torch.eye(32)[None, :, :3], "scales": torch.ones(1, 3)}
    noise = diagnostics.AnswerNoise(model, 1)
    with torch.no_grad():
        clean = model(**p, use_cache=False).logits
        with noise.active(basis, 5.0, torch.Generator().manual_seed(0)):
            noisy = model(**p, use_cache=False).logits
    torch.testing.assert_close(noisy[:, :-1], clean[:, :-1])
    assert not torch.allclose(noisy[:, -1], clean[:, -1])


# End to end ----------------------------------------------------------------------------------

@pytest.fixture
def world(monkeypatch, tmp_path):
    rng = random.Random(0)
    detours, _ = draw_detour_layouts(56, rng, set(), [4, 5], [1, 1])
    sets = {"fresh/pool": [labelled(d, i) for i, d in enumerate(detours[:44])],
            "fresh/h": [labelled(d, i) for i, d in enumerate(detours[44:48])],
            "fresh/h78": [labelled(d, i) for i, d in enumerate(detours[48:50])],
            "test": [{k: v for k, v in labelled(d, i).items() if k != "text_output"}  # released rows: no label
                     for i, d in enumerate(detours[50:54])]}
    for module in (fair_rl, diagnostics):
        monkeypatch.setattr(module, "records", lambda root, split: sets[split])

    def encode(processor, root, split, row, training=False):
        return prompt(row["map_id"])
    for module in (diagnostics, expert_iteration):
        monkeypatch.setattr(module, "encode", encode)
    monkeypatch.setattr(diagnostics, "GREEDY", diagnostics.GREEDY | {"suppress_tokens": SUPPRESS})
    b = json.loads(open("configs/text_rl_fair.json").read())
    c = json.loads(open("configs/latent_rl_fair.json").read())
    for rl in (b, c):
        rl["train_split"] = "fresh/pool"
        rl["generation"] = rl["generation"] | {"max_new_tokens": 4, "suppress_tokens": SUPPRESS}
    c["latent"] |= {"layer": 2, "basis_k": 3}
    monkeypatch.setattr(diagnostics, "rl_settings", lambda settings, arm: copy.deepcopy(b if arm == "b" else c))
    settings = diagnostics.load_settings("configs/latent_diagnostics.json", smoke=True)
    settings |= {"layers": [1, 2], "basis_k": 3, "max_new_tokens": 4, "decision_max_dims": 3}
    settings["fair"] |= {"pool_split": "fresh/pool", "round1_heldout_split": "fresh/h", "heldout_split": "fresh/h78",
                         "pilot_maps": 3}
    settings["answer_noise"] |= {"layers": [1, 2]}
    settings["probe"] |= {"inlp_iterations": 2, "inlp_dims_per_iteration": 4, "max_map_dims": 8}
    settings["latent_search"] |= {"layers": [1, 2], "scales": [1, 2], "batch": 2}
    settings["checkpoints"] |= {"noise_layer": 2, "swap_every": 1}
    settings["head_fits"] |= {"layer": 2, "scales": [1], "learning_rates": [0.01]}
    settings["smoke"] |= {"capture_maps": 44, "checkpoint_maps": 6}
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    (pilot / "summary.json").write_text(json.dumps({"proceed": True, "checks": {"valid": True},
                                                    "rules": {"prompts_per_step": 2, "scale": 1.0}}))
    generator = torch.Generator().manual_seed(0)
    torch.save({"directions": torch.linalg.qr(torch.randn(4, 32, 32, generator=generator)).Q[:, :, :3].contiguous(),
                "scales": torch.rand(4, 3, generator=generator) + 0.5, "captured": [0.9] * 4, "k": 3, "maps": 10},
               pilot / "basis.pt")
    return settings, pilot


def test_every_part_end_to_end(world, tmp_path):
    settings, pilot = world
    config = json.loads(open("configs/livr.json").read()) | {"gradient_checkpointing": False}
    sweep = tmp_path / "sweep"
    diagnostics.run_sweep(tiny(), Processor(), Evaluator(), "unused", sweep, settings, [1, 2], True, True, {})
    maps = [json.loads(line) for line in (sweep / "maps.jsonl").read_text().splitlines()]
    assert len(maps) == 3 and len(maps[0]["text"]["successes"]) == settings["text_rollouts"]
    expected = {f"{f}/{l}/1" for f in diagnostics.FAMILIES for l in (1, 2)} | {"answer/1/1", "answer/2/1"}
    assert set(maps[0]["configs"]) == expected
    assert all(len(c["successes"]) == settings["rollouts"] for c in maps[0]["configs"].values())
    assert set(maps[0]["soft"]["2"]) == {"none"} | {f"{f}/1" for f in diagnostics.FAMILIES}
    bases = json.loads((sweep / "bases.json").read_text())
    assert set(bases) >= {"1", "2"} and len(bases["2"]["pca_energy_inside_map_subspace"]) == 4

    model, reference = tiny(), tiny().to(torch.bfloat16)
    grads = tmp_path / "grads"
    rows = diagnostics.run_gradients(model, reference, Processor(), Evaluator(), config, "unused", grads, settings,
                                     pilot, {})
    assert len(rows) == 1 and (grads / "head-tuples.pt").exists()
    cell = rows[0]
    if "b_text_zero_offset_vs_none" in cell:  # the offset replay at zero is B′'s update
        assert all(v["relative_difference"] < 1e-3 for v in cell["b_text_zero_offset_vs_none"].values()
                   if v["stored_norm"] > 1e-12)
    if "c_training_gradient" in cell:
        assert cell["c_training_gradient"]["total"] > 0 and "latent_heads" in cell["c_latent_detached"]["groups"]
        assert set(cell["c_latent_detached"]["groups"]) == {"latent_heads"}  # detached: only the heads learn
    search = diagnostics.run_search(model, Processor(), Evaluator(), "unused", grads, settings, pilot)
    assert search["pilot_maps"] == 3

    base = tiny()
    runs = {"livr": "livr", "b42": "b42"}
    checkpoints = tmp_path / "checkpoints"
    diagnostics.run_checkpoints(lambda path: (copy.deepcopy(base), Processor()), runs, Evaluator(), "unused",
                                checkpoints, settings, pilot)
    assert (checkpoints / "b42.jsonl").exists() and (checkpoints / "b42-swap.jsonl").exists()

    for name in runs:
        changed = copy.deepcopy(base)
        if name != "livr":
            with torch.no_grad():
                changed.model.layers[0].mlp.up_proj.weight.add_(1.0)
        changed.save_pretrained(tmp_path / name)
    weights = diagnostics.run_weights({n: tmp_path / n for n in runs}, tmp_path / "weights")
    assert weights["difference_norm"]["b42"]["layers 1-8"] > 0
    assert weights["difference_norm"]["b42"].get("layers 25-36", 0) == 0

    summary = diagnostics.analyse([sweep], grads, checkpoints, tmp_path / "weights", tmp_path / "analysis", settings,
                                  list(runs))
    assert summary["text"]["maps"] == 3 and summary["null"] is not None
    assert set(summary["configs"]) == expected and summary["head_fits"]
    assert (tmp_path / "analysis" / "report.md").exists()


def test_unbiased_mixed_share_and_pass_at():
    assert diagnostics.mixed_unbiased(0, 64, 8) == 0 and diagnostics.mixed_unbiased(64, 64, 8) == 0
    assert math.isclose(diagnostics.mixed_unbiased(1, 64, 8), 8 / 64)
    assert math.isclose(diagnostics.pass_at(1, 64, 8), 8 / 64)


def test_text_baseline_alone(world, tmp_path):
    settings, _ = world
    diagnostics.run_sweep(tiny(), Processor(), Evaluator(), "unused", tmp_path / "text", settings, [], True, False, {})
    maps = [json.loads(line) for line in (tmp_path / "text" / "maps.jsonl").read_text().splitlines()]
    assert len(maps) == 3 and all(len(m["text"]["successes"]) == settings["text_rollouts"] and not m["configs"]
                                  for m in maps)
    assert not (tmp_path / "text" / "states.pt").exists()


def test_noise_size_check_is_one_on_average():
    generator = torch.Generator().manual_seed(0)
    basis = {"directions": torch.linalg.qr(torch.randn(4, 32, 32, generator=generator)).Q[:, :, :3],
             "scales": torch.rand(4, 3, generator=generator) + 0.5}
    family = diagnostics.matched(basis, diagnostics.slot_energy(basis))
    for scale in (0.1, 0.25, 4):
        offset, _ = diagnostics.draw(family, scale, 20000, generator, "cpu")
        measured = offset.double().square().sum(-1).mean(0)
        expected = torch.tensor(diagnostics.expected_energy(family, scale))
        assert torch.allclose(measured / expected, torch.ones(4, dtype=torch.float64), atol=0.03)
    iso = diagnostics.isotropic(diagnostics.slot_energy(basis), 32)
    offset, _ = diagnostics.draw(iso, 0.25, 20000, generator, "cpu")
    ratio = offset.double().square().sum(-1).mean(0) / torch.tensor(diagnostics.expected_energy(iso, 0.25))
    assert torch.allclose(ratio, torch.ones(4, dtype=torch.float64), atol=0.03)


def test_confirmation_sweep_and_reading_end_to_end(world, tmp_path):
    settings, _ = world
    first = tmp_path / "first"
    diagnostics.run_sweep(tiny(), Processor(), Evaluator(), "unused", first, settings, [2], False, False, {})
    text = tmp_path / "text"
    diagnostics.run_sweep(tiny(), Processor(), Evaluator(), "unused", text, settings, [], True, False, {})
    confirm_settings = copy.deepcopy(settings) | {"families": ["pca", "isotropic"], "scales": [1], "repeats": 2,
                                                  "seed": 43, "map_range": [1, 3],
                                                  "confirm": {"config": "pca/2/1", "secondary": ["isotropic/2/1"]}}
    run = tmp_path / "confirm"
    diagnostics.run_sweep(tiny(), Processor(), Evaluator(), "unused", run, confirm_settings, [2], False, False, {},
                          bases_from=first)
    maps = [json.loads(line) for line in (run / "maps.jsonl").read_text().splitlines()]
    pilot = [json.loads(line)["row_index"] for line in (text / "maps.jsonl").read_text().splitlines()]
    assert [m["row_index"] for m in maps] == pilot[1:3]  # the range, in the pilot's order
    assert set(maps[0]["configs"]) == {"pca/2/1", "pca/2/1#2", "isotropic/2/1", "isotropic/2/1#2"}
    assert all(len(c["successes"]) == settings["rollouts"] for c in maps[0]["configs"].values())
    assert len(maps[0]["configs"]["pca/2/1"]["energy_ratio"]) == 4 and "energy_ratio" not in maps[0]["configs"]["pca/2/1#2"]
    assert maps[0]["configs"]["pca/2/1"]["eps"] != maps[0]["configs"]["pca/2/1#2"]["eps"]  # independent groups
    saved = json.loads((run / "bases.json").read_text())
    digests = {name: diagnostics.tensor_digest(f) for name, f in torch.load(first / "bases.pt")["families"][2].items()}
    assert saved["sha256"]["2"] == digests and not (run / "states.pt").exists()

    summary = diagnostics.confirm(run, text, confirm_settings, tmp_path / "reading", reference=first)
    assert summary["checks"]["complete"] and summary["checks"]["greedy_route_mismatches"] == 0
    assert summary["primary"]["groups_per_map"] == 2 and summary["primary"]["maps"] == 2
    assert summary["secondary"]["isotropic/2/1"]["maps"] == 2
    assert summary["pooled_first_group"]["reference_maps"] == 3
    assert (tmp_path / "reading" / "confirm.md").exists()
    # The full analysis ignores the repeats' keys.
    assert "pca/2/1#2" not in diagnostics.analyse([run, text], None, None, None, tmp_path / "a", confirm_settings)["configs"]


def test_confirmation_reading_on_planted_groups(tmp_path):
    def write(path, rows):
        path.mkdir()
        (path / "maps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    base = {"label": "R", "greedy_route": "R", "greedy_success": False, "soft": {}, "decision": {}}
    ones, zeros = [True] * 8, [False] * 8
    mixed = [True] + [False] * 7
    text = [base | {"row_index": i, "configs": {}, "text": {"successes": s}}
            for i, s in enumerate([[False] * 64, [True] * 64, [True] * 32 + [False] * 32, [False] * 64])]
    write(tmp_path / "text", text)
    run = [base | {"row_index": 1, "configs": {"pca/24/0.25": {"successes": zeros}, "pca/24/0.25#2": {"successes": mixed}}},
           base | {"row_index": 2, "configs": {"pca/24/0.25": {"successes": mixed}, "pca/24/0.25#2": {"successes": mixed}}},
           base | {"row_index": 3, "configs": {"pca/24/0.25": {"successes": mixed}, "pca/24/0.25#2": {"successes": zeros}}}]
    write(tmp_path / "run", run)
    settings = {"rule": {"mixed_gain": 0.02, "max_success_drop": 0.02, "bootstrap": 200, "break_drop": 0.5},
                "rollouts": 8, "map_range": [1, 4], "confirm": {"config": "pca/24/0.25"}}
    summary = diagnostics.confirm(tmp_path / "run", tmp_path / "text", settings, tmp_path / "out")
    c = summary["primary"]
    # Per map, the mean over both groups of (mixed - text's unbiased mixed share for 8):
    # row 1 (text always): (0 + 1)/2; row 2 (text half): 1 - m; row 3 (text never): (1 + 0)/2.
    m = diagnostics.mixed_unbiased(32, 64, 8)
    assert math.isclose(c["mixed_gain"], (0.5 + (1 - m) + 0.5) / 3)
    # First group only: row 1 is a break (1.0 -> 0.0), row 3 a rescue; row 2 neither.
    assert (c["rescues"], c["breaks"]) == (1, 1)
    assert math.isclose(c["success"], (1 / 16 + 2 / 16 + 1 / 16) / 3)
    assert math.isclose(c["success_drop"], (1 + 0.5 + 0) / 3 - c["success"])
    assert summary["checks"]["complete"] and not summary["confirmed"]  # the success drop fails the rule
