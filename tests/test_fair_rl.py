"""B′ and C′ (docs/STRUCTURED_EXPLORATION_SPEC.md, 2026-11-17 amendment) on the tiny stand-in model."""
import copy
import json
import random

import pytest
import torch

from spara import detour_scaling, explore_probe, fair_rl, text_rl
from spara.fair_rl import (compare, pilot_rules, prompts_per_step, recipe, run_pilot, sampler_check, sign_test,
                           train_arm, evaluate_arm)
from spara.fresh_maps import answer_text, draw_detour_layouts, is_direct, label_route
from spara.latent_rl import BasisHeads, LatentAdjuster, add_latent_heads, head_input, latent_objective
from spara.runtime import read_config, require_final_rl_checkpoint
from spara.text_rl import stream_batch

SUPPRESS = [2, 3, 4, 5, 10, 11, 12, 13]  # image markers and latent tokens
LATENT_C = {"layer": 2, "basis_k": 3, "sigma_init": 0.5, "log_sigma_bounds": [-9.21, 0.6931],
            "head_hidden_size": None, "head_learning_rate": 0.01, "logprob_reduction": "sum"}


def prompt_of(batch):
    prompt = {k: v for k, v in batch.items() if k != "labels"}
    prompt["input_ids"] = prompt["input_ids"][:, :11]
    prompt["attention_mask"] = prompt["attention_mask"][:, :11]
    return prompt


def stage2(model):
    model.config.livr_stage = 2
    return model


def basis_for(slots=4, size=32, k=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    directions = torch.linalg.qr(torch.randn(slots, size, size, generator=generator)).Q[:, :, :k]
    return {"directions": directions.contiguous(), "scales": torch.rand(slots, k, generator=generator) + 0.5,
            "captured": [0.9] * slots, "k": k, "maps": 10}


class Tokenizer:
    def decode(self, ids, skip_special_tokens):
        return " ".join(map(str, ids.tolist()))


class Processor:
    tokenizer = Tokenizer()

    def save_pretrained(self, path):
        pass


class Evaluator:
    """Labels succeed; other answers succeed when their last token is even. Deterministic."""
    def score(self, row, text):
        if text == row.get("text_output"):
            return {"extracted_answer": "R", "success": True}
        tokens = text.split()
        moves = "".join("UDLR"[int(t) % 4] for t in tokens[-3:])
        return {"extracted_answer": moves, "success": bool(tokens) and int(tokens[-1]) % 2 == 0}


def test_stream_batches_are_full_and_run_across_epochs():
    first, position = stream_batch(5, 3, 7, 0)
    second, position = stream_batch(5, 3, 7, position)
    epoch0 = torch.randperm(5, generator=torch.Generator().manual_seed(7)).tolist()
    epoch1 = torch.randperm(5, generator=torch.Generator().manual_seed(8)).tolist()
    assert first == epoch0[:3] and second == epoch0[3:] + epoch1[:1] and position == 6


def rl_settings(path, **extra):
    rl = json.loads(open(path).read())
    rl |= {"steps": 3, "prompts_per_step": 2, "group_size": 4, "learning_rate": 0.01,
           "generation": rl["generation"] | {"max_new_tokens": 3, "suppress_tokens": SUPPRESS}}
    return rl | extra


def run_loop(model, tiny_batch, tmp_path, monkeypatch, name, scorer, rl):
    monkeypatch.setattr(text_rl, "records", lambda root, split: list(range(5)))
    monkeypatch.setattr(text_rl, "encode", lambda *args, training=True: prompt_of(tiny_batch))
    config = read_config("configs/livr.json") | {"gradient_checkpointing": False}
    reference = copy.deepcopy(model).requires_grad_(False)
    torch.manual_seed(0)
    final = text_rl.train_text_rl(model, reference, Processor(), "unused", tmp_path / name, config, rl,
                                  "livr/final", scorer=scorer)
    steps = [json.loads(line) for line in (tmp_path / name / "steps.jsonl").read_text().splitlines()]
    rollouts = [json.loads(line) for line in (tmp_path / name / "rollouts.jsonl").read_text().splitlines()]
    return final, steps, rollouts, config


def test_the_loop_streams_full_batches_and_trains_only_on_mixed_groups(tiny_model, tiny_batch, tmp_path,
                                                                       monkeypatch):
    model = stage2(tiny_model)
    rl = rl_settings("configs/text_rl_fair.json")
    scorer = lambda row, text: float(row == 0 or int(text.split()[-1]) % 2 == 0)  # map 0 always succeeds
    final, steps, rollouts, config = run_loop(model, tiny_batch, tmp_path, monkeypatch, "run", scorer, rl)
    assert [s["step"] for s in steps] == [1, 2, 3] and all(s["groups"] == 2 for s in steps)
    expected = stream_batch(5, 2, 42, 0)[0] + stream_batch(5, 2, 42, 2)[0] + stream_batch(5, 4, 42, 4)[0][:2]
    assert [r["row"] for r in rollouts] == expected  # the third step crosses into the second epoch
    assert all(len(r["completions"]) == 4 for r in rollouts)
    for s, (a, b) in zip(steps, zip(rollouts[::2], rollouts[1::2])):
        mixed = sum(0 < sum(r["rewards"]) < 4 for r in (a, b))
        assert s["trained_groups"] == mixed == s["groups"] - s["zero_variance_groups"]
        assert s["updated"] == bool(mixed) and (s["loss"] is None) == (not mixed)
    assert [s["verifier_calls"] for s in steps] == [8, 16, 24]
    progress = require_final_rl_checkpoint(final, config, rl)
    assert progress["optimizer_steps"] == 3 and progress["verifier_calls"] == 24
    with pytest.raises(ValueError, match="completed text-only RL"):
        require_final_rl_checkpoint(final, config, rl | {"steps": 4})


def test_a_step_without_mixed_groups_changes_nothing(tiny_model, tiny_batch, tmp_path, monkeypatch):
    model = stage2(tiny_model)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    _, steps, _, _ = run_loop(model, tiny_batch, tmp_path, monkeypatch, "uniform", lambda row, text: 1.0,
                              rl_settings("configs/text_rl_fair.json"))
    assert not any(s["updated"] for s in steps) and all(s["grad_norm"] == 0.0 for s in steps)
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


def test_sampler_check_catches_inherited_processors(tiny_model, tiny_batch):
    model = stage2(tiny_model).eval()
    model.generation_config.repetition_penalty = 1.05  # as the real checkpoint's generation config
    model.generation_config.top_k = 50
    prompt = prompt_of(tiny_batch)
    fair = json.loads(open("configs/text_rl_fair.json").read())["generation"]
    old = json.loads(open("configs/text_rl.json").read())["generation"]
    torch.manual_seed(0)
    assert sampler_check(model, prompt, fair | {"max_new_tokens": 4, "suppress_tokens": SUPPRESS}, 3) < 1e-5
    assert sampler_check(model, prompt, old | {"max_new_tokens": 4, "suppress_tokens": SUPPRESS}, 3) > 1e-3


def test_basis_heads_start_at_livr_and_sample_in_the_basis(tiny_model, tiny_batch):
    model = stage2(tiny_model).eval()
    with pytest.raises(ValueError, match="basis"):
        add_latent_heads(model, LATENT_C)
    heads = add_latent_heads(model, LATENT_C, basis_for())
    assert isinstance(heads, BasisHeads) and model.config.livr_latent_heads["basis_k"] == 3
    adjuster = LatentAdjuster(model, 2)
    prompt = prompt_of(tiny_batch)
    with torch.no_grad():
        plain = model(**tiny_batch).logits
        with adjuster.mean() as state:
            mean = model(**tiny_batch).logits
        with adjuster.capture() as captured:
            model(**prompt)
        with adjuster.sample() as sampled:
            model(**text_rl.repeat_prompt(prompt, 5))
    torch.testing.assert_close(mean, plain, atol=0, rtol=0)  # mu = 0: C′ starts where B′ starts
    assert torch.equal(state["records"][0]["mean"], torch.zeros(1, 4, 3))
    torch.testing.assert_close(state["records"][0]["sigma"], torch.full((1, 4, 3), 0.5))
    delta = sampled["records"][0]["actions"] - captured["priors"][0].float()
    coefficients = heads.coefficients(delta)
    torch.testing.assert_close(heads.offset(coefficients), delta, atol=1e-5, rtol=1e-4)  # in the basis
    assert 0.2 < coefficients.std() < 1.0 and not torch.allclose(coefficients[0], coefficients[1])


def test_basis_actions_replay_to_the_cached_logits_and_survive_save(tiny_model, tiny_batch, tmp_path):
    model = stage2(tiny_model).eval()
    heads = add_latent_heads(model, LATENT_C, basis_for())
    adjuster = LatentAdjuster(model)
    prompt = prompt_of(tiny_batch)
    torch.manual_seed(3)
    with torch.no_grad(), adjuster.sample() as state:
        generated = model.generate(**text_rl.repeat_prompt(prompt, 3), do_sample=True, max_new_tokens=4,
                                   suppress_tokens=SUPPRESS, use_cache=True, output_logits=True,
                                   return_dict_in_generate=True)
    actions, logits = state["records"][0]["actions"], torch.stack(generated.logits, dim=1)
    for row in range(3):
        completion = generated.sequences[row, 11:]
        with torch.no_grad(), adjuster.replay(actions[row:row + 1]):
            uncached = text_rl.token_logprobs(model, prompt, completion, 1.0)
        cached = logits[row].log_softmax(-1).gather(-1, completion[:, None]).squeeze(-1)
        torch.testing.assert_close(uncached, cached, atol=1e-5, rtol=1e-5)
    from spara.model import LivrQwen
    with torch.no_grad():
        heads.mean_out.normal_()
    model.save_pretrained(tmp_path)
    loaded = LivrQwen.from_pretrained(tmp_path, attn_implementation="eager")
    assert isinstance(loaded.latent_heads, BasisHeads)
    for (name, a), (_, b) in zip(heads.state_dict().items(), loaded.latent_heads.state_dict().items()):
        torch.testing.assert_close(a, b, atol=0, rtol=0, msg=name)


def test_basis_objective_moves_the_mean_toward_rewarded_coefficients(tiny_model):
    model = stage2(tiny_model)
    heads = add_latent_heads(model, LATENT_C, basis_for())
    prior = torch.randn(1, 4, 32)
    action = prior + heads.offset(torch.full((1, 4, 3), 0.3))
    optimizer = torch.optim.SGD(heads.parameters(), lr=0.05)
    for _ in range(5):
        optimizer.zero_grad()
        loss, stats = latent_objective(model, prior, action, 1.0, LATENT_C, 0.2)
        loss.backward()
        optimizer.step()
    mean, _ = heads(head_input(prior))
    assert mean.mean() > 0.01 and stats["sigma_mean"] > 0
    assert heads.directions.grad is None  # the basis is fixed


def test_pilot_rules():
    settings = json.loads(open("configs/fair_rl.json").read())
    assert prompts_per_step(0.125, settings) == 32 and prompts_per_step(0.1, settings) == 40
    assert prompts_per_step(0.06, settings) == 48 and prompts_per_step(0.04, settings) is None
    sets, noise = {"valid": True}, {"families": {"structured": {"chosen": 4.0}}}
    mixed = [{"successes": [True, False], "truncated": [False, False]}] * 2
    uniform = [{"successes": [True, True], "truncated": [False, True]}] * 2
    assert pilot_rules(sets, 0.0, mixed, noise, settings)["stop"] == []
    stops = pilot_rules({"valid": False}, 0.1, uniform, {"families": {"structured": {"chosen": None}}}, settings)["stop"]
    assert len(stops) == 4  # validity, no signal, the token cap, no scale


def test_recipe_takes_the_pilots_settings():
    pilot = {"proceed": True, "checks": {"valid": True}, "rules": {"prompts_per_step": 32, "scale": 4.0}}
    b, c = (json.loads(open(p).read()) for p in ("configs/text_rl_fair.json", "configs/latent_rl_fair.json"))
    settings = json.loads(open("configs/fair_rl.json").read())
    assert recipe(b, "b", 43, pilot, settings)["prompts_per_step"] == 32
    assert recipe(b, "b", 43, pilot, settings)["seed"] == 43
    latent = recipe(c, "c", 42, pilot, settings)["latent"]
    assert latent["sigma_init"] == 4.0 and latent["log_sigma_bounds"][1] == pytest.approx(2.7726, abs=1e-4)
    assert c["latent"]["sigma_init"] is None  # the recipe file is not modified
    with pytest.raises(ValueError, match="B′ uses"):
        recipe(c, "b", 42, pilot, settings)
    with pytest.raises(ValueError, match="pilot stopped"):
        recipe(b, "b", 42, pilot | {"proceed": False}, settings)
    with pytest.raises(ValueError, match="pilot measured"):
        recipe(b, "b", 42, pilot, settings | {"pool_split": "fresh/other"})


def test_sign_test_sums_over_seeds():
    keys = list(range(4))
    first = [{0: 1, 1: 1, 2: 0, 3: 0}, {0: 1, 1: 0, 2: 0, 3: 1}]
    second = [{0: 0, 1: 1, 2: 0, 3: 0}, {0: 0, 1: 0, 2: 1, 3: 0}]
    result = sign_test(first, second, keys)
    assert result["first"] == [2, 2] and result["second"] == [1, 1]
    assert result["maps_first_ahead"] == 2 and result["maps_second_ahead"] == 1  # map 1 ties at 1-1


# End to end: pilot, both arms for two seeds, and the readings ----------------------------------

def labelled(desc, i):
    return {"map_id": i, "map_desc": desc, "text_output": answer_text(label_route(desc)),
            "image_input": f"img/level_{len(desc)}/{i}.png"}


@pytest.fixture
def world(tiny_batch, monkeypatch):
    rng = random.Random(0)
    detours, _ = draw_detour_layouts(12, rng, set(), [4, 5], [1, 1])
    direct, _ = draw_detour_layouts(1, rng, set(), [4], [1], keep=is_direct)
    sets = {"fresh/pool": [labelled(d, i) for i, d in enumerate(detours[:6])],
            "fresh/h": [labelled(d, i) for i, d in enumerate(detours[6:8])],
            "fresh/h78": [labelled(d, i) for i, d in enumerate(detours[8:10])],
            "train": [labelled(detours[10], 0)],
            "test": [labelled(detours[11], 0), labelled(direct[0], 1)]}
    for module in (fair_rl, text_rl, explore_probe):
        monkeypatch.setattr(module, "records", lambda root, split: sets[split])
    def prompt(*args, **kwargs):  # args[3] is the row: each map gets its own image, so latents differ
        encoded = prompt_of(tiny_batch)
        return encoded | {"pixel_values": encoded["pixel_values"] + 0.05 * args[3]["map_id"]}
    for module in (fair_rl, text_rl, explore_probe, detour_scaling):
        monkeypatch.setattr(module, "encode", prompt)

    def benchmark(model, processor, config, root, output, manifest):
        output.mkdir()
        (output / "responses.jsonl").write_text("".join(
            json.dumps({"row_index": i, "success": i == 0}) + "\n" for i in range(2)))
    monkeypatch.setattr(fair_rl, "run_benchmark", benchmark)
    settings = json.loads(open("configs/fair_rl.json").read()) | {
        "pool_split": "fresh/pool", "round1_heldout_split": "fresh/h", "heldout_split": "fresh/h78",
        "layer": 2, "basis_k": 3, "pilot_maps": 4, "sampler_rollouts": 2, "scales": [1, 2, 4, 8], "draws": 2,
        "min_mixed_share": 0.0, "max_truncated_share": 1.0, "identity_maps": 2, "moved_maps": 0, "max_new_tokens": 3}
    return sets, settings


def test_pilot_arms_and_readings_end_to_end(world, tiny_model, tmp_path):
    sets, settings = world
    config = read_config("configs/livr.json") | {"gradient_checkpointing": False, "seed": 0}
    base = stage2(tiny_model).eval()
    rl_b = rl_settings("configs/text_rl_fair.json", steps=2, train_split="fresh/pool")
    rl_c = rl_settings("configs/latent_rl_fair.json", steps=2, train_split="fresh/pool")
    rl_c["latent"] |= {"layer": 2, "basis_k": 3, "head_learning_rate": 0.01}

    pilot = run_pilot(copy.deepcopy(base), Processor(), Evaluator(), "unused", tmp_path / "pilot", settings, rl_b, {})
    assert pilot["checks"]["sets"]["valid"] and pilot["checks"]["sampler_max_tv"] < 1e-5
    assert pilot["pilot_rows"] == stream_batch(6, 4, 42, 0)[0]
    assert pilot["basis"]["k"] == 3 and (tmp_path / "pilot/basis.pt").exists()
    livr = [json.loads(line) for line in (tmp_path / "pilot/greedy-livr.jsonl").read_text().splitlines()]
    assert [r["population"] for r in livr].count("moved") == 4 and len(livr) == 2 + 1 + 2 + 4
    summary = json.loads((tmp_path / "pilot/summary.json").read_text())
    if not summary["proceed"]:  # the tiny model may give no qualifying scale; force the rules for the arms
        summary["rules"] |= {"prompts_per_step": 2, "scale": 0.5, "stop": []}
        summary["proceed"] = True
    summary["rules"]["prompts_per_step"] = 2
    (tmp_path / "pilot/summary.json").write_text(json.dumps(summary))
    torch.save(basis_for(), tmp_path / "pilot/basis.pt")

    runs = {"b": {}, "c": {}}
    for seed in (1, 2):
        for arm, rl in (("b", rl_b), ("c", rl_c)):
            model, reference = copy.deepcopy(base), copy.deepcopy(base)
            out = tmp_path / f"{arm}{seed}"
            adjuster = train_arm(model, reference, Processor(), Evaluator(), config, "unused", out, settings, rl, arm,
                                 seed, tmp_path / "pilot", "livr/final", {})
            evaluate_arm(model, Processor(), Evaluator(), config, "unused", out, settings, arm, adjuster, {})
            runs[arm][seed] = out
    arm_c = json.loads((tmp_path / "c1/arm-summary.json").read_text())
    assert arm_c["identity"]["valid"] and arm_c["rl"]["latent"]["sigma_init"] == 0.5
    assert set(arm_c["greedy"]) == {"mean", "zero", "sample"} and "plain" in json.loads(
        (tmp_path / "b1/arm-summary.json").read_text())["greedy"]
    livr_outcomes = tmp_path / "livr.jsonl"
    livr_outcomes.write_text("".join(json.dumps({"row_index": i, "success": True}) + "\n" for i in range(2)))

    result = compare(runs, tmp_path / "pilot", settings, livr_outcomes, "unused", evaluator=Evaluator())
    assert result["validity"]["valid"], result["validity"]
    assert result["validity"]["pairing"] == {1: True, 2: True}
    assert all(r["valid"] for r in result["validity"]["rescored"].values())
    assert result["primary"]["maps"] == 3 and result["policy_moved"]["moved"]
    assert result["primary"]["reading"] != "" and "transfer" in result["secondary"]
    assert set(result["diagnostics"]["rescues_and_breaks"]) == {1, 2}
    assert result["secondary"]["benchmark_vs_l"]["b1"]["detour"]["maps"] == 1

    # A run that trained on other maps breaks the pairing and withholds the readings.
    rollouts = (runs["c"][2] / "rl/rollouts.jsonl").read_text().splitlines()
    changed = [json.dumps(json.loads(line) | {"row": 99}) for line in rollouts[:1]] + rollouts[1:]
    (runs["c"][2] / "rl/rollouts.jsonl").write_text("\n".join(changed) + "\n")
    broken = compare(runs, tmp_path / "pilot", settings, livr_outcomes, "unused")
    assert not broken["validity"]["valid"] and broken["primary"]["reading"].startswith("withheld")
    # A policy that did not move from LIVR gives no reading on latent exploration.
    unmoved = compare(runs, tmp_path / "pilot", settings | {"moved_maps": 99}, livr_outcomes, "unused")
    assert unmoved["policy_moved"]["extend_to"] == 250
