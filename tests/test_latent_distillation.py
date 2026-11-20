"""CPU tests of latent distillation (docs/LATENT_DISTILLATION_SPEC.md) on a tiny model."""
import json
import random

import pytest
import torch

from spara import detour_scaling, expert_iteration, training
from spara import latent_distillation as ld
from spara.fresh_maps import answer_text, draw_detour_layouts
from spara.latent_rl import LatentAdjuster, tensor_digest
from spara.model import latent_queries
from spara.runtime import read_config
from test_latent_diagnostics import ALPHABET, EOS, SUPPRESS, Processor, labelled, prompt, tiny

LAYER = 2


class SavingProcessor(Processor):
    def save_pretrained(self, path):
        (path / "processor.json").write_text("{}")


class Evaluator:
    """Boxed answers (labels, a found route's canonical answer) are accepted; a generated answer is
    accepted when its text has an odd length (the tiny model rarely writes moves)."""

    def score(self, row, text):
        answer = text.split("<|im_start|>assistant")[-1]
        moves = "".join(c for c in answer if c in "UDLR")
        success = answer.startswith("\\boxed{") or len(answer) % 2 == 1
        return {"answer": answer, "extracted_answer": moves, "success": success, "raw_response": text}


def encoded(row, training=True):
    batch = prompt(row["map_id"])
    if not training:
        return batch
    answer = [14 + ALPHABET.index(c) for c in row["text_output"]] + [EOS]
    ids = torch.cat([batch["input_ids"], torch.tensor([answer])], -1)
    labels = ids.clone()
    labels[:, :batch["input_ids"].shape[1]] = -100
    return batch | {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


def family_of(seed=0):
    generator = torch.Generator().manual_seed(seed)
    directions = torch.linalg.qr(torch.randn(4, 32, 32, generator=generator)).Q[:, :, :3].contiguous()
    return {"kind": "slots", "directions": directions, "weights": torch.rand(4, 3, generator=generator) + 2.0}


@pytest.fixture
def world(monkeypatch, tmp_path):
    rng = random.Random(0)
    detours, _ = draw_detour_layouts(40, rng, set(), [4, 5], [1, 1])
    sets = {"fresh/pool": [labelled(d, i) for i, d in enumerate(detours[:12])],
            "fresh/h": [labelled(d, 100 + i) for i, d in enumerate(detours[12:16])],
            "fresh/h78": [labelled(d, 200 + i) for i, d in enumerate(detours[16:18])],
            "fresh/probe": [labelled(d, 300 + i) for i, d in enumerate(detours[18:20])],
            "train": [labelled(d, 400 + i) for i, d in enumerate(detours[20:24])],
            "test": [labelled(d, 500 + i) for i, d in enumerate(detours[24:27])]}
    records = lambda root, split: sets[split]
    for module in (ld, expert_iteration, detour_scaling, training):
        monkeypatch.setattr(module, "records", records)
    monkeypatch.setattr(ld, "encode", lambda processor, root, split, row, training=True, **kw: encoded(row, training))
    monkeypatch.setattr(training, "encode", lambda processor, root, split, row, **kw: encoded(row))
    monkeypatch.setattr(ld, "GREEDY", ld.GREEDY | {"suppress_tokens": SUPPRESS})

    def benchmark(model, processor, config, root, output, manifest):
        output.mkdir()
        (output / "responses.jsonl").write_text("".join(
            json.dumps({"row_index": i, "success": i % 2 == 0, "raw_response": "\\boxed{U}" if i % 2 == 0 else "UU"}) + "\n"
            for i in range(3)))
    monkeypatch.setattr(ld, "run_benchmark", benchmark)
    family = family_of()
    torch.save({"families": {LAYER: {"pca": family}}}, tmp_path / "bases.pt")
    settings = json.loads(open("configs/latent_distillation.json").read())
    settings |= {"pool_split": "fresh/pool", "heldout_split": "fresh/h", "transfer_split": "fresh/h78",
                 "other_fresh": ["fresh/probe", "fresh/h78"], "maps": 12, "smoke_maps": 4, "max_new_tokens": 6,
                 "extra_examples": 12, "seen_maps": 2}
    settings["noise"] = {"layer": LAYER, "family": "pca", "sha256": tensor_digest(family)}
    settings["search"] |= {"draws": [8, 4, 4], "smoke_draws": [4, 0, 4], "batch": 4}
    return settings, tmp_path / "bases.pt", family, sets


def config():
    return read_config("configs/livr.json") | {"gradient_checkpointing": False}


# Mechanics -----------------------------------------------------------------------------------

def test_hidden_states_index_is_the_adjusted_layer_and_a_push_lands_there():
    """out.hidden_states[L] is the output of decoder layer L (one-based), the state LatentAdjuster(L)
    reads and edits: a push delta at L makes hidden_states[L] at the latents z + delta."""
    model = tiny().eval()
    batch = prompt(1)
    positions = latent_queries(batch["input_ids"], model.config.livr_latent_ids)[0].nonzero().squeeze(-1)
    adjuster = LatentAdjuster(model, LAYER)
    with torch.no_grad():
        out = model(**batch, use_cache=False, output_hidden_states=True)
        z = ld.capture_state(model, adjuster, batch)
        assert torch.allclose(out.hidden_states[LAYER][0, positions], z, atol=1e-6)
        assert not torch.allclose(out.hidden_states[LAYER - 1][0, positions], z, atol=1e-3)
        delta = torch.randn(1, 4, 32)
        with adjuster.offset(delta):
            pushed = model(**batch, use_cache=False, output_hidden_states=True)
    assert torch.allclose(pushed.hidden_states[LAYER][0, positions], z + delta[0], atol=1e-5)
    assert not torch.allclose(pushed.logits, out.logits)


def test_latent_loss_is_zero_at_the_target_matches_the_formula_and_skips_other_rows(world):
    settings, _, family, sets = world
    model = tiny()
    row = sets["fresh/pool"][0]
    z = torch.randn(4, 32)
    delta = 2.0 * torch.einsum("sdk,sk,sk->sd", family["directions"], family["weights"], torch.randn(4, 3))
    targets = {row["map_id"]: {"z": z, "delta": delta, "row_index": 0}}
    data = [("train", sets["train"][0]), ("fresh/pool", row), ("fresh/pool", sets["fresh/pool"][1])]
    objective = ld.LatentTarget(data, "fresh/pool", targets, family, LAYER, model.config.livr_latent_ids, 0.5)
    batch = encoded(row)
    positions = latent_queries(batch["input_ids"], model.config.livr_latent_ids)[0].nonzero().squeeze(-1)
    states = [torch.randn(1, batch["input_ids"].shape[1], 32) for _ in range(4)]
    states[LAYER][0, positions] = z + delta
    loss, metrics = objective.loss(states, 1, batch)
    assert float(loss) < 1e-5 and abs(metrics["push_fraction"] - 1.0) < 1e-5
    states[LAYER][0, positions] = z  # no push learned: about the squared scale
    loss, metrics = objective.loss(states, 1, batch)
    energy = family["weights"].square().sum(-1)
    e = (torch.einsum("sdk,sd->sk", family["directions"], -delta).square().sum(-1) / energy).sqrt()
    manual = torch.where(e < 1, 0.5 * e.square(), e - 0.5).mean()
    assert torch.isclose(loss, manual) and abs(metrics["push_fraction"]) < 1e-5
    # Drift outside the push's subspace is not penalised.
    outside = torch.randn(4, 32)
    outside -= torch.einsum("sdk,sek,se->sd", family["directions"], family["directions"], outside)
    states[LAYER][0, positions] = z + delta + outside
    assert float(objective.loss(states, 1, batch)[0]) < 1e-4
    for index in (0, 2):  # a released row; a pool map without a target
        loss, metrics = objective.loss(states, index, encoded(data[index][1]))
        assert float(loss) == 0 and metrics == {"latent_loss": None, "push_fraction": None}
    assert objective.prepare(None, None) == {"rows_with_target": 1, "rows": 3}


def test_train_stage_with_the_offloaded_optimizer_matches_the_regular_one(world, tmp_path):
    _, _, _, sets = world
    data = [("train", r) for r in sets["train"]] + [("fresh/pool", r) for r in sets["fresh/pool"][:4]]
    recipe = config() | {"epochs_per_stage": 1, "gradient_accumulation": 4, "warmup_ratio": 0.0}
    plain, offloaded = tiny(), tiny()
    training.train_stage(plain, Processor(), "unused", tmp_path / "plain", recipe, 2, data=data, save_final=False)
    training.train_stage(offloaded, Processor(), "unused", tmp_path / "off", recipe, 2, data=data, save_final=False,
                         offload_optimizer=True)
    for (name, a), (_, b) in zip(plain.named_parameters(), offloaded.named_parameters()):
        assert torch.allclose(a, b, atol=1e-6), name
    assert any(not torch.equal(a, b) for a, b in zip(plain.parameters(), tiny().parameters()))


# End to end ----------------------------------------------------------------------------------

def test_search_records_the_push_it_applied_and_replay_reproduces(world, tmp_path):
    settings, basis, family, sets = world
    summary = ld.run_search(tiny(), Processor(), Evaluator(), "unused", tmp_path / "search", settings, basis, {})
    rows = [json.loads(l) for l in (tmp_path / "search/search.jsonl").read_text().splitlines()]
    assert len(rows) == 12 and summary["found"] == sum(r["solved"] for r in rows)
    pushed = [r for r in rows if r["solved"] and r["scale"] is not None]
    assert pushed, "the fixture should give at least one map found by a push"
    saved = torch.load(tmp_path / "search/targets.pt")
    model = tiny().eval()
    adjuster = LatentAdjuster(model, LAYER)
    for r in pushed:
        n = saved["row_index"].index(r["row_index"])
        expected = r["scale"] * torch.einsum("sdk,sk,sk->sd", family["directions"], family["weights"],
                                             torch.tensor(r["eps"]))
        assert torch.allclose(saved["delta"][n], expected, atol=1e-5)
        # z is LIVR's own state, and the push reproduces the route alone.
        assert torch.allclose(saved["z"][n], ld.capture_state(model, adjuster, prompt(r["map_id"])), atol=1e-5)
        assert r["reproduced"] and r["answer"] == answer_text(r["route"])
    unpushed = [r for r in rows if r["solved"] and r["scale"] is None]
    assert all(r["greedy_success"] for r in unpushed)
    assert all(float(saved["delta"][saved["row_index"].index(r["row_index"])].abs().max()) == 0 for r in unpushed)
    assert summary["checks"]["valid"] and summary["checks"]["reproduced_share"] == 1.0
    assert summary["basis_orthonormal_error"] < 1e-5
    settings["noise"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256"):
        ld.run_search(tiny(), Processor(), Evaluator(), "unused", tmp_path / "bad", settings, basis, {})


def test_both_arms_train_on_the_same_rows_and_compare_reads_them(world, tmp_path):
    settings, basis, family, sets = world
    ld.run_search(tiny(), Processor(), Evaluator(), "unused", tmp_path / "search", settings, basis, {})
    runs = {}
    for arm in ld.ARMS:
        for seed in settings["seeds"]:
            runs[f"{arm}{seed}"] = tmp_path / f"{arm}{seed}"
            ld.run_train(tiny(), SavingProcessor(), Evaluator(), config(), "unused", runs[f"{arm}{seed}"], settings, arm,
                         seed, tmp_path / "search", basis, {}, offload=True)
    summaries = {n: json.loads((p / "summary.json").read_text()) for n, p in runs.items()}
    assert len({s["rows_sha256"] for s in summaries.values()}) == 1
    assert all(s["training"]["optimizer_steps"] == s["expected_steps"] == 2 for s in summaries.values())  # 4 + 12 rows
    assert summaries["D42"]["latent"]["gradient_ratio"] == 0.5 and summaries["S42"]["latent"] is None
    steps = [json.loads(l) for l in (runs["D42"] / "stage2/steps.jsonl").read_text().splitlines()]
    assert any(s["aux_scale"] for s in steps)  # the per-step ratio was applied
    reference = summaries["D42"]["latent"]["training_reference"]
    assert reference["maps"] == summaries["D42"]["target_fit"]["before"]["pushed"]["maps"]  # pushed maps only
    assert reference["mean_mismatch_over_push_norm"] < 1e-3
    steps = [json.loads(l) for l in (runs["D42"] / "stage2/steps.jsonl").read_text().splitlines()]
    assert all("latent_loss_all" in s for s in steps) and any(s["latent_loss"] is not None for s in steps)
    before = summaries["D42"]["target_fit"]["before"]
    assert before["unpushed"]["maps"] == 0 or before["unpushed"]["loss"] < 1e-5
    if before["pushed"]["maps"]:
        assert abs(before["pushed"]["push_fraction"]) < 1e-4
    greedy = [json.loads(l) for l in (runs["S42"] / "greedy.jsonl").read_text().splitlines()]
    assert {g["population"] for g in greedy} == {"heldout", "test_detour", "heldout78", "seen"}
    livr = tmp_path / "livr.jsonl"
    livr.write_text("".join(json.dumps(g | {"success": False}) + "\n" for g in greedy if g["population"] != "seen"))
    bench = tmp_path / "livr-bench.jsonl"
    bench.write_text("".join(json.dumps({"row_index": i, "success": False}) + "\n" for i in range(3)))
    result = ld.compare({n: str(p) for n, p in runs.items()}, tmp_path / "search", livr, bench, "unused", Evaluator(),
                        settings)
    assert result["validity"]["valid"], result["validity"]
    assert result["validity"]["valid_if_replay_amendment_accepted"]
    assert result["primary"]["reading_if_replay_amendment_accepted"] == result["primary"]["reading"]
    assert result["primary"]["maps"] == 4 + sum(1 for r in sets["test"] if ld.is_detour(r["map_desc"]))
    assert result["primary"]["reading"] in ("the latent target helps", "the latent target hurts",
                                            "no detectable difference")


def test_verdict_needs_both_seeds_the_same_way():
    test = {"p": 0.01, "maps_first_ahead": 20, "maps_second_ahead": 5, "first": [60, 50], "second": [50, 51]}
    assert ld.verdict(test, True, "helps", "hurts", "none") == "none"
    assert ld.verdict(test | {"second": [50, 45]}, True, "helps", "hurts", "none") == "helps"
    assert ld.verdict(test | {"second": [50, 45]}, False, "helps", "hurts", "none").startswith("withheld")


def test_a_search_below_the_replay_bar_trains_only_when_accepted_and_is_flagged(world, tmp_path):
    settings, basis, family, sets = world
    ld.run_search(tiny(), Processor(), Evaluator(), "unused", tmp_path / "search", settings, basis, {})
    path = tmp_path / "search/summary.json"
    summary = json.loads(path.read_text())
    summary["checks"] |= {"valid": False, "reproduced_share": 0.9}
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="validity"):
        ld.run_train(tiny(), SavingProcessor(), Evaluator(), config(), "unused", tmp_path / "refused", settings, "S",
                     42, tmp_path / "search", basis, {})
    out = ld.run_train(tiny(), SavingProcessor(), Evaluator(), config(), "unused", tmp_path / "accepted", settings,
                       "S", 42, tmp_path / "search", basis, {}, accept_search=True)
    assert out["search_accepted_below_replay_bar"]


def test_huber_gives_small_errors_small_gradients_and_bounds_large_ones():
    energy = torch.tensor([4.0, 1.0, 9.0, 0.25])
    for size, bound in ((1e-3, 1e-3), (100.0, None)):
        error = (size * torch.randn(4, 3)).requires_grad_()
        ld.huber(error, energy).backward()
        per_slot = (error.grad.square().sum(-1) * energy).sqrt() * 4  # gradient in push units, per slot
        if bound:
            assert float(per_slot.max()) < 10 * bound
        else:
            assert torch.allclose(per_slot, torch.ones(4), atol=1e-4)


def test_normalised_step_sets_the_latent_gradient_to_the_ratio_of_the_route_gradient(world):
    settings, _, family, sets = world
    pool = sets["fresh/pool"]
    targets = {pool[0]["map_id"]: {"z": torch.randn(4, 32), "delta": torch.randn(4, 32), "row_index": 0}}
    data = [("train", sets["train"][0]), ("fresh/pool", pool[0]), ("fresh/pool", pool[1])]
    model = tiny()
    objective = ld.LatentTarget(data, "fresh/pool", targets, family, LAYER, model.config.livr_latent_ids, 0.5)
    trainable = [p for p in model.parameters() if p.requires_grad]
    extra = {k: [] for k in objective.metric_keys}

    def grads(term, rows):
        model.zero_grad(set_to_none=True)
        for i in rows:
            split, row = data[i]
            batch = encoded(row)
            out = model(**batch, use_cache=False, output_hidden_states=True)
            value = out.loss if term == "route" else objective.loss(out.hidden_states, i, batch)[0]
            (value / 3).backward()
        return [torch.zeros_like(p) if p.grad is None else p.grad.clone() for p in trainable]

    route, aux = grads("route", [0, 1, 2]), grads("aux", [1])
    norm = lambda gs: torch.sqrt(sum(g.square().sum() for g in gs))
    expected = [r + 0.5 * norm(route) / norm(aux) * a for r, a in zip(route, aux)]
    model.zero_grad(set_to_none=True)
    _, record = training.normalised_step(model, Processor(), "unused", data, [0, 1, 2], objective, trainable, 0.5, {},
                                         extra)
    assert record["aux_rows"] == 1 and extra["latent_loss_all"]
    got = lambda: [torch.zeros_like(p) if p.grad is None else p.grad for p in trainable]
    for g, e in zip(got(), expected):
        assert torch.allclose(g, e, atol=1e-6, rtol=1e-4)
    model.zero_grad(set_to_none=True)
    _, record = training.normalised_step(model, Processor(), "unused", data, [0, 2], objective, trainable, 0.5, {}, extra)
    assert record["aux_rows"] == 0 and record["aux_scale"] is None
    plain = grads("route", [0, 2])
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(got(), plain))
