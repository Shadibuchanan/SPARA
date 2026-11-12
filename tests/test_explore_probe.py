import json

import pytest
import torch

from spara import explore_probe
from spara.explore_probe import choose_noise, principal_basis, structured_offsets, summarise_sweep
from spara.latent_rl import LatentAdjuster

SUPPRESS = [2, 3, 4, 5, 10, 11, 12, 13]  # image markers and latent tokens


def prompt_of(batch):
    prompt = {k: v for k, v in batch.items() if k != "labels"}
    prompt["input_ids"] = prompt["input_ids"][:, :11]
    prompt["attention_mask"] = prompt["attention_mask"][:, :11]
    return prompt


def test_capture_records_the_layer_output_and_changes_nothing(tiny_model, tiny_batch):
    model = tiny_model.eval()
    adjuster = LatentAdjuster(model, 2)
    with torch.no_grad():
        plain = model(**tiny_batch, output_hidden_states=True)
        with adjuster.capture() as state:
            captured = model(**tiny_batch).logits
    torch.testing.assert_close(captured, plain.logits, atol=0, rtol=0)
    # hidden_states[2] is the output of decoder layer 2; the latents sit at positions 5 to 8.
    torch.testing.assert_close(state["priors"][0], plain.hidden_states[2][:, 5:9], atol=0, rtol=0)


def test_basis_ignores_size_offsets_and_recovers_spread():
    generator = torch.Generator().manual_seed(0)
    n, d = 400, 16
    sizes = [3 + i % 2 for i in range(n)]
    u = torch.linalg.qr(torch.randn(d, 2, generator=generator))[0]
    offsets = torch.randn(2, d, generator=generator) * 50  # one large offset per grid size
    coefficients = torch.randn(n, 2, generator=generator) * torch.tensor([3.0, 1.0])
    slot = coefficients @ u.T + offsets[torch.tensor(sizes) - 3] + 0.01 * torch.randn(n, d, generator=generator)
    states = torch.stack([slot, 2 * slot], dim=1)  # two latent slots
    basis = principal_basis(states, sizes, 2)
    assert basis["directions"].shape == (2, d, 2) and basis["scales"].shape == (2, 2)
    assert abs(float(basis["directions"][0, :, 0] @ u[:, 0])) > 0.999
    torch.testing.assert_close(basis["scales"][0], torch.tensor([3.0, 1.0]), rtol=0.15, atol=0)
    torch.testing.assert_close(basis["scales"][1], torch.tensor([6.0, 2.0]), rtol=0.15, atol=0)
    assert min(basis["captured"]) > 0.99


def test_structured_offsets_follow_the_basis():
    directions = torch.linalg.qr(torch.randn(8, 2, generator=torch.Generator().manual_seed(1)))[0]
    basis = {"directions": directions[None], "scales": torch.tensor([[2.0, 0.5]])}
    offsets = structured_offsets(basis, 3.0, 20000, torch.Generator().manual_seed(2), "cpu")
    assert offsets.shape == (20000, 1, 8)
    coordinates = offsets[:, 0] @ directions
    torch.testing.assert_close(coordinates @ directions.T, offsets[:, 0], atol=1e-4, rtol=0)  # in the span
    torch.testing.assert_close(coordinates.std(0), torch.tensor([6.0, 1.5]), rtol=0.03, atol=0)


def test_sweep_size_rule():
    results = []
    for size, outcomes in ((5, [8, 8, 8, 4]), (6, [8, 3, 0, 0]), (7, [2, 5, 0, 0]), (8, [0, 0, 0, 0])):
        results += [{"grid_size": size, "greedy_success": k > 4, "sample_successes": [True] * k + [False] * (8 - k),
                     "prompt_tokens": 100 + size, "peak_gpu_gib": None} for k in outcomes]
    summary = summarise_sweep(results)
    assert summary["per_size"]["7"]["mixed_share"] == 0.5 and summary["per_size"]["8"]["all_fail_share"] == 1
    assert summary["per_size"]["6"]["greedy_failures_rescued_by_sampling"] == 1 / 3
    assert summary["training_sizes"] == [7]
    assert summarise_sweep(results, margin=0.25)["training_sizes"] == [5, 6, 7]


def cell(change, success, share):
    return {"route_change_rate": change, "success_rate": success, "rescue_share": share}


def test_noise_decision_rule():
    isotropic = {"chosen": 1.6, "scales": {"1.2": cell(0.05, 0.9, 0.1), "1.6": cell(0.12, 0.88, 0.2)}}
    better = {"chosen": 2.0, "scales": {"2.0": cell(0.11, 0.89, 0.3)}}
    decision = choose_noise({"families": {"isotropic": isotropic, "structured": better}})
    assert decision["family"] == "structured" and decision["scale"] == 2.0
    assert decision["matched_isotropic_sigma"] == 1.6
    worse = {"chosen": 2.0, "scales": {"2.0": cell(0.06, 0.9, 0.05)}}
    decision = choose_noise({"families": {"isotropic": isotropic, "structured": worse}})
    assert decision["family"] == "isotropic" and decision["scale"] == 1.6 and decision["matched_isotropic_sigma"] == 1.2
    decision = choose_noise({"families": {"isotropic": isotropic, "structured": {"chosen": None, "scales": {}}}})
    assert decision["family"] == "isotropic" and "no structured" in decision["reason"]


def test_probe_runs_end_to_end(tiny_model, tiny_batch, tmp_path, monkeypatch):
    model = tiny_model.eval()
    model.config.livr_stage = 2
    model.generation_config.suppress_tokens = SUPPRESS
    grid = lambda size: [[0] * size for _ in range(size)]
    data = {"train": [{"map_desc": grid(s)} for s in (3, 3, 3, 4, 4, 4)],
            "fresh/demo": [{"map_desc": grid(s)} for s in (5, 5, 6, 6, 7)]}
    monkeypatch.setattr(explore_probe, "records", lambda root, split: data[split])
    monkeypatch.setattr(explore_probe, "encode", lambda *args, training=True: prompt_of(tiny_batch))
    class Evaluator:
        def __init__(self, root): pass
        def score(self, row, text):
            answer = "".join("UDRL"[int(t) % 4] for t in text.split()[11:])
            return {"extracted_answer": answer, "success": answer.endswith("U")}
    monkeypatch.setattr(explore_probe, "ReferenceEvaluator", Evaluator)
    class Tokenizer:
        def decode(self, ids, skip_special_tokens): return " ".join(map(str, ids.tolist()))
    class Processor:
        tokenizer = Tokenizer()
    rl = {"latent": {"layer": 2}, "generation": {"do_sample": True, "temperature": 0.7, "top_p": 0.9, "max_new_tokens": 3}}
    torch.manual_seed(0)
    summary = explore_probe.explore_probe(model, Processor(), "unused", tmp_path / "probe", "fresh/demo", rl, {}, 0,
                                          basis_k=2, samples=4, draws=3, isotropic=(0.0, 50.0),
                                          structured=(0.0, 500.0), max_new_tokens=3)
    assert summary["basis"]["maps"] == 6 and summary["basis"]["k"] == 2
    assert set(summary["sweep"]["per_size"]) == {"5", "6", "7"}
    noise = summary["noise"]
    assert noise["rows"] == 4  # the 5 x 5 and 6 x 6 maps only
    for family in ("isotropic", "structured"):
        assert noise["families"][family]["scales"]["0.0"]["route_change_rate"] == 0
    assert noise["families"]["isotropic"]["scales"]["50.0"]["route_change_rate"] > 0
    assert "decision" in noise
    basis = torch.load(tmp_path / "probe/basis.pt")
    assert basis["directions"].shape == (4, 32, 2)
    sweep = [json.loads(line) for line in (tmp_path / "probe/sweep.jsonl").read_text().splitlines()]
    assert len(sweep) == 5 and all(len(r["sample_successes"]) == 4 for r in sweep)
    rows = [json.loads(line) for line in (tmp_path / "probe/noise.jsonl").read_text().splitlines()]
    assert all(len(r["families"]["structured"]["500.0"]["success"]) == 3 for r in rows)
    assert json.loads((tmp_path / "probe/summary.json").read_text())["noise"]["decision"] == noise["decision"]
