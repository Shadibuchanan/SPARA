import torch

from spara.latent_battery import (LatentPatch, cell_weights, latent_positions, mcnemar, probe_layers,
                                  prompt_states, swap_donors)


def _prompt(batch, prefix=11):
    return {"input_ids": batch["input_ids"][:, :prefix], "attention_mask": batch["attention_mask"][:, :prefix],
            "pixel_values": batch["pixel_values"], "image_grid_thw": batch["image_grid_thw"]}


def _greedy(model, prompt, tokens=3):
    return model.generate(**prompt, do_sample=False, max_new_tokens=tokens, min_new_tokens=tokens,
                          use_cache=True)[0, prompt["input_ids"].shape[-1]:]


def test_own_states_reproduce_logits_and_greedy_answers(tiny_model, tiny_batch):
    model = tiny_model.eval()
    patch = LatentPatch(model)
    prompt = _prompt(tiny_batch)
    positions = latent_positions(prompt["input_ids"], model.config.livr_latent_ids)
    assert positions.tolist() == [5, 6, 7, 8]
    for stage in (1, 2):
        model.config.livr_stage = stage
        states = prompt_states(model, patch, prompt, positions)
        assert states.shape[:2] == (model.config.num_hidden_layers, 11)
        with torch.no_grad():
            plain = model(**tiny_batch).logits
            plain_answer = _greedy(model, prompt)
            with patch.apply(positions, states[:, positions]):
                patched = model(**tiny_batch).logits
                patched_answer = _greedy(model, prompt)
        # States come from the prompt-only forward; the 13-token pass rounds differently.
        torch.testing.assert_close(plain, patched, atol=1e-6, rtol=1e-5)
        assert torch.equal(plain_answer, patched_answer)
    patch.remove()


def test_bottleneck_swap_gives_the_donor_answer(tiny_model, tiny_batch):
    # Same text, different image: under the bottleneck the answer reads the image
    # only through the latents, so the donor's latents give the donor's answer.
    model = tiny_model.eval()
    model.config.livr_stage = 1
    patch = LatentPatch(model)
    donor_batch = dict(tiny_batch, pixel_values=tiny_batch["pixel_values"] + 10)
    recipient, donor = _prompt(tiny_batch), _prompt(donor_batch)
    positions = latent_positions(recipient["input_ids"], model.config.livr_latent_ids)
    donor_latents = prompt_states(model, patch, donor, positions)[:, positions]
    with torch.no_grad():
        own = model(**tiny_batch).logits[:, 9:]
        target = model(**donor_batch).logits[:, 9:]
        with patch.apply(positions, donor_latents):
            swapped = model(**tiny_batch).logits[:, 9:]
            swapped_answer = _greedy(model, recipient)
        donor_answer = _greedy(model, donor)
    assert not torch.allclose(own, target)
    torch.testing.assert_close(swapped, target, atol=1e-5, rtol=1e-5)
    assert torch.equal(swapped_answer, donor_answer)
    # Outside `apply`, the hooks leave the model unchanged.
    with torch.no_grad():
        torch.testing.assert_close(model(**tiny_batch).logits[:, 9:], own, atol=0, rtol=0)
    patch.remove()


def test_cell_weights_partition_each_cell():
    # Three cells over seven merged tokens per side (a 192 px map resized to 196 px).
    weights = cell_weights(3, torch.tensor([1, 14, 14]), 2)
    assert weights.shape == (9, 49)
    torch.testing.assert_close(weights.sum(-1), torch.ones(9))
    corner = weights[0].reshape(7, 7)
    assert corner[:3, :3].gt(0).all() and corner[3:].eq(0).all() and corner[:, 3:].eq(0).all()
    # One token per cell: the weights select it.
    torch.testing.assert_close(cell_weights(2, torch.tensor([1, 4, 4]), 2), torch.eye(4))


def test_swap_donors_share_size_and_never_layout():
    maps = [[[1, 0], [0, 2]], [[1, -1], [0, 2]], [[1, 0, 0], [0, 0, 0], [0, 0, 2]],
            [[1, 0, -1], [0, 0, 0], [0, 0, 2]], [[1, 0], [0, 2]]]
    rows = [{"map_desc": m} for m in maps]
    donors = swap_donors(rows, 42)
    assert set(donors) == set(range(5))
    for recipient, donor in donors.items():
        assert len(maps[donor]) == len(maps[recipient]) and maps[donor] != maps[recipient]
    assert swap_donors([{"map_desc": maps[0]}], 42) == {}


def test_mcnemar_and_probe_layers():
    assert mcnemar(0, 0) == 1.0
    assert abs(mcnemar(0, 6) - 2 / 64) < 1e-12
    assert mcnemar(3, 3) == 1.0
    assert probe_layers(36) == [3, 7, 11, 15, 19, 23, 27, 31, 35]
    assert probe_layers(3) == [2]


def test_probe_targets_follow_shortest_routes():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).parents[1] / "scripts" / "analyze_latent_battery.py"
    spec = importlib.util.spec_from_file_location("analyze_latent_battery", path)
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    # Test map 0: goal top-left, hole between it and the start at top-right.
    t = analysis.targets([[2, -1, 1], [0, 0, 0], [0, 0, 0]], "\\boxed{DOWN, LEFT, LEFT, UP}")
    assert t["length"] == 4 and t["best_moves"] == {"D"} and t["first_move"] == "D"
    assert t["goal"] == 0 and t["start"] == 2 and t["holes"][1] == 1.0 and sum(t["holes"]) == 1
    truth = analysis.torch.tensor([0, 0, 1, 1])
    assert analysis.balanced_accuracy(analysis.torch.tensor([0, 0, 0, 1]), truth) == 0.75


def _failures():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).parents[1] / "scripts" / "analyze_battery_failures.py"
    spec = importlib.util.spec_from_file_location("analyze_battery_failures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failure_walk_follows_the_reference_rules():
    import os
    from pathlib import Path
    import pytest
    from spara.evaluation import ReferenceEvaluator
    root = Path(os.environ.get("SPARA_TEST_DATA", "data/vsp"))
    if not (root / "manifest.json").is_file():
        pytest.skip("Run spara prepare-data first")
    simulate_vsp = ReferenceEvaluator(root).simulate
    failures = _failures()
    grid = [[2, -1, 1], [0, 0, 0], [0, 0, 0]]
    for route in ("DLLU", "DLLUL", "L", "RRDLLLU", "DD", "", "UUDLLU"):
        status, _, hole = failures.walk(grid, route)
        reference = simulate_vsp(grid, route)
        assert (status == "goal") == reference["success"]
        assert (status == "hole") == (reference["status"] == "Fell in hole")
    # Off-grid moves are ignored; the route then enters the hole at (0, 1).
    assert failures.walk(grid, "RL") == ("hole", [(0, 2), (0, 1)], (0, 1))


def test_failure_statistics():
    failures = _failures()
    assert abs(failures.poisson_binomial_upper([0.5, 0.5], 1) - 0.75) < 1e-12
    assert failures.poisson_binomial_upper([0.2, 0.3], 0) == 1.0
    assert failures.auc([3, 1, 2, 0], [True, False, True, False]) == 1.0
    assert failures.auc([1, 1], [True, False]) == 0.5
    import torch
    difference, p = failures.stratified_permutation([5, 0, 5, 0], [True, False, True, False], ["a", "a", "b", "b"],
                                                     torch.Generator().manual_seed(0))
    # Within each stratum, the failed map has the larger value in half the permutations.
    assert difference == 5 and 0.2 < p < 0.3
