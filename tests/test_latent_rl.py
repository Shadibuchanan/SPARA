import copy
import json

import pytest
import torch

from spara import latent_rl, text_rl
from spara.latent_rl import LatentAdjuster, add_latent_heads, choose_sigma, latent_objective, sigma_bounds
from spara.runtime import read_config, require_final_rl_checkpoint

SUPPRESS = [2, 3, 4, 5, 10, 11, 12, 13]  # image markers and latent tokens
LATENT = {"layer": 2, "sigma_init": 0.5, "log_sigma_bounds": [-9.21, 0.6931], "head_hidden_size": None,
          "head_learning_rate": 0.01, "logprob_reduction": "mean"}


def prompt_of(batch):
    prompt = {k: v for k, v in batch.items() if k != "labels"}
    prompt["input_ids"] = prompt["input_ids"][:, :11]
    prompt["attention_mask"] = prompt["attention_mask"][:, :11]
    return prompt


def stage2(model):
    model.config.livr_stage = 2
    return model


def test_zero_adjustment_is_livr_and_nonzero_changes_answers(tiny_model, tiny_batch):
    model = stage2(tiny_model).eval()
    adjuster = LatentAdjuster(model, 2)
    with torch.no_grad():
        plain = model(**tiny_batch).logits
        with adjuster.offset(torch.zeros(1, 4, 32)):
            zero = model(**tiny_batch).logits
        add_latent_heads(model, LATENT)
        with adjuster.mean() as state:  # fresh heads: mu = 0
            mean = model(**tiny_batch).logits
        with adjuster.offset(torch.randn(1, 4, 32)):
            moved = model(**tiny_batch).logits
    torch.testing.assert_close(zero, plain, atol=0, rtol=0)
    torch.testing.assert_close(mean, plain, atol=0, rtol=0)
    assert torch.equal(state["records"][0]["mean"], torch.zeros(1, 4, 32))
    # Tokens before the latents are untouched; the answer positions after them move.
    torch.testing.assert_close(moved[:, :5], plain[:, :5], atol=0, rtol=0)
    assert not torch.allclose(moved[:, 9:], plain[:, 9:])
    # Outside a context the hooks do nothing, and removal detaches them.
    with torch.no_grad():
        torch.testing.assert_close(model(**tiny_batch).logits, plain, atol=0, rtol=0)
    adjuster.remove()
    with pytest.raises(ValueError, match="followed by another decoder layer"):
        LatentAdjuster(model, 3)


def test_rollout_actions_replay_to_the_cached_logits(tiny_model, tiny_batch):
    model = stage2(tiny_model).eval()
    add_latent_heads(model, LATENT)
    adjuster = LatentAdjuster(model)
    prompt = prompt_of(tiny_batch)
    torch.manual_seed(3)
    with torch.no_grad(), adjuster.sample() as state:
        generated = model.generate(**text_rl.repeat_prompt(prompt, 3), do_sample=True, max_new_tokens=4,
                                   suppress_tokens=SUPPRESS, use_cache=True, output_logits=True,
                                   return_dict_in_generate=True)
    assert len(state["records"]) == 1
    actions = state["records"][0]["actions"]
    assert actions.shape == (3, 4, 32) and not torch.allclose(actions[0], actions[1])
    logits = torch.stack(generated.logits, dim=1)
    for row in range(3):
        completion = generated.sequences[row, 11:]
        with torch.no_grad(), adjuster.replay(actions[row:row + 1]):
            uncached = text_rl.token_logprobs(model, prompt, completion, 1.0)
        cached = logits[row].log_softmax(-1).gather(-1, completion[:, None]).squeeze(-1)
        torch.testing.assert_close(uncached, cached, atol=1e-5, rtol=1e-5)
    # Rows with different latent actions are scored differently.
    with torch.no_grad(), adjuster.replay(actions[1:2]):
        other = text_rl.token_logprobs(model, prompt, generated.sequences[0, 11:], 1.0)
    with torch.no_grad(), adjuster.replay(actions[0:1]):
        own = text_rl.token_logprobs(model, prompt, generated.sequences[0, 11:], 1.0)
    assert not torch.allclose(other, own)


def _step_gradients(model, adjuster, prompt, completion, action, reference):
    model.zero_grad(set_to_none=True)
    with torch.no_grad(), latent_rl.replaying(LatentAdjuster(reference, 2), action):
        ref = text_rl.token_logprobs(reference, prompt, completion, 0.7)
    with adjuster.replay(action) as state:
        policy = text_rl.token_logprobs(model, prompt, completion, 0.7)
        loss, _ = text_rl.sequence_loss(policy, ref, 1.3, {"clip_epsilon": 0.2, "kl_coefficient": 0.001})
        latent_loss, stats = latent_objective(model, state["prior"], action, 1.3, LATENT, 0.2)
        (loss + latent_loss).backward()
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}, stats


def test_gradients_match_with_and_without_checkpointing(tiny_model, tiny_batch):
    model = stage2(tiny_model)
    add_latent_heads(model, LATENT)
    reference = copy.deepcopy(model).eval().requires_grad_(False)
    del reference.latent_heads
    prompt, completion = prompt_of(tiny_batch), torch.tensor([14, 15, 6])
    action = torch.randn(1, 4, 32)
    adjuster = LatentAdjuster(model)
    model.train()
    plain, stats = _step_gradients(model, adjuster, prompt, completion, action, reference)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    checkpointed, _ = _step_gradients(model, adjuster, prompt, completion, action, reference)
    assert plain.keys() == checkpointed.keys()
    assert any(name.startswith("latent_heads.") for name in plain)
    # The latent term reaches the layers below the adjustment through z.
    assert plain["model.layers.0.mlp.down_proj.weight"].abs().max() > 0
    for name in plain:
        torch.testing.assert_close(plain[name], checkpointed[name], atol=1e-6, rtol=1e-5)
    assert stats["sigma_mean"] == pytest.approx(0.5, rel=1e-5) and stats["mean_abs_mu"] == 0


def test_latent_objective_moves_the_mean_toward_rewarded_actions(tiny_model):
    model = tiny_model
    add_latent_heads(model, LATENT)
    prior = torch.randn(1, 4, 32)
    action = prior + 0.3
    optimizer = torch.optim.SGD(model.latent_heads.parameters(), lr=1.0)
    for _ in range(5):
        optimizer.zero_grad()
        latent_objective(model, prior, action, 1.0, LATENT, 0.2)[0].backward()
        optimizer.step()
    mean, _ = model.latent_heads(latent_rl.head_input(prior))
    assert mean.mean() > 0.01


def test_heads_survive_save_and_load(tiny_model, tmp_path):
    from spara.model import LivrQwen
    add_latent_heads(stage2(tiny_model), LATENT)
    with torch.no_grad():
        tiny_model.latent_heads.mean[2].weight.normal_()
    tiny_model.save_pretrained(tmp_path)
    loaded = LivrQwen.from_pretrained(tmp_path, attn_implementation="eager")
    assert loaded.config.livr_latent_layer == 2 and loaded.config.livr_latent_heads["sigma_init"] == 0.5
    for (name, a), (_, b) in zip(tiny_model.latent_heads.state_dict().items(), loaded.latent_heads.state_dict().items()):
        torch.testing.assert_close(a, b, atol=0, rtol=0, msg=name)


def test_sigma_rule_and_bounds():
    summary = {"base_success_rate": 0.9, "sigmas": {
        "0.01": {"route_change_rate": 0.02, "success_rate": 0.9},
        "0.1": {"route_change_rate": 0.10, "success_rate": 0.85},
        "0.2": {"route_change_rate": 0.30, "success_rate": 0.84},
        "0.05": {"route_change_rate": 0.11, "success_rate": 0.80},
        "1": {"route_change_rate": 0.9, "success_rate": 0.1}}}
    # 0.05 changes enough routes but loses 10 points; 0.1 passes exactly at both thresholds.
    assert choose_sigma(summary) == 0.1
    assert choose_sigma({"base_success_rate": 0.9, "sigmas": {"1": {"route_change_rate": 0.05, "success_rate": 0.9}}}) is None
    assert sigma_bounds(1.0) == [-9.21, 1.3863]


def test_latent_rl_loop_trains_heads_and_marks_phase(tiny_model, tiny_batch, tmp_path, monkeypatch):
    model = stage2(tiny_model)
    reference = copy.deepcopy(model)
    add_latent_heads(model, LATENT)
    config = read_config("configs/livr.json") | {"gradient_checkpointing": True}
    rl = json.loads(open("configs/latent_rl.json").read())
    rl |= {"epochs": 1, "prompts_per_step": 2, "group_size": 4, "learning_rate": 0.01,
           "generation": {"do_sample": True, "temperature": 0.7, "max_new_tokens": 3, "suppress_tokens": SUPPRESS},
           "latent": LATENT, "train_split": "fresh/demo"}
    splits = []  # The recipe's training split reaches both the records and the prompts.
    monkeypatch.setattr(text_rl, "records", lambda root, split: splits.append(split) or list(range(3)))
    monkeypatch.setattr(text_rl, "encode",
                        lambda processor, root, split, row, training=True: splits.append(split) or prompt_of(tiny_batch))
    class Tokenizer:
        def decode(self, ids, skip_special_tokens): return " ".join(map(str, ids.tolist()))
    class Processor:
        tokenizer = Tokenizer()
        def save_pretrained(self, path): pass
    scorer = lambda row, text: float(int(text.split()[-1]) % 2 == 0)
    heads = {n: p.detach().clone() for n, p in model.latent_heads.named_parameters()}
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    with pytest.raises(ValueError, match="both"):
        text_rl.train_text_rl(model, reference, Processor(), "unused", tmp_path / "missing", config, rl,
                              "livr/final", scorer=scorer)
    adjusters = (LatentAdjuster(model), LatentAdjuster(reference, LATENT["layer"]))
    torch.manual_seed(0)
    final = text_rl.train_text_rl(model, reference, Processor(), "unused", tmp_path / "run", config, rl,
                                  "livr/final", scorer=scorer, adjusters=adjusters)
    steps = [json.loads(line) for line in (tmp_path / "run/steps.jsonl").read_text().splitlines()]
    assert [s["phase"] for s in steps] == ["latent-rl", "latent-rl"]
    assert len(splits) > 1 and set(splits) == {"fresh/demo"}
    assert all({"sigma_mean", "mean_abs_mu", "head_lr"} <= s.keys() for s in steps)
    assert steps[0]["head_lr"] == 0.01
    assert any(not torch.equal(p, heads[n]) for n, p in model.latent_heads.named_parameters())
    for name, value in frozen.items():
        torch.testing.assert_close(value, dict(model.named_parameters())[name], atol=0, rtol=0)
    assert not hasattr(reference, "latent_heads")
    progress = json.loads((final / "complete.json").read_text())["progress"]
    assert progress["phase"] == "latent-rl" and progress["rl"]["latent"]["layer"] == 2
    with pytest.raises(ValueError, match="completed text-only RL"):
        require_final_rl_checkpoint(final, config, rl | {"epochs": 1})


def test_probe_runs_and_applies_the_sigma_rule(tiny_model, tiny_batch, tmp_path, monkeypatch):
    model = stage2(tiny_model).eval()
    model.config.livr_latent_layer = 2
    model.generation_config.suppress_tokens = SUPPRESS
    monkeypatch.setattr(latent_rl, "records", lambda *_: [{"map_desc": [[1, 2]]} for _ in range(6)])
    monkeypatch.setattr(latent_rl, "encode", lambda *args, training=True: prompt_of(tiny_batch))
    class Evaluator:
        def __init__(self, root): pass
        def score(self, row, text):
            answer = "".join("UDRL"[int(t) % 4] for t in text.split()[11:])
            return {"extracted_answer": answer, "success": answer.endswith("U")}
    monkeypatch.setattr(latent_rl, "ReferenceEvaluator", Evaluator)
    class Tokenizer:
        def decode(self, ids, skip_special_tokens): return " ".join(map(str, ids.tolist()))
    class Processor:
        tokenizer = Tokenizer()
    summary = latent_rl.latent_probe(model, Processor(), "unused", tmp_path / "probe", [0.0, 50.0], 4, 2, 0,
                                     {}, max_new_tokens=3)
    assert summary["rows"] == 4 and summary["layer"] == 2
    # Zero scale never changes a route; a huge one changes some.
    assert summary["sigmas"]["0.0"]["route_change_rate"] == 0
    assert summary["sigmas"]["50.0"]["route_change_rate"] > 0
    rows = [json.loads(line) for line in (tmp_path / "probe/rows.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and all(len(r["sigmas"]["50.0"]["route_changed"]) == 2 for r in rows)
    assert json.loads((tmp_path / "probe/summary.json").read_text())["chosen_sigma"] == summary["chosen_sigma"]
