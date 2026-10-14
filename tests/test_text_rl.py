import copy
import json
import torch
from spara import text_rl
from spara.runtime import read_config, require_final_rl_checkpoint

# Random tiny weights can emit image markers, unsupported as generated image inputs.
SUPPRESS = [2, 3, 4, 5]


def prompt_of(batch):
    prompt = {k: v for k, v in batch.items() if k != "labels"}
    prompt["input_ids"] = prompt["input_ids"][:, :11]
    prompt["attention_mask"] = prompt["attention_mask"][:, :11]
    return prompt


def test_completion_mask_and_group_advantages():
    completions = torch.tensor([[7, 8, 6, 0, 0], [7, 6, 6, 0, 0], [7, 8, 9, 7, 8]])
    expected = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 1]]).bool()
    assert torch.equal(text_rl.completion_mask(completions, [6]), expected)
    assert torch.equal(text_rl.group_advantages(torch.ones(8)), torch.zeros(8))
    advantages = text_rl.group_advantages(torch.tensor([1.0, 0, 0, 0]))
    assert advantages[0] > 0 and (advantages[1:] < 0).all()
    torch.testing.assert_close(advantages.sum(), torch.tensor(0.0))


def test_training_logprobs_match_cached_rollout(tiny_model, tiny_batch):
    model = tiny_model.eval()
    model.config.livr_stage = 2
    prompt = prompt_of(tiny_batch)
    with torch.no_grad():
        generated = model.generate(**text_rl.repeat_prompt(prompt, 3), do_sample=True, max_new_tokens=4,
                                   suppress_tokens=SUPPRESS, use_cache=True, output_logits=True,
                                   return_dict_in_generate=True)
        logits = torch.stack(generated.logits, dim=1)
        for row in range(3):
            completion = generated.sequences[row, 11:]
            uncached = text_rl.token_logprobs(model, prompt, completion, 1.0)
            cached = logits[row].log_softmax(-1).gather(-1, completion[:, None]).squeeze(-1)
            torch.testing.assert_close(uncached, cached, atol=1e-5, rtol=1e-5)


def test_loss_at_reference_and_update_direction(tiny_model, tiny_batch):
    model = tiny_model.train()
    model.config.livr_stage = 2
    reference = copy.deepcopy(model).eval().requires_grad_(False)
    rl = {"clip_epsilon": 0.2, "kl_coefficient": 0.001}
    prompt = prompt_of(tiny_batch)
    completion = torch.tensor([14, 15, 6])
    with torch.no_grad():
        ref = text_rl.token_logprobs(reference, prompt, completion, 0.7)
    policy = text_rl.token_logprobs(model, prompt, completion, 0.7)
    loss, kl = text_rl.sequence_loss(policy, ref, 1.5, rl)
    # Policy equals reference: zero KL and the surrogate reduces to -advantage.
    assert kl.item() == 0
    torch.testing.assert_close(loss, torch.tensor(-1.5))
    loss.backward()
    expected = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    model.zero_grad()
    (-1.5 * text_rl.token_logprobs(model, prompt, completion, 0.7).mean()).backward()
    for a, b in zip(expected, [p.grad for p in model.parameters() if p.grad is not None]):
        torch.testing.assert_close(a, b)
    # Zero advantage and no drift give no policy gradient at all.
    model.zero_grad()
    text_rl.sequence_loss(text_rl.token_logprobs(model, prompt, completion, 0.7), ref, 0.0, rl)[0].backward()
    assert all(p.grad.abs().max() == 0 for p in model.parameters() if p.grad is not None)
    # A rewarded completion becomes more likely after a step.
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.05)
    before = policy.detach().sum()
    for _ in range(3):
        optimizer.zero_grad()
        text_rl.sequence_loss(text_rl.token_logprobs(model, prompt, completion, 0.7), ref, 1.0, rl)[0].backward()
        optimizer.step()
    with torch.no_grad():
        assert text_rl.token_logprobs(model, prompt, completion, 0.7).sum() > before


def test_rl_loop_freezes_vision_and_resumes_exactly(tiny_model, tiny_batch, tmp_path, monkeypatch):
    tiny_model.config.livr_stage = 2
    config = read_config("configs/livr.json") | {"gradient_checkpointing": False}
    rl = json.loads(open("configs/text_rl.json").read()) | {
        "epochs": 2, "prompts_per_step": 2, "group_size": 4, "learning_rate": 0.01,
        "checkpoint_every_steps": 1,
        "generation": {"do_sample": True, "temperature": 0.7, "max_new_tokens": 3, "suppress_tokens": SUPPRESS}}
    monkeypatch.setattr(text_rl, "records", lambda *_: list(range(3)))
    def encode(*args, training=True):
        assert not training
        prompt = prompt_of(tiny_batch)
        prompt["pixel_values"] = prompt["pixel_values"] + args[-1] * 0.01
        return prompt
    monkeypatch.setattr(text_rl, "encode", encode)
    class Tokenizer:
        def decode(self, ids, skip_special_tokens): return " ".join(map(str, ids.tolist()))
    class Processor:
        tokenizer = Tokenizer()
        def save_pretrained(self, path): pass
    scorer = lambda row, text: float(int(text.split()[-1]) % 2 == 0)
    reference = copy.deepcopy(tiny_model)
    frozen = {n: p.detach().clone() for n, p in tiny_model.named_parameters() if not p.requires_grad}
    initial = {n: p.detach().clone() for n, p in tiny_model.named_parameters() if p.requires_grad}
    torch.manual_seed(0)
    final = text_rl.train_text_rl(tiny_model, reference, Processor(), "unused", tmp_path / "full",
                                  config, rl, "livr/final", scorer=scorer)
    steps = [json.loads(line) for line in (tmp_path / "full/steps.jsonl").read_text().splitlines()]
    assert [s["step"] for s in steps] == [1, 2, 3, 4] and all(s["groups"] in (1, 2) for s in steps)
    assert any(s["zero_variance_groups"] < s["groups"] for s in steps)
    params = dict(tiny_model.named_parameters())
    for name, value in frozen.items():
        torch.testing.assert_close(value, params[name], atol=0, rtol=0)
    assert any(not torch.equal(value, params[name]) for name, value in initial.items())
    assert tiny_model.config.livr_stage == 2
    for name, value in reference.named_parameters():
        assert not value.requires_grad
    # Phase-specific completion check; a 3-row fake split is not the 1,000-row run.
    progress = json.loads((final / "complete.json").read_text())["progress"]
    assert progress["phase"] == "text-rl" and progress["optimizer_steps"] == 4
    # Resuming mid-run restores optimizer and sampling RNG. Some CPU builds round
    # reloaded weights' logits differently (~1e-7), which Adam amplifies; a missing
    # RNG restore instead diverges by ~4e-2, far outside this bound.
    from spara.model import LivrQwen, freeze_vision
    resumed = LivrQwen.from_pretrained(tmp_path / "full/step-1", attn_implementation="eager")
    freeze_vision(resumed)
    text_rl.train_text_rl(resumed, reference, Processor(), "unused", tmp_path / "resumed",
                          config, rl, "livr/final", tmp_path / "full/step-1", scorer=scorer)
    for key, expected in tiny_model.state_dict().items():
        torch.testing.assert_close(expected, resumed.state_dict()[key], atol=1e-4, rtol=1e-2)
    # A step-limited pilot stops early and is marked as such.
    policy = copy.deepcopy(reference)
    freeze_vision(policy)
    pilot = text_rl.train_text_rl(policy, reference, Processor(), "unused", tmp_path / "pilot",
                                  config, rl, "livr/final", scorer=scorer, max_steps=1)
    assert pilot is None and not (tmp_path / "pilot/final").exists()
    progress = json.loads((tmp_path / "pilot/summary.json").read_text())
    assert progress["pilot"] and progress["optimizer_steps"] == 1
    assert len((tmp_path / "pilot/steps.jsonl").read_text().splitlines()) == 1
    # In-memory evaluation runs train fully but leave no weights on disk.
    policy = copy.deepcopy(reference)
    freeze_vision(policy)
    unsaved = text_rl.train_text_rl(policy, reference, Processor(), "unused", tmp_path / "unsaved",
                                    config, rl | {"seed": 7}, "livr/final", scorer=scorer, save_final=False)
    progress = json.loads((tmp_path / "unsaved/summary.json").read_text())
    assert unsaved is None and not (tmp_path / "unsaved/final").exists()
    assert "pilot" not in progress and progress["optimizer_steps"] == 4 and progress["rl"]["seed"] == 7


def test_rl_checkpoint_completion_contract(tmp_path):
    from spara.runtime import json_write
    import pytest
    config = read_config("configs/livr.json")
    rl = json.loads(open("configs/text_rl.json").read())
    complete = {"phase": "text-rl", "stage": 2, "epoch": 1, "next_row": 0, "optimizer_steps": 125,
                "config": config, "rl": rl}
    for changes in [{"phase": None}, {"next_row": 8}, {"optimizer_steps": 124}, {"pilot": True}]:
        json_write(tmp_path / "complete.json", {"progress": complete | changes})
        with pytest.raises(ValueError, match="completed text-only RL"):
            require_final_rl_checkpoint(tmp_path, config, rl)
    json_write(tmp_path / "complete.json", {"progress": complete})
    require_final_rl_checkpoint(tmp_path, config, rl)
    with pytest.raises(ValueError, match="RL recipe mismatch"):
        require_final_rl_checkpoint(tmp_path, config, rl | {"group_size": 4})
