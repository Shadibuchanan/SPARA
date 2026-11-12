import torch
import pytest
from transformers import get_cosine_schedule_with_warmup
from spara.runtime import make_optimizer, read_config, restore_training, save_checkpoint


def test_same_stage_resume_and_fresh_stage_optimizer(tiny_model, tiny_batch, tmp_path):
    config = read_config("configs/livr.json")
    optimizer = make_optimizer(tiny_model, config)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 10)
    tiny_model(**tiny_batch).loss.backward(); optimizer.step(); scheduler.step()
    # Processor persistence is covered using the real tokenizer in another test.
    class Processor:
        def save_pretrained(self, path): pass
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, tiny_model, Processor(), optimizer, scheduler, {"stage": 1, "step": 1})
    expected_rng = torch.rand(4)
    resumed = make_optimizer(tiny_model, config)
    schedule2 = get_cosine_schedule_with_warmup(resumed, 0, 10)
    progress = restore_training(checkpoint, resumed, schedule2)
    assert progress == {"stage": 1, "step": 1}
    assert schedule2.state_dict() == scheduler.state_dict()
    torch.testing.assert_close(torch.rand(4), expected_rng, atol=0, rtol=0)
    for old, new in zip(optimizer.state.values(), resumed.state.values()):
        for key in old: torch.testing.assert_close(old[key], new[key], atol=0, rtol=0)
    tiny_model.config.livr_stage = 2
    assert not make_optimizer(tiny_model, config).state



def test_prefix_learning_rates_add_groups_without_changing_the_default_two(tiny_model):
    config = read_config("configs/livr.json")
    default = make_optimizer(tiny_model, config)
    assert [g["weight_decay"] for g in default.param_groups] == [config["weight_decay"], 0.0]
    names = [n for n, p in tiny_model.named_parameters() if p.requires_grad]
    prefix = names[-1].rsplit(".", 1)[0] + "."
    split = make_optimizer(tiny_model, config, {prefix: 1e-3})
    scheduler = get_cosine_schedule_with_warmup(split, 2, 10)
    assert all(g["lr"] == 0.0 for g in split.param_groups)  # warmup starts every group at zero
    split.step(); scheduler.step()
    assert [g["initial_lr"] for g in split.param_groups] == [config["learning_rate"]] * 2 + [1e-3] * 2
    assert scheduler.get_last_lr()[-1] / scheduler.get_last_lr()[0] == pytest.approx(1e-3 / config["learning_rate"])
    by_id = {id(p): n for n, p in tiny_model.named_parameters()}
    moved = {by_id[id(p)] for g in split.param_groups[2:] for p in g["params"]}
    assert moved == {n for n in names if n.startswith(prefix)} and moved
    assert sum(len(g["params"]) for g in split.param_groups) == sum(len(g["params"]) for g in default.param_groups)

def test_mid_epoch_resume_matches_uninterrupted_training(tiny_model, tiny_batch, tmp_path, monkeypatch):
    from spara import training
    from spara.model import LivrQwen, freeze_vision
    config = read_config("configs/livr.json") | {"epochs_per_stage": 2, "checkpoint_every_steps": 1,
                                               "gradient_checkpointing": False}
    monkeypatch.setattr(training, "records", lambda *_: list(range(9)))
    def encode(*args):
        row = args[-1]
        batch = {k: v.clone() for k, v in tiny_batch.items()}
        batch["pixel_values"] += row * 0.01
        batch["input_ids"][0, 11] = batch["labels"][0, 11] = 14 + row % 3
        return batch
    monkeypatch.setattr(training, "encode", encode)
    class Processor:
        def save_pretrained(self, path): pass
    training.train_stage(tiny_model, Processor(), "unused", tmp_path / "full", config, 1)
    checkpoint = tmp_path / "full/step-1"
    resumed = LivrQwen.from_pretrained(checkpoint, attn_implementation="eager")
    freeze_vision(resumed)
    training.train_stage(resumed, Processor(), "unused", tmp_path / "resumed", config, 1, checkpoint)
    for key, expected in tiny_model.state_dict().items():
        torch.testing.assert_close(expected, resumed.state_dict()[key], atol=0, rtol=0)


def test_pilot_or_partial_checkpoint_cannot_be_a_baseline(tmp_path):
    from spara.runtime import json_write, require_final_checkpoint
    config = read_config("configs/livr.json")
    complete = {"stage": 1, "epoch": 10, "next_row": 0, "optimizer_steps": 1250, "config": config}
    for changes in [{"pilot": True}, {"epoch": 9}, {"next_row": 8}, {"optimizer_steps": 2}]:
        json_write(tmp_path / "complete.json", {"progress": complete | changes})
        with pytest.raises(ValueError, match="completed Stage"):
            require_final_checkpoint(tmp_path, 1, config)
    json_write(tmp_path / "complete.json", {"progress": complete})
    require_final_checkpoint(tmp_path, 1, config)
    with pytest.raises(ValueError, match="recipe mismatch"):
        require_final_checkpoint(tmp_path, 1, config | {"learning_rate": 0.02})


def test_final_weights_only_checkpoint_loads_without_optimizer_state(tiny_model, tmp_path):
    from spara.model import LivrQwen
    config = read_config("configs/livr.json")
    optimizer = make_optimizer(tiny_model, config)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 10)
    class Processor:
        def save_pretrained(self, path): pass
    save_checkpoint(tmp_path / "final", tiny_model, Processor(), optimizer, scheduler, {"stage": 1}, training_state=False)
    assert (tmp_path / "final/complete.json").is_file() and not (tmp_path / "final/training_state.pt").exists()
    reloaded = LivrQwen.from_pretrained(tmp_path / "final", attn_implementation="eager")
    for key, value in tiny_model.state_dict().items():
        torch.testing.assert_close(value, reloaded.state_dict()[key], atol=0, rtol=0)
    with pytest.raises(FileNotFoundError):
        restore_training(tmp_path / "final", optimizer, scheduler)
