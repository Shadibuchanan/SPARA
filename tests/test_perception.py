import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from PIL import Image
from transformers import Qwen2_5_VLConfig

from spara import perception
from spara.model import LivrQwen, freeze_vision
from spara.perception import extract, summarise


def test_extract_reads_the_option_letter():
    assert extract("B", 4) == "B" and extract("(C) leather", 4) == "C" and extract("A.", 2) == "A"
    assert extract("The answer is (D).", 4) == "D" and extract("I think B is right", 3) == "B"
    assert extract("E", 4) is None and extract("rubber", 4) is None and extract("", 2) is None
    prompt = "<image>\nQ?\n(A) Open\n(B) Closed\nAnswer with the option's letter."
    assert perception.choices(prompt) == ["Open", "Closed"]
    assert extract("closed", 2, ["Open", "Closed"]) == "B" and extract("It is open.", 2, ["Open", "Closed"]) == "A"
    assert extract("open or closed", 2, ["Open", "Closed"]) is None  # names both: no answer


def test_summarise_counts_mmvp_pairs_and_the_mean():
    rows = [{"benchmark": "mmvp", "category": "mmvp", "pair": 0, "correct": True, "letter": "A"},
            {"benchmark": "mmvp", "category": "mmvp", "pair": 0, "correct": False, "letter": None},
            {"benchmark": "mmvp", "category": "mmvp", "pair": 1, "correct": True, "letter": "B"},
            {"benchmark": "mmvp", "category": "mmvp", "pair": 1, "correct": True, "letter": "A"},
            {"benchmark": "vstar", "category": "relative_position", "correct": True, "letter": "A"},
            {"benchmark": "blink", "category": "Jigsaw", "correct": False, "letter": "B"}]
    out = summarise(rows)
    assert out["mmvp"]["accuracy"] == 0.75 and out["mmvp"]["pair_accuracy"] == 0.5 and out["mmvp"]["unparsed"] == 1
    assert out["mean"] == pytest.approx((0.75 + 1 + 0) / 3)


def write_split(root, name, rows):
    folder = root / name
    (folder / "images").mkdir(parents=True)
    out = []
    for i, row in enumerate(rows):
        paths = []
        for j in range(row.pop("images")):
            path = folder / "images" / f"{i}-{j}.png"
            Image.new("RGB", (84 + 28 * j, 56), (40 * i % 255, 90, 30 * j)).save(path)
            paths.append(str(path.relative_to(folder)))
        out.append(row | {"image_inputs": paths})
    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in out)
    (folder / "rows.jsonl").write_text(text)
    (folder / "manifest.json").write_text(json.dumps({"rows_sha256": hashlib.sha256(text.encode()).hexdigest()}))


def perception_root(root):
    letter = "Answer with the option's letter from the given choices directly."
    train = [{"text_input": f"<image>\nWhich?\n(A) x\n(B) y\n{letter}", "text_output": "AB"[i % 2], "images": 1,
              "benchmark": "zebra-cot", "category": "c", "id": str(i)} for i in range(4)]
    train.append({"text_input": f"Image 1: <image>\nImage 2: <image>\nWhich?\n{letter}", "text_output": "B",
                  "images": 2, "benchmark": "zebra-cot", "category": "c", "id": "two"})
    write_split(root, "zebra8k", train)
    for name in perception.BENCHMARKS:
        two = 2 if name == "blink" else 1
        write_split(root, name, [{"text_input": "<image>\n" * two + f"Q?\n(A) a\n(B) b\n{letter}",
                                  "text_output": "AB"[i], "images": two, "options": 2, "benchmark": name,
                                  "category": name, "id": str(i), "pair": 0} for i in range(2)])


def real_processor():
    from transformers import AutoProcessor
    path = Path(os.environ.get("SPARA_TEST_PROCESSOR", ".cache/processor"))
    if not path.is_dir():
        pytest.skip("Prepare the processor")
    return AutoProcessor.from_pretrained(path)


def loader():
    """load_model's stand-in: a fresh tiny model that reads the real processor's prompts."""
    def load(config):
        from spara.model import initialize_latents
        processor = real_processor()
        ids = processor.tokenizer.convert_tokens_to_ids(["<|image_pad|>", "<|video_pad|>", "<|vision_start|>",
                                                         "<|vision_end|>"])
        torch.manual_seed(0)
        model_config = Qwen2_5_VLConfig(
            vocab_size=len(processor.tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
            image_token_id=ids[0], video_token_id=ids[1], vision_start_token_id=ids[2], vision_end_token_id=ids[3],
            pad_token_id=processor.tokenizer.pad_token_id, eos_token_id=processor.tokenizer.convert_tokens_to_ids(
                "<|im_end|>"), rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
            vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64, "out_hidden_size": 32,
                           "num_heads": 4, "patch_size": 14, "temporal_patch_size": 2, "spatial_merge_size": 2,
                           "window_size": 28, "fullatt_block_indexes": [0]})
        model_config._attn_implementation = "eager"
        model = LivrQwen(model_config)
        if config.get("latent_tokens", 4):
            initialize_latents(model, processor)
        else:
            model.config.livr_latent_ids, model.config.livr_latent_tokens, model.config.livr_stage = [], [], 2
        freeze_vision(model)
        return model, processor
    return load


@pytest.mark.parametrize("arm", ["base", "livr", "patch"])
def test_an_arm_trains_and_scores_every_benchmark(arm, tmp_path):
    from spara.runtime import read_config
    root = tmp_path / "data"
    perception_root(root)
    config = read_config("configs/livr.json") | {"gradient_checkpointing": False, "gradient_accumulation": 2}
    settings = json.loads(Path("configs/perception.json").read_text()) | {"max_new_tokens": 3, "grid": 2,
                                                                          "save_weights": False}
    summary = perception.run_arm(arm, config, root, tmp_path / arm, settings, lambda m, p: {"described": True},
                                 loader())
    answers = [json.loads(line) for line in (tmp_path / arm / "answers.jsonl").read_text().splitlines()]
    assert len(answers) == 6 and {a["benchmark"] for a in answers} == set(perception.BENCHMARKS)
    assert set(summary["evaluation"]) == {*perception.BENCHMARKS, "mean"}
    assert json.loads((tmp_path / arm / "manifest.json").read_text())["described"]
    if arm == "base":
        assert not (tmp_path / arm / "stage1").exists()
    else:
        steps = [json.loads(line) for line in (tmp_path / arm / "stage1/steps.jsonl").read_text().splitlines()]
        assert len(steps) == 3 and all(s["stage"] == 1 for s in steps)  # 5 rows, 2 per step
    assert ("evaluation_globals_hidden" in summary) == (arm == "patch")


def test_compare_needs_the_same_rows_and_reads_patch_against_livr(tmp_path):
    keys = [("mmvp", "0"), ("vstar", "0"), ("blink", "0"), ("blink", "1")]
    for arm, right in {"base": [0], "livr": [0, 1], "patch": [0, 1, 2, 3], "scaffold": [1]}.items():
        folder = tmp_path / arm
        folder.mkdir()
        (folder / "answers.jsonl").write_text("".join(
            json.dumps({"benchmark": b, "id": i, "correct": k in right}) + "\n" for k, (b, i) in enumerate(keys)))
        (folder / "summary.json").write_text(json.dumps({"evaluation": {"mean": len(right) / 4}}))
    result = perception.compare({arm: tmp_path / arm for arm in perception.ARMS})
    assert result["items"] == 4 and result["tests"]["patch_vs_livr"]["all"]["only_first"] == 2
    assert result["reading"] == "no detectable difference between patch and LIVR latents"
    (tmp_path / "base/answers.jsonl").write_text("".join((tmp_path / "base/answers.jsonl").read_text().splitlines(True)[:3]))
    with pytest.raises(ValueError, match="same rows"):
        perception.compare({arm: tmp_path / arm for arm in perception.ARMS})


def test_hidden_rows_alternate():
    data = [("s", {"i": k}) for k in range(4)]
    assert [bool(r.get("livr_hide_globals")) for _, r in perception.hidden_every(data, 2)] == [True, False, True, False]
    assert all(r.get("livr_hide_globals") for _, r in perception.hidden_every(data, 1))
