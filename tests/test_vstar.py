import json
import os
from pathlib import Path

import pytest
from PIL import Image

from spara import vstar


def question(i, category, label="A", text=None):
    return {"image": f"{category}/img_{i}.jpg", "category": category, "question_id": str(i), "label": label,
            "text": text or "What is it?\n(A) one\n(B) two\nAnswer with the option's letter from the given choices directly."}


def release(root, rows, size=(560, 420)):
    for row in rows:
        path = root / row["image"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, "white").save(path)
    (root / vstar.QUESTIONS).write_text("".join(json.dumps(row) + "\n" for row in rows))
    (root / "manifest.json").write_text(json.dumps({"revision": vstar.REVISION}))
    return root


def full(root):
    rows = [question(i, "direct_attributes") for i in range(115)]
    return release(root, rows + [question(115 + i, "relative_position") for i in range(76)])


@pytest.mark.parametrize("response,expected", [
    ("B", "B"), (" B\n", "B"), ("(B)", "B"), ("B.", "B"), ("B) left", "B"), ("(B) left", "B"), ("B: left", "B"),
    ("Answer: B", "B"), (".(B)", "B"), (":B", "B"), (". A dog", None), ("The answer is (B).", "B"), ("the answer is B", "B"),
    ("A dog", None), ("Both", None), ("lefty", None), ("", None), ("E", None), ("C", None), ("AB", None),
])
def test_extract_reads_only_an_offered_letter(response, expected):
    assert vstar.extract(response, {"A": "one", "B": "left"})[0] == expected


def test_extract_reads_an_option_in_words_and_says_so():
    choices = {"A": "right", "B": "left"}
    assert vstar.extract("Left.", choices) == ("B", "text")
    assert vstar.extract("B", choices) == ("B", "letter")
    assert vstar.extract("left side", choices) == (None, None)
    assert vstar.extract("same", {"A": "same", "B": "Same"}) == (None, None)


def test_options_are_the_lettered_lines():
    assert vstar.options(question(0, "direct_attributes")["text"]) == {"A": "one", "B": "two"}
    assert list(vstar.options("Is (A) here?\n(A) x\n(B) y z \n(C) z\n(D) w\nAnswer")) == ["A", "B", "C", "D"]
    assert vstar.options("Q\n(A) x\n(B) y z \nAnswer")["B"] == "y z"


def test_questions_audit_counts_labels_and_images(tmp_path):
    root = full(tmp_path)
    rows = vstar.questions(root)
    assert len(rows) == 191 and rows[0]["question_id"] == "0" and rows[-1]["category"] == "relative_position"
    (root / rows[3]["image"]).unlink()
    with pytest.raises(ValueError, match="Missing or unsafe"):
        vstar.questions(root)
    release(root, rows[:-1])
    with pytest.raises(ValueError, match="115 \\+ 76"):
        vstar.questions(root)
    release(root, rows[:-1] + [question(999, "relative_position", label="D")])
    with pytest.raises(ValueError, match="label"):
        vstar.questions(root)


def test_image_file_stays_inside_the_root(tmp_path):
    (tmp_path / "data").mkdir()
    Image.new("RGB", (28, 28)).save(tmp_path / "outside.jpg")
    with pytest.raises(ValueError, match="Missing or unsafe"):
        vstar.image_file(tmp_path / "data", {"image": "../outside.jpg"})


def test_resized_keeps_small_images_and_caps_large_ones():
    small = Image.new("RGB", (560, 420))
    assert vstar.resized(small, 16384 * 784) is small
    large = vstar.resized(Image.new("RGB", (2246, 1582)), 1280 * 784)
    assert large.width % 28 == 0 and large.height % 28 == 0
    tokens = (large.width // 28) * (large.height // 28)
    assert 1100 < tokens <= 1280
    assert abs(large.width / large.height - 2246 / 1582) < 0.05


def test_summarize_splits_by_category():
    results = [{"category": "direct_attributes", "correct": True, "answer": "A", "how": "letter"},
               {"category": "direct_attributes", "correct": False, "answer": None, "how": None},
               {"category": "relative_position", "correct": True, "answer": "B", "how": "text"}]
    summary = vstar.summarize(results)
    assert summary["overall"] == {"correct": 2, "total": 3, "accuracy": 2 / 3, "unparsed": 1,
                                  "answered_in_words": 1}
    assert summary["direct_attributes"]["accuracy"] == 0.5 and summary["relative_position"]["total"] == 1


class Config:
    def __init__(self, tokens):
        self.livr_latent_tokens = tokens


class Model:
    def __init__(self, tokens):
        self.config = Config(tokens)


def test_image_latents_for_each_kind_of_model():
    assert vstar.image_latents(Model([])) == []
    assert vstar.image_latents(Model(["a", "b"])) == ["a", "b"]
    assert vstar.image_latents(Model({"globals": ["g"], "per_image": [["p0", "p1"], ["q0", "q1"]]})) == ["g", "p0", "p1"]
    with pytest.raises(ValueError, match="grid size"):
        vstar.image_latents(Model({"8": ["r"]}))


@pytest.mark.integration
def test_encode_with_the_real_processor(tmp_path):
    from transformers import AutoProcessor
    processor_path = Path(os.environ.get("SPARA_TEST_PROCESSOR", ".cache/processor"))
    if not processor_path.is_dir():
        pytest.skip("Prepare the processor")
    processor = AutoProcessor.from_pretrained(processor_path)
    root = release(tmp_path, [question(0, "direct_attributes")], size=(2246, 1582))
    row = vstar.questions(root, audit=False)[0]
    chat, tokens = vstar.encode(processor, root, row, 16384 * 784)
    assert tokens == chat["image_grid_thw"][0, 1:].prod().item() // 4
    text = processor.tokenizer.decode(chat["input_ids"][0])
    assert text.endswith("directly.<|im_end|>\n<|im_start|>assistant\n")
    assert text.count("<|vision_start|>") == 1 and "What is it?" in text
    small, fewer = vstar.encode(processor, root, row, 1280 * 784)
    assert fewer <= 1280 < tokens and fewer == small["image_grid_thw"][0, 1:].prod().item() // 4
    livr, same = vstar.encode(processor, root, row, 1280 * 784, "livr")
    assert same == fewer
    assert processor.tokenizer.decode(livr["input_ids"][0]).endswith("directly.<|im_end|>\n<|im_start|>assistant")
    with pytest.raises(ValueError, match="no latent tokens"):
        vstar.encode(processor, root, row, 1280 * 784, "chat", ["<|livr_latent_0|>"])
