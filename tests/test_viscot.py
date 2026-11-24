import hashlib
import io
import json
import tarfile

import pytest
from PIL import Image

from spara import perception, viscot


def released(i, dataset="gqa", image=None):
    image = image or f"cot/{dataset}/{i}.jpg"
    return {"dataset": dataset, "split": "train", "question_id": i, "image": [image, image + "###[1, 2, 3, 4]"],
            "conversations": [{"from": "human", "value": f"<image>\nWhat is {i}?" + viscot.BOX_REQUEST},
                              {"from": "gpt", "value": "[0.1, 0.2, 0.3, 0.4]"},
                              {"from": "human", "value": "<image>"}, {"from": "gpt", "value": f"answer {i}"}]}


def test_answer_only_drops_the_box_turn_and_the_crop():
    row = viscot.answer_only(released(7))
    assert row == {"text_input": "<image>\nWhat is 7?", "text_output": "answer 7", "image_inputs": ["cot/gqa/7.jpg"],
                   "dataset": "gqa", "question_id": 7}
    broken = released(8)
    broken["conversations"][0]["value"] = "<image>\nWhat is 8?"
    with pytest.raises(ValueError, match="Unexpected"):
        viscot.answer_only(broken)


def test_select_is_seeded_and_keeps_validation_images_out_of_training(tmp_path, monkeypatch):
    rows = [released(i, image=f"cot/gqa/{i // 2}.jpg") for i in range(60)]  # two questions per image
    monkeypatch.setattr(viscot, "LIST_ROWS", len(rows))
    monkeypatch.setattr(viscot, "download", lambda url, dest, sha: dest.write_text(json.dumps(rows)))
    first = viscot.select(tmp_path, 20, 6, seed=3)
    train, held = viscot.rows(tmp_path, "train"), viscot.rows(tmp_path, "validation")
    assert first["train"]["rows"] == len(train) == 20 and len(held) == 6
    assert all(split == viscot.SPLIT for split, _ in train)
    assert not {r["image_inputs"][0] for _, r in train} & {r["image_inputs"][0] for _, r in held}
    assert len({r["question_id"] for _, r in train + held}) == 26
    viscot.select(tmp_path, 20, 6, seed=3)
    assert viscot.rows(tmp_path, "train") == train
    viscot.select(tmp_path, 20, 6, seed=4)
    assert viscot.rows(tmp_path, "train") != train
    with pytest.raises(ValueError, match="Not enough"):
        viscot.select(tmp_path, 60, 6, seed=3)


class Response:
    def __init__(self, data, fail_after=None, quiet=False):
        self.data, self.fail_after, self.sent, self.quiet = io.BytesIO(data), fail_after, 0, quiet

    def read(self, n):
        if self.fail_after is not None and self.sent >= self.fail_after:
            if self.quiet:
                return b""  # what http.client gives when the server closes early
            raise ConnectionResetError("dropped")
        block = self.data.read(min(n, 7))
        self.sent += len(block)
        return block


def opener(parts, drop_first=False, quiet=False):
    calls = []

    def open_(request, timeout):
        start = int(request.headers.get("Range", "bytes=0-")[6:-1])
        calls.append((request.full_url, start))
        fail = 10 if drop_first and len(calls) == 1 else None
        return Response(parts[request.full_url][start:], fail, quiet)
    return open_, calls


def test_parts_join_verify_and_resume(monkeypatch):
    monkeypatch.setattr(viscot.time, "sleep", lambda s: None)
    parts = {"http://u/a": b"0123456789" * 5, "http://u/b": b"abcdefghij" * 3}
    hashes = [(url, hashlib.sha256(data).hexdigest(), len(data)) for url, data in parts.items()]
    for quiet in (False, True):  # an error, or an early end of data (job 7155589)
        open_, calls = opener(parts, drop_first=True, quiet=quiet)
        stream = viscot.Parts(hashes, opener=open_)
        assert stream.readall() == parts["http://u/a"] + parts["http://u/b"]
        assert stream.verified == ["a", "b"]
        assert calls[1] == ("http://u/a", 14)  # resumed where the connection stopped
    bad = viscot.Parts([("http://u/a", "0" * 64, 50)], opener=opener(parts)[0])
    with pytest.raises(ValueError, match="Checksum"):
        bad.readall()


def archive(members):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as tar:
        for name, size in members:
            image = io.BytesIO()
            Image.new("RGB", size, "red").save(image, format="JPEG")
            info = tarfile.TarInfo(viscot.ARCHIVE_PREFIX + name)
            info.size = image.tell()
            image.seek(0)
            tar.addfile(info, image)
    return data.getvalue()


def test_take_images_keeps_only_sampled_images_resized(tmp_path, monkeypatch):
    rows = [released(i) for i in range(3)]
    monkeypatch.setattr(viscot, "LIST_ROWS", 3)
    monkeypatch.setattr(viscot, "download", lambda url, dest, sha: dest.write_text(json.dumps(rows)))
    viscot.select(tmp_path, 2, 1, seed=0)
    data = archive([("gqa/0.jpg", (2240, 1680)), ("gqa/other.jpg", (56, 56)), ("gqa/1.jpg", (280, 280)),
                    ("gqa/2.jpg", (56, 84))])
    manifest = viscot.take_images(tmp_path, 1280, stream=io.BytesIO(data))
    folder = tmp_path / viscot.SPLIT / "cot" / "gqa"
    assert sorted(p.name for p in folder.iterdir()) == ["0.jpg", "1.jpg", "2.jpg"]
    with Image.open(folder / "0.jpg") as big, Image.open(folder / "1.jpg") as small:
        assert (big.width // 28) * (big.height // 28) <= 1280 and big.width % 28 == 0
        assert small.size == (280, 280)
    assert manifest["images"]["count"] == 3
    viscot.take_images(tmp_path, 1280, stream=io.BytesIO(b""))  # nothing missing: the archive is not read
    (folder / "2.jpg").unlink()
    with pytest.raises(ValueError, match="not in the archive"):
        viscot.take_images(tmp_path, 1280, stream=io.BytesIO(archive([("gqa/0.jpg", (56, 56))])))


def test_normalized_and_arm_rows():
    assert perception.normalized("  The  Girl. ") == "the girl"
    data = [("images", {"text_input": "<image>\nQ"})]
    assert perception.arm_data("livr", data) == data and perception.arm_data("sft", data) == data
    assert perception.arm_data("patch", data)[0][1]["livr_hide_globals"] is True
    assert "livr_hide_globals" not in data[0][1]
