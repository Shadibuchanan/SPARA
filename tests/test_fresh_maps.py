import json
import os
import random
import statistics
from pathlib import Path

import pytest
from PIL import Image

from spara import data, fresh_maps
from spara.data import image_path, layout_key, records
from spara.fresh_maps import draw_layout, shortest_path

# Released train and test layouts pooled per size: mean and s.d. of hole counts, mean
# shortest route, and the share of maps whose start is next to the goal.
RELEASED = {3: (1.40, 1.00, 2.07, 0.335), 4: (2.77, 1.43, 2.67, 0.217),
            5: (4.47, 1.83, 3.48, 0.172), 6: (6.66, 2.25, 4.25, 0.090)}


def test_shortest_path():
    assert shortest_path([[1, 2]]) == 1
    assert shortest_path([[1, -1], [0, 2]]) == 2
    assert shortest_path([[1, -1], [-1, 2]]) is None
    assert shortest_path([[2, 0, 0], [-1, -1, 0], [1, 0, 0]]) == 6


def test_layouts_match_released_statistics():
    # Per-draw statistics: 2,000 unique 3 x 3 layouts would oversample rare ones, since
    # the 3 x 3 layout space is small. Real sets draw a few hundred maps of size 5 to 8.
    rng = random.Random(0)
    for size, (holes, spread, route, adjacent) in RELEASED.items():
        layouts = []
        while len(layouts) < 2000:
            desc = draw_layout(size, rng)
            if shortest_path(desc) is not None:
                layouts.append(desc)
        counts = [sum(v == -1 for row in d for v in row) for d in layouts]
        routes = [shortest_path(d) for d in layouts]
        assert None not in routes
        assert abs(statistics.mean(counts) - holes) < 0.2
        assert abs(statistics.pstdev(counts) - spread) < 0.2
        assert abs(statistics.mean(routes) - route) < 0.25
        assert abs(sum(r == 1 for r in routes) / len(routes) - adjacent) < 0.06


@pytest.fixture
def fake_release(monkeypatch):
    """Released prompt and layouts without the real data; a blank image as the render."""
    blocked = [[1, 2], [0, 0]]
    released = {layout_key({"map_desc": blocked})}
    real_records = data.records
    monkeypatch.setattr(fresh_maps, "records",
                        lambda root, split: [{"text_input": "Map: <image>"}] if split == "train" else real_records(root, split))
    monkeypatch.setattr(fresh_maps, "released_layouts", lambda root: released)
    monkeypatch.setattr(data, "released_layouts", lambda root: released)
    monkeypatch.setattr(fresh_maps, "render", lambda desc: Image.new("RGB", (64 * len(desc),) * 2))
    monkeypatch.setattr(fresh_maps, "version", lambda name: "test")
    return blocked


def test_generated_sets_load_and_avoid_released_and_each_other(tmp_path, fake_release):
    manifest = fresh_maps.generate(tmp_path, "a", [2, 3], 8, seed=1)
    rows = records(tmp_path, "fresh/a")
    assert manifest["count"] == len(rows) == 16 and [r["map_id"] for r in rows] == list(range(16))
    assert all(r["text_input"] == "Map: <image>" and shortest_path(r["map_desc"]) for r in rows)
    assert fake_release not in [r["map_desc"] for r in rows]
    assert image_path(tmp_path, "fresh/a", rows[-1]) == (tmp_path / "fresh/a/img/level_3/15.png").resolve()
    # 2 x 2 grids have few solvable layouts, so a second set must actively avoid the first.
    fresh_maps.generate(tmp_path, "b", [2], 8, seed=1, exclude_sets=["a"])
    first = {layout_key(r) for r in rows}
    assert not first & {layout_key(r) for r in records(tmp_path, "fresh/b")}
    with pytest.raises(FileExistsError):
        fresh_maps.generate(tmp_path, "a", [2], 1, seed=2)
    with pytest.raises(ValueError, match="Sizes"):
        fresh_maps.generate(tmp_path, "c", [9], 1, seed=2)


def test_fresh_records_reject_tampering_and_released_layouts(tmp_path, fake_release, monkeypatch):
    fresh_maps.generate(tmp_path, "a", [3], 4, seed=1)
    path = tmp_path / "fresh/a/maps.jsonl"
    original = path.read_text()
    path.write_text(original.replace('"map_id": 0', '"map_id": 99'))
    with pytest.raises(ValueError, match="manifest"):
        records(tmp_path, "fresh/a")
    path.write_text(original)
    taken = json.loads(original.splitlines()[0])
    monkeypatch.setattr(data, "released_layouts", lambda root: {layout_key(taken)})
    with pytest.raises(ValueError, match="shares a layout"):
        records(tmp_path, "fresh/a")
    with pytest.raises(ValueError, match="fresh/<name>"):
        records(tmp_path, "validation")


def test_renderer_reproduces_released_images():
    pytest.importorskip("gymnasium")
    pytest.importorskip("pygame")
    import numpy as np
    root = Path(os.environ.get("SPARA_TEST_DATA", "data/vsp"))
    if not (root / "train/train_direct.jsonl").is_file():
        pytest.skip("Released VSP data not available")
    for split in ("train", "test"):
        rows = records(root, split)
        for size in (3, 4, 5, 6):
            for row in [r for r in rows if len(r["map_desc"]) == size][:5]:
                with Image.open(image_path(root, split, row)) as released:
                    expected = np.asarray(released.convert("RGB"))
                assert np.array_equal(np.asarray(fresh_maps.render(row["map_desc"])), expected)


def test_detour_sets_are_labelled_detour_maps_drawn_by_weight(tmp_path, fake_release):
    from mathruler.grader import extract_boxed_content
    from spara.fresh_maps import is_detour, label_route
    from spara.planning_probe import walk
    manifest = fresh_maps.generate(tmp_path, "d", [3, 4, 5, 6], None, seed=3, detour_count=40, weights=[1, 2, 3, 4])
    rows = records(tmp_path, "fresh/d")
    assert manifest["count"] == len(rows) == 40 and manifest["attempts"] > 40 and manifest["weights"] == [1, 2, 3, 4]
    assert all(is_detour(r["map_desc"]) for r in rows)
    for r in rows:
        route = "".join(ch for ch in extract_boxed_content(r["text_output"]) if ch in "UDLR")
        assert walk(r["map_desc"], route)[1] == "goal" and len(route) == shortest_path(r["map_desc"])
        assert route == label_route(r["map_desc"])
    assert {len(r["map_desc"]) for r in rows} <= {3, 4, 5, 6}
    fresh_maps.generate(tmp_path, "e", [4], None, seed=3, exclude_sets=["d"], detour_count=10)
    assert not {layout_key(r) for r in rows} & {layout_key(r) for r in records(tmp_path, "fresh/e")}


def test_nodetour_sets_are_labelled_direct_maps(tmp_path, fake_release):
    from mathruler.grader import extract_boxed_content
    from spara.fresh_maps import is_detour, is_direct, label_route
    from spara.planning_probe import walk
    fresh_maps.generate(tmp_path, "d", [3, 4], None, seed=3, detour_count=10)
    manifest = fresh_maps.generate(tmp_path, "n", [3, 4, 5, 6, 7, 8], None, seed=6, exclude_sets=["d"],
                                   nodetour_count=60, weights=[1] * 6)
    rows = records(tmp_path, "fresh/n")
    assert manifest["count"] == len(rows) == 60 and manifest["nodetour_count"] == 60 and "detour_count" not in manifest
    assert all(is_direct(r["map_desc"]) and not is_detour(r["map_desc"]) for r in rows)
    assert {len(r["map_desc"]) for r in rows} == {3, 4, 5, 6, 7, 8}
    for r in rows:
        (sr, sc), (gr, gc) = fresh_maps.find(r["map_desc"], 1), fresh_maps.find(r["map_desc"], 2)
        route = "".join(ch for ch in extract_boxed_content(r["text_output"]) if ch in "UDLR")
        assert walk(r["map_desc"], route)[1] == "goal" and route == label_route(r["map_desc"])
        assert len(route) == abs(sr - gr) + abs(sc - gc)  # every move heads toward the goal
    assert not {layout_key(r) for r in rows} & {layout_key(r) for r in records(tmp_path, "fresh/d")}
    assert not is_direct([[1, -1], [-1, 2]]) and not is_direct([[2, -1, 1], [0, 0, 0]])
    with pytest.raises(ValueError, match="not both"):
        fresh_maps.generate(tmp_path, "x", [3], None, seed=1, detour_count=2, nodetour_count=2)


def test_labels_break_ties_up_left_right_down():
    from spara.fresh_maps import answer_text, label_route
    open_map = [[1, 0, 0], [0, 0, 0], [0, 0, 2]]
    assert label_route(open_map) == "RRDD"  # no U or L step shortens; R comes before D
    goal_above = [[2, 0, 0], [0, 0, 0], [0, 0, 1]]
    assert label_route(goal_above) == "UULL"
    assert answer_text("RD") == "\\boxed{RIGHT, DOWN}"
    assert label_route([[1, -1], [-1, 2]]) is None


def test_detour_draws_stop_after_the_attempt_limit():
    with pytest.raises(RuntimeError, match="attempts"):
        fresh_maps.draw_detour_layouts(5, random.Random(0), set(), [2], [1], max_attempts=200)
