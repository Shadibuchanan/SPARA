"""New FrozenLake maps in the Mirage VSP format, drawn like the released ones.

Start and goal are distinct cells drawn uniformly; every other cell is a hole with
probability `hole_probability`. Unsolvable layouts, layouts in the released train
or test split, and layouts already drawn (including in `exclude_sets`) are
rejected. Images come from gymnasium's FrozenLake renderer, which reproduces every
released image pixel for pixel; requirements/fresh-maps.txt pins its versions.

A set is written to <data root>/fresh/<name>/ (maps.jsonl, img/, manifest.json) and
read back with spara.data.records(root, "fresh/<name>").

Detour-only sets (docs/DETOUR_SCALING_SPEC.md) draw each attempt's size by weight, keep
only detour maps, and label every map with a shortest route in the released answer format.

    python -m spara.fresh_maps --data-root data/vsp --name probe --sizes 5,6,7,8 --per-size 100 --seed 101
    python -m spara.fresh_maps --data-root data/vsp --name detour-pool --detour-count 2000 \
        --sizes 3,4,5,6 --weights 1,2,3,4 --seed 202 --exclude probe,detour-heldout
"""
import argparse
import json
import random
from collections import deque
from importlib.metadata import version
from pathlib import Path

from .data import FRESH, layout_key, records, released_layouts, sha256

START, GOAL, HOLE, SAFE = 1, 2, -1, 0
# The renderer keeps 64-pixel cells up to a 512-pixel window, as in the released images.
MAX_SIZE = 8


def shortest_path(desc):
    """Moves on the shortest hole-free route from start to goal, or None."""
    size = len(desc)
    cells = {(r, c): v for r, row in enumerate(desc) for c, v in enumerate(row)}
    start = next(cell for cell, v in cells.items() if v == START)
    queue, seen = deque([(start, 0)]), {start}
    while queue:
        (r, c), steps = queue.popleft()
        if cells[r, c] == GOAL:
            return steps
        for cell in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if cell in cells and cell not in seen and cells[cell] != HOLE:
                seen.add(cell)
                queue.append((cell, steps + 1))
    return None


def find(desc, value):
    return next((r, c) for r, row in enumerate(desc) for c, v in enumerate(row) if v == value)


def is_detour(desc):
    """Solvable, and the shortest safe route is longer than the grid distance."""
    (sr, sc), (gr, gc) = find(desc, START), find(desc, GOAL)
    steps = shortest_path(desc)
    return steps is not None and steps > abs(sr - gr) + abs(sc - gc)


# Ties between shortest routes are broken by this move order; it reproduces 976 of the
# 1,000 released training answers (all of which are shortest routes).
LABEL_ORDER = (("U", (-1, 0), "UP"), ("L", (0, -1), "LEFT"), ("R", (0, 1), "RIGHT"), ("D", (1, 0), "DOWN"))


def label_route(desc):
    """The shortest safe route that comes first in U, L, R, D order, as move letters."""
    size = len(desc)
    goal = find(desc, GOAL)
    distance, queue = {goal: 0}, deque([goal])
    while queue:
        r, c = queue.popleft()
        for _, (dr, dc), _ in LABEL_ORDER:
            cell = (r + dr, c + dc)
            if 0 <= cell[0] < size and 0 <= cell[1] < size and desc[cell[0]][cell[1]] != HOLE and cell not in distance:
                distance[cell] = distance[r, c] + 1
                queue.append(cell)
    cell, route = find(desc, START), ""
    if cell not in distance:
        return None
    while cell != goal:
        letter, step = next((m, (dr, dc)) for m, (dr, dc), _ in LABEL_ORDER
                            if distance.get((cell[0] + dr, cell[1] + dc)) == distance[cell] - 1)
        cell, route = (cell[0] + step[0], cell[1] + step[1]), route + letter
    return route


def answer_text(route):
    words = {m: w for m, _, w in LABEL_ORDER}
    return "\\boxed{" + ", ".join(words[m] for m in route) + "}"


def draw_layout(size, rng, hole_probability=0.2):
    start, goal = rng.sample([(r, c) for r in range(size) for c in range(size)], 2)
    desc = [[HOLE if rng.random() < hole_probability else SAFE for _ in range(size)] for _ in range(size)]
    desc[start[0]][start[1]], desc[goal[0]][goal[1]] = START, GOAL
    return desc


def draw_layouts(size, count, rng, exclude, hole_probability=0.2):
    """`count` solvable layouts not in `exclude`; adds them to `exclude`."""
    layouts = []
    while len(layouts) < count:
        desc = draw_layout(size, rng, hole_probability)
        key = layout_key({"map_desc": desc})
        if key not in exclude and shortest_path(desc) is not None:
            exclude.add(key)
            layouts.append(desc)
    return layouts


def draw_detour_layouts(count, rng, exclude, sizes, weights, hole_probability=0.2, max_attempts=2_000_000):
    """`count` detour layouts not in `exclude`, each attempt's size drawn by `weights`;
    adds them to `exclude`. Returns the layouts and the number of attempts."""
    layouts, attempts = [], 0
    while len(layouts) < count:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"Found {len(layouts)} of {count} detour layouts in {max_attempts} attempts")
        desc = draw_layout(rng.choices(sizes, weights)[0], rng, hole_probability)
        key = layout_key({"map_desc": desc})
        if key not in exclude and is_detour(desc):
            exclude.add(key)
            layouts.append(desc)
    return layouts, attempts


def render(desc):
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import gymnasium
    from PIL import Image
    letters = {SAFE: "F", HOLE: "H", START: "S", GOAL: "G"}
    env = gymnasium.make("FrozenLake-v1", desc=["".join(letters[v] for v in row) for row in desc],
                         is_slippery=False, render_mode="rgb_array")
    try:
        env.reset(seed=0)
        return Image.fromarray(env.render())
    finally:
        env.close()


def generate(root, name, sizes, per_size, seed, hole_probability=0.2, exclude_sets=(), detour_count=None,
             weights=None):
    """`per_size` maps of each size; or, with `detour_count`, that many labelled detour maps
    with each attempt's size drawn by `weights`."""
    root = Path(root)
    target = root / (FRESH + name)
    if target.exists():
        raise FileExistsError(f"{target} exists; choose a new set name")
    if not all(2 <= size <= MAX_SIZE for size in sizes):
        raise ValueError(f"Sizes must be 2 to {MAX_SIZE}")
    prompts = {row["text_input"] for row in records(root, "train")}
    if len(prompts) != 1:
        raise ValueError("Expected one released prompt for every map")
    prompt = prompts.pop()
    exclude = released_layouts(root) | {layout_key(r) for other in exclude_sets for r in records(root, FRESH + other)}
    rng = random.Random(seed)
    rows, extra = [], {}
    if detour_count:
        weights = list(weights or [1] * len(sizes))
        layouts, attempts = draw_detour_layouts(detour_count, rng, exclude, list(sizes), weights, hole_probability)
        extra = {"detour_count": detour_count, "weights": weights, "attempts": attempts,
                 "label_rule": "shortest route, first in U, L, R, D order"}
    else:
        layouts = [desc for size in sizes for desc in draw_layouts(size, per_size, rng, exclude, hole_probability)]
    for desc in layouts:
        image = f"img/level_{len(desc)}/{len(rows)}.png"
        (target / image).parent.mkdir(parents=True, exist_ok=True)
        render(desc).save(target / image)
        row = {"text_input": prompt, "image_input": image, "map_id": len(rows), "map_desc": desc}
        if detour_count:
            row["text_output"] = answer_text(label_route(desc))
        rows.append(row)
    (target / "maps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    manifest = {"name": name, "generator": "spara.fresh_maps", "seed": seed, "sizes": list(sizes),
                "per_size": per_size, "hole_probability": hole_probability, "excluded_sets": list(exclude_sets),
                "count": len(rows), "jsonl_sha256": sha256(target / "maps.jsonl"),
                "renderer": {"gymnasium": version("gymnasium"), "pygame": version("pygame")}} | extra
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(prog="spara.fresh_maps")
    parser.add_argument("--data-root", default="data/vsp")
    parser.add_argument("--name", required=True)
    parser.add_argument("--sizes", required=True, help="Comma-separated grid sizes, e.g. 5,6,7,8")
    parser.add_argument("--per-size", type=int, help="Maps per size (not with --detour-count)")
    parser.add_argument("--detour-count", type=int, help="Labelled detour maps, sizes drawn by --weights")
    parser.add_argument("--weights", default="", help="Comma-separated size weights for --detour-count")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--hole-probability", type=float, default=0.2)
    parser.add_argument("--exclude", default="", help="Comma-separated fresh set names whose layouts to avoid")
    args = parser.parse_args()
    if (args.per_size is None) == (args.detour_count is None):
        parser.error("Give exactly one of --per-size and --detour-count")
    manifest = generate(args.data_root, args.name, [int(s) for s in args.sizes.split(",")], args.per_size,
                        args.seed, args.hole_probability, [s for s in args.exclude.split(",") if s],
                        args.detour_count, [float(w) for w in args.weights.split(",")] if args.weights else None)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
