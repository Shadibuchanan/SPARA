"""Training-time helper images. They never enter evaluation inputs."""
from pathlib import Path

import numpy as np
from PIL import Image

# Five-stop inferno-like ramp: dark for value 0 (holes/unreachable), pale for 1 (goal).
RAMP = np.array([[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]], dtype=float)


def value_iteration(grid, gamma=0.85, theta=1e-7):
    """Optimal state values, transcribed from the paper's Appendix E.1.
    Codes: 1=start, 2=goal, -1=hole, 0=safe (the released map_desc encoding)."""
    grid = np.asarray(grid)
    n_rows, n_cols = grid.shape
    V = np.zeros_like(grid, dtype=float)
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    is_terminal = lambda r, c: grid[r, c] in (-1, 2)
    while True:
        delta = 0.0
        for r in range(n_rows):
            for c in range(n_cols):
                if is_terminal(r, c): continue
                v_old = V[r, c]
                V[r, c] = max(
                    ((0. if grid[nr, nc] == -1 else
                      1. if grid[nr, nc] == 2 else 0.)
                     + gamma * (0. if grid[nr, nc] in (-1, 2)
                                else V[nr, nc]))
                    if 0 <= nr < n_rows and 0 <= nc < n_cols
                    else gamma * V[r, c]
                    for dr, dc in actions
                    for nr, nc in [(r + dr, c + dc)])
                delta = max(delta, abs(v_old - V[r, c]))
        if delta < theta: break
    V[grid == -1] = 0.0; V[grid == 2] = 1.0
    return V


def colorize(values):
    scaled = np.clip(values, 0.0, 1.0) * (len(RAMP) - 1)
    low = np.floor(scaled).astype(int).clip(0, len(RAMP) - 2)
    weight = (scaled - low)[..., None]
    return ((1 - weight) * RAMP[low] + weight * RAMP[low + 1]).round().astype(np.uint8)


def value_heatmap(map_desc, size=320, gamma=0.85, theta=1e-7):
    """One flat colour block per cell, rendered at the released map-image size."""
    cells = colorize(value_iteration(map_desc, gamma, theta))
    return Image.fromarray(cells, "RGB").resize((size, size), Image.NEAREST)


def released_helper(root, row):
    """The released solved-route overlay (the paper's red-arrow alternative, Table 5)."""
    parts = Path(row["image_output"].replace("\\", "/")).parts
    if "img" not in parts:
        raise ValueError(f"Unexpected released helper path: {row['image_output']}")
    base = (Path(root) / "train").resolve()
    path = base.joinpath(*parts[parts.index("img"):]).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError(f"Missing or unsafe helper image: {path}")
    with Image.open(path) as source:
        return source.convert("RGB")


def row_helper(root, row):
    """`helper_input`, relative to the data root (scripts/prepare_perception.py)."""
    base = Path(root).resolve()
    path = base.joinpath(*Path(row["helper_input"]).parts).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError(f"Missing or unsafe helper image: {path}")
    with Image.open(path) as source:
        return source.convert("RGB")


def helper_image(root, row, config):
    if config["helper_image"] == "value_heatmap":
        settings = config["value_iteration"]
        return value_heatmap(row["map_desc"], config["helper_size"], settings["gamma"], settings["theta"])
    if config["helper_image"] == "released":
        return released_helper(root, row)
    if config["helper_image"] == "row":  # a dataset's own helper (Zebra-CoT's first reasoning image)
        return row_helper(root, row)
    raise ValueError(f"Unknown helper image: {config['helper_image']}")
