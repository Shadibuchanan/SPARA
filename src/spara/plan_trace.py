"""A written plan before the route (docs/DETOUR_SCALING_SPEC.md, arm p8000): the map's size, holes
and ends as coordinates, the cells at each distance from the goal until the start is reached, and
the path those distances give. Every section is a fixed function of the map layout, so a written
plan is scored section by section against the map's own.

    size 7
    holes: 0,3 1,1 2,5
    start 6,0 goal 0,6
    1: 0,5 1,6
    2: 0,4 2,6
    path: 6,0 U 5,0 R 5,1 ... 0,6
    \\boxed{UP, RIGHT, ...}
"""
from .fresh_maps import GOAL, HOLE, LABEL_ORDER, START, answer_text, find


def sections(desc):
    """The plan's sections by name, in the order they are written. The path is `label_route`'s:
    from the start, the first move in U, L, R, D order that lowers the distance to the goal."""
    size = len(desc)
    start, goal = find(desc, START), find(desc, GOAL)
    distance, levels = {goal: 0}, [[goal]]
    while start not in distance:
        level = []
        for r, c in levels[-1]:
            for _, (dr, dc), _ in LABEL_ORDER:
                cell = (r + dr, c + dc)
                if (0 <= cell[0] < size and 0 <= cell[1] < size and desc[cell[0]][cell[1]] != HOLE
                        and cell not in distance):
                    distance[cell] = len(levels)
                    level.append(cell)
        if not level:
            raise ValueError("The map has no safe route")
        levels.append(level)
    cells = lambda group: " ".join(f"{r},{c}" for r, c in group)
    cell, path, route = start, [], ""
    while cell != goal:
        letter, (dr, dc) = next((m, step) for m, step, _ in LABEL_ORDER
                                if distance.get((cell[0] + step[0], cell[1] + step[1])) == distance[cell] - 1)
        path.append(f"{cells([cell])} {letter}")
        cell, route = (cell[0] + dr, cell[1] + dc), route + letter
    holes = [(r, c) for r in range(size) for c in range(size) if desc[r][c] == HOLE]
    return {"size": f"size {size}", "holes": f"holes: {cells(holes) or '-'}",
            "ends": f"start {cells([start])} goal {cells([goal])}",
            "levels": "\n".join(f"{k}: {cells(level)}" for k, level in enumerate(levels) if k),
            "path": f"path: {' '.join(path)} {cells([goal])}", "route": answer_text(route)}


def with_plan(row):
    """The row with the written plan as its answer."""
    return row | {"text_output": "\n".join(sections(row["map_desc"]).values())}


def read_plan(desc, answer):
    """How a written plan compares with the map's: the first section that differs (None when the
    whole plan is the map's), and the holes it lists against the map's."""
    truth = sections(desc)
    expected = [(name, line) for name, text in truth.items() for line in text.split("\n")]
    lines = [line.strip() for line in answer.strip().split("\n")]
    wrong = next((name for i, (name, line) in enumerate(expected) if i >= len(lines) or lines[i] != line), None)
    listed = next((set(line[len("holes:"):].split()) - {"-"} for line in lines if line.startswith("holes:")), set())
    holes = set(truth["holes"][len("holes:"):].split()) - {"-"}
    return {"first_wrong": wrong, "holes": len(holes), "holes_listed": len(holes & listed),
            "holes_invented": len(listed - holes)}
