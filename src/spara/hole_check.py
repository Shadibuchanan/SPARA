"""The hole check (docs/HOLE_CHECK_SPEC.md): before each move, the answer names the agent's
cell and the directions from it that lead into a hole, then the move.

    3,2 hole ULR > D
    4,2 hole - > L
    \\boxed{DOWN, LEFT}

Lines follow the reference rules: a hole ends the route, the goal finishes it, and a move
off the grid leaves the agent where it is. Maps are square, as in VSP.
"""
import re
from collections import Counter

from mathruler.grader import extract_boxed_content

from .latent_rl import moves

MOVES = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
LINE = re.compile(r"(\d+),(\d+) hole ([UDLR]+|-) > ([UDLR])")
START, HOLE = 1, -1


def _start(desc):
    return next((r, c) for r, line in enumerate(desc) for c, value in enumerate(line) if value == START)


def _step(desc, cell, letter):
    r, c = cell[0] + MOVES[letter][0], cell[1] + MOVES[letter][1]
    return (r, c) if 0 <= r < len(desc) and 0 <= c < len(desc) else cell


def holes_around(desc, cell):
    """The directions from `cell` whose move lands in a hole, in U, D, L, R order."""
    return "".join(d for d in MOVES if (to := _step(desc, cell, d)) != cell and desc[to[0]][to[1]] == HOLE)


def check_lines(desc, route):
    """One line per move of `route` (letters), until the route falls into a hole."""
    cell, lines = _start(desc), []
    for letter in route:
        lines.append(f"{cell[0]},{cell[1]} hole {holes_around(desc, cell) or '-'} > {letter}")
        cell = _step(desc, cell, letter)
        if desc[cell[0]][cell[1]] == HOLE:
            break
    return lines


def with_hole_check(row):
    """A training row with the hole check written in front of its released-format answer."""
    route = moves(extract_boxed_content(row["text_output"]))
    return row | {"text_output": "".join(line + "\n" for line in check_lines(row["map_desc"], route))
                  + row["text_output"]}


def parse(answer):
    """(cell, holes, move) for each line before the last `\\boxed{`, or None for a line that
    does not follow the format."""
    head = answer[:answer.rfind("\\boxed{")] if "\\boxed{" in answer else answer
    parsed = []
    for text in head.strip().splitlines():
        match = LINE.fullmatch(text.strip())
        parsed.append(match and ((int(match[1]), int(match[2])), "" if match[3] == "-" else match[3], match[4]))
    return parsed


def read(desc, answer, route):
    """What an answer's lines say against the map, for the route the evaluator scored:
    whether the lines follow the format and replay to the route; per line, whether the cell
    and the hole set are right; and, if the route falls, the class of its fatal move."""
    lines = parse(answer)
    # A line per move, either to the end of the route or to the fall, with the route's moves.
    follows = (all(lines) and len(lines) in (len(route), len(check_lines(desc, route)))
               and [line[2] for line in lines] == list(route[:len(lines)]))
    cell, per_line, fatal = _start(desc), [], None
    for k, letter in enumerate(route):
        line = lines[k] if k < len(lines) else None
        truth = holes_around(desc, cell)
        if line:
            per_line.append({"cell": line[0] == cell, "holes": set(line[1]) == set(truth),
                             "missed": len(set(truth) - set(line[1])), "invented": len(set(line[1]) - set(truth))})
        to = _step(desc, cell, letter)
        if desc[to[0]][to[1]] == HOLE:
            if not line or line[2] != letter:
                fatal = "no line"
            elif line[0] != cell:
                fatal = "lost"
            elif letter in line[1]:
                fatal = "listed and taken"
            else:
                fatal = "missed"
            break
        cell = to
    return {"follows": follows, "lines": per_line, "fatal": fatal}


def line_reading(answers):
    """How the lines of `answers` (map, answer, scored route) read their maps. Walks each
    route with its lines: the first line's offset from the start (row,column); on lines
    naming the right cell, the hole directions listed against those that lead into a hole,
    with writing `-` every time as the baseline, by direction and by first or later line; and,
    on lines naming the wrong cell, whether the listed set is the true cell's or the named one's."""
    count, offsets, total = Counter(), Counter(), 0
    by_direction = {d: Counter(holes=0, listed=0, safe=0, invented=0) for d in MOVES}
    by_line = {position: Counter(holes=0, listed=0) for position in ("first", "later")}
    for desc, answer, route in answers:
        total += 1
        lines, cell = parse(answer), _start(desc)
        if lines and lines[0]:
            offsets[f"{lines[0][0][0] - cell[0]},{lines[0][0][1] - cell[1]}"] += 1
        for k, letter in enumerate(route):
            if k >= len(lines) or not lines[k]:
                break
            named, listed, _ = lines[k]
            truth = holes_around(desc, cell)
            if named == cell:
                count["right_cell"] += 1
                count["right_cell_exact"] += set(listed) == set(truth)
                count["right_cell_dash_baseline"] += truth == ""
                count["hole_directions"] += len(truth)
                count["hole_directions_listed"] += len(set(truth) & set(listed))
                count["directions_invented"] += len(set(listed) - set(truth))
                for d in MOVES:
                    by_direction[d]["holes" if d in truth else "safe"] += 1
                    by_direction[d]["listed" if d in truth else "invented"] += d in listed
                position = by_line["first" if k == 0 else "later"]
                position["holes"] += len(truth)
                position["listed"] += len(set(truth) & set(listed))
            else:
                on_grid = 0 <= named[0] < len(desc) and 0 <= named[1] < len(desc)
                count["wrong_cell"] += 1
                count["wrong_cell_matches_true_cell"] += set(listed) == set(truth)
                count["wrong_cell_matches_named_cell"] += on_grid and set(listed) == set(holes_around(desc, named))
            to = _step(desc, cell, letter)
            if desc[to[0]][to[1]] == HOLE:
                break
            cell = to
    return {"answers": total, "first_line_offset": dict(offsets.most_common()), "lines": dict(count),
            "by_direction": {d: dict(c) for d, c in by_direction.items()},
            "by_line": {position: dict(c) for position, c in by_line.items()}}
