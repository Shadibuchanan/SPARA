import pytest
from mathruler.grader import extract_boxed_content

from spara.expert_round2 import replayed
from spara.hole_check import check_lines, holes_around, line_reading, parse, read, with_hole_check

# S . H
# . H .
# . . G
DESC = [[1, 0, -1], [0, -1, 0], [0, 0, 2]]


def test_lines_name_the_cell_and_the_holes_around_it():
    assert check_lines(DESC, "DDRR") == ["0,0 hole - > D", "1,0 hole R > D", "2,0 hole - > R", "2,1 hole U > R"]


def test_a_move_off_the_grid_repeats_the_cell_and_a_fall_ends_the_lines():
    assert check_lines(DESC, "LD") == ["0,0 hole - > L", "0,0 hole - > D"]
    assert check_lines(DESC, "DRRR") == ["0,0 hole - > D", "1,0 hole R > R"]
    assert holes_around(DESC, (0, 1)) == "DR"


def test_the_check_leads_the_released_answer_and_scoring_reads_the_box():
    row = {"map_desc": DESC, "text_output": "\\boxed{DOWN, DOWN, RIGHT, RIGHT}"}
    text = with_hole_check(row)["text_output"]
    assert text == ("0,0 hole - > D\n1,0 hole R > D\n2,0 hole - > R\n2,1 hole U > R\n"
                    "\\boxed{DOWN, DOWN, RIGHT, RIGHT}")
    assert extract_boxed_content(text) == "DOWN, DOWN, RIGHT, RIGHT"
    assert parse(text)[1] == ((1, 0), "R", "D")


def test_a_right_answer_follows_and_reads_right():
    text = with_hole_check({"map_desc": DESC, "text_output": "\\boxed{DOWN, DOWN, RIGHT, RIGHT}"})["text_output"]
    result = read(DESC, text, "DDRR")
    assert result["follows"] and result["fatal"] is None
    assert all(line["cell"] and line["holes"] for line in result["lines"]) and len(result["lines"]) == 4


@pytest.mark.parametrize("lines, fatal, missed", [
    ("0,0 hole - > D\n1,0 hole R > R", "listed and taken", 0),
    ("0,0 hole - > D\n1,0 hole - > R", "missed", 1),
    ("0,0 hole - > D\n2,0 hole - > R", "lost", None),
    ("0,0 hole - > D\nright", "no line", None),
])
def test_the_line_before_the_fall_classes_the_fatal_move(lines, fatal, missed):
    result = read(DESC, lines + "\n\\boxed{DOWN, RIGHT}", "DR")
    assert result["fatal"] == fatal
    assert result["follows"] == (fatal != "no line")
    if missed is not None:
        assert result["lines"][1]["missed"] == missed and result["lines"][1]["cell"]


def test_lines_past_the_fall_still_follow_but_a_bare_box_does_not():
    assert read(DESC, "0,0 hole - > D\n1,0 hole R > R\n1,1 hole - > R\n\\boxed{D, R, R}", "DRR")["follows"]
    bare = read(DESC, "\\boxed{DOWN, RIGHT}", "DR")
    assert not bare["follows"] and bare["fatal"] == "no line"


def test_training_rows_that_do_not_replay_stop_the_arm():
    good = with_hole_check({"map_desc": DESC, "text_output": "\\boxed{DOWN, DOWN, RIGHT, RIGHT}", "map_id": 0})
    assert replayed([("train", good)]) == {"rows": 1, "replay": 1}
    bad = good | {"text_output": good["text_output"].replace("1,0 hole R", "1,0 hole -"), "map_id": 1}
    with pytest.raises(ValueError, match="do not replay"):
        replayed([("train", good), ("train", bad)])


def test_line_reading_counts_the_start_and_the_holes_listed():
    right = "0,0 hole - > D\n1,0 hole R > D\n2,0 hole - > R\n2,1 hole U > R\n\boxed{}"
    blind = "0,0 hole - > D\n1,0 hole - > D\n2,0 hole U > R\n\boxed{}"
    offset = "0,1 hole DR > D\n1,1 hole R > D\n\boxed{}"
    result = line_reading([(DESC, right, "DDRR"), (DESC, blind, "DDR"), (DESC, offset, "DD")])
    assert result["answers"] == 3 and result["first_line_offset"] == {"0,0": 2, "0,1": 1}
    lines = result["lines"]
    assert lines["right_cell"] == 7 and lines["right_cell_exact"] == 5 and lines["right_cell_dash_baseline"] == 4
    assert lines["hole_directions"] == 3 and lines["hole_directions_listed"] == 2 and lines["directions_invented"] == 1
    # The offset answer: its first line lists the named cell's holes, its second the true cell's.
    assert lines["wrong_cell"] == 2 and lines["wrong_cell_matches_named_cell"] == 1
    assert lines["wrong_cell_matches_true_cell"] == 1
    # Right-cell lines: the hole to the right of 1,0 is listed once of twice; the one above 2,1 is listed once.
    assert result["by_direction"]["R"] == {"holes": 2, "listed": 1, "safe": 5, "invented": 0}
    assert result["by_direction"]["U"] == {"holes": 1, "listed": 1, "safe": 6, "invented": 1}
    assert result["by_line"] == {"first": {"holes": 0, "listed": 0}, "later": {"holes": 3, "listed": 2}}
