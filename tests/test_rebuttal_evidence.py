import importlib.util
from pathlib import Path


def test_staged_fair_rl_outcomes_reproduce_the_recorded_comparisons(capsys):
    path = Path(__file__).parents[1] / "scripts" / "verify_fair_rl_outcomes.py"
    spec = importlib.util.spec_from_file_location("verify_fair_rl_outcomes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main() == 0
    printed = capsys.readouterr().out
    assert '"first": [77, 86], "second": [92, 96]' in printed  # C' against B'
    assert '"first": [98, 98], "second": [92, 96]' in printed  # C'' against B'


def test_away_move_analysis_reproduces_the_rebuttal_counts():
    path = Path(__file__).parents[1] / "scripts" / "analyze_away_moves.py"
    spec = importlib.util.spec_from_file_location("analyze_away_moves", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    counts = module.main()
    assert {arm: (c["solved"], c["made_needed_away_moves"]) for arm, c in counts.items()} == {
        "EB3": (83, 189), "e3": (197, 274), "n24000": (273, 322), "p8000": (395, 398)}
