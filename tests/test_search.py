import math
import random

import pytest

from spara import search
from spara.fresh_maps import draw_layouts

from spara.route_search import FrozenLakeEnvironment, toward_goal_policy
from spara.search import (ALIVE, DEAD, DONE, STOP, NoisyJudge, fixed_draw, guided_search, renormalise,
                          surviving_route, uniform_policy, verdict_search)
from test_detour_probe import WIDE
from test_planning_probe import Evaluator, GRID

# Start (0, 0), goal (2, 0); holes leave one way round, so every step has one viable move.
CORRIDOR = [[1, 0, 0], [-1, -1, 0], [2, 0, 0]]


def problem(grid, name="map"):
    return {"id": name, "split": "test", "row_index": 0, "row": {"map_id": 0, "map_desc": grid}}


def toward_goal(requests, log=None):
    """LIVR's failure without a model; `log` records each round's batch size."""
    if log is not None:
        log.append(len(requests))
    return toward_goal_policy(requests)


def walk_answer(env, p, actions):
    state = env.start(p)
    for a in actions:
        state, outcome = env.step(p, state, a)
    return outcome


def test_renormalise_sums_to_one():
    values = renormalise({"a": -1.0, "b": -2.0, "c": -50.0})
    assert abs(sum(math.exp(v) for v in values.values()) - 1) < 1e-12 and values["a"] > values["b"]


def test_search_finds_the_detour_a_toward_goal_policy_misses():
    env = FrozenLakeEnvironment(Evaluator())
    [result] = guided_search(env, toward_goal, [problem(WIDE)], budget=64, max_depth=16)
    assert result["solved"] and result["stop"] == "solved" and len(result["actions"]) == 5
    assert walk_answer(env, problem(WIDE), result["actions"]) == DONE
    assert result["answer"] == "\\boxed{" + ", ".join({"U": "UP", "D": "DOWN", "L": "LEFT", "R": "RIGHT"}[a]
                                                      for a in result["actions"]) + "}"
    first = {tuple(t["actions"]): t for t in result["transitions"]}
    assert first[("R",)]["outcomes"]["R"] == DEAD  # the toward-goal move into the hole is pruned
    assert first[()]["outcomes"]["L"] == "repeat"  # off the grid: the agent stays, a repeat
    assert first[("R",)]["scores"]["R"] > first[("R",)]["scores"]["U"]  # the policy still preferred it


def test_forced_moves_need_no_policy_call():
    env = FrozenLakeEnvironment(Evaluator())
    [result] = guided_search(env, lambda requests: pytest.fail("no call expected"), [problem(CORRIDOR)], 64, 16)
    assert result["solved"] and result["policy_calls"] == 0 and result["forced_expansions"] == 6
    assert result["actions"] == list("RRDDLL")


def test_budget_and_depth_cap():
    env = FrozenLakeEnvironment(Evaluator())
    [result] = guided_search(env, toward_goal, [problem(WIDE)], budget=0, max_depth=16)
    assert not result["solved"] and result["stop"] == "budget" and result["policy_calls"] == 0
    [result] = guided_search(env, toward_goal, [problem(WIDE)], budget=64, max_depth=4)  # shortest route is 5
    assert not result["solved"] and result["stop"] == "frontier"
    assert any("depth" in t["outcomes"].values() for t in result["transitions"])


def test_searches_run_in_lockstep_and_match_separate_runs():
    env = FrozenLakeEnvironment(Evaluator())
    problems = [problem(WIDE, "a"), problem(GRID, "b"), problem(CORRIDOR, "c")]
    sizes = []
    together = guided_search(env, lambda r: toward_goal(r, sizes), problems, 64, 16)
    assert max(sizes) == 2  # the corridor needs no calls; the other two share rounds
    for p, result in zip(problems, together):
        [alone] = guided_search(env, toward_goal, [p], 64, 16)
        assert alone["actions"] == result["actions"] and alone["policy_calls"] == result["policy_calls"]


def test_uniform_policy_counts_the_calls_it_would_make():
    env = FrozenLakeEnvironment(Evaluator())
    [result] = guided_search(env, uniform_policy, [problem(WIDE)], 64, 16)
    assert result["solved"] and result["policy_calls"] > 0


def test_policy_scores_must_cover_viable_moves():
    env = FrozenLakeEnvironment(Evaluator())
    with pytest.raises(ValueError, match="finite score"):
        guided_search(env, lambda r: [{"U": 0.0} for _ in r], [problem(WIDE)], 64, 16)


class Counting:
    """A puzzle with no grid and no image: reach `target` from 1 with "+1", "x2" and "-3".
    Below 1 or above twice the target, the attempt is dead. The verifier re-reads the
    answer text."""
    def start(self, target):
        return 1

    def actions(self, target, state):
        return ["x2", "+1", "-3"]

    def step(self, target, state, action):
        value = {"x2": state * 2, "+1": state + 1, "-3": state - 3}[action]
        return value, DEAD if not 1 <= value <= 2 * target else DONE if value == target else ALIVE

    def key(self, state):
        return state

    def answer(self, actions):
        return " ".join(actions)

    def reward(self, target, answer):
        value = 1
        for action in answer.split():
            value, outcome = self.step(target, value, action)
            if outcome == DEAD:
                return 0.0
        return float(value == target)


def test_a_second_environment_with_no_grid():
    doubling = lambda requests: [{c: (0.0 if c == "x2" else -5.0) for c in cands} for _, _, cands in requests]
    results = guided_search(Counting(), doubling, [8, 7, 1000], budget=200, max_depth=40)
    for target, result in zip([8, 7, 1000], results):
        assert result["solved"] and Counting().reward(target, result["answer"]) == 1
    assert results[0]["actions"] == ["x2", "x2", "x2"]  # the policy's preference leads straight there


def test_rejected_answers_do_not_count():
    class Strict(Counting):
        def reward(self, target, answer):
            return 0.0
    [result] = guided_search(Strict(), uniform_policy, [4], budget=50, max_depth=10)
    assert not result["solved"] and result["rejected"] > 0


# Final verdict only -------------------------------------------------------------------

class VerdictOnly(FrozenLakeEnvironment):
    """FrozenLake with the step simulator and states taken away."""
    def step(self, *args):
        pytest.fail("verdict_search must not use the step simulator")

    def key(self, *args):
        pytest.fail("verdict_search must not use states")


def test_verdict_search_uses_only_the_final_verdict():
    env = VerdictOnly(Evaluator())
    [result] = verdict_search(env, toward_goal, [problem(WIDE)], budget=2_000, max_depth=16)
    assert result["solved"] and result["costs_nondecreasing"]
    assert walk_answer(FrozenLakeEnvironment(Evaluator()), problem(WIDE), result["actions"]) == DONE
    assert result["verifier_calls"] == result["rejected"] + 1 == len(result["verdicts"])
    assert result["rejected"] > 0  # the straight routes through the hole are tried and refused first
    assert [v["accepted"] for v in result["verdicts"]].count(True) == 1


def test_verdict_search_verifies_answers_in_order_of_probability():
    env = VerdictOnly(Evaluator())
    [result] = verdict_search(env, uniform_policy, [problem(GRID)], budget=2_000, max_depth=16)
    lengths = [len(v["actions"]) for v in result["verdicts"]]
    assert lengths[0] == 0 and lengths == sorted(lengths)  # uniform: shortest answers first
    assert result["solved"] and len(result["actions"]) == 4 and result["costs_nondecreasing"]
    assert result["policy_calls"] > 1 + 4 + 16 + 64 + 256  # every prefix up to 4 moves is scored first
    assert all(set(e["scores"]) == {"U", "D", "L", "R", STOP} for e in result["expansions"])


def test_verdict_search_budget_depth_and_scores():
    env = VerdictOnly(Evaluator())
    big, = verdict_search(env, toward_goal, [problem(WIDE)], budget=2_000, max_depth=16)
    small, = verdict_search(env, toward_goal, [problem(WIDE)], budget=big["policy_calls"] - 1, max_depth=16)
    assert not small["solved"] and small["stop"] == "budget"
    assert small["verdicts"] == big["verdicts"][:len(small["verdicts"])]  # the budget only truncates
    capped, = verdict_search(env, toward_goal, [problem(WIDE)], budget=2_000, max_depth=3)
    assert not capped["solved"] and capped["stop"] == "frontier"  # the shortest safe route has 5 moves
    assert max(len(v["actions"]) for v in capped["verdicts"]) == 3
    assert max(len(e["actions"]) for e in capped["expansions"]) == 2  # capped prefixes end with no call
    with pytest.raises(ValueError, match="ending included"):
        verdict_search(env, lambda r: [{c: 0.0 for c in cands if c != STOP} for _, _, cands in r],
                       [problem(WIDE)], 10, 16)


def test_verdict_search_on_the_puzzle_with_no_grid():
    doubling = lambda requests: [{c: (0.0 if c == "x2" else -5.0) for c in cands} for _, _, cands in requests]
    results = verdict_search(Counting(), doubling, [8, 7], budget=2_000, max_depth=12)
    assert all(r["solved"] and Counting().reward(t, r["answer"]) == 1 for t, r in zip([8, 7], results))
    assert results[0]["actions"] == ["x2", "x2", "x2"]


# Noisy judges -------------------------------------------------------------------------

def test_fixed_draws_repeat_and_spread():
    assert fixed_draw(0, "map", (1, 2), "U") == fixed_draw(0, "map", (1, 2), "U")
    draws = [fixed_draw(seed, "map") for seed in range(2_000)]
    assert all(0 <= d < 1 for d in draws) and 0.45 < sum(d < 0.5 for d in draws) / len(draws) < 0.55


def judge(miss, false_alarm, seed=0):
    return NoisyJudge(FrozenLakeEnvironment(Evaluator()), miss, false_alarm, seed, lambda p: p["id"])


def outcomes(env, p):
    cells = [(r, c) for r in range(len(WIDE)) for c in range(len(WIDE))]
    return {(cell, a): env.step(p, cell, a)[1] for cell in cells for a in "UDLR"}


def test_noisy_judge_rates_and_consistency():
    p, exact = problem(WIDE), outcomes(FrozenLakeEnvironment(Evaluator()), problem(WIDE))
    assert outcomes(judge(0, 0), p) == exact
    flipped = outcomes(judge(1, 0), p)
    assert all(flipped[k] == (ALIVE if v == DEAD else v) for k, v in exact.items())  # every hole missed
    alarmed = outcomes(judge(0, 1), p)
    assert all(alarmed[k] == (DEAD if v == ALIVE else v) for k, v in exact.items())  # arriving is never misjudged
    assert outcomes(judge(0.3, 0.3, seed=4), p) == outcomes(judge(0.3, 0.3, seed=4), p)
    errors = lambda rate: {k for k, v in outcomes(judge(rate, rate, seed=4), p).items() if v != exact[k]}
    assert errors(0.1) <= errors(0.3) <= errors(0.6) and errors(0.6)  # nested across rates
    assert judge(0.2, 0.2).reward(p, "\\boxed{RIGHT, RIGHT}") == 0.0  # the verifier is the real one


def test_zero_noise_reproduces_the_exact_search():
    problems = [problem(WIDE, "a"), problem(GRID, "b"), problem(CORRIDOR, "c")]
    exact = guided_search(FrozenLakeEnvironment(Evaluator()), toward_goal, problems, 64, 16)
    noisy = guided_search(judge(0, 0), toward_goal, problems, 64, 16)
    assert [(r["actions"], r["policy_calls"]) for r in exact] == [(r["actions"], r["policy_calls"]) for r in noisy]


def test_a_judges_search_order_does_not_depend_on_the_budget():
    p = problem(WIDE)
    for seed in range(6):
        big, = guided_search(judge(0.3, 0.2, seed), uniform_policy, [p], 1_000, 16)
        small, = guided_search(judge(0.3, 0.2, seed), uniform_policy, [p], 3, 16)
        seen = lambda r: [(t["actions"], t["outcomes"]) for t in r["transitions"]]
        assert seen(small) == seen(big)[:len(seen(small))]  # the budget only truncates
        assert small["solved"] == (big["solved"] and big["policy_calls"] <= 3)


# Start (0, 0), goal (2, 2). The only safe route is D, R, R, D; the hole at (0, 1) is the
# way the policy prefers.
TRAP = [[1, -1, 0], [0, 0, 0], [-1, -1, 2]]


class MissesTheTrap(FrozenLakeEnvironment):
    def step(self, p, state, action):
        child, outcome = super().step(p, state, action)
        return child, ALIVE if child == (0, 1) else outcome


def test_a_missed_hole_can_poison_the_closed_set():
    prefers_right = lambda r: [{c: {"R": 0.0, "D": -1.0}.get(c, -5.0) for c in cands} for _, _, cands in r]
    env, p = FrozenLakeEnvironment(Evaluator()), problem(TRAP)
    [result] = guided_search(MissesTheTrap(Evaluator()), prefers_right, [p], 64, 16)
    assert not result["solved"] and result["stop"] == "frontier" and result["rejected"] == 1
    assert surviving_route(env, MissesTheTrap(Evaluator()), p, 16)  # poisoned, not blocked
    assert guided_search(env, prefers_right, [p], 64, 16)[0]["actions"] == list("DRRD")
    assert surviving_route(env, env, p, 16) and not surviving_route(env, judge(0, 1), p, 16)  # blocked


# Tolerating a noisy judge (step 1a″) ----------------------------------------------------

def test_reopening_changes_nothing_under_an_exact_judge():
    rng = random.Random(3)
    problems = [problem(WIDE, "a"), problem(GRID, "b"), problem(CORRIDOR, "c")]
    problems += [problem(desc, f"r{n}-{i}") for n in (4, 5, 6) for i, desc in enumerate(draw_layouts(n, 4, rng, set()))]
    env = FrozenLakeEnvironment(Evaluator())
    for policy in (toward_goal, uniform_policy):
        plain = guided_search(env, policy, problems, 64, 36)
        again = guided_search(env, policy, problems, 64, 36, reopen=True)
        assert [(r["actions"], r["policy_calls"], r["expansions"]) for r in plain] == \
               [(r["actions"], r["policy_calls"], r["expansions"]) for r in again]
        assert guided_search(env, policy, problems, 64, 36, veto=math.inf) == plain


def test_reopening_undoes_a_poisoned_closed_set():
    prefers_right = lambda r: [{c: {"R": 0.0, "D": -1.0}.get(c, -5.0) for c in cands} for _, _, cands in r]
    [result] = guided_search(MissesTheTrap(Evaluator()), prefers_right, [problem(TRAP)], 64, 16, reopen=True)
    assert result["solved"] and result["actions"] == list("DRRD")
    assert result["rejected"] >= 1 and result["reopened"] > 0


def test_a_soft_veto_recovers_from_false_alarms():
    env, p = FrozenLakeEnvironment(Evaluator()), problem(WIDE)
    [hard] = guided_search(judge(0, 1), toward_goal, [p], 64, 16)
    assert not hard["solved"] and hard["stop"] == "frontier"  # every safe move vetoed
    for veto in (1.0, 3.0, 10.0):
        [soft] = guided_search(judge(0, 1), toward_goal, [p], 64, 16, veto=veto)
        assert soft["solved"] and walk_answer(env, p, soft["actions"]) == DONE
    [exact_soft] = guided_search(env, toward_goal, [p], 64, 16, veto=10.0)
    [exact] = guided_search(env, toward_goal, [p], 64, 16)
    assert exact_soft["actions"] == exact["actions"]  # judged-viable moves keep their order


def test_robust_search_order_does_not_depend_on_the_budget():
    p = problem(WIDE)
    for seed in range(6):
        big, = guided_search(judge(0.3, 0.3, seed), uniform_policy, [p], 1_000, 16, veto=3.0, reopen=True)
        small, = guided_search(judge(0.3, 0.3, seed), uniform_policy, [p], 3, 16, veto=3.0, reopen=True)
        seen = lambda r: [(t["actions"], t["outcomes"]) for t in r["transitions"]]
        assert seen(small) == seen(big)[:len(seen(small))]
        assert small["solved"] == (big["solved"] and big["policy_calls"] <= 3)


def test_the_expansion_guard_stops_a_search():
    p = problem(WIDE)
    [result] = search._lockstep([search._search(judge(0, 1), p, 1_000, 16, 1.0, True, max_expansions=3)],
                                uniform_policy, [p])
    assert result["stop"] == "expansions" and result["expansions"] == 3


def test_the_carried_setting_reports_a_tie_and_breaks_it_by_the_whole_curve():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("search_sweep", Path(__file__).parents[1] / "scripts/search_sweep.py")
    sweep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep)

    def variant(eps, median, curve):
        rates = ["0", "0.01", "0.05"]
        return {"priors": {"toward_goal": {
            "exact": {"policy_calls": {"median": median}},
            "conditions": {"transition/both": {
                "epsilon_90": {"1000": eps},
                "curve": {r: {"solved_within": {"1000": {"mean": m}}} for r, m in zip(rates, curve)}}}}}}
    report = {"a": variant(0.05, 6, [97, 95, 89]), "b": variant(0.05, 6, [97, 96, 90]),
              "c": variant(0.05, 7, [97, 97, 97]), "d": variant(0.01, 5, [97, 97, 80])}
    carried = sweep.carried(report)
    assert carried["tied_on_rule"] == ["a", "b"] and carried["variant"] == "b" and "tiebreak" in carried
    assert sweep.carried({"a": report["a"], "c": report["c"]})["variant"] == "a"  # no tie: the rule alone
    crossed = {"a": variant(0.05, 6, [97, 97, 80]), "b": variant(0.05, 6, [97, 90, 90])}
    assert sweep.carried(crossed)["variant"] is None  # neither curve dominates
