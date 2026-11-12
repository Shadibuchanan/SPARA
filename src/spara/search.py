"""Policy-guided search with step-simulator pruning, for any search environment
(docs/GUIDED_SEARCH_SPEC.md).

A search environment says which actions exist, what each does, and whether a finished
answer is accepted. A policy gives log-probabilities of candidate actions after a
prefix, for a batch of requests. The search expands prefixes best first by the policy's
probability renormalised over the children the simulator leaves viable. It records every
transition it sees. Nothing here knows about grids, images or models.

Two harder settings (step 1a′): `verdict_search` uses only the verifier's accept or
reject, with no step simulator; `NoisyJudge` replaces the simulator's judgments with ones
that err at set rates.
"""
import hashlib
import heapq
import itertools
import math
from typing import Hashable, Protocol, Sequence

ALIVE, DEAD, DONE = "alive", "dead", "done"
# Logged outcomes for viable children the search drops: state already expanded, or past the cap.
REPEAT, DEPTH = "repeat", "depth"
STOP = "<end>"  # verdict_search's extra candidate: end the answer here


class Environment(Protocol):
    def start(self, problem): ...

    def actions(self, problem, state) -> Sequence[str]: ...

    def step(self, problem, state, action) -> tuple:
        """(next state, ALIVE | DEAD | DONE)."""

    def key(self, state) -> Hashable: ...

    def answer(self, actions) -> str: ...

    def reward(self, problem, answer) -> float:
        """The authoritative verifier; 1 accepts the answer."""


class Policy(Protocol):
    def __call__(self, requests):
        """[(problem, actions so far, candidates)] -> [{candidate: log-probability}]."""


def uniform_policy(requests):
    """Every candidate scored equally: the search without a learned prior."""
    return [{c: 0.0 for c in candidates} for _, _, candidates in requests]


def renormalise(scores):
    """Log-probabilities over the given candidates, summing to one."""
    top = max(scores.values())
    total = top + math.log(sum(math.exp(v - top) for v in scores.values()))
    return {a: v - total for a, v in scores.items()}


def _search(env, problem, budget, max_depth, veto=math.inf, reopen=False, max_expansions=50_000):
    """One problem's search, as a generator. It yields (actions, candidates) whenever it
    needs the policy, receives the scores, and returns the result.

    `veto` is the cost in nats of a move the judge calls dead: infinite drops it (the
    default); finite defers it, and the search walks on from where the move led.
    `reopen` re-opens the states a refused answer's route closed and re-expands the
    parents whose children were dropped as repeats of them (step 1a″)."""
    counter = itertools.count()
    soft = math.isfinite(veto)
    frontier = [(0.0, next(counter), (), env.start(problem), False)]  # last: re-expand a closed state
    closed, closer, parked, transitions = {}, {}, {}, []
    calls = forced = rejected = verified = reopened = 0
    solution, stop = None, "frontier"
    while frontier:
        if len(transitions) >= max_expansions:
            stop = "expansions"
            break
        cost, _, actions, state, again = heapq.heappop(frontier)
        key = env.key(state)
        if key in closed and not again:
            continue
        if key not in closed:  # a re-expanded parent whose own state was re-opened is closed again
            closed[key], closer[actions] = actions, key
        children = {}
        for action in env.actions(problem, state):
            child, outcome = env.step(problem, state, action)
            if outcome not in (ALIVE, DEAD, DONE):
                raise ValueError(f"Unknown step outcome {outcome!r}")
            walks_on = outcome == ALIVE or (soft and outcome == DEAD)
            if (outcome != DEAD or soft) and env.key(child) in closed:
                outcome = REPEAT
            elif walks_on and len(actions) + 1 >= max_depth:
                outcome = DEPTH  # an unfinished route at the cap cannot finish within it
            if outcome == REPEAT and reopen and env.key(child) != key:
                parked.setdefault(env.key(child), []).append((cost, actions, state))
            children[action] = (child, outcome)
        viable = [a for a, (_, o) in children.items() if o in (ALIVE, DONE)]
        deferred = [a for a, (_, o) in children.items() if o == DEAD] if soft else []
        record = {"actions": list(actions), "outcomes": {a: o for a, (_, o) in children.items()}, "scores": None}
        transitions.append(record)
        scores = None
        if len(viable) == 1:
            forced += 1
            logp = {viable[0]: 0.0}
        elif viable:
            if calls >= budget:
                stop = "budget"
                break
            calls += 1
            scores = yield list(actions), list(children)
            if any(a not in scores or not math.isfinite(scores[a]) for a in viable + deferred):
                raise ValueError("The policy must give a finite score to every viable candidate")
            record["scores"] = {a: float(scores[a]) for a in children if a in scores}
            logp = renormalise({a: scores[a] for a in viable})
        else:
            logp = {}
        if deferred:
            pool = viable + deferred
            q = renormalise({a: scores[a] for a in pool}) if scores else {a: -math.log(len(pool)) for a in pool}
            for action in deferred:
                heapq.heappush(frontier, (cost + veto - q[action], next(counter), actions + (action,),
                                          children[action][0], False))
        for action in sorted(viable, key=lambda a: -logp[a]):
            child, outcome = children[action]
            path = actions + (action,)
            if outcome == DONE:
                verified += 1
                if env.reward(problem, env.answer(path)) >= 1:
                    solution = {"actions": list(path), "log_likelihood": logp[action] - cost}
                    break
                rejected += 1
                if reopen:
                    for i in range(1, len(path)):
                        state_key = closer.get(path[:i])
                        if state_key is not None and closed.get(state_key) == path[:i]:
                            del closed[state_key]
                            reopened += 1
                            for parent in parked.pop(state_key, []):
                                heapq.heappush(frontier, (parent[0], next(counter), parent[1], parent[2], True))
            else:
                heapq.heappush(frontier, (cost - logp[action], next(counter), path, child, False))
        if solution:
            stop = "solved"
            break
    return {"solved": solution is not None, "actions": solution and solution["actions"],
            "answer": solution and env.answer(solution["actions"]),
            "log_likelihood": solution and solution["log_likelihood"], "stop": stop,
            "policy_calls": calls, "verifier_calls": verified, "expansions": len(transitions),
            "forced_expansions": forced, "rejected": rejected, "reopened": reopened, "transitions": transitions}


def guided_search(env, policy, problems, budget, max_depth, veto=math.inf, reopen=False, max_expansions=50_000):
    """Search every problem; the searches advance in lockstep, so each round sends all
    pending expansions to the policy in one call. `budget` counts policy calls per problem.
    `veto` and `reopen` make the search tolerate a noisy judge (see `_search`);
    `max_expansions` caps the work when expansions are costly even without a call."""
    return _lockstep([_search(env, p, budget, max_depth, veto, reopen, max_expansions) for p in problems],
                     policy, problems)


def _lockstep(searches, policy, problems):
    results, pending = [None] * len(problems), {}

    def advance(i, value):
        try:
            pending[i] = searches[i].send(value)
        except StopIteration as stop:
            results[i] = stop.value

    for i in range(len(searches)):
        advance(i, None)
    while pending:
        requests, pending = pending, {}
        replies = policy([(problems[i], *requests[i]) for i in requests])
        if len(replies) != len(requests):
            raise ValueError("The policy must answer every request")
        for i, scores in zip(requests, replies):
            advance(i, scores)
    return results


# Final verdict only ---------------------------------------------------------------------

def _verdict_search(env, problem, budget, max_depth, log):
    """One problem's search with no step simulator: only `actions` (asked with state None,
    so the action set must not depend on the state), `answer` and `reward`. Complete
    answers are verified best first by their probability under the policy, which also
    scores STOP. A popped prefix costs a policy call; a popped ended answer costs a
    verifier call; a prefix at the depth cap is ended without a call."""
    counter = itertools.count()
    frontier = [(0.0, next(counter), False, ())]  # (cost, order, ended, actions)
    calls = verified = 0
    expansions, verdicts = [], []
    previous, nondecreasing = 0.0, True
    solution, stop = None, "frontier"
    while frontier:
        cost, _, ended, actions = heapq.heappop(frontier)
        nondecreasing &= cost >= previous - 1e-9
        previous = cost
        if ended:
            verified += 1
            accepted = env.reward(problem, env.answer(actions)) >= 1
            if log:
                verdicts.append({"actions": list(actions), "accepted": accepted})
            if accepted:
                solution, stop = {"actions": list(actions), "log_likelihood": -cost}, "solved"
                break
            continue
        if calls >= budget:
            stop = "budget"
            break
        calls += 1
        candidates = [*env.actions(problem, None), STOP]
        scores = yield list(actions), candidates
        if any(c not in scores or not math.isfinite(scores[c]) for c in candidates):
            raise ValueError("The policy must give a finite score to every candidate, ending included")
        if log:
            expansions.append({"actions": list(actions), "scores": {c: float(scores[c]) for c in candidates}})
        logp = renormalise({c: scores[c] for c in candidates})
        heapq.heappush(frontier, (cost - logp[STOP], next(counter), True, actions))
        for action in candidates[:-1]:
            path = actions + (action,)
            heapq.heappush(frontier, (cost - logp[action], next(counter), len(path) >= max_depth, path))
    return {"solved": solution is not None, "actions": solution and solution["actions"],
            "answer": solution and env.answer(solution["actions"]),
            "log_likelihood": solution and solution["log_likelihood"], "stop": stop,
            "policy_calls": calls, "verifier_calls": verified, "rejected": verified - (solution is not None),
            "costs_nondecreasing": nondecreasing, "expansions": expansions, "verdicts": verdicts}


def verdict_search(env, policy, problems, budget, max_depth, log=True):
    """Final-verdict-only search over every problem, in lockstep like `guided_search`. It is
    the de-duplicated best case of sampling with a final check. `log=False` drops the
    per-expansion records, for large budgets."""
    return _lockstep([_verdict_search(env, p, budget, max_depth, log) for p in problems], policy, problems)


# Noisy judges ---------------------------------------------------------------------------

def fixed_draw(*parts):
    """A number in [0, 1) fixed by `parts`: the same query always gets the same draw."""
    digest = hashlib.blake2b("|".join(map(repr, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


class NoisyJudge:
    """An environment whose step judgments err per transition: a dead move is judged alive
    with probability `miss` (the search walks on from where the move led), an alive move
    judged dead with probability `false_alarm`. Arriving (done) is judged correctly. Each
    (problem, state, action) has one fixed draw per seed, so asking again gives the same
    judgment and the errors at a rate include those at every smaller rate. The verifier
    is the real one."""

    def __init__(self, env, miss, false_alarm, seed, problem_key):
        self.env, self.miss, self.false_alarm, self.seed, self.problem_key = env, miss, false_alarm, seed, problem_key

    def __getattr__(self, name):  # start, actions, key, answer and reward are the real ones
        return getattr(self.env, name)

    def step(self, problem, state, action):
        child, outcome = self.env.step(problem, state, action)
        rate = self.miss if outcome == DEAD else self.false_alarm if outcome == ALIVE else 0.0
        if rate and fixed_draw(self.seed, self.problem_key(problem), self.env.key(state), action) < rate:
            outcome = ALIVE if outcome == DEAD else DEAD
        return child, outcome


def surviving_route(env, judge, problem, max_depth):
    """Whether some route the real step simulator accepts is judged viable at every step
    by `judge`, within `max_depth` moves (breadth first). When a search with that judge
    empties its frontier anyway, a state on the route was first reached through a missed
    dead end and closed: the search's error, not the judge's."""
    layer, seen = [env.start(problem)], {env.key(env.start(problem))}
    for _ in range(max_depth):
        following = []
        for state in layer:
            for action in env.actions(problem, state):
                child, true = env.step(problem, state, action)
                if DEAD in (true, judge.step(problem, state, action)[1]):
                    continue
                if true == DONE:
                    return True
                if env.key(child) not in seen:
                    seen.add(env.key(child))
                    following.append(child)
        layer = following
    return False
