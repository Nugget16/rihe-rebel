# eval.py
from __future__ import annotations
import time
import math
import random
from typing import List

from rihe_engine import RIHEEngine
from policies import random_policy, heuristic_policy, Policy, MCCFRPolicyWrapper

from mccfr import train_mccfr, AvgStrategyPolicy

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def stdev(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def ci95_of_mean(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return 1.96 * stdev(xs) / math.sqrt(len(xs))

def play_one_hand(engine: RIHEEngine, p0: Policy, p1: Policy) -> int:
    """
    Returns profit for player 0 for this hand (delta stack0).
    """
    start0 = engine.stacks[0]
    start1 = engine.stacks[1]

    engine.deal_hand()

    while engine.h is not None and not engine.h.terminal:
        if engine.is_chance_node():
            c = engine.rng.choice(engine.chance_outcomes())
            engine.apply_chance(c)
            continue

        to_act = engine.current_player()
        a = p0(engine, 0) if to_act == 0 else p1(engine, 1)
        engine.apply(a)

    end0 = engine.stacks[0]
    end1 = engine.stacks[1]

    assert (end0 - start0) == -(end1 - start1)
    return end0 - start0


def run_match(
    n_hands: int,
    seed: int,
    p0: Policy,
    p1: Policy,
    starting_stack: int = 100_000,
    quiet: bool = False,
) -> dict:
    rng = random.Random(seed)
    engine = RIHEEngine(rng)
    engine.reset(starting_stack)

    profits: List[float] = []
    resets_total = 0
    resets_p0 = 0
    resets_p1 = 0

    t0 = time.time()
    for _ in range(n_hands):
        # If someone can't ante, record who busted and reset.
        if not engine.can_deal():
            if engine.stacks[0] < engine.ANTE:
                resets_p0 += 1
            if engine.stacks[1] < engine.ANTE:
                resets_p1 += 1
            resets_total += 1
            engine.reset(starting_stack)

        prof = play_one_hand(engine, p0, p1)
        profits.append(float(prof))

    t1 = time.time()

    m = mean(profits)
    ci = ci95_of_mean(profits)
    dt = (t1 - t0) if (t1 - t0) > 0 else 1e-9

    result = {
        "seed": seed,
        "hands": n_hands,
        "starting_stack": starting_stack,
        "mean_profit_p0": m,
        "ci95": ci,
        "resets_total": resets_total,
        "resets_p0": resets_p0,
        "resets_p1": resets_p1,
        "runtime_s": (t1 - t0),
        "hands_per_s": (n_hands / dt),
    }

    if not quiet:
        print("=== Match result ===")
        print(f"Seed: {seed}")
        print(f"Hands: {n_hands}")
        print(f"Starting stack (per reset): {starting_stack} each")
        print(f"Bankroll resets: total={resets_total} | p0_busts={resets_p0} | p1_busts={resets_p1}")
        print(f"Mean profit/hand for P0: {m:.4f} chips  (95% CI ± {ci:.4f})")
        print(f"Runtime: {t1 - t0:.2f}s  ({n_hands/dt:.1f} hands/s)")
        print()

    return result


if __name__ == "__main__":
    
    run_match(n_hands=200_000, seed=0, p0=random_policy, p1=heuristic_policy, starting_stack=100_000)
    run_match(n_hands=200_000, seed=1, p0=heuristic_policy, p1=random_policy, starting_stack=100_000)
    
    # tables = train_mccfr(iterations=2000, seed=0, log_every=100)
    # mccfr_pol = MCCFRPolicyWrapper(AvgStrategyPolicy(tables))
 
    # run_match(n_hands=200_000, seed=0, p0=mccfr_pol, p1=heuristic_policy, starting_stack=100_000)
    # run_match(n_hands=200_000, seed=1, p0=heuristic_policy, p1=mccfr_pol, starting_stack=100_000)