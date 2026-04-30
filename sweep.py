from __future__ import annotations
import time
import random
import csv
from typing import Tuple, List

from tqdm import tqdm

from eval import run_match
from policies import random_policy, heuristic_policy, Policy, MCCFRPolicyWrapper
from mccfr import (
    CFRTables,
    AvgStrategyPolicy,
    external_sampling_mccfr,
)
from rihe_engine import RIHEEngine


def eval_two_seats(
    n_hands: int,
    seed_base: int,
    pA: Policy,
    pB: Policy,
    starting_stack: int,
) -> Tuple[float, float]:
    """
    Returns (ev_A, ev_B) in chips/hand, where ev_A is A's seat-averaged EV vs B.
    """
    res1 = run_match(
        n_hands=n_hands,
        seed=seed_base,
        p0=pA,
        p1=pB,
        starting_stack=starting_stack,
        quiet=True,
    )
    ev_A_as_P0 = res1["mean_profit_p0"]

    res2 = run_match(
        n_hands=n_hands,
        seed=seed_base + 1,
        p0=pB,
        p1=pA,
        starting_stack=starting_stack,
        quiet=True,
    )
    ev_B_as_P0 = res2["mean_profit_p0"]
    ev_A_as_P1 = -ev_B_as_P0

    ev_A = 0.5 * (ev_A_as_P0 + ev_A_as_P1)
    ev_B = -ev_A
    return ev_A, ev_B

def continue_train_mccfr(
    tables: CFRTables,
    additional_iterations: int,
    seed: int,
    starting_stack: int = 100_000,
    desc: str = "MCCFR train",
    log_every: int = 0,
) -> CFRTables:
    """
    Continue training an existing MCCFR table for additional_iterations.
    """
    if additional_iterations <= 0:
        return tables

    rng = random.Random(seed)
    iterator = tqdm(
        range(1, additional_iterations + 1),
        total=additional_iterations,
        desc=desc,
        unit="iter",
        leave=False,
    )

    for t in iterator:
        for update_player in (0, 1):
            eng = RIHEEngine(random.Random(rng.randrange(1 << 30)))
            eng.reset(starting_stack)
            eng.deal_hand()

            start_stack0 = eng.stacks[0]

            external_sampling_mccfr(
                tables=tables,
                engine=eng,
                update_player=update_player,
                rng=rng,
                start_stack0=start_stack0,
                reach_p0=1.0,
                reach_p1=1.0,
            )

        if log_every and (t % log_every == 0):
            tqdm.write(
                f"[mccfr] +{t}/{additional_iterations} iters "
                f"| infosets={len(tables.regret_sum)}"
            )

    return tables


def main():
    starting_stack = 100_000
    n_hands = 100_000
    train_seed_base = 0
    policy_seed = 0

    budgets = [
        200,
        500,
        1_000,
        2_000,
        5_000,
        10_000,
        20_000,
        50_000,
        100_000,
        200_000,
        300_000,
        400_000,
        500_000,
        750_000,
        1_000_000,
    ]

    txt_out = "sweep_results.txt"
    csv_out = "sweep_results.csv"

    tables = CFRTables()
    prev_budget = 0

    header = (
        f"{'iters':>10} | {'vs random':>10} | {'vs heuristic':>12} | "
        f"{'infosets':>10} | {'time (s)':>9}"
    )
    rule = "-" * len(header)

    print("\nIncremental MCCFR sweep (seat-avg EV in chips/hand)\n")
    print(header)
    print(rule)

    with open(txt_out, "w") as f:
        f.write("Incremental MCCFR sweep (seat-avg EV in chips/hand)\n\n")
        f.write(header + "\n")
        f.write(rule + "\n")

    with open(csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iters", "vs_random", "vs_heuristic", "infosets", "time_s"])

    outer = tqdm(budgets, desc="Sweep budgets", unit="budget")

    for total_budget in outer:
        delta = total_budget - prev_budget
        outer.set_postfix_str(f"target={total_budget:,}, add={delta:,}")

        t0 = time.time()

        tqdm.write(f"\n[train] continuing to {total_budget:,} iterations (+{delta:,})")

        seg_seed = train_seed_base + total_budget

        continue_train_mccfr(
            tables=tables,
            additional_iterations=delta,
            seed=seg_seed,
            starting_stack=starting_stack,
            desc=f"Train to {total_budget:,}",
            log_every=max(1, delta // 5) if delta >= 5 else 0,
        )

        pol = MCCFRPolicyWrapper(
            AvgStrategyPolicy(tables, rng=random.Random(policy_seed))
        )

        tqdm.write(f"[eval] {total_budget:,} vs random...")
        ev_vs_rand, _ = eval_two_seats(
            n_hands=n_hands,
            seed_base=100,
            pA=pol,
            pB=random_policy,
            starting_stack=starting_stack,
        )

        tqdm.write(f"[eval] {total_budget:,} vs heuristic...")
        ev_vs_heur, _ = eval_two_seats(
            n_hands=n_hands,
            seed_base=200,
            pA=pol,
            pB=heuristic_policy,
            starting_stack=starting_stack,
        )

        dt = time.time() - t0
        infosets = len(tables.regret_sum)

        row = (
            f"{total_budget:10d} | "
            f"{ev_vs_rand:10.3f} | "
            f"{ev_vs_heur:12.3f} | "
            f"{infosets:10d} | "
            f"{dt:9.2f}"
        )

        print(row)

        with open(txt_out, "a") as f:
            f.write(row + "\n")

        with open(csv_out, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([total_budget, ev_vs_rand, ev_vs_heur, infosets, dt])

        prev_budget = total_budget

    print(f"\nDone. Clean table written to {txt_out} and {csv_out}.")

if __name__ == "__main__":
    main()