import pickle
import random
from eval import run_match
from mccfr import AvgStrategyPolicy
from policies import random_policy, heuristic_policy, MCCFRPolicyWrapper
from gsi_policy import GSIPolicy

# Load MCCFR
with open("mccfr_checkpoint.pkl", "rb") as f:
    data = pickle.load(f)
tables = data["tables"]
print(f"Loaded MCCFR: {data['iters']:,} iterations, {len(tables.regret_sum):,} infosets")
mccfr = MCCFRPolicyWrapper(AvgStrategyPolicy(tables, rng=random.Random(0)))

gsi = GSIPolicy(java_dir="GSI_extract")

n_hands = 100_000
stack = 100_000

def seat_avg(p0_result, p1_result):
    return 0.5 * (p0_result["mean_profit_p0"] - p1_result["mean_profit_p0"])

print("\nMCCFR vs random:")
r1 = run_match(n_hands, seed=0, p0=mccfr, p1=random_policy, starting_stack=stack)
r2 = run_match(n_hands, seed=1, p0=random_policy, p1=mccfr, starting_stack=stack)
print(f"  Seat-averaged EV: {seat_avg(r1,r2):.3f}")

print("\nMCCFR vs heuristic:")
r3 = run_match(n_hands, seed=2, p0=mccfr, p1=heuristic_policy, starting_stack=stack)
r4 = run_match(n_hands, seed=3, p0=heuristic_policy, p1=mccfr, starting_stack=stack)
print(f"  Seat-averaged EV: {seat_avg(r3,r4):.3f}")

print("\nMCCFR vs GSI:")
r5 = run_match(n_hands, seed=4, p0=mccfr, p1=gsi, starting_stack=stack)
r6 = run_match(n_hands, seed=5, p0=gsi, p1=mccfr, starting_stack=stack)
print(f"  Seat-averaged EV: {seat_avg(r5,r6):.3f}")

gsi.close()