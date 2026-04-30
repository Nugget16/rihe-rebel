import pickle
import random
from eval import run_match
from mccfr import AvgStrategyPolicy
from value_net import ValueNet, get_device
from rebel_policy import RebelPolicy
from gsi_policy import GSIPolicy
from policies import random_policy, heuristic_policy, MCCFRPolicyWrapper

# Load MCCFR
with open("mccfr_checkpoint.pkl", "rb") as f:
    data = pickle.load(f)
tables = data["tables"]
mccfr_pol_raw = AvgStrategyPolicy(tables, rng=random.Random(0))
mccfr_pol = MCCFRPolicyWrapper(mccfr_pol_raw)

# Load ReBeL
device = get_device()
net = ValueNet()
net.load("value_net_v4.pt", device)
rebel = RebelPolicy(
    net=net,
    mccfr_policy=mccfr_pol_raw,
    n_search_iters=10,
    max_depth=3,
)

gsi = GSIPolicy(java_dir="GSI_extract")

n = 10_000
stack = 100_000

def seat_avg(r1, r2):
    return 0.5 * (r1["mean_profit_p0"] - r2["mean_profit_p0"])

print("ReBeL vs random:")
r1 = run_match(n, seed=0, p0=rebel, p1=random_policy, starting_stack=stack)
r2 = run_match(n, seed=1, p0=random_policy, p1=rebel, starting_stack=stack)
print(f"  Seat-averaged EV: {seat_avg(r1,r2):.3f}")

print("\nReBeL vs heuristic:")
r3 = run_match(n, seed=2, p0=rebel, p1=heuristic_policy, starting_stack=stack)
r4 = run_match(n, seed=3, p0=heuristic_policy, p1=rebel, starting_stack=stack)
print(f"  Seat-averaged EV: {seat_avg(r3,r4):.3f}")

print("\nReBeL vs MCCFR:")
r5 = run_match(n, seed=4, p0=rebel, p1=mccfr_pol, starting_stack=stack)
r6 = run_match(n, seed=5, p0=mccfr_pol, p1=rebel, starting_stack=stack)
print(f"  Seat-averaged EV: {seat_avg(r5,r6):.3f}")

print("\nReBeL vs GSI:")
r7 = run_match(n, seed=6, p0=rebel, p1=gsi, starting_stack=stack)
r8 = run_match(n, seed=7, p0=gsi, p1=rebel, starting_stack=stack)
print(f"  Seat-averaged EV: {seat_avg(r7,r8):.3f}")

gsi.close()