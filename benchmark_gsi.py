from eval import run_match
from gsi_policy import GSIPolicy
from policies import random_policy, heuristic_policy

java_dir = "GSI_extract"
gsi = GSIPolicy(java_dir=java_dir)

r1 = run_match(n_hands=100_000, seed=0, p0=gsi, p1=random_policy, starting_stack=100_000)
r2 = run_match(n_hands=100_000, seed=1, p0=random_policy, p1=gsi, starting_stack=100_000)

ev_as_p0 = r1["mean_profit_p0"]
ev_as_p1 = -r2["mean_profit_p0"]
print(f"GSI EV as P0: {ev_as_p0:.3f} ± {r1['ci95']:.3f}")
print(f"GSI EV as P1: {ev_as_p1:.3f} ± {r2['ci95']:.3f}")
print(f"Seat-averaged: {0.5*(ev_as_p0+ev_as_p1):.3f}")
print()
gsi.print_stats()

print("\nGSI vs heuristic:")
r3 = run_match(n_hands=100_000, seed=2, p0=gsi, p1=heuristic_policy, starting_stack=100_000)
r4 = run_match(n_hands=100_000, seed=3, p0=heuristic_policy, p1=gsi, starting_stack=100_000)
ev_as_p0 = r3["mean_profit_p0"]
ev_as_p1 = -r4["mean_profit_p0"]
print(f"GSI EV as P0: {ev_as_p0:.3f} ± {r3['ci95']:.3f}")
print(f"GSI EV as P1: {ev_as_p1:.3f} ± {r4['ci95']:.3f}")
print(f"Seat-averaged: {0.5*(ev_as_p0+ev_as_p1):.3f}")

gsi.close()