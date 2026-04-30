from rihe_engine import RIHEEngine
from policies import random_policy
import random

e = RIHEEngine(random.Random(0))
for i in range(5):
    e.reset(500)
    e.deal_hand()
    while not e.h.terminal:
        a = e.random_action()
        e.apply(a)

    print("\n".join(e.h.history))
    print("\nHAND OVER.\n")
