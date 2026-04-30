import pickle
import random
from rihe_engine import RIHEEngine, card_str
from mccfr import AvgStrategyPolicy
from value_net import ValueNet, get_device
from rebel_policy import RebelPolicy

# Load MCCFR
with open("mccfr_checkpoint.pkl", "rb") as f:
    data = pickle.load(f)
tables = data["tables"]
mccfr_pol = AvgStrategyPolicy(tables, rng=random.Random(0))

# Load value net
device = get_device()
net = ValueNet()
net.load("value_net.pt", device)

# Build ReBeL policy
rebel = RebelPolicy(
    net=net,
    mccfr_policy=mccfr_pol,
    n_search_iters=50,
    max_depth=2,
)

# Play 3 hands
rng = random.Random(42)
engine = RIHEEngine(rng)
engine.reset(1000)

for hand in range(3):
    engine.deal_hand()
    print(f"\n--- Hand {hand+1} ---")
    print(f"P0 hole: {card_str(engine.h.hole[0])}")

    while not engine.h.terminal:
        if engine.is_chance_node():
            c = rng.choice(engine.chance_outcomes())
            engine.apply_chance(c)
            print(f"  Dealt: {card_str(c)}")
            continue

        player = engine.current_player()
        legal = engine.legal_actions()

        if player == 0:
            action = rebel(engine, 0)
            print(f"  P0 (ReBeL) legal={legal} -> {action}")
        else:
            action = mccfr_pol.sample_action(engine, 1)
            print(f"  P1 (MCCFR) -> {action}")

        engine.apply(action)

    print(f"  Result: {engine.h.terminal_reason}")
