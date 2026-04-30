import random
from rihe_engine import RIHEEngine, card_str
from gsi_policy import GSIPolicy
from policies import random_policy

gsi = GSIPolicy(java_dir="GSI_extract", debug=True)
rng = random.Random(42)
engine = RIHEEngine(rng)
engine.reset(1000)

for hand in range(3):
    engine.deal_hand()
    print(f"\n--- Hand {hand+1} ---")
    print(f"P0 hole: {card_str(engine.h.hole[0])}")
    print(f"P1 hole: {card_str(engine.h.hole[1])}")

    while not engine.h.terminal:
        if engine.is_chance_node():
            c = rng.choice(engine.chance_outcomes())
            engine.apply_chance(c)
            print(f"  Dealt: {card_str(c)}")
            continue

        player = engine.current_player()
        legal = engine.legal_actions()
        if player == 0:
            action = gsi(engine, 0)
            print(f"  P0 (GSI) -> {action}")
        else:
            action = random_policy(engine, 1)
            engine.apply(action)
            print(f"  P1 (random) -> {action}")
            continue

        engine.apply(action)

    print(f"  Result: {engine.h.terminal_reason}")

gsi.close()