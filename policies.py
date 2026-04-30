from __future__ import annotations
from typing import Callable
from rihe_engine import RIHEEngine, hand_strength_3card
from mccfr import AvgStrategyPolicy, CFRTables
from gsi_policy import GSIPolicy
from rebel_policy import RebelPolicy
import random
from value_net import ValueNet, get_device
import torch

Policy = Callable[[RIHEEngine, int], str]

def random_policy(engine: RIHEEngine, player: int) -> str:
    return engine.random_action()

def _rank(card) -> int:
    return card[0]

def heuristic_policy(engine: RIHEEngine, player: int) -> str:
    """
      - Turn (full info): bet/raise with Pair+; call with decent high-card; fold weak high-card vs bet.
      - Preflop/Flop: bet sometimes with high hole / pair-on-board potential; otherwise check/call.
    """
    h = engine.h
    assert h is not None and h.round is not None
    r = h.round

    hole = h.hole[player]
    board = [c for c in [h.flop, h.turn] if c is not None]

    facing_bet = (r.current_bet > 0)

    if h.street == 2 and h.flop is not None and h.turn is not None:
        cards = [hole, h.flop, h.turn]
        cat, tiebreak = hand_strength_3card(cards)

        # Pair or better -> aggressive
        if cat >= 2:
            if facing_bet:
                return "raise" if "raise" in r.legal_actions() else "call"
            return "bet"

        # High card only: decide based on top rank
        top = max(_rank(c) for c in cards)
        if facing_bet:
            # call with A/K/Q/J or better; otherwise fold
            return "call" if top >= 11 else "fold"
        else:
            # bluff a little with very high card
            return "bet" if top >= 13 else "check"

    # Flop street (have hole + flop)
    if h.street == 1 and h.flop is not None:
        hole_rank = _rank(hole)
        flop_rank = _rank(h.flop)
        suited = (hole[1] == h.flop[1])
        paired = (hole_rank == flop_rank)

        if facing_bet:
            # call if paired or high hole or suited high; else fold sometimes
            if paired or hole_rank >= 12 or (suited and hole_rank >= 10):
                return "call"
            return "fold"
        else:
            # bet if paired or strong high card; else check
            if paired or hole_rank >= 13:
                return "bet"
            return "check"

    # Preflop (only hole)
    hole_rank = _rank(hole)
    if facing_bet:
        # call with Q+; fold with low junk
        return "call" if hole_rank >= 12 else "fold"
    else:
        # bet with A/K sometimes, otherwise check
        return "bet" if hole_rank >= 13 else "check"
    

class MCCFRPolicyWrapper:
    def __init__(self, avg_policy: AvgStrategyPolicy):
        self.avg = avg_policy
    def __call__(self, engine, player: int) -> str:
        return self.avg.sample_action(engine, player)
    
def make_gsi_policy(java_dir: str) -> GSIPolicy:
    return GSIPolicy(java_dir=java_dir)

def make_rebel_policy(
    net_path: str,
    tables: CFRTables,
    starting_stack: int = 100_000,
    n_search_iters: int = 100,
    max_depth: int = 3,
) -> RebelPolicy:
    device = get_device()
    net = ValueNet()
    net.load(net_path, device)
    mccfr_pol = AvgStrategyPolicy(tables, rng=random.Random(0))
    return RebelPolicy(
        net=net,
        mccfr_policy=mccfr_pol,
        starting_stack=starting_stack,
        n_search_iters=n_search_iters,
        max_depth=max_depth,
    )