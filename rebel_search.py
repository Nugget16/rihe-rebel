from __future__ import annotations

import random
from typing import Dict, List, Optional
import torch
import numpy as np

from rihe_engine import RIHEEngine
from mccfr import CFRTables, regret_matching, sample_from_dist
from belief import BeliefState
from pbs import encode_pbs
from value_net import ValueNet

Action = str


# Local CFR tables for decision-time search

class SearchTables:
    """Regret/strategy tables local to one search call."""

    def __init__(self):
        self.regret_sum:  Dict[tuple, Dict[Action, float]] = {}
        self.strat_sum:   Dict[tuple, Dict[Action, float]] = {}

    def ensure(self, I: tuple, legal: List[Action]):
        if I not in self.regret_sum:
            self.regret_sum[I] = {a: 0.0 for a in legal}
            self.strat_sum[I]  = {a: 0.0 for a in legal}

    def get_sigma(self, I: tuple, legal: List[Action]) -> Dict[Action, float]:
        self.ensure(I, legal)
        return regret_matching(self.regret_sum[I], legal)

    def get_avg_strategy(self, I: tuple, legal: List[Action]) -> Dict[Action, float]:
        ss = self.strat_sum.get(I, {})
        total = sum(ss.get(a, 0.0) for a in legal)
        if total <= 1e-12:
            p = 1.0 / len(legal)
            return {a: p for a in legal}
        return {a: ss.get(a, 0.0) / total for a in legal}


# Depth-limited CFR traversal

def _infoset_key(engine: RIHEEngine, player: int) -> tuple:
    """Simple infoset key for search: (player, hole_card, street, history)."""
    h = engine.h
    from infosets import compress_history
    return (player, h.hole[player], h.street, compress_history(engine))


def _value_net_eval(
    engine: RIHEEngine,
    player: int,
    belief: BeliefState,
    net: ValueNet,
    device: torch.device,
    starting_stack: int,
) -> float:
    """Use value network to estimate value at a leaf node."""
    pbs = encode_pbs(engine, player, belief, starting_stack)
    return net.predict(pbs, device) * starting_stack  # denormalize


def cfr_search(
    engine: RIHEEngine,
    update_player: int,
    rng: random.Random,
    tables: SearchTables,
    belief: BeliefState,
    net: ValueNet,
    device: torch.device,
    starting_stack: int,
    start_stack0: int,
    depth: int,
    max_depth: int,
    reach_p0: float = 1.0,
    reach_p1: float = 1.0,
) -> float:
    """
    Depth-limited CFR traversal.
    At max_depth or terminal, use value network to evaluate.
    Returns utility for update_player.
    """
    # Terminal node
    if engine.is_terminal():
        u0 = float(engine.stacks[0] - start_stack0)
        return u0 if update_player == 0 else -u0

    # Depth limit reached - use value network
    if depth >= max_depth and not engine.is_chance_node():
        return _value_net_eval(engine, update_player, belief, net, device, starting_stack)

    # Chance node
    if engine.is_chance_node():
        outcomes = engine.chance_outcomes()
        card = outcomes[rng.randrange(len(outcomes))]
        snap = engine.snapshot()
        belief_copy = BeliefState()
        belief_copy.beliefs = belief.beliefs.copy()
        engine.apply_chance(card)
        belief_copy.observe_card(card)
        val = cfr_search(
            engine, update_player, rng, tables, belief_copy,
            net, device, starting_stack, start_stack0,
            depth + 1, max_depth, reach_p0, reach_p1,
        )
        engine.restore(snap)
        return val

    # Player node
    to_act = engine.current_player()
    legal = engine.legal_actions()
    I = _infoset_key(engine, to_act)
    tables.ensure(I, legal)
    sigma = tables.get_sigma(I, legal)

    # Opponent node - sample one action, update belief
    if to_act != update_player:
        a = sample_from_dist(rng, sigma)

        # Update opponent reach and strategy sum
        opp_reach = reach_p0 if to_act == 0 else reach_p1
        for act in legal:
            tables.strat_sum[I][act] += opp_reach * sigma[act]

        # Update belief
        belief_copy = BeliefState()
        belief_copy.beliefs = belief.beliefs.copy()
        from mccfr import AvgStrategyPolicy
        # Simple belief update: weight by sigma directly
        for idx in range(52):
            if belief_copy.beliefs[idx] == 0.0:
                continue
            from belief import ALL_CARDS
            card = ALL_CARDS[idx]
            snap = engine.snapshot()
            engine.h.hole[to_act] = card
            from infosets import infoset_key
            I_card = _infoset_key(engine, to_act)
            tables.ensure(I_card, legal)
            sigma_card = tables.get_sigma(I_card, legal)
            belief_copy.beliefs[idx] *= sigma_card.get(a, 1e-9)
            engine.restore(snap)
        belief_copy._normalize()

        snap = engine.snapshot()
        engine.apply(a)

        if to_act == 0:
            val = cfr_search(
                engine, update_player, rng, tables, belief_copy,
                net, device, starting_stack, start_stack0,
                depth + 1, max_depth, reach_p0 * sigma[a], reach_p1,
            )
        else:
            val = cfr_search(
                engine, update_player, rng, tables, belief_copy,
                net, device, starting_stack, start_stack0,
                depth + 1, max_depth, reach_p0, reach_p1 * sigma[a],
            )

        engine.restore(snap)
        return val

    # Update-player node - enumerate all actions
    opponent_reach = reach_p1 if update_player == 0 else reach_p0

    # Accumulate strategy sum
    for a in legal:
        tables.strat_sum[I][a] += opponent_reach * sigma[a]

    action_util: Dict[Action, float] = {}
    node_util = 0.0

    for a in legal:
        snap = engine.snapshot()
        engine.apply(a)
        u = cfr_search(
            engine, update_player, rng, tables, belief,
            net, device, starting_stack, start_stack0,
            depth + 1, max_depth, reach_p0, reach_p1,
        )
        engine.restore(snap)
        action_util[a] = u
        node_util += sigma[a] * u

    # Regret update
    for a in legal:
        tables.regret_sum[I][a] += action_util[a] - node_util

    return node_util


def run_search(
    engine: RIHEEngine,
    player: int,
    belief: BeliefState,
    net: ValueNet,
    device: torch.device,
    starting_stack: int,
    n_iters: int = 100,
    max_depth: int = 3,
    seed: int = 0,
) -> Dict[Action, float]:
    """
    Run depth-limited CFR search from the current game state.
    Returns the average strategy (action probabilities) for `player`.
    """
    rng = random.Random(seed)
    tables = SearchTables()
    start_stack0 = engine.stacks[0]
    legal = engine.legal_actions()

    for _ in range(n_iters):
        for update_player in (0, 1):
            snap = engine.snapshot()
            cfr_search(
                engine=engine,
                update_player=update_player,
                rng=rng,
                tables=tables,
                belief=belief,
                net=net,
                device=device,
                starting_stack=starting_stack,
                start_stack0=start_stack0,
                depth=0,
                max_depth=max_depth,
                reach_p0=1.0,
                reach_p1=1.0,
            )
            engine.restore(snap)

    # Get average strategy for player at current infoset
    I = _infoset_key(engine, player)
    return tables.get_avg_strategy(I, legal)