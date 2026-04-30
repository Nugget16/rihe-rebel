from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import random

from tqdm import tqdm

from rihe_engine import RIHEEngine
from infosets import infoset_key

Infoset = Tuple
Action = str

def regret_matching(regrets: Dict[Action, float], legal: List[Action]) -> Dict[Action, float]:
    """Regret-matching distribution over legal actions."""
    pos = [max(0.0, regrets.get(a, 0.0)) for a in legal]
    s = sum(pos)
    if s <= 1e-12:
        p = 1.0 / len(legal)
        return {a: p for a in legal}
    return {a: v / s for a, v in zip(legal, pos)}


def sample_from_dist(rng: random.Random, dist: Dict[Action, float]) -> Action:
    x = rng.random()
    acc = 0.0
    last = None
    for a, p in dist.items():
        acc += p
        last = a
        if x <= acc:
            return a
    assert last is not None
    return last


@dataclass
class CFRTables:
    regret_sum: Dict[Infoset, Dict[Action, float]]
    strat_sum: Dict[Infoset, Dict[Action, float]]

    def __init__(self):
        self.regret_sum = {}
        self.strat_sum = {}

    def ensure(self, I: Infoset, legal: List[Action]) -> None:
        rs = self.regret_sum.get(I)
        if rs is None:
            self.regret_sum[I] = {a: 0.0 for a in legal}
        else:
            for a in legal:
                rs.setdefault(a, 0.0)

        ss = self.strat_sum.get(I)
        if ss is None:
            self.strat_sum[I] = {a: 0.0 for a in legal}
        else:
            for a in legal:
                ss.setdefault(a, 0.0)


class AvgStrategyPolicy:
    """Policy induced by average strategy (strat_sum)."""
    def __init__(self, tables: CFRTables, rng: Optional[random.Random] = None):
        self.tables = tables
        self.rng = rng or random.Random()

    def action_probs(self, engine: RIHEEngine, player: int) -> Dict[Action, float]:
        legal = engine.legal_actions()
        I = infoset_key(engine, player)
        ss = self.tables.strat_sum.get(I)

        if not ss:
            p = 1.0 / len(legal)
            return {a: p for a in legal}

        total = sum(ss.get(a, 0.0) for a in legal)
        if total <= 1e-12:
            p = 1.0 / len(legal)
            return {a: p for a in legal}

        return {a: ss.get(a, 0.0) / total for a in legal}

    def sample_action(self, engine: RIHEEngine, player: int) -> Action:
        return sample_from_dist(self.rng, self.action_probs(engine, player))

def terminal_u_update_player(engine: RIHEEngine, start_stack0: int, update_player: int) -> float:
    u0 = float(engine.stacks[0] - start_stack0)
    return u0 if update_player == 0 else -u0

def external_sampling_mccfr(
    tables: CFRTables,
    engine: RIHEEngine,
    update_player: int,
    rng: random.Random,
    start_stack0: int,
    reach_p0: float = 1.0,
    reach_p1: float = 1.0,
) -> float:
    """
    Two-player zero-sum external-sampling MCCFR.
    """
    if engine.is_terminal():
        return terminal_u_update_player(engine, start_stack0, update_player)

    # Chance node
    if engine.is_chance_node():
        outcomes = engine.chance_outcomes()
        a = outcomes[rng.randrange(len(outcomes))]

        snap = engine.snapshot()
        engine.apply_chance(a)
        val = external_sampling_mccfr(
            tables, engine, update_player, rng, start_stack0,
            reach_p0, reach_p1,
        )
        engine.restore(snap)
        return val

    # Player node
    to_act = engine.current_player()
    legal = engine.legal_actions()
    I = infoset_key(engine, to_act)
    tables.ensure(I, legal)
    sigma = regret_matching(tables.regret_sum[I], legal)

    # Opoonent node
    # Sample one action, update opponent's reach, and recurse.
    if to_act != update_player:
        a = sample_from_dist(rng, sigma)
        snap = engine.snapshot()
        engine.apply(a)

        if to_act == 0:
            val = external_sampling_mccfr(
                tables, engine, update_player, rng, start_stack0,
                reach_p0 * sigma[a],
                reach_p1,
            )
        else:
            val = external_sampling_mccfr(
                tables, engine, update_player, rng, start_stack0,
                reach_p0,
                reach_p1 * sigma[a],
            )

        engine.restore(snap)
        return val

    # Update-player node
    opponent_reach = reach_p1 if update_player == 0 else reach_p0

    for a in legal:
        tables.strat_sum[I][a] += opponent_reach * sigma[a]

    action_util: Dict[Action, float] = {}
    node_util = 0.0

    for a in legal:
        snap = engine.snapshot()
        engine.apply(a)
        u = external_sampling_mccfr(
            tables, engine, update_player, rng, start_stack0,
            reach_p0, reach_p1,
        )

        engine.restore(snap)
        action_util[a] = u
        node_util += sigma[a] * u

    for a in legal:
        tables.regret_sum[I][a] += action_util[a] - node_util

    return node_util


def train_mccfr(
    iterations: int,
    seed: int = 0,
    starting_stack: int = 100_000,
    log_every: int = 5_000,
    show_progress: bool = True,
) -> CFRTables:
    """
    Train using explicit-chance external-sampling MCCFR.
    One traversal for player 0 and one traversal for player 1 per iteration.
    """
    rng = random.Random(seed)
    tables = CFRTables()

    iterator = range(1, iterations + 1)
    if show_progress:
        iterator = tqdm(iterator, total=iterations, desc="MCCFR train", unit="iter")

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
            msg = f"iter {t}/{iterations} | infosets={len(tables.regret_sum)}"
            if show_progress:
                tqdm.write(f"[mccfr] {msg}")
            else:
                print(f"[mccfr] {msg}")

    return tables