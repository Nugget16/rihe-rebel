from __future__ import annotations

import random
from typing import Optional

from rihe_engine import RIHEEngine
from mccfr import AvgStrategyPolicy, CFRTables, sample_from_dist
from belief import BeliefState
from value_net import ValueNet, get_device
from rebel_search import run_search

class RebelPolicy:
    """
    ReBeL-style agent combining:
      - PBS belief tracking over opponent hole cards
      - Depth-limited CFR search at decision time
      - Learned value network to evaluate leaf nodes beyond search depth
    """

    def __init__(
        self,
        net: ValueNet,
        mccfr_policy: AvgStrategyPolicy,
        starting_stack: int = 100_000,
        n_search_iters: int = 100,
        max_depth: int = 3,
        fallback_rng: Optional[random.Random] = None,
    ):
        self.net = net
        self.mccfr_policy = mccfr_policy
        self.starting_stack = starting_stack
        self.n_search_iters = n_search_iters
        self.max_depth = max_depth
        self.device = get_device()
        self.fallback_rng = fallback_rng or random.Random()
        self.net.to(self.device)
        self.net.eval()

        self._belief: Optional[BeliefState] = None
        self._last_hand_index: int = -1
        self._last_player: int = -1
        self._search_seed: int = 0

    def _maybe_reset_belief(self, engine: RIHEEngine, player: int):
        """Reset belief state if a new hand has started."""
        h = engine.h
        hand_idx = engine.hand_index

        if hand_idx != self._last_hand_index or player != self._last_player:
            self._belief = BeliefState()
            community = [c for c in [h.flop, h.turn] if c is not None]
            self._belief.reset(h.hole[player], community)
            self._last_hand_index = hand_idx
            self._last_player = player
            self._search_seed = random.randrange(1 << 30)

    def update_belief_from_history(self, engine: RIHEEngine, player: int):
        """
        Replay the hand history to bring belief up to date.
        """
        h = engine.h
        assert h is not None
        opponent = 1 - player

        community = [c for c in [h.flop, h.turn] if c is not None]
        self._belief = BeliefState()
        self._belief.reset(h.hole[player], [])

        for ev in h.history:
            if ev.startswith("deal_flop:"):
                card_str = ev.split(":")[1]
                card = _parse_card_token(card_str)
                if card is not None:
                    self._belief.observe_card(card)
            elif ev.startswith("deal_turn:"):
                card_str = ev.split(":")[1]
                card = _parse_card_token(card_str)
                if card is not None:
                    self._belief.observe_card(card)
            elif ev.startswith("s") and ":" in ev:
                try:
                    parts = ev.split(":")
                    acting_player = int(parts[1][1:])
                    action = parts[2]
                    if acting_player == opponent and action != "fold":
                        self._belief.update(
                            engine, opponent, action, self.mccfr_policy
                        )
                except Exception:
                    continue

    def __call__(self, engine: RIHEEngine, player: int) -> str:
        h = engine.h
        assert h is not None and h.round is not None

        legal = engine.legal_actions()

        self._maybe_reset_belief(engine, player)
        self.update_belief_from_history(engine, player)

        try:
            # Run depth-limited CFR search
            strategy = run_search(
                engine=engine,
                player=player,
                belief=self._belief,
                net=self.net,
                device=self.device,
                starting_stack=self.starting_stack,
                n_iters=self.n_search_iters,
                max_depth=self.max_depth,
                seed=self._search_seed,
            )
            self._search_seed += 1

            # Sample action from searched strategy
            action = sample_from_dist(self.fallback_rng, strategy)

            if action not in legal:
                return self.fallback_rng.choice(legal)

            return action

        except Exception as e:
            # Fallback to MCCFR policy if search fails
            return self.mccfr_policy.sample_action(engine, player)

from rihe_engine import RANK_STR

_RANK_STR_INV = {v: k for k, v in RANK_STR.items()}

def _parse_card_token(token: str):
    """Parse a card token like 'A♣' or '10♥' back to (rank, suit)."""
    try:
        suit = token[-1]
        rank_str = token[:-1]
        rank = _RANK_STR_INV.get(rank_str)
        if rank is None:
            return None
        return (rank, suit)
    except Exception:
        return None