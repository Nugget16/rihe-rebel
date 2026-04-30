from __future__ import annotations
import numpy as np

from rihe_engine import RIHEEngine, RANKS, SUITS
from mccfr import AvgStrategyPolicy

# All 52 cards in GSI encoding order (suit-major, clubs-first, ace-low)
_SUITS = ["♣", "♦", "♥", "♠"]
_RANKS = [14, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]  # A,2,3..K

ALL_CARDS = [(r, s) for s in _SUITS for r in _RANKS]

assert len(ALL_CARDS) == 52


def _card_to_idx(card) -> int:
    """Convert (rank, suit) to 0-51 index matching ALL_CARDS order."""
    rank, suit = card
    suit_idx = _SUITS.index(suit)
    rank_idx = _RANKS.index(rank)
    return suit_idx * 13 + rank_idx


class BeliefState:
    """
    Maintains probability distribution over opponent's hole card.
    """

    def __init__(self):
        self.beliefs = np.ones(52, dtype=np.float32)

    def reset(self, my_card, community_cards: list):
        """
        Initialize beliefs at start of hand.
        Zero out my cards and community cards.
        """
        self.beliefs = np.ones(52, dtype=np.float32)
        known = [my_card] + [c for c in community_cards if c is not None]
        for card in known:
            self.beliefs[_card_to_idx(card)] = 0.0
        self._normalize()

    def observe_card(self, card):
        """
        Zero out newly dealt card.
        """
        self.beliefs[_card_to_idx(card)] = 0.0
        self._normalize()

    def update(self, engine: RIHEEngine, opponent: int, action: str, policy: AvgStrategyPolicy):
        """
        Update after observing opponent take `action`.
        """
        snap = engine.snapshot()
        h = engine.h

        for idx in range(52):
            if self.beliefs[idx] == 0.0:
                continue

            card = ALL_CARDS[idx]

            # Temporarily set opponent's hole card and query policy
            h.hole[opponent] = card
            probs = policy.action_probs(engine, opponent)
            self.beliefs[idx] *= probs.get(action, 1e-9)

        # Restore engine to original state
        engine.restore(snap)
        self._normalize()

    def _normalize(self):
        total = self.beliefs.sum()
        if total > 1e-12:
            self.beliefs /= total
        else:
            # Fallback: uniform over non-zero entries
            nonzero = self.beliefs > 0
            if nonzero.any():
                self.beliefs[nonzero] = 1.0 / nonzero.sum()

    def as_vector(self) -> np.ndarray:
        """Return the 52-float belief vector for use as network input."""
        return self.beliefs.copy()