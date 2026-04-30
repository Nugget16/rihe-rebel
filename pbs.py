from __future__ import annotations
import numpy as np
from rihe_engine import RIHEEngine
from belief import BeliefState, _card_to_idx

# Input dimension breakdown:
# street one-hot: 3
# my hole card: 52
# flop card: 52
# turn card: 52
# betting history: 3 (bh per street, normalized)
# belief distribution: 52
# pot + in-front: 3
# total: 217

PBS_DIM = 217

_POT_NORM = 200_000.0

def encode_pbs(
    engine: RIHEEngine,
    player: int,
    belief: BeliefState,
    starting_stack: int = 100_000,
) -> np.ndarray:
    """
    Encode the current game state from player's perspective into a
    217-dimensional public-belief state vector.
    """
    h = engine.h
    assert h is not None

    vec = np.zeros(PBS_DIM, dtype=np.float32)
    offset = 0

    # Street one-hot (3 dims)
    street = min(h.street, 2)
    vec[offset + street] = 1.0
    offset += 3

    # Hole card one-hot (52 dims)
    my_card = h.hole[player]
    vec[offset + _card_to_idx(my_card)] = 1.0
    offset += 52

    # Flop card one-hot (52 dims)
    if h.flop is not None:
        vec[offset + _card_to_idx(h.flop)] = 1.0
    offset += 52

    # Turn card one-hot (52 dims)
    if h.turn is not None:
        vec[offset + _card_to_idx(h.turn)] = 1.0
    offset += 52

    # Betting history (3 dims, normalized 0-1)
    # Use raw bh integers from HISTORY_TO_BH mapping, normalized by max bh (17)
    from infosets import compress_history
    full_hist = compress_history(engine)
    streets = full_hist.split("|") if full_hist else [""]
    from gsi_policy import HISTORY_TO_BH
    for i in range(3):
        if i < len(streets):
            bh = HISTORY_TO_BH.get(streets[i], 0)
        else:
            bh = 0
        vec[offset + i] = bh / 17.0
    offset += 3

    # Belief distribution (52 dims)
    vec[offset:offset + 52] = belief.as_vector()
    offset += 52

    # Pot + in-front amounts (3 dims, normalized)
    norm = float(starting_stack) * 2.0
    vec[offset]     = h.pot / norm
    vec[offset + 1] = h.in_front[player] / norm
    vec[offset + 2] = h.in_front[1 - player] / norm
    offset += 3

    assert offset == PBS_DIM, f"Expected {PBS_DIM} dims, got {offset}"
    return vec