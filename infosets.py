from __future__ import annotations
from typing import Optional, Tuple
from rihe_engine import RIHEEngine, Card, RANK_STR

def card_token(c: Optional[Card]) -> str:
    if c is None:
        return "-"
    r, s = c
    return f"{RANK_STR[r]}{s}"

_ACTION_MAP = {
    "check": "c",
    "bet": "b",
    "call": "k",   # k for call (avoid c/collision)
    "raise": "r",
    "fold": "f",
}

def compress_history(engine: RIHEEngine) -> str:
    h = engine.h
    assert h is not None

    parts = []
    cur_street = None
    for ev in h.history:
        if not ev.startswith("s"):
            continue
        try:
            s_part, p_part, a_part = ev.split(":")
            street = int(s_part[1:])
            action = a_part
        except Exception:
            continue

        if cur_street is None:
            cur_street = street
        elif street != cur_street:
            parts.append("|")
            cur_street = street

        parts.append(_ACTION_MAP.get(action, "?"))

    return "".join(parts)

def infoset_key(engine: RIHEEngine, player: int) -> Tuple:
    """
    Infoset = what player can observe:
      - player id
      - player's hole card
      - public cards known
      - compact betting history
    """
    h = engine.h
    assert h is not None and h.round is not None

    return (
        player,
        card_token(h.hole[player]),
        card_token(h.flop if h.street >= 1 else None),
        card_token(h.turn if h.street >= 2 else None),
        compress_history(engine),
    )