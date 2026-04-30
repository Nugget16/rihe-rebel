from __future__ import annotations

import random
import subprocess
from typing import Optional

from rihe_engine import RIHEEngine, RANKS, SUITS
from infosets import compress_history

_SUIT_IDX = {"♣": 0, "♦": 1, "♥": 2, "♠": 3}
_RANK_IDX = {14: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6,
             8: 7, 9: 8, 10: 9, 11: 10, 12: 11, 13: 12}

def card_to_gsi(card) -> int:
    """Convert engine (rank, suit) tuple to GSI 0-51 integer."""
    rank, suit = card
    return _SUIT_IDX[suit] * 13 + _RANK_IDX[rank]

# Betting history mapping
HISTORY_TO_BH = {
    "":      0,
    "c":     1,
    "cc":    2,
    "b":     3,
    "bk":    4,
    "br":    5,
    "brk":   6,
    "brr":   7,
    "brrk":  8,
    "cb":    12,
    "cbk":   13,
    "cbr":   14,
    "cbrk":  15,
    "cbrr":  16,
    "cbrrk": 17,
}

# Action index -> engine action string
_CHECK_BET_BHS = {0, 1}
_FCR_BHS       = {3, 5, 7, 12, 14, 16}

_CHECK_BET_ACTIONS = ["check", "bet"]
_FCR_ACTIONS       = ["fold", "call", "raise"]

def action_idx_to_str(bh: int, idx: int) -> str:
    if bh in _CHECK_BET_BHS:
        return _CHECK_BET_ACTIONS[idx]
    elif bh in _FCR_BHS:
        return _FCR_ACTIONS[idx]
    else:
        raise ValueError(f"Unexpected bh={bh} for action mapping")

# Per-street history extraction

def per_street_histories(engine: RIHEEngine) -> list[str]:
    full = compress_history(engine)
    if not full:
        return [""]
    return full.split("|")

# GSI strategy server subprocess wrapper

class GSIPolicy:
    def __init__(self, java_dir: str, fallback_rng: Optional[random.Random] = None,
                 debug: bool = False):
        self.java_dir = java_dir
        self.fallback_rng = fallback_rng or random.Random()
        self.debug = debug
        self._proc: Optional[subprocess.Popen] = None
        self._n_queries = 0
        self._n_not_found = 0
        self._n_errors = 0
        self._n_illegal = 0
        self._start()

    def _start(self):
        self._proc = subprocess.Popen(
            ["java", "-cp", ".", "StrategyServer"],
            cwd=self.java_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        line = self._proc.stdout.readline().strip()
        if line != "READY":
            raise RuntimeError(f"StrategyServer did not start cleanly: {line!r}")

    def _query(self, query: str) -> str:
        assert self._proc is not None
        self._proc.stdin.write(query + "\n")
        self._proc.stdin.flush()
        response = self._proc.stdout.readline().strip()
        if self.debug:
            print(f"  [GSI] query={query!r} -> {response!r}")
        return response

    def close(self):
        if self._proc is not None:
            try:
                self._proc.stdin.write("quit\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()
            self._proc = None

    def __del__(self):
        self.close()

    def print_stats(self):
        total = self._n_queries
        if total == 0:
            print("No queries made.")
            return
        print(f"GSI query stats ({total} total):")
        print(f"  NOT_FOUND : {self._n_not_found:6d} ({self._n_not_found/total:.1%})")
        print(f"  Errors    : {self._n_errors:6d} ({self._n_errors/total:.1%})")
        print(f"  Illegal   : {self._n_illegal:6d} ({self._n_illegal/total:.1%})")
        good = total - self._n_not_found - self._n_errors - self._n_illegal
        print(f"  Good      : {good:6d} ({good/total:.1%})")

    def get_action(self, engine: RIHEEngine, player: int) -> str:
        h = engine.h
        assert h is not None and h.round is not None

        legal = engine.legal_actions()
        self._n_queries += 1

        try:
            query, bh_current = self._build_query(engine, player)
        except (KeyError, ValueError) as e:
            if self.debug:
                print(f"  [GSI] fallback (build error): {e}")
            self._n_errors += 1
            return self.fallback_rng.choice(legal)

        response = self._query(query)

        if response == "NOT_FOUND":
            self._n_not_found += 1
            return self.fallback_rng.choice(legal)

        if response.startswith("ERROR"):
            if self.debug:
                print(f"  [GSI] fallback ({response})")
            self._n_errors += 1
            return self.fallback_rng.choice(legal)

        try:
            probs_part, idx_part = response.split("|")
            action_idx = int(idx_part.strip())
            action_str = action_idx_to_str(bh_current, action_idx)
        except Exception as e:
            if self.debug:
                print(f"  [GSI] fallback (parse error): {e}")
            self._n_errors += 1
            return self.fallback_rng.choice(legal)

        if action_str not in legal:
            if self.debug:
                print(f"  [GSI] fallback (illegal action {action_str!r}, legal={legal})")
            self._n_illegal += 1
            return self.fallback_rng.choice(legal)

        return action_str

    def _build_query(self, engine: RIHEEngine, player: int):
        h = engine.h
        street = h.street

        histories = per_street_histories(engine)
        while len(histories) <= street:
            histories.append("")

        bh_current = HISTORY_TO_BH[histories[street]]
        hole_gsi = card_to_gsi(h.hole[player])

        if self.debug:
            from rihe_engine import card_str
            print(f"  [GSI] street={street} player={player} "
                  f"hole={card_str(h.hole[player])}({hole_gsi}) "
                  f"histories={histories} bh_current={bh_current}")

        if street == 0:
            query = f"0 {hole_gsi} {bh_current}"

        elif street == 1:
            bh1 = HISTORY_TO_BH[histories[0]]
            bh2 = bh_current
            flop_gsi = card_to_gsi(h.flop)
            query = f"1 {hole_gsi} {bh1} {flop_gsi} {bh2}"

        else:
            bh1 = HISTORY_TO_BH[histories[0]]
            bh2 = HISTORY_TO_BH[histories[1]]
            bh3 = bh_current
            flop_gsi = card_to_gsi(h.flop)
            turn_gsi = card_to_gsi(h.turn)
            query = f"2 {hole_gsi} {bh1} {flop_gsi} {bh2} {turn_gsi} {bh3}"

        return query, bh_current

    def __call__(self, engine: RIHEEngine, player: int) -> str:
        return self.get_action(engine, player)