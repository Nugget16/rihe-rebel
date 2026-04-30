from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import random

# Cards / Hand evaluation (3-card poker)

RANKS = list(range(2, 15))
SUITS = ["♠", "♥", "♦", "♣"]
RANK_STR = {**{r: str(r) for r in range(2, 11)}, 11: "J", 12: "Q", 13: "K", 14: "A"}

Card = Tuple[int, str]

def card_str(c: Optional[Card]) -> str:
    if c is None:
        return "??"
    r, s = c
    return f"{RANK_STR[r]}{s}"

def card_token(c: Card) -> str:
    r, s = c
    return f"{RANK_STR[r]}{s}"

def make_deck() -> List[Card]:
    return [(r, s) for r in RANKS for s in SUITS]

def is_flush(cards: List[Card]) -> bool:
    return len({s for _, s in cards}) == 1

def straight_high(ranks: List[int]) -> Optional[int]:
    r = sorted(ranks)
    if len(set(r)) != 3:
        return None
    if r == [2, 3, 14]:
        return 3
    if r == [12, 13, 14]:
        return 14
    if r[0] + 1 == r[1] and r[1] + 1 == r[2]:
        return r[2]
    return None

def hand_strength_3card(cards: List[Card]) -> Tuple[int, Tuple[int, ...]]:
    """
      6 Straight flush
      5 Trips
      4 Straight
      3 Flush
      2 Pair
      1 High card
    """
    ranks = [r for r, _ in cards]
    ranks_sorted = tuple(sorted(ranks, reverse=True))
    flush = is_flush(cards)
    sh = straight_high(ranks)

    counts: Dict[int, int] = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1

    if flush and sh is not None:
        return (6, (sh,))
    if len(counts) == 1:
        trip_rank = next(iter(counts.keys()))
        return (5, (trip_rank,))
    if sh is not None:
        return (4, (sh,))
    if flush:
        return (3, ranks_sorted)
    if len(counts) == 2:
        pair_rank = max(r for r, c in counts.items() if c == 2)
        kicker = min(r for r, c in counts.items() if c == 1)
        return (2, (pair_rank, kicker))
    return (1, ranks_sorted)

def compare_hands(p0: List[Card], p1: List[Card]) -> int:
    h0 = hand_strength_3card(p0)
    h1 = hand_strength_3card(p1)
    return (h0 > h1) - (h0 < h1)

# Action is either a player action ("check","bet","call","raise","fold") OR a Card at chance nodes.
Action = Any  # str for players, Card for chance

PLAYER_ACTIONS = {"check", "bet", "call", "raise", "fold"}

@dataclass
class RoundState:
    bet_size: int
    raises_used: int = 0
    current_bet: int = 0
    contrib_in_round: List[int] = field(default_factory=lambda: [0, 0])
    pending_response: bool = False
    prev_action: Optional[str] = None

    def legal_actions(self) -> List[str]:
        if self.current_bet == 0:
            return ["check", "bet"]
        acts = ["fold", "call"]
        if self.raises_used < 3:
            acts.append("raise")
        return acts

    def call_amount(self, player: int) -> int:
        return max(0, self.current_bet - self.contrib_in_round[player])

@dataclass
class HandState:
    first_to_act: int
    to_act: int
    street: int # 0 pre, 1 flop, 2 turn, 3 terminal
    phase: str # "preflop_bet","deal_flop","flop_bet","deal_turn","turn_bet","terminal"
    hole: List[Card]
    flop: Optional[Card] = None
    turn: Optional[Card] = None
    pot: int = 0 # committed pot
    in_front: List[int] = field(default_factory=lambda: [0, 0]) # uncommitted current-street chips
    round: Optional[RoundState] = None
    history: List[str] = field(default_factory=list)
    runout_to_showdown: bool = False # if True, after chance runout go straight to showdown (no more betting)
    terminal: bool = False
    terminal_reason: str = ""
    showdown_result: Optional[int] = None

class RIHEEngine:
    ANTE = 5
    BET_SIZES = [10, 20, 20] # preflop, flop, turn

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()
        self.stacks = [0, 0]
        self.hand_index = 0
        self.deck: List[Card] = []
        self.h: Optional[HandState] = None

    def is_terminal(self) -> bool:
        return self.h is None or self.h.terminal

    def is_chance_node(self) -> bool:
        return self.h is not None and (self.h.phase in ("deal_flop", "deal_turn"))

    def chance_outcomes(self) -> List[Card]:
        """Legal chance actions (remaining cards)."""
        assert self.h is not None and self.is_chance_node()
        return list(self.deck)

    def apply_chance(self, a: Card) -> None:
        """Apply a chance action (deal a specific remaining card)."""
        assert self.h is not None and self.is_chance_node()

        h = self.h
        if a not in self.deck:
            raise ValueError("Chance action card not in deck")

        # Remove dealt card from deck
        self.deck.remove(a)

        if h.phase == "deal_flop":
            h.history.append("/flop")
            h.flop = a
            h.history.append(f"deal_flop:{card_token(a)}")

            if h.runout_to_showdown:
                # Runout continues: need turn next
                h.phase = "deal_turn"
                h.street = 2
                h.to_act = h.first_to_act # irrelevant at chance
                h.round = None
                return

            # Start flop betting
            self._start_betting_round(street=1)

        elif h.phase == "deal_turn":
            h.history.append("/turn")
            h.turn = a
            h.history.append(f"deal_turn:{card_token(a)}")

            if h.runout_to_showdown:
                # Finish runout: go straight to showdown
                h.phase = "terminal"
                h.street = 2
                h.round = None
                self._resolve_showdown()
                return

            # Start turn betting
            self._start_betting_round(street=2)

        else:
            raise RuntimeError("apply_chance called in non-chance phase")

    def current_player(self) -> int:
        assert self.h is not None and (not self.is_terminal()) and (not self.is_chance_node())
        return self.h.to_act

    # Snapshot/restore for fast traversal
    def snapshot(self):
        """Snapshot of all mutable state (faster than deepcopy)."""
        h = self.h
        if h is None:
            h_snap = None
        else:
            r = h.round
            r_snap = None
            if r is not None:
                r_snap = RoundState(
                    bet_size=r.bet_size,
                    raises_used=r.raises_used,
                    current_bet=r.current_bet,
                    contrib_in_round=r.contrib_in_round.copy(),
                    pending_response=r.pending_response,
                    prev_action=r.prev_action,
                )
            h_snap = HandState(
                first_to_act=h.first_to_act,
                to_act=h.to_act,
                street=h.street,
                phase=h.phase,
                hole=h.hole.copy(),
                flop=h.flop,
                turn=h.turn,
                pot=h.pot,
                in_front=h.in_front.copy(),
                round=r_snap,
                history=h.history.copy(),
                runout_to_showdown=h.runout_to_showdown,
                terminal=h.terminal,
                terminal_reason=h.terminal_reason,
                showdown_result=h.showdown_result,
            )
        return (self.stacks.copy(), self.hand_index, self.deck.copy(), h_snap)

    def restore(self, snap) -> None:
        stacks, hand_index, deck, h_snap = snap
        self.stacks = stacks
        self.hand_index = hand_index
        self.deck = deck
        self.h = h_snap

    def reset(self, starting_stack: int):
        self.stacks = [starting_stack, starting_stack]
        self.hand_index = 0
        self.h = None

    def can_deal(self) -> bool:
        return self.stacks[0] >= self.ANTE and self.stacks[1] >= self.ANTE

    def deal_hand(self):
        if not self.can_deal():
            raise RuntimeError("Not enough chips to ante.")

        first = self.hand_index % 2
        self.hand_index += 1

        self.deck = make_deck()
        self.rng.shuffle(self.deck)

        # antes
        self.stacks[0] -= self.ANTE
        self.stacks[1] -= self.ANTE
        pot = 2 * self.ANTE

        hole0 = self.deck.pop()
        hole1 = self.deck.pop()

        rs = RoundState(bet_size=self.BET_SIZES[0])
        self.h = HandState(
            first_to_act=first,
            to_act=first,
            street=0,
            phase="preflop_bet",
            hole=[hole0, hole1],
            pot=pot,
            in_front=[0, 0],
            round=rs,
            history=[],
            runout_to_showdown=False,
            terminal=False,
        )

        h = self.h
        h.history.append(f"first_to_act:p{first}")
        h.history.append(f"deal_hole:p0:{card_token(hole0)}")
        h.history.append(f"deal_hole:p1:{card_token(hole1)}")
        h.history.append("/preflop")

    # Betting and progression
    def _commit_in_front_to_pot(self):
        """Move current-street chips into the committed pot and clear in_front."""
        assert self.h is not None
        self.h.pot += self.h.in_front[0] + self.h.in_front[1]
        self.h.in_front = [0, 0]

    def _start_betting_round(self, street: int):
        """Enter a betting phase (street 0/1/2) with correct bet size and first-to-act."""
        assert self.h is not None
        h = self.h

        h.street = street
        h.to_act = h.first_to_act
        h.round = RoundState(bet_size=self.BET_SIZES[street])

        if street == 0:
            h.phase = "preflop_bet"
        elif street == 1:
            h.phase = "flop_bet"
        elif street == 2:
            h.phase = "turn_bet"
        else:
            raise ValueError("invalid betting street")

    def _advance_after_round_end(self):
        """Called when a betting round ends via check/check or bet/call."""
        assert self.h is not None
        h = self.h

        # commit bets-in-front from this round
        self._commit_in_front_to_pot()

        if h.street == 0:
            # go to explicit chance node: deal flop
            h.phase = "deal_flop"
            h.street = 1
            h.round = None
            return

        if h.street == 1:
            # go to explicit chance node: deal turn
            h.phase = "deal_turn"
            h.street = 2
            h.round = None
            return

        if h.street == 2:
            # showdown
            h.phase = "terminal"
            self._resolve_showdown()
            return

    def _resolve_showdown(self):
        assert self.h is not None
        h = self.h

        self._commit_in_front_to_pot()

        if h.flop is None:
            h.history.append("/flop")
            h.flop = self.deck.pop()
            h.history.append(f"deal_flop:{card_token(h.flop)}")
        if h.turn is None:
            h.history.append("/turn")
            h.turn = self.deck.pop()
            h.history.append(f"deal_turn:{card_token(h.turn)}")

        h.street = 3
        h.phase = "terminal"
        h.terminal = True
        h.terminal_reason = "showdown"

        p0 = [h.hole[0], h.flop, h.turn]
        p1 = [h.hole[1], h.flop, h.turn]
        res = compare_hands(p0, p1)
        h.showdown_result = res

        if res == 1:
            self.stacks[0] += h.pot
        elif res == -1:
            self.stacks[1] += h.pot
        else:
            self.stacks[0] += h.pot // 2
            self.stacks[1] += h.pot - (h.pot // 2)

    def _win_by_fold(self, folder: int):
        assert self.h is not None
        h = self.h
        winner = 1 - folder

        if h.round is not None and h.round.current_bet > 0:
            uncalled = h.round.call_amount(folder)
            if uncalled > 0:
                refund = min(uncalled, h.in_front[winner])
                h.in_front[winner] -= refund
                self.stacks[winner] += refund

        self._commit_in_front_to_pot()

        h.terminal = True
        h.street = 3
        h.phase = "terminal"
        h.terminal_reason = f"p{folder} folded"
        self.stacks[winner] += h.pot

    def legal_actions(self) -> List[Action]:
        """
        If chance node: legal actions are remaining cards in deck (Card tuples).
        If player node: legal actions are poker actions.
        """
        assert self.h is not None

        if self.is_terminal():
            return []

        if self.is_chance_node():
            return self.chance_outcomes()

        assert self.h.round is not None
        return list(self.h.round.legal_actions())

    def apply(self, action: Action):
        """
        Apply either a chance action (Card) if in chance phase, or a player action (str).
        """
        assert self.h is not None
        h = self.h

        if h.terminal:
            return

        # Chance nodes
        if self.is_chance_node():
            if not isinstance(action, tuple) or len(action) != 2:
                raise ValueError("Chance action must be a Card tuple (rank, suit).")
            self.apply_chance(action)
            return

        # Player nodes
        assert h.round is not None
        r = h.round
        p = h.to_act

        if not isinstance(action, str) or action not in PLAYER_ACTIONS:
            raise ValueError("Player action must be one of: check, bet, call, raise, fold")

        if action not in self.legal_actions():
            raise ValueError(f"Illegal action {action} (legal={self.legal_actions()})")

        h.history.append(f"s{h.street}:p{p}:{action}")

        def pay_from_stack(amount: int) -> tuple[int, bool]:
            """
            Returns (paid, all_in_triggered).
            If stack < amount, pay full remaining stack and signal all-in.
            """
            if amount < 0:
                raise ValueError("negative pay")

            paid = min(self.stacks[p], amount)
            self.stacks[p] -= paid
            h.in_front[p] += paid
            return paid, (paid < amount)

        def refund_unmatched_and_runout_to_showdown():
            if h.in_front[0] > h.in_front[1]:
                diff = h.in_front[0] - h.in_front[1]
                h.in_front[0] -= diff
                self.stacks[0] += diff
            elif h.in_front[1] > h.in_front[0]:
                diff = h.in_front[1] - h.in_front[0]
                h.in_front[1] -= diff
                self.stacks[1] += diff

            # Commit matched at-risk amounts
            self._commit_in_front_to_pot()

            # Enter runout mode
            h.runout_to_showdown = True
            h.round = None

            # Decide which chance phase we need next
            if h.flop is None:
                h.phase = "deal_flop"
                h.street = 1
            elif h.turn is None:
                h.phase = "deal_turn"
                h.street = 2
            else:
                # already have both (rare)
                h.phase = "terminal"
                self._resolve_showdown()

        if action == "check":
            if r.current_bet != 0:
                raise ValueError("Cannot check facing a bet")

            # two checks ends round
            if r.prev_action == "check":
                r.prev_action = None
                self._advance_after_round_end()
                return

            r.prev_action = "check"
            h.to_act = 1 - h.to_act
            return

        if action == "bet":
            if r.current_bet != 0:
                raise ValueError("Cannot bet when a bet exists (use raise)")

            r.current_bet = r.bet_size
            paid, all_in = pay_from_stack(r.bet_size)
            r.contrib_in_round[p] += paid
            r.pending_response = True
            r.prev_action = "bet"

            if all_in:
                refund_unmatched_and_runout_to_showdown()
                return

            h.to_act = 1 - h.to_act
            return

        if action == "fold":
            self._win_by_fold(folder=p)
            return

        if action == "call":
            amt = r.call_amount(p)
            paid, all_in = pay_from_stack(amt)
            r.contrib_in_round[p] += paid

            if all_in:
                refund_unmatched_and_runout_to_showdown()
                return

            # call ends round (resolves pending response)
            if r.pending_response:
                r.pending_response = False
                r.prev_action = None
                self._advance_after_round_end()
                return

            h.to_act = 1 - h.to_act
            return

        if action == "raise":
            if r.current_bet == 0:
                raise ValueError("Cannot raise when no bet exists (use bet)")
            if r.raises_used >= 3:
                raise ValueError("Raise cap reached")

            call_amt = r.call_amount(p)
            total = call_amt + r.bet_size

            paid, all_in = pay_from_stack(total)
            r.contrib_in_round[p] += paid

            # If can't cover full call+raise, treat as all-in and run out to showdown
            if all_in:
                refund_unmatched_and_runout_to_showdown()
                return

            # Normal full raise
            r.current_bet += r.bet_size
            r.raises_used += 1
            r.pending_response = True
            r.prev_action = "raise"
            h.to_act = 1 - h.to_act
            return

        raise RuntimeError("Unknown action")

    def random_action(self) -> Action:
        """Random legal action for whichever node type we're at (chance or player)."""
        legal = self.legal_actions()
        return self.rng.choice(legal)