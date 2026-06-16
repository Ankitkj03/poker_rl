"""2-player No-Limit Texas Hold'em environment."""

import numpy as np
from collections import deque
from poker_rl.cards import make_deck, best_hand, card_str


class PokerEnv:
    """
    Heads-up NLHE.  Player 0 = dealer/small-blind, Player 1 = big-blind.

    Observation (113 floats, from the acting player's POV):
        [0:52]   hole cards (one-hot)
        [52:104] community cards (one-hot)
        [104]    pot  / STARTING_STACK
        [105]    my stack  / STARTING_STACK
        [106]    opp stack / STARTING_STACK
        [107]    amount to call / STARTING_STACK
        [108]    1 if I am the dealer (SB)
        [109:113] street one-hot  (preflop / flop / turn / river)

    Actions:
        0 FOLD  1 CHECK/CALL  2 RAISE_MIN  3 RAISE_POT  4 ALL_IN

    Reward: chips won (>0) or lost (<0), normalised by STARTING_STACK.
            Returned from the perspective of the player who just acted.
    """

    FOLD, CALL, RAISE_MIN, RAISE_POT, ALL_IN = 0, 1, 2, 3, 4
    N_ACTIONS  = 5
    STATE_DIM  = 113
    BIG_BLIND  = 2
    SMALL_BLIND = 1
    STARTING_STACK = 200   # 100 BBs

    # ------------------------------------------------------------------
    def __init__(self, starting_stack: int = None):
        if starting_stack:
            self.STARTING_STACK = starting_stack
        self.stacks = [self.STARTING_STACK, self.STARTING_STACK]
        self._new_hand()

    # ------------------------------------------------------------------
    def reset(self):
        self.stacks = [self.STARTING_STACK, self.STARTING_STACK]
        return self._new_hand()

    def _new_hand(self):
        deck = make_deck()
        np.random.shuffle(deck)
        self.hole       = [deck[:2], deck[2:4]]
        self.deck       = deck
        self.community  = []
        self.pot        = 0
        self.bets       = [0, 0]   # chips in per street
        self.street     = 0        # 0=preflop 1=flop 2=turn 3=river
        self.deck_ptr   = 4
        self.done       = False

        # Post blinds: P0=SB, P1=BB
        self._put_in(0, self.SMALL_BLIND)
        self._put_in(1, self.BIG_BLIND)

        # Preflop: SB (P0) acts first in heads-up, BB (P1) has option after
        self.action_q      = deque([0, 1])
        self.current_player = 0
        return self._obs(0)

    # ------------------------------------------------------------------
    def step(self, action: int):
        assert not self.done, "Hand is over — call reset() or _new_hand()."
        p   = self.action_q.popleft()
        opp = 1 - p

        to_call  = max(0, self.bets[opp] - self.bets[p])
        min_raise = to_call + max(self.BIG_BLIND, to_call)
        raised   = False

        if action == self.FOLD:
            return self._end(winner=opp)

        elif action == self.CALL:
            call_amt = min(to_call, self.stacks[p])
            self._put_in(p, call_amt)
            # If all-in for less, return uncalled bet to opponent
            if call_amt < to_call:
                excess = to_call - call_amt
                self.pot          -= excess
                self.stacks[opp]  += excess
                self.bets[opp]    -= excess

        elif action == self.RAISE_MIN:
            if self.stacks[p] <= to_call:
                self._put_in(p, self.stacks[p])    # all-in call
            else:
                amt = min(min_raise, self.stacks[p])
                self._put_in(p, amt)
                raised = True

        elif action == self.RAISE_POT:
            if self.stacks[p] <= to_call:
                self._put_in(p, self.stacks[p])
            else:
                pot_raise = to_call + self.pot
                amt = min(max(pot_raise, min_raise), self.stacks[p])
                self._put_in(p, amt)
                raised = True

        elif action == self.ALL_IN:
            amt = self.stacks[p]
            if amt > to_call:
                raised = True
            self._put_in(p, amt)

        # After a raise the opponent must respond (add once if not present)
        if raised and opp not in self.action_q:
            self.action_q.append(opp)

        if len(self.action_q) == 0:
            return self._next_street()

        self.current_player = self.action_q[0]
        return self._obs(self.current_player), 0.0, False, {}

    # ------------------------------------------------------------------
    def _put_in(self, player: int, amount: int):
        amount = min(amount, self.stacks[player])
        self.stacks[player] -= amount
        self.bets[player]   += amount
        self.pot            += amount

    def _next_street(self):
        if self.street == 3:
            return self._showdown()

        self.street   += 1
        self.bets      = [0, 0]
        self.action_q  = deque()   # will be set below

        if self.street == 1:
            self.community.extend(self.deck[self.deck_ptr:self.deck_ptr + 3])
            self.deck_ptr += 3
        else:
            self.community.append(self.deck[self.deck_ptr])
            self.deck_ptr += 1

        # If either player is already all-in, run remaining streets out
        if self.stacks[0] == 0 or self.stacks[1] == 0:
            return self._run_it_out()

        # Post-flop: OOP player (P1/BB) acts first
        self.action_q       = deque([1, 0])
        self.current_player = 1
        return self._obs(1), 0.0, False, {}

    def _run_it_out(self):
        """Deal remaining community cards without betting and go to showdown."""
        while self.street < 3:
            self.street += 1
            if self.street == 1:
                self.community.extend(self.deck[self.deck_ptr:self.deck_ptr + 3])
                self.deck_ptr += 3
            else:
                self.community.append(self.deck[self.deck_ptr])
                self.deck_ptr += 1
        return self._showdown()

    def _showdown(self):
        s0 = best_hand(self.hole[0] + self.community)
        s1 = best_hand(self.hole[1] + self.community)
        if   s0 > s1: winner = 0
        elif s1 > s0: winner = 1
        else:         winner = -1   # chop
        return self._end(winner)

    def _end(self, winner: int):
        if winner == -1:
            half = self.pot // 2
            self.stacks[0] += half
            self.stacks[1] += self.pot - half
            reward = 0.0
        else:
            self.stacks[winner] += self.pot
            raw = self.pot if winner == 0 else -self.pot
            reward = raw / self.STARTING_STACK

        self.done           = True
        self.current_player = -1
        info = {'winner': winner, 'pot': self.pot}
        return self._obs(0), float(reward), True, info

    # ------------------------------------------------------------------
    def _obs(self, player: int) -> np.ndarray:
        obs = np.zeros(self.STATE_DIM, dtype=np.float32)
        for c in self.hole[player]:
            obs[c] = 1.0
        for c in self.community:
            obs[52 + c] = 1.0
        norm = float(self.STARTING_STACK)
        obs[104] = self.pot / norm
        obs[105] = self.stacks[player] / norm
        obs[106] = self.stacks[1 - player] / norm
        obs[107] = max(0, self.bets[1 - player] - self.bets[player]) / norm
        obs[108] = 1.0 if player == 0 else 0.0
        obs[109 + self.street] = 1.0
        return obs

    def get_action_mask(self) -> np.ndarray:
        """Boolean mask of legal actions for current_player."""
        p        = self.current_player
        to_call  = max(0, self.bets[1 - p] - self.bets[p])
        mask     = np.ones(self.N_ACTIONS, dtype=bool)
        if to_call == 0:
            mask[self.FOLD] = False        # no need to fold if free to check
        if self.stacks[p] <= to_call:
            mask[self.RAISE_MIN] = False   # can't raise if can barely call
            mask[self.RAISE_POT] = False
        return mask

    # ------------------------------------------------------------------
    def render(self):
        street_names = ['Preflop', 'Flop', 'Turn', 'River']
        h0 = [card_str(c) for c in self.hole[0]]
        h1 = [card_str(c) for c in self.hole[1]]
        cm = [card_str(c) for c in self.community]
        print(f"\n=== {street_names[self.street]} ===")
        print(f"  P0 (SB/dealer): {h0}  stack={self.stacks[0]}")
        print(f"  P1 (BB):        {h1}  stack={self.stacks[1]}")
        print(f"  Board: {cm}   Pot: {self.pot}")
        print(f"  Street bets: P0={self.bets[0]}  P1={self.bets[1]}")
        if not self.done:
            print(f"  To act: Player {self.current_player}")
