"""Card representation and hand evaluation for Texas Hold'em."""

import numpy as np
from itertools import combinations
from collections import Counter

RANKS = '23456789TJQKA'   # index 0-12
SUITS = 'cdhs'            # index 0-3


def make_deck():
    return list(range(52))


def card_rank(c):
    return c >> 2


def card_suit(c):
    return c & 3


def card_str(c):
    return RANKS[card_rank(c)] + SUITS[card_suit(c)]


def str_to_card(s):
    """'As' -> 51, 'Tc' -> 32, etc."""
    return RANKS.index(s[0]) * 4 + SUITS.index(s[1])


def _eval5(cards):
    """
    Evaluate 5 cards. Returns a comparable tuple — higher is better.
    Categories: 8=str-flush, 7=quads, 6=boat, 5=flush, 4=straight,
                3=trips, 2=two-pair, 1=pair, 0=high-card
    """
    ranks = sorted([card_rank(c) for c in cards], reverse=True)
    suits = [card_suit(c) for c in cards]

    is_flush = len(set(suits)) == 1

    rank_set = sorted(set(ranks))
    is_straight = False
    straight_high = -1
    if len(rank_set) == 5:
        if rank_set[-1] - rank_set[0] == 4:
            is_straight = True
            straight_high = rank_set[-1]
        elif rank_set == [0, 1, 2, 3, 12]:   # wheel A-2-3-4-5
            is_straight = True
            straight_high = 3

    if is_flush and is_straight:
        return (8, straight_high)

    cnt = Counter(ranks)
    freq = sorted(cnt.items(), key=lambda x: (x[1], x[0]), reverse=True)
    groups = [f[1] for f in freq]
    group_ranks = [f[0] for f in freq]

    if groups[0] == 4:
        return (7, group_ranks[0], group_ranks[1])
    if groups[0] == 3 and groups[1] == 2:
        return (6, group_ranks[0], group_ranks[1])
    if is_flush:
        return (5, *ranks)
    if is_straight:
        return (4, straight_high)
    if groups[0] == 3:
        return (3, group_ranks[0], *group_ranks[1:])
    if groups[0] == 2 and groups[1] == 2:
        return (2, group_ranks[0], group_ranks[1], group_ranks[2])
    if groups[0] == 2:
        return (1, group_ranks[0], *group_ranks[1:])
    return (0, *ranks)


def best_hand(cards):
    """Best 5-card hand from 5-7 cards."""
    if len(cards) <= 5:
        return _eval5(cards)
    return max(_eval5(list(combo)) for combo in combinations(cards, 5))


def hand_name(score):
    names = ['High Card', 'Pair', 'Two Pair', 'Trips',
             'Straight', 'Flush', 'Full House', 'Quads', 'Straight Flush']
    return names[score[0]]
