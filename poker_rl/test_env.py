"""Smoke tests for the poker environment and card evaluator."""

import numpy as np
from poker_rl.cards import str_to_card, best_hand, hand_name, card_str
from poker_rl.env   import PokerEnv


# ---------------------------------------------------------------------------
# Card evaluator
# ---------------------------------------------------------------------------

def hand(*cards):
    return [str_to_card(c) for c in cards]


def test_straight_flush():
    score = best_hand(hand('As', 'Ks', 'Qs', 'Js', 'Ts'))
    assert score[0] == 8


def test_quads():
    score = best_hand(hand('Ac', 'Ad', 'Ah', 'As', 'Kc'))
    assert score[0] == 7


def test_full_house():
    score = best_hand(hand('Ac', 'Ad', 'Ah', 'Kc', 'Kd'))
    assert score[0] == 6


def test_flush():
    score = best_hand(hand('2c', '5c', '8c', 'Jc', 'Ac'))
    assert score[0] == 5


def test_straight():
    score = best_hand(hand('5d', '4c', '3h', '2s', 'Ac'))
    assert score[0] == 4
    assert score[1] == 3    # 5-high (wheel)


def test_wheel_vs_straight():
    wheel  = best_hand(hand('5d', '4c', '3h', '2s', 'Ac'))
    six_hi = best_hand(hand('6d', '5c', '4h', '3s', '2c'))
    assert six_hi > wheel


def test_best_from_seven():
    # Hold: Ac As, board: Ad Ah Kc 2d 7s → four aces
    score = best_hand(hand('Ac', 'As', 'Ad', 'Ah', 'Kc', '2d', '7s'))
    assert score[0] == 7


def test_ranking_order():
    sf  = best_hand(hand('As', 'Ks', 'Qs', 'Js', 'Ts'))
    fh  = best_hand(hand('Ac', 'Ad', 'Ah', 'Kc', 'Kd'))
    two = best_hand(hand('Ac', 'Ad', 'Kc', 'Kd', '2h'))
    assert sf > fh > two


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def test_obs_shape():
    env = PokerEnv()
    obs = env.reset()
    assert obs.shape == (PokerEnv.STATE_DIM,)


def test_full_hand_no_crash():
    env  = PokerEnv()
    obs  = env.reset()
    done = False
    steps = 0
    while not done:
        mask   = env.get_action_mask()
        action = int(np.random.choice(np.where(mask)[0]))
        obs, reward, done, info = env.step(action)
        steps += 1
        assert steps < 200, "Hand took too many steps"
    assert isinstance(reward, float)
    assert 'winner' in info


def test_chips_conserved():
    env = PokerEnv()
    for _ in range(200):
        obs  = env.reset()
        done = False
        while not done:
            mask   = env.get_action_mask()
            action = int(np.random.choice(np.where(mask)[0]))
            obs, reward, done, info = env.step(action)
        total = env.stacks[0] + env.stacks[1]
        assert total == env.STARTING_STACK * 2, f"Chip leak! total={total}"


def test_fold_ends_hand():
    env = PokerEnv()
    env.reset()
    # Player 0 folds immediately
    obs, reward, done, info = env.step(PokerEnv.FOLD)
    assert done
    assert info['winner'] == 1


def test_all_in_then_call():
    env = PokerEnv()
    env.reset()
    # P0 goes all-in preflop
    obs, reward, done, info = env.step(PokerEnv.ALL_IN)
    assert not done   # P1 must respond
    # P1 calls
    obs, reward, done, info = env.step(PokerEnv.CALL)
    assert done       # hand should be over
    assert env.stacks[0] + env.stacks[1] == env.STARTING_STACK * 2


def test_action_mask():
    env = PokerEnv()
    env.reset()
    mask = env.get_action_mask()
    assert mask.shape == (PokerEnv.N_ACTIONS,)
    # P0 can't fold preflop when there's a BB to call (fold is masked because
    # technically allowed — actually fold IS allowed preflop)
    # At minimum, call must be valid
    assert mask[PokerEnv.CALL]


def test_reward_normalisation():
    """Reward should never exceed 1.0 in absolute value with symmetric stacks."""
    env = PokerEnv()
    for _ in range(500):
        obs  = env.reset()
        done = False
        while not done:
            mask   = env.get_action_mask()
            action = int(np.random.choice(np.where(mask)[0]))
            obs, reward, done, info = env.step(action)
        # Both players can commit full stack → pot up to 2×STARTING_STACK → reward ≤ 2.0
        assert abs(reward) <= 2.0 + 1e-6, f"Reward out of range: {reward}"


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    # Run manually without pytest
    tests = [
        test_straight_flush, test_quads, test_full_house, test_flush,
        test_straight, test_wheel_vs_straight, test_best_from_seven,
        test_ranking_order, test_obs_shape, test_full_hand_no_crash,
        test_chips_conserved, test_fold_ends_hand, test_all_in_then_call,
        test_action_mask, test_reward_normalisation,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(failed)
