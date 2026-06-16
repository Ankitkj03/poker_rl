"""Evaluate a trained agent against fixed baselines."""

import numpy as np
from poker_rl.env   import PokerEnv
from poker_rl.agent import DQNAgent


# ---------------------------------------------------------------------------
# Baseline policies  (obs: np.ndarray, mask: np.ndarray) -> int
# ---------------------------------------------------------------------------

def random_policy(obs, mask):
    valid = np.where(mask)[0]
    return int(np.random.choice(valid))


def call_policy(obs, mask):
    return 1   # always check/call


def fold_policy(obs, mask):
    return 0 if mask[0] else 1   # fold when allowed, else check


def aggressive_policy(obs, mask):
    """Always raise/all-in if possible, else call."""
    for a in (4, 3, 2, 1):   # ALL_IN → RAISE_POT → RAISE_MIN → CALL
        if mask[a]:
            return a
    return 0


# ---------------------------------------------------------------------------

def _make_agent_policy(agent: DQNAgent):
    def policy(obs, mask):
        return agent.act(obs, mask, greedy=True)
    return policy


def run_match(
    policy_0,
    policy_1,
    n_hands: int = 2_000,
    starting_stack: int = 200,
) -> tuple[float, float]:
    """
    Returns (mean_reward_p0, stderr).
    Reward is normalised by starting_stack (so ±1 means winning/losing a full stack).
    """
    env     = PokerEnv(starting_stack=starting_stack)
    rewards = []

    for _ in range(n_hands):
        obs  = env.reset()
        done = False

        while not done:
            p    = env.current_player
            mask = env.get_action_mask()
            curr_obs = obs

            if p == 0:
                action = policy_0(curr_obs, mask)
            else:
                action = policy_1(env._obs(1), mask)

            obs, reward, done, info = env.step(action)

        rewards.append(reward)

    arr  = np.array(rewards)
    mean = arr.mean()
    std  = arr.std() / np.sqrt(len(arr))
    return float(mean), float(std)


def evaluate_agent(agent: DQNAgent, n_hands: int = 2_000):
    """Print win-rate vs several baselines."""
    ap = _make_agent_policy(agent)
    baselines = {
        'Random':     random_policy,
        'Call-only':  call_policy,
        'Aggressive': aggressive_policy,
    }

    print(f"\n{'Opponent':<15} {'Avg reward':>12}  {'±95% CI':>10}")
    print('-' * 42)
    for name, bl in baselines.items():
        mean, se = run_match(ap, bl, n_hands=n_hands)
        ci = 1.96 * se
        print(f"  {name:<13} {mean:>+12.4f}  ±{ci:.4f}")
    print()


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse, sys

    p = argparse.ArgumentParser()
    p.add_argument('model', help='Path to saved .pt checkpoint')
    p.add_argument('--hands', type=int, default=5_000)
    args = p.parse_args()

    agent = DQNAgent()
    agent.load(args.model)
    evaluate_agent(agent, n_hands=args.hands)
