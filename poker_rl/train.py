"""Self-play training loop for the DQN poker agent."""

import numpy as np
import os
from poker_rl.env   import PokerEnv
from poker_rl.agent import DQNAgent


def run_episode(env: PokerEnv, agent: DQNAgent, train: bool = True):
    """
    Play one hand with both seats using the same agent policy.
    All transitions get the hand's final reward (Monte-Carlo credit assignment).
    Returns (reward_p0, avg_loss).
    """
    obs  = env.reset()
    done = False

    history = []   # (player, obs, action, mask)

    while not done:
        p    = env.current_player
        mask = env.get_action_mask()
        curr_obs = obs.copy()          # obs is already from current player's POV
        action   = agent.act(curr_obs, mask)

        obs, reward, done, info = env.step(action)
        history.append((p, curr_obs, action, mask, obs.copy()))

    # Push all transitions — final reward from each player's perspective
    for p, s, a, m, ns in history:
        r = reward if p == 0 else -reward
        agent.push(s, a, r, ns, True)

    # Train once per step in the episode
    losses = []
    if train:
        for _ in range(len(history)):
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

    return reward, (float(np.mean(losses)) if losses else 0.0)


# ---------------------------------------------------------------------------
def train(
    n_episodes:  int   = 200_000,
    save_every:  int   = 20_000,
    log_every:   int   = 2_000,
    save_dir:    str   = 'models',
    resume:      str   = None,
    **agent_kwargs,
) -> DQNAgent:
    os.makedirs(save_dir, exist_ok=True)

    env   = PokerEnv()
    agent = DQNAgent(**agent_kwargs)
    if resume:
        agent.load(resume)
        print(f"Resumed from {resume} | steps={agent.steps:,} eps={agent.eps:.3f}")

    reward_window = []
    loss_window   = []

    for ep in range(1, n_episodes + 1):
        r, loss = run_episode(env, agent)
        reward_window.append(r)
        loss_window.append(loss)

        if ep % log_every == 0:
            avg_r = np.mean(reward_window[-log_every:])
            avg_l = np.mean([l for l in loss_window[-log_every:] if l > 0] or [0.0])
            print(
                f"ep {ep:>8,} | "
                f"avg_reward {avg_r:+.4f} | "
                f"loss {avg_l:.5f} | "
                f"eps {agent.eps:.3f} | "
                f"buf {len(agent.buffer):,}"
            )

        if ep % save_every == 0:
            path = os.path.join(save_dir, f'dqn_{ep:07d}.pt')
            agent.save(path)
            print(f"  [saved {path}]")

    agent.save(os.path.join(save_dir, 'dqn_final.pt'))
    return agent


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int,   default=200_000)
    p.add_argument('--lr',       type=float, default=1e-3)
    p.add_argument('--hidden',   type=int,   default=256)
    p.add_argument('--save-dir', type=str,   default='models')
    p.add_argument('--resume',   type=str,   default=None)
    args = p.parse_args()

    train(
        n_episodes = args.episodes,
        lr         = args.lr,
        hidden     = args.hidden,
        save_dir   = args.save_dir,
        resume     = args.resume,
    )
