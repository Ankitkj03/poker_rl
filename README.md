# poker_rl

A reinforcement learning agent for heads-up No-Limit Texas Hold'em, built from scratch in PyTorch.

Trains a Double DQN agent via self-play — no hand-crafted heuristics, no lookup tables. After 50k episodes:

```
Opponent          Avg reward    ±95% CI
------------------------------------------
  Random             +0.2171   ±0.0393
  Call-only          +0.0662   ±0.0446
  Aggressive         +0.0307   ±0.0469
```

## Structure

```
poker_rl/
├── cards.py      # Card encoding and 5/7-card hand evaluator
├── env.py        # Heads-up NLHE game engine
├── agent.py      # Double DQN with action masking and replay buffer
├── train.py      # Self-play training loop
├── evaluate.py   # Evaluation vs rule-based baselines
└── test_env.py   # 15 unit tests (no external test runner needed)
```

## Installation

```bash
pip install torch numpy
```

Tested on Python 3.10+, PyTorch 2.0+, CPU-only.

## Training

```bash
# Fresh run
python -m poker_rl.train --episodes 50000 --save-dir models

# Resume from a checkpoint
python -m poker_rl.train --episodes 50000 --resume models/dqn_final.pt --save-dir models
```

Checkpoints save every 20k episodes to `models/dqn_XXXXXXX.pt`. Final model saved to `models/dqn_final.pt`.

Log output:
```
ep    2,000 | avg_reward +0.0193 | loss 0.07182 | eps 0.927 | buf 6,441
ep    4,000 | avg_reward +0.0406 | loss 0.12957 | eps 0.858 | buf 12,973
...
```

## Evaluation

```bash
python -m poker_rl.evaluate models/dqn_final.pt --hands 5000
```

Runs the agent against three rule-based baselines over N hands and reports mean reward ± 95% CI:

| Baseline | Strategy |
|---|---|
| Random | Uniformly random over legal actions |
| Call-only | Always check/call, never raises |
| Aggressive | Always tries All-In → Raise-Pot → Raise-Min → Call |

## Tests

```bash
python -m poker_rl.test_env
```

15 tests covering hand evaluation (straight flush through high card, wheel, 7-card best-hand), environment correctness (chip conservation, fold ends hand, all-in-then-call), and reward normalization. No pytest required.

## How it works

**Environment:** Heads-up NLHE, SB=1 / BB=2 / starting stack=200. Five actions: Fold, Call, Raise-Min, Raise-Pot, All-In. A deque-based action queue handles the BB option, re-raises, and all-in-for-less correctly.

**State (113-dim float32):**

| Slice | Content |
|---|---|
| `[0:52]` | Hole cards (one-hot) |
| `[52:104]` | Community cards (one-hot) |
| `[104:109]` | Pot, stack 0, stack 1, amount to call, dealer flag — normalized by starting stack |
| `[109:113]` | Street one-hot (preflop / flop / turn / river) |

**Agent:** Double DQN — online network picks the best next action, target network evaluates it (decouples selection from evaluation to reduce overestimation). Target network synced every 500 gradient steps. Illegal actions masked to -1e9 before argmax.

**Credit assignment:** Monte-Carlo. All transitions in a hand receive the terminal reward (`±pot / 200`). No intermediate rewards, so discount factor γ barely matters in practice.

**Self-play:** Both seats share one policy. The "opponent" is the agent itself, so the training distribution shifts each episode — this is standard in self-play RL and is stabilized by a large replay buffer (200k) and slow epsilon decay.

## Why not CFR?

Poker is a POMDP — the opponent's hole cards are hidden, so the true game state is not fully observable. CFR handles this correctly by reasoning over *information sets* (all histories consistent with what a player can observe) and converges to Nash equilibrium. Libratus and Pluribus use CFR variants.

DQN approximates the POMDP as an MDP using only observable state. It learns to beat common opponents but does not converge to Nash — a sufficiently adaptive opponent can still find exploits.
