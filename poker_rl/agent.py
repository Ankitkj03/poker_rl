"""Double DQN agent with action masking and experience replay."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os


class QNet(nn.Module):
    def __init__(self, state_dim: int = 113, n_actions: int = 5, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 200_000):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s,  dtype=np.float32),
            np.array(a,  dtype=np.int64),
            np.array(r,  dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d,  dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    """Double DQN with epsilon-greedy exploration and action masking."""

    def __init__(
        self,
        state_dim:      int   = 113,
        n_actions:      int   = 5,
        hidden:         int   = 256,
        lr:             float = 1e-3,
        gamma:          float = 0.99,
        eps_start:      float = 1.0,
        eps_end:        float = 0.05,
        eps_decay:      int   = 80_000,   # steps to decay over
        target_update:  int   = 500,
        batch_size:     int   = 256,
        buffer_cap:     int   = 200_000,
        device:         str   = None,
    ):
        self.n_actions    = n_actions
        self.gamma        = gamma
        self.eps          = eps_start
        self.eps_end      = eps_end
        self.eps_decay    = eps_decay
        self.target_update = target_update
        self.batch_size   = batch_size
        self.steps        = 0

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.q        = QNet(state_dim, n_actions, hidden).to(self.device)
        self.q_target = QNet(state_dim, n_actions, hidden).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        self.optim  = optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_cap)

    # ------------------------------------------------------------------
    def act(self, state: np.ndarray, mask: np.ndarray = None, greedy: bool = False) -> int:
        """Return an action.  Pass greedy=True for evaluation (no exploration)."""
        self.eps = self.eps_end + (1.0 - self.eps_end) * np.exp(-self.steps / self.eps_decay)

        if not greedy and np.random.random() < self.eps:
            valid = np.where(mask)[0] if mask is not None else np.arange(self.n_actions)
            return int(np.random.choice(valid))

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.q(state_t).squeeze(0).cpu().numpy()

        if mask is not None:
            q_vals[~mask] = -1e9

        return int(np.argmax(q_vals))

    # ------------------------------------------------------------------
    def push(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.q_target.load_state_dict(self.q.state_dict())

    def train_step(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, ns, d = self.buffer.sample(self.batch_size)

        s_t  = torch.FloatTensor(s).to(self.device)
        a_t  = torch.LongTensor(a).to(self.device)
        r_t  = torch.FloatTensor(r).to(self.device)
        ns_t = torch.FloatTensor(ns).to(self.device)
        d_t  = torch.FloatTensor(d).to(self.device)

        # Current Q
        q_curr = self.q(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            best_a = self.q(ns_t).argmax(1)
            q_next = self.q_target(ns_t).gather(1, best_a.unsqueeze(1)).squeeze(1)
            target = r_t + self.gamma * q_next * (1.0 - d_t)

        loss = nn.functional.smooth_l1_loss(q_curr, target)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.optim.step()

        return loss.item()

    # ------------------------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'q':      self.q.state_dict(),
            'optim':  self.optim.state_dict(),
            'steps':  self.steps,
            'eps':    self.eps,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q.load_state_dict(ckpt['q'])
        self.q_target.load_state_dict(ckpt['q'])
        self.optim.load_state_dict(ckpt['optim'])
        self.steps = ckpt['steps']
        self.eps   = ckpt['eps']
