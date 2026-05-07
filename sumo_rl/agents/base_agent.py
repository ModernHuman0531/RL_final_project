"""
DQN Agent with Acceptance-Rejection (AR) safety constraint.

The AR method (proposed approach) guarantees C(s)-feasibility with zero
QP overhead: sample an action from the greedy policy; if it is invalid,
resample from the valid subset uniformly. Expected extra samples per step
is 1 / P(valid) — typically < 2 for binary action spaces.

Architecture
------------
  state  →  FC(128, ReLU)  →  FC(128, ReLU)  →  Q(|A|)
  Target network (hard-copy every target_update_freq steps).
  Replay buffer (deque, capacity replay_capacity).
  Epsilon-greedy exploration with linear decay.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Q-Network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    DQN with AR safety projection for a single intersection.

    Args:
        state_dim:           Dimension of the flattened state vector per intersection.
        action_dim:          Number of discrete actions (2 for NS/EW).
        num_intersections:   Number of intersections controlled simultaneously.
        lr:                  Adam learning rate.
        gamma:               Discount factor.
        epsilon_start:       Initial exploration rate.
        epsilon_end:         Minimum exploration rate.
        epsilon_decay_steps: Steps over which epsilon decays linearly.
        batch_size:          Mini-batch size for gradient updates.
        replay_capacity:     Maximum transitions stored.
        target_update_freq:  Hard-copy online → target network every N steps.
        device:              "cpu" or "cuda".
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 2,
        num_intersections: int = 1,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50_000,
        batch_size: int = 64,
        replay_capacity: int = 50_000,
        target_update_freq: int = 500,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_intersections = num_intersections
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)

        # Epsilon schedule
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps

        # Networks — one shared network for all intersections
        self.online_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.replay = ReplayBuffer(replay_capacity)
        self.steps = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, traffic_signals: list) -> np.ndarray:
        """
        AR-safe epsilon-greedy action selection.

        For each intersection:
          - With prob epsilon: sample uniformly from valid actions (safe explore).
          - Otherwise:         take greedy action; if invalid, resample from valid subset.

        Args:
            state:           Flattened joint state (num_intersections * state_dim).
            traffic_signals: List of TrafficSignalEnv objects.
        Returns:
            np.ndarray of shape (num_intersections,) with valid actions.
        """
        actions = []
        for i, signal in enumerate(traffic_signals):
            s_i = state[i * self.state_dim: (i + 1) * self.state_dim]
            valid = signal.get_valid_actions()
            valid_indices = [a for a, v in enumerate(valid) if v == 1]

            if random.random() < self.epsilon:
                actions.append(random.choice(valid_indices))
            else:
                actions.append(self._ar_greedy(s_i, valid_indices))

        self.steps += 1
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        return np.array(actions, dtype=np.int32)

    def store(self, state, action, reward, next_state, done):
        """Push a single transition into the replay buffer."""
        self.replay.push(state, action, reward, next_state, done)

    def update(self) -> float | None:
        """
        Sample a mini-batch and perform one gradient step.
        Returns the loss value, or None if the buffer is too small.
        """
        if len(self.replay) < self.batch_size:
            return None

        s, a, r, ns, d = self.replay.sample(self.batch_size)
        s  = torch.FloatTensor(s).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        r  = torch.FloatTensor(r).to(self.device)
        d  = torch.FloatTensor(d).to(self.device)

        # Use only the first intersection's slice for the shared network.
        # For multi-intersection, this averages gradients across all slices.
        q_vals = self._batch_q(s, a)
        with torch.no_grad():
            next_q = self._batch_target(ns)
        targets = r + self.gamma * next_q * (1.0 - d)

        loss = self.loss_fn(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def reset(self):
        """Called at the start of each episode — nothing to reset for DQN."""
        pass

    def save(self, path: str):
        torch.save(self.online_net.state_dict(), path)

    def load(self, path: str):
        self.online_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.online_net.state_dict())

    def get_q_fn(self):
        """Return a callable suitable for SPRePlusAgent.set_policy()."""
        def policy_fn(state: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return self.online_net(t).squeeze(0).cpu().numpy()
        return policy_fn

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ar_greedy(self, state: np.ndarray, valid_indices: list) -> int:
        """Greedy action; fall back to valid subset if greedy is infeasible."""
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.online_net(t).squeeze(0).cpu().numpy()
        greedy = int(np.argmax(q))
        if greedy in valid_indices:
            return greedy
        # AR rejection: pick best valid action
        valid_q = {a: q[a] for a in valid_indices}
        return max(valid_q, key=valid_q.get)

    def _batch_q(self, states: torch.Tensor, actions) -> torch.Tensor:
        """Q(s, a) for the sampled actions — handles multi-intersection batches."""
        n = self.num_intersections
        total_q = torch.zeros(states.shape[0], device=self.device)
        for i in range(n):
            s_i = states[:, i * self.state_dim: (i + 1) * self.state_dim]
            # actions is (batch,) for single intersection, sum across intersections otherwise
            if n == 1:
                a_i = torch.LongTensor(actions).to(self.device)
            else:
                a_i = torch.LongTensor([act[i] if hasattr(act, '__len__') else act
                                        for act in actions]).to(self.device)
            q_i = self.online_net(s_i).gather(1, a_i.unsqueeze(1)).squeeze(1)
            total_q += q_i
        return total_q / n

    def _batch_target(self, next_states: torch.Tensor) -> torch.Tensor:
        """max Q_target(s', ·) averaged across intersections."""
        n = self.num_intersections
        total = torch.zeros(next_states.shape[0], device=self.device)
        for i in range(n):
            ns_i = next_states[:, i * self.state_dim: (i + 1) * self.state_dim]
            total += self.target_net(ns_i).max(1)[0]
        return total / n
