"""
Deep Q-Network (DQN) Agent
===========================
Production implementation of DQN with:
  - Target network for stable training
  - Epsilon-greedy exploration with decay
  - Gradient clipping
  - Huber loss for robustness

Reference: Mnih et al., "Human-level control through deep reinforcement learning" (Nature 2015)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, List
from copy import deepcopy
import logging

from ..models.q_network import QNetwork
from ..training.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class DQNAgent:
    """
    Deep Q-Network agent with experience replay and target network.

    Key innovations over vanilla Q-learning:
      1. Neural network function approximator (generalizes across states)
      2. Experience replay buffer (breaks temporal correlations)
      3. Target network (stabilizes bootstrapping targets)

    Args:
        state_dim: Input state dimension
        action_dim: Number of discrete actions
        hidden_dims: Q-network hidden layer sizes
        lr: Learning rate
        gamma: Discount factor
        epsilon_start: Initial exploration rate
        epsilon_end: Minimum exploration rate
        epsilon_decay: Multiplicative decay per step
        buffer_capacity: Replay buffer size
        batch_size: Mini-batch size for updates
        target_update_freq: Steps between target network syncs
        device: Training device
    """

    def __init__(
        self,
        state_dim: int = 512,
        action_dim: int = 4,
        hidden_dims: List[int] = [256, 128],
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 50_000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: str = "cpu",
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)

        # Online and target networks
        self.q_net = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net = deepcopy(self.q_net)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.loss_fn = nn.HuberLoss()

        self.steps = 0
        self.losses: List[float] = []
        logger.info(f"DQNAgent: state_dim={state_dim}, action_dim={action_dim}")

    def select_action(self, state_emb: torch.Tensor) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            if state_emb.dim() == 1:
                state_emb = state_emb.unsqueeze(0)
            return self.q_net(state_emb).argmax(dim=-1).item()

    def push(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def update(self, embeddings: torch.Tensor) -> Optional[float]:
        """
        Sample mini-batch and perform one gradient update.

        Args:
            embeddings: Full node embedding matrix (N, embed_dim)

        Returns:
            loss: Scalar loss value, or None if buffer not ready
        """
        if not self.buffer.is_ready(self.batch_size):
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # State embeddings
        state_embs = embeddings[states]
        next_state_embs = embeddings[next_states]

        # Current Q-values
        q_vals = self.q_net(state_embs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (no gradient through target net)
        with torch.no_grad():
            next_q = self.target_net(next_state_embs).max(1)[0]
            targets = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        # Decay exploration
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.steps += 1

        # Sync target network periodically
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        loss_val = loss.item()
        self.losses.append(loss_val)
        return loss_val

    def save(self, path: str) -> None:
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "epsilon": self.epsilon,
        }, path)

    def load(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ck["q_net"])
        self.target_net.load_state_dict(ck["target_net"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.steps = ck["steps"]
        self.epsilon = ck["epsilon"]
