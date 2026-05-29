"""
Double DQN Agent
=================
Addresses Q-value overestimation in DQN by decoupling action selection
from action evaluation using separate online and target networks.

Update rule:
  a* = argmax_a Q_online(s', a)        # select with online net
  target = r + γ * Q_target(s', a*)   # evaluate with target net

Reference: van Hasselt et al., "Deep RL with Double Q-Learning" (AAAI 2016)
"""

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from typing import List, Optional
import logging

from ..models.q_network import QNetwork
from ..training.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class DoubleDQNAgent:
    """
    Double DQN: decouples action selection from evaluation to reduce overestimation.

    Key difference from DQN:
      DQN target:       r + γ * max_a Q_target(s', a)
      Double DQN target: r + γ * Q_target(s', argmax_a Q_online(s', a))

    Args:
        state_dim: Input state embedding dimension
        action_dim: Number of discrete actions
        hidden_dims: Q-network hidden layer sizes
        lr: Adam learning rate
        gamma: Discount factor
        epsilon_start / epsilon_end / epsilon_decay: ε-greedy schedule
        buffer_capacity: Replay buffer size
        batch_size: Gradient update batch size
        target_update_freq: Steps between target net syncs
        device: torch device
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

        self.q_net = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net = deepcopy(self.q_net)
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.loss_fn = nn.HuberLoss()
        self.steps = 0
        self.losses: List[float] = []
        logger.info(f"DoubleDQNAgent initialized: state_dim={state_dim}, actions={action_dim}")

    def select_action(self, state_emb: torch.Tensor) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            if state_emb.dim() == 1:
                state_emb = state_emb.unsqueeze(0)
            return self.q_net(state_emb).argmax(dim=-1).item()

    def push(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def update(self, embeddings: torch.Tensor) -> Optional[float]:
        if not self.buffer.is_ready(self.batch_size):
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        state_embs = embeddings[states]
        next_state_embs = embeddings[next_states]

        # Current Q-values
        q_vals = self.q_net(state_embs).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: online net selects, target net evaluates
            best_actions = self.q_net(next_state_embs).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_state_embs).gather(1, best_actions).squeeze(1)
            targets = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.steps += 1

        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        loss_val = loss.item()
        self.losses.append(loss_val)
        return loss_val
