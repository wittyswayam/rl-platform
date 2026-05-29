"""
Experience Replay Buffer
========================
Standard and Prioritized Experience Replay (PER) implementations.

PER samples transitions with probability proportional to their TD error,
focusing learning on the most informative experiences. Uses a sum-tree
data structure for O(log N) priority updates and sampling.

References:
  - Lin, "Self-improving reactive agents based on RL, planning, and teaching" (1992)
  - Schaul et al., "Prioritized Experience Replay" (ICLR 2016)
"""

import numpy as np
import torch
from typing import Tuple, Optional, NamedTuple
from collections import deque
import random
import logging

logger = logging.getLogger(__name__)


class Transition(NamedTuple):
    """Single environment transition."""
    state: int
    action: int
    reward: float
    next_state: int
    done: bool


class ReplayBuffer:
    """
    Standard uniform experience replay buffer.

    Stores transitions and samples uniformly at random.
    Essential for breaking temporal correlations in RL training.

    Args:
        capacity: Maximum number of transitions to store
        seed: Random seed for reproducibility
    """

    def __init__(self, capacity: int = 100_000, seed: int = 42):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        random.seed(seed)
        np.random.seed(seed)
        logger.info(f"ReplayBuffer: capacity={capacity:,}")

    def push(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        """Add a transition to the buffer."""
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states = torch.tensor([t.state for t in batch], dtype=torch.long)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        next_states = torch.tensor([t.next_state for t in batch], dtype=torch.long)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        return len(self) >= batch_size


class SumTree:
    """
    Binary sum tree for O(log N) priority sampling.

    Maintains a binary tree where leaf nodes store priorities
    and internal nodes store the sum of their children.
    Enables efficient proportional sampling.

    Args:
        capacity: Number of leaf nodes (must be power of 2 for efficiency)
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.size = 0

    def _propagate(self, idx: int, change: float) -> None:
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Find leaf with cumulative sum s."""
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Total priority sum."""
        return self.tree[0]

    def add(self, priority: float, data: object) -> None:
        """Add data with given priority."""
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_idx: int, priority: float) -> None:
        """Update priority at tree index."""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, s: float) -> Tuple[int, float, object]:
        """
        Sample leaf with cumulative sum s.

        Returns:
            (tree_idx, priority, data)
        """
        tree_idx = self._retrieve(0, s)
        data_idx = tree_idx - self.capacity + 1
        return tree_idx, self.tree[tree_idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer.

    Samples transitions with probability proportional to TD error,
    focusing learning on the most "surprising" experiences.

    Key parameters:
      - alpha: Priority exponent (0=uniform, 1=full prioritization)
      - beta: Importance sampling exponent (0=no correction, 1=full)
      - beta_increment: Per-sample increase in beta (anneals to 1.0)

    Args:
        capacity: Maximum buffer size
        alpha: Priority exponent for sampling
        beta_start: Initial importance sampling weight
        beta_increment: Beta annealing rate
        eps: Small constant to prevent zero priorities
    """

    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_increment: float = 0.001,
        eps: float = 1e-6,
    ):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = beta_increment
        self.eps = eps
        self.max_priority = 1.0
        logger.info(
            f"PrioritizedReplayBuffer: capacity={capacity:,}, "
            f"alpha={alpha}, beta_start={beta_start}"
        )

    def push(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        """Add transition with maximum priority (new experiences get high priority)."""
        transition = Transition(state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, transition)

    def sample(
        self, batch_size: int
    ) -> Tuple[Tuple, np.ndarray, np.ndarray]:
        """
        Sample batch with importance sampling weights.

        Splits total priority into batch_size equal segments and
        samples one transition per segment for stability.

        Args:
            batch_size: Number of transitions

        Returns:
            transitions: Batched transition tensors
            indices: Tree indices (for priority updates)
            weights: Importance sampling weights (IS correction)
        """
        self.beta = min(1.0, self.beta + self.beta_increment)

        indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float64)
        transitions = []

        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            tree_idx, priority, data = self.tree.get(s)
            indices[i] = tree_idx
            priorities[i] = priority
            transitions.append(data)

        # Importance sampling weights to correct for non-uniform sampling
        sampling_probs = priorities / self.tree.total()
        weights = (self.tree.size * sampling_probs) ** (-self.beta)
        weights /= weights.max()  # Normalize so max weight = 1.0

        # Unpack transitions into tensors
        states = torch.tensor([t.state for t in transitions], dtype=torch.long)
        actions = torch.tensor([t.action for t in transitions], dtype=torch.long)
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32)
        next_states = torch.tensor([t.next_state for t in transitions], dtype=torch.long)
        dones = torch.tensor([t.done for t in transitions], dtype=torch.float32)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)

        return (states, actions, rewards, next_states, dones, weights_tensor), indices

    def update_priorities(
        self, indices: np.ndarray, td_errors: np.ndarray
    ) -> None:
        """
        Update priorities based on new TD errors.

        Args:
            indices: Tree indices from sample()
            td_errors: Absolute TD errors for each sampled transition
        """
        priorities = (np.abs(td_errors) + self.eps) ** self.alpha
        for idx, priority in zip(indices, priorities):
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return self.tree.size

    def is_ready(self, batch_size: int) -> bool:
        return len(self) >= batch_size
