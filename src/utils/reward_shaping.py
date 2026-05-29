"""
Reward Shaping
==============
Auxiliary reward signals to accelerate learning in sparse/delayed reward settings.

Implements:
  - Potential-based reward shaping (guaranteed policy invariance)
  - Curiosity-driven intrinsic rewards (prediction error)
  - Count-based exploration bonuses
  - Temporal distance shaping
  - InferNet-based reward densification

Key theoretical guarantee (Ng et al., 1999):
  Any potential-based shaping F(s,s') = γΦ(s') - Φ(s) preserves
  the optimal policy of the original MDP.

Reference:
  Ng et al., "Policy Invariance Under Reward Transformations" (ICML 1999)
"""

import torch
import numpy as np
from typing import Dict, Optional, Callable
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class RewardShaper(ABC):
    """Abstract base class for reward shaping strategies."""

    @abstractmethod
    def shape(
        self, state: int, next_state: int, reward: float, done: bool
    ) -> float:
        """Return shaped reward for a transition."""

    def reset(self) -> None:
        """Reset any episode-level state."""


class PotentialBasedShaper(RewardShaper):
    """
    Potential-based reward shaping with policy invariance guarantee.

    Shaped reward: r' = r + γΦ(s') - Φ(s)

    Φ can be any state potential function. Common choices:
      - Negative graph distance to goal
      - Predicted value function
      - Domain heuristics

    Args:
        potential_fn: Maps state index → scalar potential
        gamma: Discount factor (must match agent's gamma)
    """

    def __init__(
        self,
        potential_fn: Callable[[int], float],
        gamma: float = 0.99,
    ):
        self.potential_fn = potential_fn
        self.gamma = gamma

    def shape(self, state: int, next_state: int, reward: float, done: bool) -> float:
        if done:
            shaped = reward - self.potential_fn(state)
        else:
            shaped = reward + self.gamma * self.potential_fn(next_state) - self.potential_fn(state)
        return shaped


class CountBasedExplorationBonus(RewardShaper):
    """
    Count-based intrinsic exploration bonus: r_bonus = β / sqrt(N(s))

    Encourages visiting less-explored states by providing higher
    bonus rewards for rare states. N(s) is the visit count.

    Args:
        beta: Bonus scaling factor
        decay: Per-step decay of bonus (annealing)
        min_bonus: Minimum bonus floor
    """

    def __init__(
        self,
        beta: float = 0.1,
        decay: float = 0.999,
        min_bonus: float = 0.001,
    ):
        self.beta = beta
        self.decay = decay
        self.min_bonus = min_bonus
        self.visit_counts: Dict[int, int] = {}
        self.current_beta = beta

    def shape(self, state: int, next_state: int, reward: float, done: bool) -> float:
        self.visit_counts[next_state] = self.visit_counts.get(next_state, 0) + 1
        n = self.visit_counts[next_state]
        bonus = self.current_beta / np.sqrt(n)
        bonus = max(bonus, self.min_bonus)
        self.current_beta = max(self.current_beta * self.decay, self.min_bonus)
        return reward + bonus

    def reset(self) -> None:
        self.current_beta = self.beta


class TemporalDistanceShaper(RewardShaper):
    """
    Distance-based shaping using shortest path to reward nodes.

    Provides dense guidance by rewarding progress toward known
    reward-bearing states, computed via BFS on the graph.

    Args:
        adjacency: Graph adjacency dict {node: {action: next_node}}
        goal_nodes: Set of high-reward node indices
        gamma: Discount factor
        scale: Shaping magnitude
    """

    def __init__(
        self,
        adjacency: Dict,
        goal_nodes,
        gamma: float = 0.99,
        scale: float = 0.1,
    ):
        self.gamma = gamma
        self.scale = scale
        self.distances = self._compute_distances(adjacency, goal_nodes)

    def _compute_distances(self, adjacency, goal_nodes) -> Dict[int, float]:
        """BFS shortest distances from all nodes to nearest goal."""
        from collections import deque
        dist = {g: 0 for g in goal_nodes}
        queue = deque(goal_nodes)

        while queue:
            node = queue.popleft()
            for _, neighbor in adjacency[node].items():
                if neighbor not in dist:
                    dist[neighbor] = dist[node] + 1
                    queue.append(neighbor)

        # Nodes with no path get large distance
        max_dist = max(dist.values(), default=1)
        num_nodes = max(adjacency.keys()) + 1
        for n in range(num_nodes):
            if n not in dist:
                dist[n] = max_dist * 2

        return dist

    def _potential(self, node: int) -> float:
        d = self.distances.get(node, 100)
        return -self.scale * d

    def shape(self, state: int, next_state: int, reward: float, done: bool) -> float:
        if done:
            return reward
        return reward + self.gamma * self._potential(next_state) - self._potential(state)


class InferNetShaper(RewardShaper):
    """
    InferNet-based reward densification.

    Uses the trained reward prediction network to provide dense
    reward signals at every step, reducing credit assignment delay.

    Args:
        infernet: Trained InferNet model
        node2vec: Trained Node2Vec model
        scale: Shaping bonus scale
        use_as_bonus: True=add to env reward, False=replace it
    """

    def __init__(
        self,
        infernet,
        node2vec,
        scale: float = 0.5,
        use_as_bonus: bool = True,
    ):
        self.infernet = infernet
        self.node2vec = node2vec
        self.scale = scale
        self.use_as_bonus = use_as_bonus

    def shape(self, state: int, next_state: int, reward: float, done: bool) -> float:
        with torch.no_grad():
            node_idx = torch.tensor([next_state], dtype=torch.long)
            emb = self.node2vec(node_idx)
            predicted = self.infernet(emb).squeeze().item()

        bonus = self.scale * predicted
        if self.use_as_bonus:
            return reward + bonus
        return predicted  # Full replacement


class CompositeShaper(RewardShaper):
    """
    Combines multiple reward shapers via weighted summation.

    Args:
        shapers: List of (RewardShaper, weight) tuples
    """

    def __init__(self, shapers: list):
        self.shapers = shapers  # [(shaper, weight), ...]

    def shape(self, state: int, next_state: int, reward: float, done: bool) -> float:
        total = 0.0
        for shaper, weight in self.shapers:
            total += weight * shaper.shape(state, next_state, reward, done)
        return total

    def reset(self) -> None:
        for shaper, _ in self.shapers:
            shaper.reset()
