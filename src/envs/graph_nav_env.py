"""
Graph Navigation Environment
==============================
Production-grade RL environment for graph navigation with delayed rewards.

Implements the core 8×8 grid graph environment from the original research,
extended with a gym-compatible interface, configurable reward structures,
episode statistics, and environment vectorization support.

The environment models a sparse reward navigation problem where:
  - State: Current node in the graph (encoded as embedding index)
  - Actions: 4 cardinal directions (Up, Down, Left, Right)
  - Rewards: +1.0 at coin nodes, 0.0 elsewhere
  - Challenge: Coins are at only 3/64 locations → extreme sparsity
"""

import numpy as np
import torch
from typing import Optional, Tuple, Dict, Any, Set
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# Action encoding
ACTION_UP = 0
ACTION_RIGHT = 1
ACTION_DOWN = 2
ACTION_LEFT = 3

DIRECTION_DELTAS = {
    ACTION_UP: (-1, 0),
    ACTION_RIGHT: (0, 1),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
}


@dataclass
class EnvConfig:
    """Configuration for the graph navigation environment."""
    grid_size: int = 8
    coin_nodes: Set[int] = field(default_factory=lambda: {10, 30, 50})
    coin_reward: float = 1.0
    step_penalty: float = 0.0
    max_steps: int = 128
    random_start: bool = True
    fixed_start_node: int = 0


@dataclass
class StepResult:
    """Result from a single environment step."""
    next_state: int
    reward: float
    done: bool
    info: Dict[str, Any]


class GraphNavEnv:
    """
    Graph navigation environment with configurable reward structure.

    Supports:
      - Configurable grid sizes (N×N)
      - Custom coin placement
      - Episode statistics tracking
      - Gymnasium-compatible interface
      - Efficient adjacency computation

    Args:
        config: EnvConfig specifying environment parameters
        device: Torch device for tensor operations
    """

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        device: str = "cpu",
    ):
        self.config = config or EnvConfig()
        self.device = device
        self.G = self.config.grid_size
        self.num_nodes = self.G * self.G
        self.num_actions = 4

        # Build adjacency list
        self.adjacency = self._build_adjacency()

        # Episode state
        self.current_node: int = 0
        self.steps_taken: int = 0
        self.episode_reward: float = 0.0
        self.coins_collected: int = 0

        # Statistics
        self.episode_count: int = 0
        self.total_steps: int = 0

        logger.info(
            f"GraphNavEnv: {self.G}×{self.G} grid, "
            f"coins at {self.config.coin_nodes}"
        )

    def _build_adjacency(self) -> Dict[int, Dict[int, int]]:
        """
        Build action-indexed adjacency list for the grid graph.

        Returns:
            adjacency: {node: {action: next_node}} dict
                       Missing keys indicate invalid moves (boundary)
        """
        adj = {}
        for x in range(self.G):
            for y in range(self.G):
                node = self._to_node(x, y)
                adj[node] = {}
                for action, (dx, dy) in DIRECTION_DELTAS.items():
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.G and 0 <= ny < self.G:
                        adj[node][action] = self._to_node(nx, ny)
        return adj

    def _to_node(self, x: int, y: int) -> int:
        return self.G * x + y

    def _to_coord(self, node: int) -> Tuple[int, int]:
        return node // self.G, node % self.G

    def get_edge_index(self) -> torch.Tensor:
        """
        Return edge index tensor for PyTorch Geometric compatibility.

        Returns:
            edge_index: LongTensor of shape (2, num_edges)
        """
        edges = []
        for node, neighbors in self.adjacency.items():
            for _, neighbor in neighbors.items():
                edges.append([node, neighbor])
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def reset(self, start_node: Optional[int] = None) -> int:
        """
        Reset environment to start of episode.

        Args:
            start_node: Override random start if specified

        Returns:
            initial_state: Starting node index
        """
        if start_node is not None:
            self.current_node = start_node
        elif self.config.random_start:
            self.current_node = np.random.randint(0, self.num_nodes)
        else:
            self.current_node = self.config.fixed_start_node

        self.steps_taken = 0
        self.episode_reward = 0.0
        self.coins_collected = 0
        self.episode_count += 1

        return self.current_node

    def step(self, action: int) -> StepResult:
        """
        Execute action and return transition result.

        If action is invalid (boundary), agent stays in place.

        Args:
            action: Action index (0=Up, 1=Right, 2=Down, 3=Left)

        Returns:
            StepResult with next_state, reward, done, info
        """
        # Move if valid, else stay
        neighbors = self.adjacency[self.current_node]
        next_node = neighbors.get(action, self.current_node)

        # Compute reward
        reward = self.config.coin_reward if next_node in self.config.coin_nodes else 0.0
        reward -= self.config.step_penalty

        # Update state
        prev_node = self.current_node
        self.current_node = next_node
        self.steps_taken += 1
        self.episode_reward += reward
        self.total_steps += 1

        if reward > 0:
            self.coins_collected += 1

        done = self.steps_taken >= self.config.max_steps

        info = {
            "prev_node": prev_node,
            "next_node": next_node,
            "coin_hit": reward > 0,
            "steps_taken": self.steps_taken,
            "episode_reward": self.episode_reward,
            "coins_collected": self.coins_collected,
        }
        return StepResult(next_node, reward, done, info)

    def sample_episode(
        self,
        policy: Optional[Dict[int, int]] = None,
        start_node: Optional[int] = None,
    ) -> Dict[str, list]:
        """
        Sample a complete episode trajectory.

        Args:
            policy: {state: action} dict; random if None
            start_node: Episode starting node

        Returns:
            trajectory: Dict with states, actions, rewards lists
        """
        state = self.reset(start_node)
        states, actions, rewards = [state], [], []

        for _ in range(self.config.max_steps):
            if policy is not None:
                action = policy.get(state, np.random.randint(0, self.num_actions))
            else:
                action = np.random.randint(0, self.num_actions)

            result = self.step(action)
            actions.append(action)
            rewards.append(result.reward)
            states.append(result.next_state)
            state = result.next_state

        return {"states": states, "actions": actions, "rewards": rewards}

    def render_policy(self, policy: Dict[int, int]) -> str:
        """
        Render policy as ASCII grid with directional arrows.

        Args:
            policy: {state: action} mapping

        Returns:
            grid_str: ASCII art policy visualization
        """
        symbols = {
            ACTION_UP: "↑",
            ACTION_RIGHT: "→",
            ACTION_DOWN: "↓",
            ACTION_LEFT: "←",
        }
        lines = [f"Policy Grid ({self.G}×{self.G}):"]
        lines.append("  " + "  ".join(str(i) for i in range(self.G)))

        for row in range(self.G):
            row_str = f"{row} "
            for col in range(self.G):
                node = self._to_node(row, col)
                if node in self.config.coin_nodes:
                    row_str += "$ "
                else:
                    action = policy.get(node, ACTION_UP)
                    row_str += symbols[action] + " "
            lines.append(row_str)

        return "\n".join(lines)

    @property
    def observation_space_size(self) -> int:
        return self.num_nodes

    @property
    def action_space_size(self) -> int:
        return self.num_actions
