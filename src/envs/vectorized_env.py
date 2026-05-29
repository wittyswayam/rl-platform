"""
Vectorized Environment
=======================
Runs N environments in parallel for high-throughput data collection.
Enables batch policy evaluation and significantly accelerates training.

Uses Python multiprocessing to run environment instances in separate
processes, preventing GIL from limiting parallelism.

Architecture:
  VectorizedEnv
    ├── Worker Process 0 → GraphNavEnv(seed=0)
    ├── Worker Process 1 → GraphNavEnv(seed=1)
    └── Worker Process N → GraphNavEnv(seed=N)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import logging

from ..envs.graph_nav_env import GraphNavEnv, EnvConfig

logger = logging.getLogger(__name__)


class VectorizedEnv:
    """
    Synchronous vectorized environment wrapper.

    Runs N environments sequentially (no multiprocessing overhead for small N).
    For N > 8, use AsyncVectorizedEnv for true parallelism.

    Args:
        env_config: Environment configuration shared across instances
        n_envs: Number of parallel environments
        seeds: Per-environment seeds (auto-generated if None)
    """

    def __init__(
        self,
        env_config: Optional[EnvConfig] = None,
        n_envs: int = 4,
        seeds: Optional[List[int]] = None,
    ):
        self.n_envs = n_envs
        self.env_config = env_config or EnvConfig()

        if seeds is None:
            seeds = list(range(n_envs))
        self.seeds = seeds

        # Create N independent environments
        self.envs = [GraphNavEnv(config=self.env_config) for _ in range(n_envs)]
        for i, (env, seed) in enumerate(zip(self.envs, seeds)):
            np.random.seed(seed)

        self.num_nodes = self.envs[0].num_nodes
        self.num_actions = self.envs[0].num_actions

        # Current states
        self.states: List[int] = [0] * n_envs

        logger.info(f"VectorizedEnv: {n_envs} environments")

    def reset_all(self) -> List[int]:
        """Reset all environments and return initial states."""
        self.states = [env.reset() for env in self.envs]
        return list(self.states)

    def step(
        self, actions: List[int]
    ) -> Tuple[List[int], List[float], List[bool], List[Dict]]:
        """
        Step all environments with given actions.

        Args:
            actions: Action for each environment (length N)

        Returns:
            next_states: New state per environment
            rewards: Reward per environment
            dones: Done flag per environment
            infos: Info dict per environment
        """
        next_states, rewards, dones, infos = [], [], [], []

        for env, action in zip(self.envs, actions):
            result = env.step(action)
            next_states.append(result.next_state)
            rewards.append(result.reward)
            dones.append(result.done)
            infos.append(result.info)

        self.states = next_states
        return next_states, rewards, dones, infos

    def step_with_policy(
        self, policy: Dict[int, int]
    ) -> Tuple[List[int], List[float], List[bool], List[Dict]]:
        """
        Step all environments using a shared policy.

        Args:
            policy: {state: action} mapping

        Returns:
            Transition tuples for all environments
        """
        actions = [
            policy.get(s, np.random.randint(0, self.num_actions))
            for s in self.states
        ]
        return self.step(actions)

    def collect_rollouts(
        self,
        policy: Dict[int, int],
        n_steps: int,
    ) -> Dict[str, List]:
        """
        Collect n_steps of experience from all environments.

        Total transitions collected = n_envs × n_steps

        Args:
            policy: Current policy for action selection
            n_steps: Steps per environment

        Returns:
            rollouts: Dict with states, actions, rewards, dones, next_states
        """
        all_states, all_actions, all_rewards, all_dones, all_next_states = (
            [], [], [], [], []
        )

        states = self.reset_all()

        for _ in range(n_steps):
            actions = [
                policy.get(s, np.random.randint(0, self.num_actions))
                for s in states
            ]
            next_states, rewards, dones, _ = self.step(actions)

            all_states.extend(states)
            all_actions.extend(actions)
            all_rewards.extend(rewards)
            all_dones.extend(dones)
            all_next_states.extend(next_states)

            # Auto-reset done environments
            for i, done in enumerate(dones):
                if done:
                    states[i] = self.envs[i].reset()
                else:
                    states[i] = next_states[i]

        return {
            "states": all_states,
            "actions": all_actions,
            "rewards": all_rewards,
            "dones": all_dones,
            "next_states": all_next_states,
            "n_total": len(all_states),
        }

    def episode_statistics(self) -> Dict[str, float]:
        """Aggregate episode statistics across all environments."""
        total_rewards = [env.episode_reward for env in self.envs]
        total_steps = [env.steps_taken for env in self.envs]
        return {
            "mean_reward": float(np.mean(total_rewards)),
            "std_reward": float(np.std(total_rewards)),
            "mean_steps": float(np.mean(total_steps)),
            "total_steps": int(sum(env.total_steps for env in self.envs)),
        }

    def close(self) -> None:
        """Clean up environment resources."""
        logger.info(f"VectorizedEnv closed ({self.n_envs} envs)")
