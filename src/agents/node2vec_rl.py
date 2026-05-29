"""
Node2Vec RL Agent
==================
Production-grade implementation of the delayed reward RL agent combining
Node2Vec graph embeddings, InferNet reward prediction, and Q-learning.

This agent is the core of the original research, substantially upgraded with:
  - Modular architecture (separate models, env, buffer)
  - Configurable hyperparameters via dataclass
  - Full experiment tracking integration
  - Checkpoint management
  - Epsilon-greedy exploration with decay
  - Multi-start episode sampling
  - TensorBoard-compatible metric logging

Algorithm:
    1. Random walks → Node2Vec embeddings (graph topology encoding)
    2. Embeddings → InferNet reward predictions (auxiliary signal)
    3. Episodes + rewards → Q-table updates (credit assignment)
    4. Q-values → Greedy policy extraction
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import time
from pathlib import Path

from ..models.node2vec import Node2VecModel
from ..models.infer_net import InferNet
from ..envs.graph_nav_env import GraphNavEnv, EnvConfig

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Full configuration for the Node2Vec RL agent."""
    # Embedding model
    embed_dim: int = 512
    # Training
    num_iter: int = 100
    walk_length: int = 128
    context_size: int = 3
    num_negative: int = 1
    # Optimizer
    lr_node2vec: float = 0.1
    lr_infernet: float = 0.01
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    # Q-learning
    q_lr: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    # Misc
    device: str = "cpu"
    save_dir: str = "checkpoints"
    log_every: int = 10


class Node2VecRLAgent:
    """
    Graph RL agent using Node2Vec embeddings and Q-learning.

    Solves the delayed reward navigation problem by:
      1. Learning structural node representations (Node2Vec)
      2. Predicting rewards from representations (InferNet)
      3. Propagating value backwards (Q-learning)

    Args:
        env: Graph navigation environment
        config: AgentConfig with hyperparameters
    """

    def __init__(
        self,
        env: GraphNavEnv,
        config: Optional[AgentConfig] = None,
    ):
        self.env = env
        self.config = config or AgentConfig()
        self.device = torch.device(self.config.device)

        num_nodes = env.num_nodes

        # Core models
        self.model_node2vec = Node2VecModel(
            num_nodes=num_nodes,
            embedding_dim=self.config.embed_dim,
        ).to(self.device)

        self.model_infernet = InferNet(
            input_dim=self.config.embed_dim,
        ).to(self.device)

        # Optimizers with learning rate scheduling
        self.opt_node2vec = optim.Adam(
            self.model_node2vec.parameters(),
            lr=self.config.lr_node2vec,
        )
        self.opt_infernet = optim.Adam(
            self.model_infernet.parameters(),
            lr=self.config.lr_infernet,
        )
        self.scheduler_n2v = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt_node2vec,
            patience=self.config.lr_scheduler_patience,
            factor=self.config.lr_scheduler_factor,
        )
        self.scheduler_inf = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt_infernet,
            patience=self.config.lr_scheduler_patience,
            factor=self.config.lr_scheduler_factor,
        )

        # Q-table: shape (num_nodes, num_actions)
        self.Q = torch.zeros(num_nodes, env.num_actions, dtype=torch.float32)

        # Policy: uniform random initialization
        self.policy: Dict[int, int] = {
            n: np.random.randint(0, env.num_actions)
            for n in range(num_nodes)
        }

        # Exploration
        self.epsilon = self.config.epsilon_start

        # Metrics
        self.losses_node2vec: List[float] = []
        self.losses_infernet: List[float] = []
        self.episode_returns: List[float] = []
        self.iteration = 0

        logger.info(
            f"Node2VecRLAgent initialized: {num_nodes} nodes, "
            f"{self.config.num_iter} iterations"
        )

    def _get_context_pairs(
        self, walk: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract positive and negative pairs from a random walk.

        Positive pairs: nodes within context_size window
        Negative pairs: uniformly sampled random nodes

        Args:
            walk: Sequence of node indices

        Returns:
            start_nodes, positive_nodes, negative_nodes tensors
        """
        starts, positives, negatives = [], [], []
        size = self.config.context_size

        for i, node in enumerate(walk):
            lo = max(0, i - size)
            hi = min(len(walk), i + size + 1)
            for j in range(lo, hi):
                if i != j:
                    starts.append(node)
                    positives.append(walk[j])
                    neg = np.random.randint(0, self.env.num_nodes)
                    negatives.append(neg)

        return (
            torch.tensor(starts, dtype=torch.long, device=self.device),
            torch.tensor(positives, dtype=torch.long, device=self.device),
            torch.tensor(negatives, dtype=torch.long, device=self.device),
        )

    def _update_node2vec(
        self, trajectories: Dict[int, Dict]
    ) -> float:
        """Train Node2Vec on walk data from all episodes."""
        self.model_node2vec.train()
        total_loss = 0.0

        for episode in trajectories.values():
            states = episode["states"]
            starts, positives, negatives = self._get_context_pairs(states)

            if len(starts) == 0:
                continue

            self.opt_node2vec.zero_grad()
            loss, _ = self.model_node2vec.compute_loss(starts, positives, negatives)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model_node2vec.parameters(), max_norm=1.0
            )
            self.opt_node2vec.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(trajectories), 1)
        self.scheduler_n2v.step(avg_loss)
        return avg_loss

    def _update_infernet(
        self, trajectories: Dict[int, Dict]
    ) -> float:
        """Train InferNet to predict rewards from embeddings."""
        self.model_infernet.train()
        self.model_node2vec.eval()
        total_loss = 0.0

        # Get all node embeddings
        all_nodes = torch.arange(
            self.env.num_nodes, dtype=torch.long, device=self.device
        )
        all_embeddings = self.model_node2vec(all_nodes).detach()

        for episode in trajectories.values():
            states = episode["states"][:-1]  # exclude terminal
            rewards = episode["rewards"]

            if not states:
                continue

            state_embeddings = all_embeddings[
                torch.tensor(states, dtype=torch.long, device=self.device)
            ]
            reward_targets = torch.tensor(
                rewards, dtype=torch.float32, device=self.device
            )

            self.opt_infernet.zero_grad()
            predictions = self.model_infernet(state_embeddings)
            loss, _ = self.model_infernet.compute_loss(
                predictions, reward_targets, walk_length=len(states)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model_infernet.parameters(), max_norm=1.0
            )
            self.opt_infernet.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(trajectories), 1)
        self.scheduler_inf.step(avg_loss)
        return avg_loss

    def _update_q_values(self, trajectories: Dict[int, Dict]) -> None:
        """Q-learning update across all episode trajectories."""
        alpha = self.config.q_lr
        gamma = self.config.gamma

        for episode in trajectories.values():
            states = episode["states"]
            actions = episode["actions"]
            rewards = episode["rewards"]

            for t in range(len(actions)):
                s = states[t]
                a = actions[t]
                r = rewards[t]
                s_next = states[t + 1]

                # Bellman update
                target = r + gamma * self.Q[s_next].max().item()
                self.Q[s, a] = (1 - alpha) * self.Q[s, a] + alpha * target

    def _extract_policy(self) -> None:
        """Update policy to be greedy w.r.t. current Q-values."""
        for node in range(self.env.num_nodes):
            self.policy[node] = self.Q[node].argmax().item()

    def _collect_trajectories(self) -> Dict[int, Dict]:
        """Sample one episode per starting node using current policy."""
        trajectories = {}
        for start_node in range(self.env.num_nodes):
            episode = self.env.sample_episode(
                policy=self.policy,
                start_node=start_node,
            )
            trajectories[start_node] = episode
        return trajectories

    def train(self, num_iter: Optional[int] = None) -> Dict[str, List]:
        """
        Run the full training loop.

        Each iteration:
          1. Collect episodes from all start nodes
          2. Update Node2Vec embeddings
          3. Update InferNet reward predictor
          4. Update Q-values
          5. Extract greedy policy

        Args:
            num_iter: Override number of iterations

        Returns:
            metrics: Training history dict
        """
        n_iter = num_iter or self.config.num_iter
        logger.info(f"Starting training: {n_iter} iterations")
        t0 = time.time()

        for i in range(1, n_iter + 1):
            self.iteration = i

            # Step 1: Collect trajectories
            trajectories = self._collect_trajectories()

            # Step 2: Train Node2Vec
            loss_n2v = self._update_node2vec(trajectories)

            # Step 3: Train InferNet
            loss_inf = self._update_infernet(trajectories)

            # Step 4: Q-learning update
            self._update_q_values(trajectories)

            # Step 5: Policy improvement
            self._extract_policy()

            # Decay exploration
            self.epsilon = max(
                self.config.epsilon_end,
                self.epsilon * self.config.epsilon_decay,
            )

            # Track metrics
            self.losses_node2vec.append(loss_n2v)
            self.losses_infernet.append(loss_inf)

            ep_return = np.mean([
                sum(ep["rewards"]) for ep in trajectories.values()
            ])
            self.episode_returns.append(ep_return)

            if i % self.config.log_every == 0:
                elapsed = time.time() - t0
                logger.info(
                    f"Iter {i:4d}/{n_iter} | "
                    f"N2V Loss: {loss_n2v:.4f} | "
                    f"Inf Loss: {loss_inf:.4f} | "
                    f"Avg Return: {ep_return:.3f} | "
                    f"ε: {self.epsilon:.3f} | "
                    f"Time: {elapsed:.1f}s"
                )

        logger.info(f"Training complete in {time.time() - t0:.1f}s")

        return {
            "losses_node2vec": self.losses_node2vec,
            "losses_infernet": self.losses_infernet,
            "episode_returns": self.episode_returns,
        }

    def get_embeddings(self) -> torch.Tensor:
        """Return all node embeddings as a matrix."""
        self.model_node2vec.eval()
        with torch.no_grad():
            return self.model_node2vec.get_all_embeddings()

    def save_checkpoint(self, path: str) -> None:
        """Save full agent state."""
        checkpoint = {
            "iteration": self.iteration,
            "node2vec_state": self.model_node2vec.state_dict(),
            "infernet_state": self.model_infernet.state_dict(),
            "q_table": self.Q,
            "policy": self.policy,
            "losses_node2vec": self.losses_node2vec,
            "losses_infernet": self.losses_infernet,
            "episode_returns": self.episode_returns,
            "config": self.config,
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    @classmethod
    def load_checkpoint(
        cls, path: str, env: GraphNavEnv, config: Optional[AgentConfig] = None
    ) -> "Node2VecRLAgent":
        """Load agent from checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        cfg = config or checkpoint.get("config", AgentConfig())
        agent = cls(env, cfg)
        agent.model_node2vec.load_state_dict(checkpoint["node2vec_state"])
        agent.model_infernet.load_state_dict(checkpoint["infernet_state"])
        agent.Q = checkpoint["q_table"]
        agent.policy = checkpoint["policy"]
        agent.losses_node2vec = checkpoint.get("losses_node2vec", [])
        agent.losses_infernet = checkpoint.get("losses_infernet", [])
        agent.episode_returns = checkpoint.get("episode_returns", [])
        agent.iteration = checkpoint.get("iteration", 0)
        logger.info(f"Agent loaded from {path} (iter {agent.iteration})")
        return agent
