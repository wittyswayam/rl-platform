"""
Distributed RL Trainer
=======================
Scalable distributed training using multiprocessing rollout workers.
Implements a parameter-server style architecture:

  ┌─────────────────────────────────────────────┐
  │              Central Learner                 │
  │  (holds parameters, performs gradient updates)│
  └────────────────┬────────────────────────────┘
                   │ broadcast weights
       ┌───────────┼──────────────┐
       ▼           ▼              ▼
  ┌─────────┐ ┌─────────┐ ┌─────────┐
  │Worker 0 │ │Worker 1 │ │Worker N │
  │(rollout)│ │(rollout)│ │(rollout)│
  └────┬────┘ └────┬────┘ └────┬────┘
       └───────────┴──────────→ experience queue

Workers collect trajectories using the current policy, sending
experience to a central queue. The learner dequeues batches and
performs parameter updates, broadcasting weights back to workers.

This architecture is inspired by IMPALA and Ape-X.
Reference: Espeholt et al., "IMPALA: Scalable Distributed Deep-RL" (ICML 2018)
"""

import torch
import torch.multiprocessing as mp
import numpy as np
import queue
import time
import logging
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

from ..envs.graph_nav_env import GraphNavEnv, EnvConfig
from ..models.node2vec import Node2VecModel
from ..models.infer_net import InferNet

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed RL training."""
    num_workers: int = 4
    episodes_per_worker: int = 16
    num_iterations: int = 50
    embed_dim: int = 512
    lr: float = 0.01
    gamma: float = 0.99
    device: str = "cpu"
    sync_every: int = 5  # iterations between weight broadcasts


def _rollout_worker(
    worker_id: int,
    shared_policy: Dict,
    result_queue: mp.Queue,
    config: EnvConfig,
    n_episodes: int,
    stop_event: mp.Event,
) -> None:
    """
    Standalone rollout worker process.

    Runs in a separate process. Reads policy from shared memory,
    collects episodes, and pushes trajectories to result_queue.

    Args:
        worker_id: Unique worker identifier
        shared_policy: Shared-memory dict {state: action}
        result_queue: Queue for sending collected experience to learner
        config: Environment configuration
        n_episodes: Episodes to collect per signal
        stop_event: Signal to terminate worker
    """
    np.random.seed(worker_id * 42)
    env = GraphNavEnv(config=config)

    logger.info(f"Worker {worker_id} started (PID: {mp.current_process().pid})")

    while not stop_event.is_set():
        trajectories = []
        for ep_idx in range(n_episodes):
            start = np.random.randint(0, env.num_nodes)
            episode = env.sample_episode(
                policy=dict(shared_policy),
                start_node=start,
            )
            trajectories.append(episode)

        try:
            result_queue.put(
                {"worker_id": worker_id, "trajectories": trajectories},
                timeout=5.0,
            )
        except queue.Full:
            pass  # Skip if learner is backlogged

    logger.info(f"Worker {worker_id} terminated")


class DistributedTrainer:
    """
    Parameter-server distributed RL trainer.

    Spawns N rollout workers that collect experience in parallel
    while the central learner performs gradient updates.

    Throughput scales approximately linearly with num_workers
    for rollout-bound workloads (environment simulation).

    Args:
        env_config: Environment configuration shared across workers
        agent_config: Training hyperparameters
        dist_config: Distributed system configuration
    """

    def __init__(
        self,
        env_config: Optional[EnvConfig] = None,
        dist_config: Optional[DistributedConfig] = None,
    ):
        self.env_config = env_config or EnvConfig()
        self.dist_config = dist_config or DistributedConfig()
        self.device = torch.device(self.dist_config.device)

        # Central environment (for validation)
        self.env = GraphNavEnv(config=self.env_config)
        num_nodes = self.env.num_nodes

        # Central models (learner side)
        self.model_n2v = Node2VecModel(
            num_nodes=num_nodes,
            embedding_dim=self.dist_config.embed_dim,
        )
        self.model_inf = InferNet(input_dim=self.dist_config.embed_dim)

        # Shared policy (updated by learner, read by workers)
        # Uses mp.Manager dict for cross-process sharing
        self.manager = mp.Manager()
        self.shared_policy = self.manager.dict(
            {n: np.random.randint(0, 4) for n in range(num_nodes)}
        )

        # Experience queue
        self.result_queue: mp.Queue = mp.Queue(maxsize=100)
        self.stop_event = mp.Event()

        # Workers
        self.workers: List[mp.Process] = []
        self.metrics: List[Dict[str, Any]] = []

        logger.info(
            f"DistributedTrainer: {self.dist_config.num_workers} workers, "
            f"device={self.dist_config.device}"
        )

    def _spawn_workers(self) -> None:
        """Spawn rollout worker processes."""
        for worker_id in range(self.dist_config.num_workers):
            p = mp.Process(
                target=_rollout_worker,
                args=(
                    worker_id,
                    self.shared_policy,
                    self.result_queue,
                    self.env_config,
                    self.dist_config.episodes_per_worker,
                    self.stop_event,
                ),
                daemon=True,
            )
            p.start()
            self.workers.append(p)
            logger.info(f"Spawned worker {worker_id} (PID {p.pid})")

    def _collect_from_queue(
        self, timeout: float = 10.0
    ) -> List[Dict]:
        """
        Drain the result queue, collecting worker trajectories.

        Args:
            timeout: Maximum seconds to wait for results

        Returns:
            all_trajectories: Combined trajectories from all workers
        """
        all_trajectories = []
        deadline = time.time() + timeout

        while time.time() < deadline and len(all_trajectories) < (
            self.dist_config.num_workers * self.dist_config.episodes_per_worker
        ):
            try:
                result = self.result_queue.get(timeout=1.0)
                all_trajectories.extend(result["trajectories"])
            except queue.Empty:
                break

        return all_trajectories

    def _learner_update(self, trajectories: List[Dict]) -> Dict[str, float]:
        """
        Central learner gradient update on collected trajectories.

        Args:
            trajectories: List of episode dicts

        Returns:
            metrics: Training loss metrics
        """
        if not trajectories:
            return {}

        opt_n2v = torch.optim.Adam(self.model_n2v.parameters(), lr=self.dist_config.lr)
        opt_inf = torch.optim.Adam(self.model_inf.parameters(), lr=self.dist_config.lr)

        total_n2v_loss = 0.0
        total_inf_loss = 0.0

        for episode in trajectories:
            states = episode["states"]
            rewards = episode["rewards"]

            # Node2Vec update
            self.model_n2v.train()
            if len(states) > 2:
                starts = torch.tensor(states[:-1], dtype=torch.long)
                positives = torch.tensor(states[1:], dtype=torch.long)
                negatives = torch.randint(0, self.env.num_nodes, (len(starts),))

                opt_n2v.zero_grad()
                loss_n2v, _ = self.model_n2v.compute_loss(starts, positives, negatives)
                loss_n2v.backward()
                opt_n2v.step()
                total_n2v_loss += loss_n2v.item()

            # InferNet update
            self.model_n2v.eval()
            self.model_inf.train()
            if len(states) > 1:
                state_tensors = torch.tensor(states[:-1], dtype=torch.long)
                with torch.no_grad():
                    embs = self.model_n2v(state_tensors)
                reward_targets = torch.tensor(rewards, dtype=torch.float32)

                opt_inf.zero_grad()
                preds = self.model_inf(embs)
                loss_inf, _ = self.model_inf.compute_loss(
                    preds, reward_targets, walk_length=len(states)
                )
                loss_inf.backward()
                opt_inf.step()
                total_inf_loss += loss_inf.item()

        n = max(len(trajectories), 1)
        return {
            "loss_node2vec": total_n2v_loss / n,
            "loss_infernet": total_inf_loss / n,
            "n_trajectories": len(trajectories),
        }

    def _update_shared_policy(self) -> None:
        """
        Recompute Q-values from current models and broadcast to workers.
        Simple greedy policy from predicted reward landscape.
        """
        self.model_n2v.eval()
        self.model_inf.eval()
        with torch.no_grad():
            all_nodes = torch.arange(self.env.num_nodes, dtype=torch.long)
            embs = self.model_n2v(all_nodes)
            reward_preds = self.model_inf(embs).squeeze()

        # Simple greedy: move toward highest predicted reward neighbor
        for node in range(self.env.num_nodes):
            neighbors = self.env.adjacency[node]
            if not neighbors:
                continue
            best_action = max(
                neighbors.keys(),
                key=lambda a: reward_preds[neighbors[a]].item(),
            )
            self.shared_policy[node] = best_action

    def train(self) -> List[Dict[str, Any]]:
        """
        Run distributed training loop.

        Returns:
            metrics: Per-iteration training metrics
        """
        logger.info("Starting distributed training...")
        mp.set_start_method("spawn", force=True)

        self._spawn_workers()
        time.sleep(1.0)  # Allow workers to initialize

        t_start = time.time()
        for iteration in range(1, self.dist_config.num_iterations + 1):

            # Collect from workers
            trajectories = self._collect_from_queue()

            # Learner update
            iter_metrics = self._learner_update(trajectories)
            iter_metrics["iteration"] = iteration
            iter_metrics["wall_time_s"] = time.time() - t_start
            iter_metrics["throughput_eps"] = len(trajectories) / max(
                iter_metrics["wall_time_s"] / iteration, 1e-6
            )

            # Broadcast updated policy to workers
            if iteration % self.dist_config.sync_every == 0:
                self._update_shared_policy()
                logger.info(
                    f"[{iteration}] Synced policy. "
                    f"N2V={iter_metrics.get('loss_node2vec', 0):.4f} | "
                    f"Inf={iter_metrics.get('loss_infernet', 0):.4f} | "
                    f"Throughput={iter_metrics['throughput_eps']:.1f} ep/s"
                )

            self.metrics.append(iter_metrics)

        # Shutdown workers
        self.stop_event.set()
        for p in self.workers:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()

        logger.info(
            f"Distributed training complete in {time.time() - t_start:.1f}s"
        )
        return self.metrics
