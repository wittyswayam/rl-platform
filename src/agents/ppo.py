"""
Proximal Policy Optimization (PPO) Agent
==========================================
State-of-the-art on-policy policy gradient method with clipped surrogate objective.

PPO prevents destructively large policy updates by clipping the probability ratio,
providing a stable and sample-efficient alternative to TRPO without 2nd-order optimization.

Key design:
  - Clipped surrogate objective: L^CLIP
  - GAE (Generalized Advantage Estimation)
  - Value function coefficient + entropy bonus
  - Multiple epochs per data collection

Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (arXiv 2017)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

from ..models.actor_critic import ActorCriticNetwork

logger = logging.getLogger(__name__)


@dataclass
class PPORollout:
    """Container for collected on-policy experience."""
    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class PPOAgent:
    """
    PPO agent with clipped surrogate objective and GAE.

    Training loop:
      1. Collect T timesteps with current policy
      2. Compute GAE advantages
      3. Run K epochs of mini-batch gradient updates
      4. Clip ratio to [1-ε, 1+ε] for stability

    Args:
        state_dim: Input state dimension
        action_dim: Number of discrete actions
        lr: Optimizer learning rate
        gamma: Reward discount factor
        gae_lambda: GAE smoothing parameter (0=TD, 1=MC)
        clip_eps: PPO clipping epsilon
        value_coef: Value loss weight in total loss
        entropy_coef: Entropy bonus weight (exploration)
        max_grad_norm: Gradient clipping threshold
        n_epochs: Update epochs per rollout
        batch_size: Mini-batch size
        device: Training device
    """

    def __init__(
        self,
        state_dim: int = 512,
        action_dim: int = 4,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.actor_critic = ActorCriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
        ).to(self.device)

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr, eps=1e-5)
        self.losses: List[float] = []
        self.steps = 0
        logger.info(f"PPOAgent: state_dim={state_dim}, action_dim={action_dim}, clip_eps={clip_eps}")

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generalized Advantage Estimation (GAE-λ).

        GAE interpolates between TD(1) and MC returns:
            A_t^GAE = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
            where δ_t = r_t + γV(s_{t+1}) - V(s_t)

        λ=0: Low variance, high bias (TD)
        λ=1: High variance, low bias (MC)
        λ=0.95: Practical sweet spot

        Args:
            rewards: Episode rewards (T,)
            values: Value estimates (T,)
            dones: Episode done flags (T,)
            next_value: Bootstrap value for last state

        Returns:
            advantages: GAE advantage estimates (T,)
            returns: Discounted returns for value training (T,)
        """
        T = len(rewards)
        advantages = torch.zeros(T, device=self.device)
        gae = 0.0

        for t in reversed(range(T)):
            next_v = next_value if t == T - 1 else values[t + 1].item()
            next_done = dones[t].item()
            delta = rewards[t] + self.gamma * next_v * (1 - next_done) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - next_done) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(self, rollout: PPORollout) -> Dict[str, float]:
        """
        PPO mini-batch update with clipped surrogate objective.

        Total loss:
            L = L^CLIP - c1 * L^VF + c2 * H[π]

        where:
            L^CLIP = E[min(r_t * A_t, clip(r_t, 1±ε) * A_t)]
            L^VF   = MSE(V(s), returns)
            H[π]   = Policy entropy (exploration bonus)

        Args:
            rollout: Collected on-policy experience

        Returns:
            metrics: Loss components for logging
        """
        # Normalize advantages (reduces variance)
        adv = (rollout.advantages - rollout.advantages.mean()) / (
            rollout.advantages.std() + 1e-8
        )

        policy_losses, value_losses, entropy_losses = [], [], []
        T = len(rollout.states)

        for _ in range(self.n_epochs):
            # Shuffle for mini-batch updates
            indices = torch.randperm(T)

            for start in range(0, T, self.batch_size):
                idx = indices[start : start + self.batch_size]

                states_b = rollout.states[idx]
                actions_b = rollout.actions[idx]
                old_log_probs_b = rollout.log_probs[idx]
                returns_b = rollout.returns[idx]
                adv_b = adv[idx]

                # Re-evaluate actions under current policy
                new_log_probs, entropy, values = self.actor_critic.evaluate_actions(
                    states_b, actions_b
                )

                # Probability ratio
                ratio = torch.exp(new_log_probs - old_log_probs_b)

                # Clipped surrogate objective
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values.squeeze(), returns_b)

                # Entropy bonus (encourage exploration)
                entropy_loss = -entropy.mean()

                # Combined loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(-entropy_loss.item())

        self.steps += T
        metrics = {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropy_losses),
        }
        self.losses.append(metrics["policy_loss"])
        return metrics

    def get_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[int, float]:
        """Sample or greedily select action."""
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if deterministic:
                return self.actor_critic.get_greedy_action(state), 0.0
            action, log_prob, _, _ = self.actor_critic.get_action_and_value(state)
            return action.item(), log_prob.item()
