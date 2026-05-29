"""
Actor-Critic Network
====================
Shared-encoder actor-critic architecture for policy gradient methods (PPO, A2C).

The shared backbone enables efficient feature extraction, while separate
output heads specialize for policy (actor) and value (critic) estimation.

Architecture:
    State → Shared MLP → Actor Head (policy logits)
                       → Critic Head (state value)

References:
  - Mnih et al., "Asynchronous Methods for Deep RL" (ICML 2016)
  - Schulman et al., "Proximal Policy Optimization" (arXiv 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class ActorCriticNetwork(nn.Module):
    """
    Shared-encoder actor-critic for discrete action spaces.

    Args:
        state_dim: Input state dimension
        action_dim: Number of discrete actions
        shared_dims: Hidden dimensions for shared backbone
        actor_dims: Actor-specific hidden layers
        critic_dims: Critic-specific hidden layers
        activation: Activation function (default: Tanh for policy stability)
    """

    def __init__(
        self,
        state_dim: int = 512,
        action_dim: int = 4,
        shared_dims: List[int] = [256, 128],
        actor_dims: List[int] = [64],
        critic_dims: List[int] = [64],
        activation: str = "tanh",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        act_fn = nn.Tanh() if activation == "tanh" else nn.ReLU()

        # Shared encoder
        shared_layers = []
        in_dim = state_dim
        for h in shared_dims:
            shared_layers.extend([nn.Linear(in_dim, h), act_fn])
            in_dim = h
        self.shared = nn.Sequential(*shared_layers)

        # Actor head: outputs logits for policy distribution
        actor_layers = []
        for h in actor_dims:
            actor_layers.extend([nn.Linear(in_dim, h), act_fn])
            in_dim = h
        actor_layers.append(nn.Linear(in_dim, action_dim))
        self.actor = nn.Sequential(*actor_layers)

        # Critic head: outputs scalar state value estimate
        in_dim = shared_dims[-1]
        critic_layers = []
        for h in critic_dims:
            critic_layers.extend([nn.Linear(in_dim, h), act_fn])
            in_dim = h
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        self._init_weights()
        logger.info(f"ActorCriticNetwork: state_dim={state_dim}, actions={action_dim}")

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute policy logits and state value simultaneously.

        Args:
            state: State representation (B, state_dim)

        Returns:
            logits: Policy logits (B, action_dim) — use with Categorical dist
            value: State value estimate (B, 1)
        """
        features = self.shared(state)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def get_action_and_value(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action, log probability, entropy, and value in one pass.

        Args:
            state: State representation (B, state_dim)

        Returns:
            action: Sampled action (B,)
            log_prob: Log probability of action (B,)
            entropy: Policy entropy (B,)
            value: State value estimate (B, 1)
        """
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def evaluate_actions(
        self, state: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and value for given state-action pairs.

        Used during PPO update to compute ratio π_new(a|s) / π_old(a|s).

        Args:
            state: State batch (B, state_dim)
            actions: Action batch (B,)

        Returns:
            log_prob: Log probability under current policy (B,)
            entropy: Policy entropy (B,)
            value: State values (B, 1)
        """
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value

    def get_greedy_action(self, state: torch.Tensor) -> int:
        """Deterministic greedy action (argmax policy)."""
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            logits, _ = self.forward(state)
            return logits.argmax(dim=-1).item()
