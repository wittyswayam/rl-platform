"""
Q-Network Architectures
========================
Standard and Dueling Q-Networks for value-based deep RL.

Implements:
  - QNetwork: Standard deep Q-network (DQN)
  - DuelingQNetwork: Dueling architecture separating V(s) and A(s,a)

The dueling architecture provides better policy evaluation by decoupling
state value from action advantage, improving learning in states where
action choice doesn't significantly affect the outcome.

References:
  - Mnih et al., "Human-level control through deep reinforcement learning" (Nature 2015)
  - Wang et al., "Dueling Network Architectures for Deep RL" (ICML 2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class QNetwork(nn.Module):
    """
    Standard Deep Q-Network for discrete action spaces.

    Maps state representations to Q-values for all actions simultaneously.
    Uses a multi-layer MLP with configurable hidden dimensions.

    Args:
        state_dim: Dimension of input state representation
        action_dim: Number of discrete actions
        hidden_dims: List of hidden layer widths
        dropout_rate: Dropout probability for regularization
    """

    def __init__(
        self,
        state_dim: int = 512,
        action_dim: int = 4,
        hidden_dims: List[int] = [256, 128],
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Build MLP layers dynamically
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_dim))

        self.network = nn.Sequential(*layers)
        self._init_weights()

        logger.info(
            f"QNetwork: {state_dim} → {hidden_dims} → {action_dim} actions"
        )

    def _init_weights(self) -> None:
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for all actions given a state.

        Args:
            state: State representation (B, state_dim)

        Returns:
            q_values: Q(s,a) for all actions (B, action_dim)
        """
        return self.network(state)

    def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        Epsilon-greedy action selection.

        Args:
            state: Single state tensor (state_dim,) or (1, state_dim)
            epsilon: Exploration probability

        Returns:
            action: Selected action index
        """
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.action_dim, (1,)).item()

        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            q_values = self.forward(state)
            return q_values.argmax(dim=-1).item()


class DuelingQNetwork(nn.Module):
    """
    Dueling Deep Q-Network with separate value and advantage streams.

    The architecture decomposes Q(s,a) into:
        Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))

    This enables the network to learn state values independently from
    action advantages, significantly improving learning efficiency in
    states where action choice has minimal impact.

    Args:
        state_dim: Dimension of input state representation
        action_dim: Number of discrete actions
        shared_dims: Shared encoder hidden layers
        value_dims: Value stream hidden layers
        advantage_dims: Advantage stream hidden layers
    """

    def __init__(
        self,
        state_dim: int = 512,
        action_dim: int = 4,
        shared_dims: List[int] = [256],
        value_dims: List[int] = [128],
        advantage_dims: List[int] = [128],
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature extraction
        self.shared_encoder = self._build_mlp(state_dim, shared_dims)
        shared_out_dim = shared_dims[-1]

        # Value stream: V(s) → scalar
        self.value_stream = self._build_mlp(shared_out_dim, value_dims, out_dim=1)

        # Advantage stream: A(s,a) → action_dim
        self.advantage_stream = self._build_mlp(
            shared_out_dim, advantage_dims, out_dim=action_dim
        )

        self._init_weights()
        logger.info(
            f"DuelingQNetwork: {state_dim} → shared{shared_dims} → "
            f"V{value_dims}+A{advantage_dims} → {action_dim} actions"
        )

    def _build_mlp(
        self,
        in_dim: int,
        hidden_dims: List[int],
        out_dim: Optional[int] = None,
    ) -> nn.Sequential:
        layers = []
        curr_dim = in_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(curr_dim, h_dim), nn.ReLU()])
            curr_dim = h_dim
        if out_dim is not None:
            layers.append(nn.Linear(curr_dim, out_dim))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values using dueling decomposition.

        Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))

        The mean-centering of advantages ensures identifiability:
        given Q(s,a), we can uniquely recover V(s) and A(s,a).

        Args:
            state: State representation (B, state_dim)

        Returns:
            q_values: Q(s,a) for all actions (B, action_dim)
        """
        features = self.shared_encoder(state)

        # Value: single scalar per state
        value = self.value_stream(features)  # (B, 1)

        # Advantages: one per action
        advantages = self.advantage_stream(features)  # (B, A)

        # Combine: subtract mean advantage for identifiability
        q_values = value + advantages - advantages.mean(dim=-1, keepdim=True)
        return q_values

    def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """Epsilon-greedy action selection."""
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.action_dim, (1,)).item()
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            return self.forward(state).argmax(dim=-1).item()
