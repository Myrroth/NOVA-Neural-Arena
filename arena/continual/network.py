"""New-lineage visual network with independent control and world optimizers.

This module never changes or loads the frozen universal campaign's weights.
The decoder inherited from that architecture is unused and frozen here.
"""

from __future__ import annotations

from arena.universal.network import VisualQNetwork


class ContinualQNetwork(VisualQNetwork):
    """Keep a copy initialization while allowing the full recurrent-state range."""

    architecture_version = "continual-drqn-world-residual-v2"
    # Hidden states are in [-1, 1], so a transition can require a delta of 2.
    # The inherited zero-initialized delta rows still initially predict a copy.
    model_delta_scale = 2.0

    def __init__(self, *, latent_size: int = 64, hidden_size: int = 64):
        super().__init__(latent_size=latent_size, hidden_size=hidden_size)
        self.decoder.requires_grad_(False)

    def control_parameters(self):
        """Fresh tuple containing exactly encoder, recurrent memory and Q-head."""
        return tuple(
            parameter
            for module in (self.encoder, self.memory, self.q_head)
            for parameter in module.parameters()
        )

    def world_parameters(self):
        """World fitting updates the transition model, never control weights."""
        return tuple(self.transition.parameters())
