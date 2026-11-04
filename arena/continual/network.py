"""New-lineage visual network with independent control and world optimizers.

This module never changes or loads the frozen universal campaign's weights.
The decoder inherited from that architecture is unused and frozen here.
"""

from __future__ import annotations

import torch

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

    def sequence(
        self,
        pixels: torch.Tensor,
        prev_actions: torch.Tensor,
        prev_rewards: torch.Tensor,
        hidden: torch.Tensor | None = None,
        resets: torch.Tensor | None = None,
    ):
        """Encode independent frames together, then unroll memory causally.

        The fixed encoder has only frame-local Conv/ReLU/Linear/LayerNorm/Tanh
        operations, with no BatchNorm or dropout. A single B*T encoder batch
        therefore cannot mix observations across time. Changing that encoder
        contract requires revisiting this equivalence. ``step`` is unchanged.

        Resets occur BEFORE their marked observations; previous actions and
        rewards are never reset implicitly. Padding/loss masks remain owned by
        replay and the learner. No hidden state or latent is detached here.
        """
        if (
            not isinstance(pixels, torch.Tensor)
            or pixels.ndim != 5
            or pixels.shape[0] < 1
            or pixels.shape[1] < 1
            or tuple(pixels.shape[2:]) != self.observation_shape
            or pixels.dtype != torch.uint8
        ):
            raise ValueError("pixels must be uint8 with shape (batch, time, 3, 32, 32)")
        batch, time = pixels.shape[:2]
        if (
            not isinstance(prev_actions, torch.Tensor)
            or not isinstance(prev_rewards, torch.Tensor)
            or tuple(prev_actions.shape) != (batch, time)
            or tuple(prev_rewards.shape) != (batch, time)
        ):
            raise ValueError("previous inputs must have shape (batch, time)")
        if resets is None:
            resets = torch.zeros(batch, time, dtype=torch.bool, device=pixels.device)
        if (
            not isinstance(resets, torch.Tensor)
            or resets.dtype != torch.bool
            or tuple(resets.shape) != (batch, time)
        ):
            raise ValueError("resets must be bool with shape (batch, time)")
        state = self.initial(batch, pixels.device) if hidden is None else hidden
        if self._hidden(state) != batch:
            raise ValueError("initial hidden batch differs from pixels")
        previous = (
            self._actions(prev_actions.reshape(-1), batch * time)
            .to(dtype=state.dtype)
            .reshape(batch, time, self.action_count)
        )
        if not prev_rewards.is_floating_point() or not bool(
            torch.isfinite(prev_rewards).all()
        ):
            raise ValueError(
                "prev_rewards must be finite floats with shape (batch, time)"
            )
        rewards = prev_rewards.to(dtype=state.dtype).clamp(-1, 1).unsqueeze(-1)
        # Reshape, rather than view, also accepts strided replay slices and
        # burn/unroll splits without imposing a new contiguity requirement.
        encoded = self.encoder(
            pixels.reshape(batch * time, *self.observation_shape)
            .to(dtype=state.dtype)
            .div(255.0)
        ).reshape(batch, time, self.latent_size)
        values, states = [], []
        for t in range(time):
            state = torch.where(resets[:, t, None], torch.zeros_like(state), state)
            memory_input = torch.cat(
                (encoded[:, t], previous[:, t], rewards[:, t]), dim=-1
            )
            state = self.memory(memory_input, state)
            values.append(self.q_head(state))
            states.append(state)
        return torch.stack(values, dim=1), torch.stack(states, dim=1), encoded

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
