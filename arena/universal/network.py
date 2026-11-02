"""One visual recurrent Q-network and learned dynamics for every prepared game.

There are no game identifiers, separate game heads, expert actions, or geometry
features here. The only observations are pixels, previous input and reward.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class VisualQNetwork(nn.Module):
    """Pixels -> encoder latent -> recurrent memory -> a single joypad Q-head.

    ``latent`` is the CNN representation. ``hidden`` is the history-conditioned
    representation used by Q and the transition model; they remain different
    concepts even when both dimensions happen to be 64.
    """

    action_count = 10
    observation_shape = (3, 32, 32)
    architecture_version = "visual-drqn-world-residual-v1"
    model_delta_scale = .1

    def __init__(self, *, latent_size: int = 64, hidden_size: int = 64):
        super().__init__()
        for name, value in (("latent_size", latent_size), ("hidden_size", hidden_size)):
            if type(value) is not int or not 1 <= value <= 512:
                raise ValueError(f"{name} must be an integer in 1..512")
        self.latent_size = self.latent_dim = latent_size
        self.hidden_size = self.hidden_dim = hidden_size
        self.num_actions = self.action_count
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 4 * 4, latent_size),
            nn.LayerNorm(latent_size), nn.Tanh(),
        )
        self.memory = nn.GRUCell(latent_size + self.action_count + 1, hidden_size)
        self.q_head = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(),
                                    nn.Linear(64, self.action_count))
        self.transition = nn.Sequential(
            nn.Linear(hidden_size + self.action_count, 128), nn.ReLU(),
            nn.Linear(128, hidden_size + 2),
        )
        # A generic persistence prior, not a game strategy: before learning the
        # transition model predicts the current memory. Reward/termination rows
        # retain their ordinary initialization and all rows remain trainable.
        with torch.no_grad():
            self.transition[-1].weight[:hidden_size].zero_()
            self.transition[-1].bias[:hidden_size].zero_()
        # Reconstruction anchors the visual encoder, not a game-specific label.
        self.decoder = nn.Sequential(nn.Linear(latent_size, 128), nn.ReLU(),
                                     nn.Linear(128, 3 * 8 * 8), nn.Sigmoid())

    def initial(self, batch: int, device=None) -> torch.Tensor:
        if type(batch) is not int or batch < 1:
            raise ValueError("batch must be a positive integer")
        parameter = next(self.parameters())
        return torch.zeros(batch, self.hidden_size,
                           device=parameter.device if device is None else device,
                           dtype=parameter.dtype)

    def _hidden(self, hidden: torch.Tensor) -> int:
        if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 2
                or hidden.shape[0] < 1 or hidden.shape[1] != self.hidden_size
                or not hidden.is_floating_point()):
            raise ValueError("hidden must have shape (batch, hidden_size)")
        return hidden.shape[0]

    def _actions(self, actions: torch.Tensor, batch: int) -> torch.Tensor:
        if (not isinstance(actions, torch.Tensor) or actions.dtype != torch.long
                or tuple(actions.shape) != (batch,)
                or bool(((actions < 0) | (actions >= self.action_count)).any())):
            raise ValueError("actions must be int64 joypad indices with shape (batch,)")
        return F.one_hot(actions, self.action_count)

    def step(self, pixels: torch.Tensor, prev_actions: torch.Tensor,
             prev_rewards: torch.Tensor, hidden: torch.Tensor):
        """Observe a BCHW uint8 frame and return Q, new memory, CNN latent."""
        batch = self._hidden(hidden)
        if (not isinstance(pixels, torch.Tensor) or pixels.dtype != torch.uint8
                or tuple(pixels.shape) != (batch, *self.observation_shape)):
            raise ValueError("pixels must be uint8 with shape (batch, 3, 32, 32)")
        previous = self._actions(prev_actions, batch).to(dtype=hidden.dtype)
        if (not isinstance(prev_rewards, torch.Tensor)
                or tuple(prev_rewards.shape) != (batch,)
                or not prev_rewards.is_floating_point()
                or not bool(torch.isfinite(prev_rewards).all())):
            raise ValueError("prev_rewards must be finite floats with shape (batch,)")
        latent = self.encoder(pixels.to(dtype=hidden.dtype).div(255.0))
        memory_input = torch.cat((latent, previous,
                                  prev_rewards.to(dtype=hidden.dtype).clamp(-1, 1).unsqueeze(-1)), dim=-1)
        new_hidden = self.memory(memory_input, hidden)
        return self.q_head(new_hidden), new_hidden, latent

    def sequence(self, pixels: torch.Tensor, prev_actions: torch.Tensor,
                 prev_rewards: torch.Tensor, hidden: torch.Tensor | None = None,
                 resets: torch.Tensor | None = None):
        """Unroll BTCHW frames, resetting memory BEFORE marked observations.

        Previous actions/rewards are explicit inputs and are not silently
        replaced. Replay records zero previous input at genuine episode starts.
        Padding resets may be marked by replay; its ``valid`` mask governs loss.
        """
        if (not isinstance(pixels, torch.Tensor) or pixels.ndim != 5
                or pixels.shape[0] < 1 or pixels.shape[1] < 1
                or tuple(pixels.shape[2:]) != self.observation_shape
                or pixels.dtype != torch.uint8):
            raise ValueError("pixels must be uint8 with shape (batch, time, 3, 32, 32)")
        batch, time = pixels.shape[:2]
        if tuple(prev_actions.shape) != (batch, time) or tuple(prev_rewards.shape) != (batch, time):
            raise ValueError("previous inputs must have shape (batch, time)")
        if resets is None:
            resets = torch.zeros(batch, time, dtype=torch.bool, device=pixels.device)
        if resets.dtype != torch.bool or tuple(resets.shape) != (batch, time):
            raise ValueError("resets must be bool with shape (batch, time)")
        state = self.initial(batch, pixels.device) if hidden is None else hidden
        if self._hidden(state) != batch:
            raise ValueError("initial hidden batch differs from pixels")
        values, states, latents = [], [], []
        for t in range(time):
            state = torch.where(resets[:, t, None], torch.zeros_like(state), state)
            q, state, latent = self.step(pixels[:, t], prev_actions[:, t], prev_rewards[:, t], state)
            values.append(q)
            states.append(state)
            latents.append(latent)
        return (torch.stack(values, dim=1), torch.stack(states, dim=1),
                torch.stack(latents, dim=1))

    def model(self, hidden: torch.Tensor, actions: torch.Tensor):
        """Predict NEXT recurrent memory, reward and termination logits.

        The next-memory target must be detached recurrent state after observing
        the actual next image, never the encoder-only latent passed to Q.
        """
        batch = self._hidden(hidden)
        one_hot = self._actions(actions, batch).to(dtype=hidden.dtype)
        prediction = self.transition(torch.cat((hidden, one_hot), dim=-1))
        next_hidden = (hidden + self.model_delta_scale *
                       torch.tanh(prediction[:, :self.hidden_size])).clamp(-1, 1)
        return (next_hidden,
                prediction[:, self.hidden_size], prediction[:, self.hidden_size + 1])

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Reconstruct an 8x8 RGB thumbnail from the CNN latent, in [0, 1]."""
        if (not isinstance(latent, torch.Tensor) or latent.ndim != 2
                or latent.shape[0] < 1 or latent.shape[1] != self.latent_size):
            raise ValueError("latent must have shape (batch, latent_size)")
        return self.decoder(latent).reshape(-1, 3, 8, 8)

    def plan_values(self, hidden: torch.Tensor, mask: torch.Tensor,
                    gamma: float = .98) -> torch.Tensor:
        """One learned lookahead step; the caller controls its mixing strength.

        Uses the same constant legal-input mask in the predicted next state.
        There are no per-game rules. Invalid current actions return -infinity.
        This function does not imply that the model's predictions are accurate.
        """
        batch = self._hidden(hidden)
        if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
                or tuple(mask.shape) != (batch, self.action_count)
                or not bool(mask.any(dim=-1).all())):
            raise ValueError("mask must provide at least one legal input per batch item")
        if type(gamma) not in (int, float) or not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must be finite and in [0, 1]")
        actions = torch.arange(self.action_count, device=hidden.device).expand(batch, -1)
        states = hidden[:, None, :].expand(-1, self.action_count, -1)
        predicted, rewards, done_logits = self.model(
            states.reshape(-1, self.hidden_size), actions.reshape(-1))
        next_q = self.q_head(predicted).reshape(batch, self.action_count, self.action_count)
        future = next_q.masked_fill(~mask[:, None, :], -torch.inf).max(dim=-1).values
        values = rewards.reshape(batch, self.action_count) + float(gamma) * (
            1 - torch.sigmoid(done_logits.reshape(batch, self.action_count))) * future
        return values.masked_fill(~mask, -torch.inf)
