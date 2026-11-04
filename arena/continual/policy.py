"""Frozen, observation-only inference for the continual-learning lineage.

The caller owns episode boundaries, previous executed inputs and mechanical
control availability. This controller has no environment, game identity,
telemetry, replay, optimizer or random action source. Calibration is copied
when constructed so later training-side refreshes cannot alter an episode.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch

from .calibration import GateCalibration
from .network import ContinualQNetwork


class FrozenPolicyController:
    """One local recurrent history using an already-frozen CPU/eval network.

    Network parameters must remain frozen through every alias for the lifetime
    of the controller. Ordinary in-place parameter changes, replacement,
    device changes and train-mode changes invalidate this controller. It never
    freezes or switches the caller's network modes on the caller's behalf.

    ``gru_reset`` intervenes on the current recurrent input only. Its learned
    lookahead is the ordinary hypothetical continuation from that intervened
    state, not a model of an independently optimized GRU-reset policy. Only
    the periodic phase guard disables planning at scheduled context boundaries.
    """

    def __init__(
        self,
        network: ContinualQNetwork,
        *,
        calibration: GateCalibration,
        policy_revision: int,
        context_length: int = 16,
        gamma: float = 0.98,
    ):
        if not isinstance(network, ContinualQNetwork):
            raise ValueError("a continual v2 network is required")
        if not isinstance(calibration, GateCalibration):
            raise ValueError("explicit calibration state is required")
        if type(policy_revision) is not int or not 0 <= policy_revision <= 2**63 - 1:
            raise ValueError("policy revision must be a nonnegative integer")
        if type(context_length) is not int or not 2 <= context_length <= 64:
            raise ValueError("context length must be an integer in 2..64")
        if (
            type(gamma) not in (int, float)
            or not math.isfinite(gamma)
            or not 0 <= gamma <= 1
        ):
            raise ValueError("gamma must be finite and in [0,1]")
        self.network = network
        self.policy_revision = policy_revision
        self.context_length = context_length
        self.gamma = float(gamma)
        self._calibration = GateCalibration.from_state_dict(calibration.state_dict())
        self._initial_stamp = self._network_stamp(check_finite=True)
        self.reset()

    def _network_stamp(self, *, check_finite=False):
        modules = []
        for name, module in self.network.named_modules():
            if module.training:
                raise ValueError(
                    "the policy network and all submodules must already be in eval mode"
                )
            modules.append((name, id(module), type(module)))
        values = []
        for name, tensor in (
            *self.network.named_parameters(),
            *self.network.named_buffers(),
        ):
            if tensor.device.type != "cpu" or tensor.requires_grad:
                raise ValueError(
                    "policy parameters/buffers must already be frozen on CPU"
                )
            if check_finite and not bool(torch.isfinite(tensor).all()):
                raise ValueError("policy parameters/buffers must be finite")
            try:
                version = tensor._version
            except RuntimeError as exc:
                raise ValueError(
                    "policy tensors must support mutation/version tracking"
                ) from exc
            values.append(
                (name, id(tensor), version, tuple(tensor.shape), tensor.dtype)
            )
        return tuple(modules), tuple(values)

    def _check_frozen(self):
        if self._network_stamp() != self._initial_stamp:
            raise ValueError(
                "frozen policy weights or modules changed; construct a new controller"
            )

    def reset(self):
        """Start a new local recurrent history without RNG or model mutation."""
        self._check_frozen()
        self._observations = 0
        self._hidden = self.network.initial(1)

    def fork(self):
        """Copy this inference history for a same-state counterfactual.

        Weights remain the same frozen object. Calibration and recurrent state
        are separate copies so advancing or resetting either branch cannot
        change the other branch's decision history. No RNG is consulted.
        """
        self._check_frozen()
        # The network and its construction-time immutable stamp are already
        # validated. Copying this small wrapper avoids rescanning every weight
        # for finiteness for every same-state counterfactual.
        branch = copy.copy(self)
        branch._calibration = GateCalibration.from_state_dict(
            self._calibration.state_dict()
        )
        branch._hidden = self._hidden.detach().clone()
        branch._check_frozen()
        return branch

    @staticmethod
    def _inputs(observation, previous_action, previous_reward, mask):
        if (
            not isinstance(observation, np.ndarray)
            or observation.dtype != np.uint8
            or observation.shape != (32, 32, 3)
        ):
            raise ValueError(
                "observation must be an RGB uint8 array with shape (32,32,3)"
            )
        if (
            isinstance(previous_action, (bool, np.bool_))
            or not isinstance(previous_action, (int, np.integer))
            or not 0 <= previous_action < 10
        ):
            raise ValueError("previous_action must be an integer joypad index in 0..9")
        if (
            isinstance(previous_reward, (bool, np.bool_, complex, np.complexfloating))
            or not isinstance(previous_reward, (int, float, np.integer, np.floating))
            or not math.isfinite(float(previous_reward))
        ):
            raise ValueError("previous_reward must be a finite real number")
        if isinstance(mask, torch.Tensor):
            if mask.device.type != "cpu":
                raise ValueError("mechanical mask must be on CPU")
            legal = mask.detach().clone()
        else:
            available = np.asarray(mask)
            if available.dtype != np.bool_:
                raise ValueError("mechanical mask must contain boolean values")
            legal = torch.from_numpy(available.copy())
        if (
            legal.dtype != torch.bool
            or tuple(legal.shape) != (10,)
            or not bool(legal.any())
        ):
            raise ValueError("mechanical mask must be bool[10] with a legal control")
        pixels = torch.from_numpy(
            np.ascontiguousarray(observation.transpose(2, 0, 1))
        ).unsqueeze(0)
        previous = torch.tensor([int(previous_action)], dtype=torch.long)
        clipped = max(-1.0, min(1.0, float(previous_reward)))
        reward = torch.tensor([clipped], dtype=torch.float32)
        return pixels, previous, reward, legal.unsqueeze(0)

    def act(
        self,
        observation,
        previous_action,
        previous_reward,
        mask,
        *,
        world_on=True,
        pixels_hidden=False,
        gru_reset=False,
    ):
        """Consume exactly one current observation and commit its local memory.

        Returned Q vectors are masked float tensors with shape [10], memory
        tensors have shape [1,H], and every returned tensor is a detached copy.
        ``hidden_before`` is the effective input memory AFTER any reset before
        the current image. A failed input/decision never advances the context.
        """
        if any(
            type(value) is not bool for value in (world_on, pixels_hidden, gru_reset)
        ):
            raise ValueError("ablation flags must be boolean")
        pixels, previous, reward, legal = self._inputs(
            observation, previous_action, previous_reward, mask
        )
        self._check_frozen()
        periodic_reset = self._observations % self.context_length == 0
        context_reset = periodic_reset or gru_reset
        next_periodic_reset = (self._observations + 1) % self.context_length == 0
        with torch.inference_mode():
            hidden_before = self.network.initial(1) if context_reset else self._hidden
            seen = torch.zeros_like(pixels) if pixels_hidden else pixels
            q, hidden, _ = self.network.step(seen, previous, reward, hidden_before)
            direct = q.masked_fill(~legal, -torch.inf)
            if not bool(torch.isfinite(direct[legal]).all()) or not bool(
                torch.isfinite(hidden).all()
            ):
                raise ValueError("nonfinite frozen policy output")
            mix = torch.zeros(1, dtype=torch.float32)
            if world_on and not next_periodic_reset:
                mix = self._calibration.weights(legal, revision=self.policy_revision)
            # Report the declared exact scalar cap, not float32's slightly
            # larger binary representation of .2. Tensor arithmetic below
            # still matches the trainer's calibrated mixing precisely.
            weight = self._calibration.max_mix if bool(mix[0] > 0) else 0.0
            model_values = None
            mixed = direct.clone()
            if weight > 0:
                model_values = self.network.plan_values(hidden, legal, gamma=self.gamma)
                if tuple(model_values.shape) != (1, 10) or not bool(
                    torch.isfinite(model_values[legal]).all()
                ):
                    raise ValueError("nonfinite or invalid learned lookahead")
                model_values = model_values.masked_fill(~legal, -torch.inf)
                # Arithmetic is restricted to legal finite entries. A zero mix
                # returns direct Q verbatim and never evaluates 0 * infinity.
                mixed[legal] = (1 - mix[0]) * q[legal] + mix[0] * model_values[legal]
            action = int(mixed.argmax(dim=-1).item())
        self._check_frozen()
        # Clone OUTSIDE inference_mode so consumers can edit their own returned
        # tensors without aliasing the controller or inference-tensor restrictions.
        result = {
            "action": action,
            "q_direct": direct[0].detach().clone(),
            "model_values": (
                None if model_values is None else model_values[0].detach().clone()
            ),
            "q_mixed": mixed[0].detach().clone(),
            "hidden": hidden.detach().clone(),
            "hidden_before": hidden_before.detach().clone(),
            "planning_weight": weight,
            "context_reset": bool(context_reset),
        }
        self._hidden = hidden
        self._observations += 1
        return result
