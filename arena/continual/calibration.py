"""Conservative, revision-bound lookahead gating from reserved calibration data.

Only the caller can establish that a batch belongs to a separate calibration
buffer. No game identity or test result is accepted. A refresh replaces all old
statistics, and an absent, unsupported, poor or stale model receives zero mix.
Serialized statistics are not scientific evidence; authenticate the enclosing
checkpoint before restoring them.
"""

from __future__ import annotations

import copy
import math

import torch


_METRICS = (
    "model_mse",
    "copy_mse",
    "mse_ratio",
    "reward_mae",
    "reward_baseline_mae",
    "reward_mean",
    "terminal_brier",
    "terminal_prior_brier",
    "terminal_prior",
)


def _revision(value):
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        raise ValueError("parameter revision must be a nonnegative integer")
    return value


def _number(value, name, *, minimum=None, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    value = float(value)
    if (
        minimum is not None
        and value < minimum
        or maximum is not None
        and value > maximum
    ):
        raise ValueError(f"{name} is outside its valid range")
    return value


def _cold_stats():
    return [
        {
            "action": i,
            "support": 0,
            **{key: None for key in _METRICS},
            "calibrated": False,
        }
        for i in range(10)
    ]


class GateCalibration:
    """One action-conditioned calibration record for one parameter revision.

    Reward and terminal baselines are constants estimated independently for
    each action, exclusively in the supplied calibration batch. Copy-MSE zero
    supplies no defined relative-improvement evidence and keeps that action off.
    """

    schema_version = 1
    max_mse_ratio = 0.9
    reward_mae_slack = 0.02
    terminal_brier_slack = 0.01

    def __init__(self, *, max_mix: float = 0.2, min_action_support: int = 8):
        self.max_mix = _number(max_mix, "max_mix", minimum=0, maximum=0.2)
        if (
            type(min_action_support) is not int
            or not 1 <= min_action_support <= 2**31 - 1
        ):
            raise ValueError("min_action_support must be a positive integer")
        self.min_action_support = min_action_support
        self.revision = None
        self._stats = _cold_stats()

    @property
    def stats(self):
        """Return a detached primitive copy, never mutable internal records."""
        return copy.deepcopy(self._stats)

    def _passes(self, stat):
        return (
            stat["support"] >= self.min_action_support
            and stat["copy_mse"] > 0
            and stat["model_mse"] <= self.max_mse_ratio * stat["copy_mse"]
            and stat["reward_mae"]
            <= stat["reward_baseline_mae"] + self.reward_mae_slack
            and stat["terminal_brier"]
            <= stat["terminal_prior_brier"] + self.terminal_brier_slack
        )

    @torch.no_grad()
    def refresh(
        self,
        predicted_h,
        target_h,
        copy_h,
        predicted_rewards,
        rewards,
        done_logits,
        terminated,
        actions,
        valid,
        *,
        revision,
    ):
        """Replace current action statistics; no gradients or game IDs retained.

        Hidden arrays have shape [..., H]; all other arrays have shape [...].
        Only valid transitions contribute. Invalid input clears the old gate
        before raising so callers cannot accidentally retain a previous pass.
        """
        revision = _revision(revision)
        if self.revision is not None and revision < self.revision:
            raise ValueError("calibration cannot move to an older parameter revision")
        self.revision, self._stats = revision, _cold_stats()
        tensors = [
            torch.as_tensor(value).detach().cpu()
            for value in (
                predicted_h,
                target_h,
                copy_h,
                predicted_rewards,
                rewards,
                done_logits,
                terminated,
                actions,
                valid,
            )
        ]
        ph, th, ch, pr, reward, logits, terminal, action, available = tensors
        if (
            ph.ndim < 2
            or ph.shape[-1] < 1
            or ph.shape != th.shape
            or ph.shape != ch.shape
            or not all(x.is_floating_point() for x in (ph, th, ch, pr, reward, logits))
        ):
            raise ValueError(
                "hidden predictions/targets/copies and scalar predictions must be floats"
            )
        leading = ph.shape[:-1]
        if any(
            x.shape != leading
            for x in (pr, reward, logits, terminal, action, available)
        ):
            raise ValueError(
                "calibration scalar shapes must match the hidden batch dimensions"
            )
        if available.dtype != torch.bool or action.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError("valid must be bool and actions must be integer indices")
        if terminal.is_complex():
            raise ValueError("terminal targets must be binary")
        keep = available.reshape(-1)
        ph, th, ch = (
            x.reshape(-1, x.shape[-1])[keep].to(torch.float64) for x in (ph, th, ch)
        )
        pr, reward, logits, terminal = (
            x.reshape(-1)[keep].to(torch.float64)
            for x in (pr, reward, logits, terminal)
        )
        action = action.reshape(-1)[keep].to(torch.int64)
        if (
            any(
                not bool(torch.isfinite(x).all())
                for x in (ph, th, ch, pr, reward, logits, terminal)
            )
            or bool(((action < 0) | (action >= 10)).any())
            or bool(((terminal != 0) & (terminal != 1)).any())
        ):
            raise ValueError(
                "valid calibration transitions must be finite with legal actions and binary terminals"
            )
        errors = (ph - th).square().mean(-1)
        copy_errors = (ch - th).square().mean(-1)
        probabilities = torch.sigmoid(logits)
        stats = _cold_stats()
        for i in range(10):
            selected = action == i
            support = int(selected.sum())
            if not support:
                continue
            model_mse = float(errors[selected].mean())
            copy_mse = float(copy_errors[selected].mean())
            reward_values, terminal_values = reward[selected], terminal[selected]
            reward_mean, terminal_prior = float(reward_values.mean()), float(
                terminal_values.mean()
            )
            stat = {
                "action": i,
                "support": support,
                "model_mse": model_mse,
                "copy_mse": copy_mse,
                "mse_ratio": model_mse / copy_mse if copy_mse > 0 else None,
                "reward_mae": float((pr[selected] - reward_values).abs().mean()),
                "reward_baseline_mae": float(
                    (reward_values - reward_mean).abs().mean()
                ),
                "reward_mean": reward_mean,
                "terminal_brier": float(
                    (probabilities[selected] - terminal_values).square().mean()
                ),
                "terminal_prior_brier": float(
                    (terminal_values - terminal_prior).square().mean()
                ),
                "terminal_prior": terminal_prior,
            }
            if any(
                value is not None and not math.isfinite(value)
                for key, value in stat.items()
                if key not in ("action", "support")
            ):
                raise ValueError("calibration statistics overflowed their finite range")
            stat["calibrated"] = bool(self._passes(stat))
            stats[i] = stat
        self._stats = stats
        return self.state_dict()

    def weights(self, mask: torch.Tensor, *, revision):
        """Return max_mix only where every legal input has a current pass."""
        revision = _revision(revision)
        if (
            not isinstance(mask, torch.Tensor)
            or mask.dtype != torch.bool
            or mask.ndim != 2
            or mask.shape[0] < 1
            or mask.shape[1] != 10
        ):
            raise ValueError("mechanical mask must be bool with shape (batch, 10)")
        off = torch.zeros(mask.shape[0], dtype=torch.float32, device=mask.device)
        if self.revision != revision:
            return off
        passed = torch.tensor(
            [x["calibrated"] for x in self._stats], dtype=torch.bool, device=mask.device
        )
        enabled = mask.any(-1) & ((~mask) | passed[None, :]).all(-1)
        return torch.where(enabled, torch.full_like(off, self.max_mix), off)

    def state_dict(self):
        """Finite JSON primitives only; no tensors or graphs survive refresh."""
        return {
            "schema_version": self.schema_version,
            "max_mix": self.max_mix,
            "min_action_support": self.min_action_support,
            "revision": self.revision,
            "stats": self.stats,
        }

    @classmethod
    def from_state_dict(cls, state):
        expected = {
            "schema_version",
            "max_mix",
            "min_action_support",
            "revision",
            "stats",
        }
        if (
            type(state) is not dict
            or set(state) != expected
            or type(state["schema_version"]) is not int
            or state["schema_version"] != cls.schema_version
        ):
            raise ValueError("invalid calibration state schema")
        obj = cls(
            max_mix=state["max_mix"], min_action_support=state["min_action_support"]
        )
        revision = state["revision"]
        if revision is not None:
            revision = _revision(revision)
        stats = state["stats"]
        if type(stats) is not list or len(stats) != 10:
            raise ValueError(
                "calibration state must contain exactly ten action records"
            )
        keys = {"action", "support", "calibrated", *_METRICS}
        checked = []
        for i, record in enumerate(stats):
            if (
                type(record) is not dict
                or set(record) != keys
                or type(record["action"]) is not int
                or record["action"] != i
                or type(record["support"]) is not int
                or not 0 <= record["support"] <= 2**31 - 1
                or type(record["calibrated"]) is not bool
            ):
                raise ValueError("invalid action calibration record")
            stat = copy.deepcopy(record)
            if stat["support"] == 0:
                if stat != _cold_stats()[i]:
                    raise ValueError(
                        "unsupported action statistics must be empty and off"
                    )
            else:
                if revision is None:
                    raise ValueError("supported calibration needs an explicit revision")
                for key in _METRICS:
                    if key == "mse_ratio" and stat[key] is None:
                        continue
                    minimum = None if key == "reward_mean" else 0
                    maximum = (
                        1
                        if key
                        in ("terminal_brier", "terminal_prior_brier", "terminal_prior")
                        else None
                    )
                    stat[key] = _number(
                        stat[key], key, minimum=minimum, maximum=maximum
                    )
                ratio = (
                    stat["model_mse"] / stat["copy_mse"]
                    if stat["copy_mse"] > 0
                    else None
                )
                if (
                    (ratio is None) != (stat["mse_ratio"] is None)
                    or ratio is not None
                    and not math.isclose(
                        ratio, stat["mse_ratio"], rel_tol=1e-12, abs_tol=1e-15
                    )
                    or stat["calibrated"] != bool(obj._passes(stat))
                ):
                    raise ValueError(
                        "calibration ratios or pass flags disagree with their metrics"
                    )
            checked.append(stat)
        obj.revision, obj._stats = revision, checked
        return obj
