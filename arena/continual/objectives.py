"""Tensor-only continual-learning objectives; no game or policy is constructed."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def _finite_float(name: str, value: torch.Tensor, dimensions: int) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != dimensions
        or any(size < 1 for size in value.shape)
        or not value.is_floating_point()
    ):
        raise ValueError(
            f"{name} must be a nonempty floating tensor of rank {dimensions}"
        )
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def _boolean(
    name: str, value: torch.Tensor, shape: tuple, device: torch.device
) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.bool
        or tuple(value.shape) != shape
        or value.device != device
    ):
        raise ValueError(
            f"{name} must be a bool tensor with shape {shape} on the data device"
        )


def _math_dtype(*values: torch.Tensor) -> torch.dtype:
    # Accumulate low-precision model outputs in float32 while preserving float64
    # mathematical fixtures and callers. Casts preserve student gradients.
    return (
        torch.float64
        if any(x.dtype == torch.float64 for x in values)
        else torch.float32
    )


def masked_sequence_mean(
    per_step: torch.Tensor, valid: torch.Tensor, row_weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Mean within each valid sequence, then weight equally across sequences.

    An eight-transition sequence and a one-transition sequence have the same
    default influence. Entirely padded rows have no weight. No effective rows
    (including all-zero supplied weights) produce a differentiable scalar zero.
    Nonfinite padding is rejected rather than hidden by multiplication by zero.
    """
    _finite_float("per_step", per_step, 2)
    _boolean("valid", valid, tuple(per_step.shape), per_step.device)
    if row_weights is not None:
        _finite_float("row_weights", row_weights, 1)
        if (
            tuple(row_weights.shape) != (per_step.shape[0],)
            or row_weights.device != per_step.device
        ):
            raise ValueError("row_weights must match batch size and data device")
        if bool((row_weights < 0).any()):
            raise ValueError("row_weights must be nonnegative")
    dtype = _math_dtype(per_step, *(() if row_weights is None else (row_weights,)))
    values = per_step.to(dtype=dtype)
    counts = valid.sum(dim=1)
    row_means = (
        torch.where(valid, values, torch.zeros_like(values))
        / counts.clamp_min(1).to(dtype=dtype).unsqueeze(1)
    ).sum(dim=1)
    weights = (
        torch.ones_like(row_means)
        if row_weights is None
        else row_weights.to(dtype=dtype)
    )
    weights = torch.where(counts > 0, weights, torch.zeros_like(weights))
    maximum = weights.max()
    if not bool(maximum > 0):
        return values.reshape(-1)[0] * 0
    # Scaling before summing avoids overflow for finite, very large weights.
    weights = weights / maximum
    result = (row_means * (weights / weights.sum())).sum()
    if not bool(torch.isfinite(result)):
        raise ValueError("masked sequence mean is not finite")
    return result


def policy_distillation(
    student_q: torch.Tensor,
    teacher_q: torch.Tensor,
    masks: torch.Tensor,
    valid: torch.Tensor,
    temp: float = 1.0,
) -> torch.Tensor:
    """Masked KL(teacher || student), T²-scaled and balanced per sequence.

    Teacher outputs are detached. Only mechanically legal actions enter either
    distribution. A padded position may have an empty action mask; a valid one
    must contain at least one legal action. No teacher or student is executed.
    """
    _finite_float("student_q", student_q, 3)
    _finite_float("teacher_q", teacher_q, 3)
    if teacher_q.shape != student_q.shape or teacher_q.device != student_q.device:
        raise ValueError("teacher_q must match student_q shape and device")
    _boolean("masks", masks, tuple(student_q.shape), student_q.device)
    _boolean("valid", valid, tuple(student_q.shape[:2]), student_q.device)
    if type(temp) not in (int, float) or not math.isfinite(temp) or temp <= 0:
        raise ValueError("temp must be finite and positive")
    has_action = masks.any(dim=-1)
    if bool((valid & ~has_action).any()):
        raise ValueError("every valid position needs a legal action")
    safe_masks = masks.clone()
    safe_masks[..., 0] |= ~has_action  # neutral calculation on ignored padding
    dtype = _math_dtype(student_q, teacher_q)
    student = student_q.to(dtype=dtype) / float(temp)
    teacher = teacher_q.detach().to(dtype=dtype) / float(temp)
    if not bool(torch.isfinite(student[safe_masks]).all()) or not bool(
        torch.isfinite(teacher[safe_masks]).all()
    ):
        raise ValueError("temperature-scaled logits must be finite")
    log_student = F.log_softmax(student.masked_fill(~safe_masks, -torch.inf), dim=-1)
    log_teacher = F.log_softmax(teacher.masked_fill(~safe_masks, -torch.inf), dim=-1)
    probabilities = log_teacher.exp()
    # Zero illegal log-probabilities BEFORE subtracting, avoiding 0 * (inf-inf).
    log_student = torch.where(safe_masks, log_student, torch.zeros_like(log_student))
    log_teacher = torch.where(safe_masks, log_teacher, torch.zeros_like(log_teacher))
    # Multiply twice instead of computing a Python T**2 scalar, so unrepresentable
    # temperatures are caught by finite-tensor validation rather than OverflowError.
    per_step = (
        (probabilities * (log_teacher - log_student)).sum(dim=-1)
        * float(temp)
        * float(temp)
    )
    return masked_sequence_mean(per_step, valid)


@torch.no_grad()
def n_step_targets(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    valid: torch.Tensor,
    bootstrap_values: torch.Tensor,
    gamma: float,
    n_step: int,
) -> torch.Tensor:
    """Detached finite-horizon returns from real transitions and successors.

    ``bootstrap_values[:, t+1]`` must describe the actual successor of transition
    t, including a time-limit truncation (never its autoreset observation).
    Termination cuts bootstrap. Truncation, padding, available-data end or the
    n-step horizon stop reward accumulation and bootstrap that real successor.
    No reward beyond an episode boundary or invalid transition is borrowed.
    If both terminal flags are true, true termination takes precedence.
    """
    _finite_float("rewards", rewards, 2)
    _finite_float("bootstrap_values", bootstrap_values, 2)
    shape, device = tuple(rewards.shape), rewards.device
    for name, value in (
        ("terminated", terminated),
        ("truncated", truncated),
        ("valid", valid),
    ):
        _boolean(name, value, shape, device)
    if (
        tuple(bootstrap_values.shape) != (shape[0], shape[1] + 1)
        or bootstrap_values.device != device
    ):
        raise ValueError(
            "bootstrap_values must have shape (batch, time+1) on the data device"
        )
    if (
        type(gamma) not in (int, float)
        or not math.isfinite(gamma)
        or not 0 <= gamma <= 1
    ):
        raise ValueError("gamma must be finite and in [0, 1]")
    if type(n_step) is not int or n_step < 1:
        raise ValueError("n_step must be a positive integer")
    dtype = _math_dtype(rewards, bootstrap_values)
    reward = rewards.detach().to(dtype=dtype)
    bootstrap = bootstrap_values.detach().to(dtype=dtype)
    targets = torch.zeros_like(reward)
    for anchor in range(shape[1]):
        active = valid[:, anchor].clone()
        total = torch.zeros(shape[0], dtype=dtype, device=device)
        discount = 1.0
        for distance in range(min(n_step, shape[1] - anchor)):
            index = anchor + distance
            real = active & valid[:, index]
            total += torch.where(
                real, discount * reward[:, index], torch.zeros_like(total)
            )
            discount *= float(gamma)
            terminal = real & terminated[:, index]
            boundary = truncated[:, index].clone()
            if index + 1 == shape[1] or distance + 1 == n_step:
                boundary |= True
            else:
                boundary |= ~valid[:, index + 1]
            finish = real & boundary & ~terminal
            total += torch.where(
                finish, discount * bootstrap[:, index + 1], torch.zeros_like(total)
            )
            active = real & ~terminal & ~boundary
            if not bool(active.any()):
                break
        targets[:, anchor] = total
    if not bool(torch.isfinite(targets).all()):
        raise ValueError("computed n-step targets must be finite")
    return targets
