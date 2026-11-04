"""Bounded observation-only evaluation for the independent continual v2 lineage.

This function does not authorize an experiment, load checkpoints, fit models or
select demonstrations. The caller owns the finite protocol and authorization.
The original controller is forked, so even its recurrent history is preserved.
"""

from __future__ import annotations

import hashlib
import json
import math
import random

import numpy as np
import torch

from arena.universal.envs import GAMES, VisualConfig, make_game
from arena.universal.evaluation import (
    _fingerprint,
    _mask,
    _numpy_state_equal,
    _pixels,
    _training_baseline,
    _vector,
)

from .policy import FrozenPolicyController


POLICIES = frozenset(
    (
        "shared",
        "scratch",
        "untrained",
        "random",
        "world_off",
        "gru_reset",
        "pixels_hidden",
    )
)
RNG_NAMESPACE = "NNA-CONTINUAL-V2-evaluation-v1"
TARGET_MODE = "continuous_gru_one_step"
EVALUATION_SCHEMA = "continual-v2-evaluation-v1"


def _local_rng(model_seed, game, environment_seed, stage, policy):
    identity = [RNG_NAMESPACE, model_seed, game, environment_seed, stage, policy]
    digest = hashlib.sha256(
        json.dumps(identity, separators=(",", ":")).encode()
    ).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _logged_vector(values, mask):
    return None if values is None else _vector(values.unsqueeze(0), mask)


def _raw_model_values(controller, decision, mask):
    """Audit lookahead independently of whether the gate admitted it to control."""
    values = decision["model_values"]
    if values is None:
        with torch.inference_mode():
            values = controller.network.plan_values(
                decision["hidden"], mask, gamma=controller.gamma
            )
        if tuple(values.shape) != (1, 10):
            raise ValueError("invalid raw learned lookahead shape")
        values = values[0]
    if not bool(torch.isfinite(values[mask[0]]).all()):
        raise ValueError("nonfinite raw learned lookahead")
    return values


def evaluate(
    controller: FrozenPolicyController | None,
    model_seed: int,
    stage: str,
    policy: str,
    game: str,
    environment_seed: int,
    max_steps: int = 400,
    *,
    training_baselines=None,
    record_world_predictions: bool | None = None,
    record_trajectory: bool = False,
    include_frames: bool = False,
    before_step=None,
    after_step=None,
    env_factory=None,
    difficulty: float = 1.0,
    scenario: str = "standard",
):
    """Return ``episode, action_pairs, world_predictions, trajectory``.

    All neural decisions go through ``FrozenPolicyController.act`` with only
    RGB, the previously executed input, clipped reward and mechanical mask.
    Separate ablation episodes start independent histories. Shared-after action
    pairs instead fork the SAME current history before each actual observation;
    they never advance the actual trajectory or feed their actions back to it.

    World targets are the ordinary one-step continuous GRU successor of the
    actual current memory, next RGB, chosen action and clipped reward. At a
    periodic boundary this is explicitly NOT the next reset control state.
    ``next_context_boundary`` distinguishes such targets. True termination is
    the terminal target; time-limit truncation is not death. Baselines must be
    supplied from training and are never estimated using evaluation rewards.

    ``model_values`` logs raw hypothetical lookahead, including when its gate
    is zero. ``q_mixed`` always comes verbatim from the controller, never from
    this logging calculation. Recording forecasts defaults to shared-after.
    No scientific seed allowlist or run budget is granted by this low-level API.
    """
    if policy not in POLICIES or game not in GAMES:
        raise ValueError("unknown evaluation policy or prepared game")
    if (
        type(model_seed) is not int
        or model_seed < 0
        or type(environment_seed) is not int
        or environment_seed < 0
    ):
        raise ValueError("model/environment seeds must be nonnegative integers")
    if type(max_steps) is not int or not 1 <= max_steps <= 400:
        raise ValueError("evaluation episode max_steps must be 1..400")
    if not isinstance(stage, str) or not stage:
        raise ValueError("evaluation stage must be a nonempty audit label")
    VisualConfig(max_steps=max_steps, difficulty=difficulty, scenario=scenario)
    if game != "agar" and scenario != "standard":
        raise ValueError("threat/prey scenarios are specific to Agar-like")
    if env_factory is not None and not callable(env_factory):
        raise ValueError("env_factory must be callable")
    if any(
        hook is not None and not callable(hook) for hook in (before_step, after_step)
    ):
        raise ValueError("budget hooks must be callable")
    shared_after = stage == "after_adaptation" and policy == "shared"
    if record_world_predictions is None:
        record_world_predictions = shared_after
    for value in (record_world_predictions, record_trajectory, include_frames):
        if type(value) is not bool:
            raise ValueError("recording flags must be boolean")
    if record_world_predictions and not shared_after:
        raise ValueError(
            "world prediction rows use only final shared-after trajectories"
        )
    if include_frames and not record_trajectory:
        raise ValueError("frames require trajectory recording")
    if policy == "random":
        if model_seed != 0 or controller is not None:
            raise ValueError("random control uses model_seed=0 and no controller")
        network = baseline = None
    else:
        if not isinstance(controller, FrozenPolicyController) or model_seed == 0:
            raise ValueError(
                "neural policies need an independent model seed and frozen v2 controller"
            )
        network = controller.network
        baseline = (
            dict(_training_baseline(training_baselines, game))
            if record_world_predictions
            else None
        )

    before_python = random.getstate()
    before_numpy = np.random.get_state()
    before_torch = torch.random.get_rng_state().clone()
    before_weights = _fingerprint(network) if network is not None else None
    modes = (
        [(module, module.training) for module in network.modules()]
        if network is not None
        else []
    )
    generator = _local_rng(model_seed, game, environment_seed, stage, policy)
    counterfactuals, predictions, trajectory = [], [], []
    env = None
    try:
        actor = None if controller is None else controller.fork()
        if actor is not None:
            actor.reset()
        factory = make_game if env_factory is None else env_factory
        env = factory(
            game, max_steps=max_steps, difficulty=difficulty, scenario=scenario
        )
        observation, info = env.reset(seed=environment_seed)
        mechanical_mask = _mask(info)
        current_mask = mechanical_mask
        previous_action, previous_reward = 0, 0.0
        raw_return = clipped_return = 0.0
        final_info = info
        for step in range(1, max_steps + 1):
            if before_step is not None:
                before_step()
            _pixels(observation)
            # Some renderers reuse their framebuffer in step(); preserve the
            # image actually used for this decision before they overwrite it.
            current_rgb = observation.copy() if record_trajectory else None
            decision = raw_values = None
            # This independent phase matches the fresh actor's reset schedule.
            next_boundary = bool(actor is not None and step % actor.context_length == 0)
            if actor is None:
                action = int(generator.choice(np.flatnonzero(current_mask.numpy()[0])))
            else:
                before_observation = actor.fork() if shared_after else None
                decision = actor.act(
                    observation,
                    previous_action,
                    previous_reward,
                    current_mask[0],
                    world_on=policy != "world_off",
                    pixels_hidden=policy == "pixels_hidden",
                    gru_reset=policy == "gru_reset",
                )
                action = decision["action"]
                if shared_after or record_trajectory:
                    raw_values = _raw_model_values(actor, decision, current_mask)
                if shared_after:
                    reset = before_observation.fork().act(
                        observation,
                        previous_action,
                        previous_reward,
                        current_mask[0],
                        gru_reset=True,
                    )
                    hidden_pixels = before_observation.act(
                        observation,
                        previous_action,
                        previous_reward,
                        current_mask[0],
                        pixels_hidden=True,
                    )
                    # World-off at this state is exactly the same direct Q,
                    # including its tie rule; it does not need a fourth observe.
                    counterfactuals.append(
                        {
                            "model_seed": model_seed,
                            "stage": stage,
                            "policy": policy,
                            "game": game,
                            "environment_seed": environment_seed,
                            "step": step,
                            "action_default": action,
                            "action_world_off": int(
                                decision["q_direct"].argmax().item()
                            ),
                            "action_gru_reset": reset["action"],
                            "action_pixels_hidden": hidden_pixels["action"],
                            "q_direct": _logged_vector(
                                decision["q_direct"], current_mask
                            ),
                            "model_values": _logged_vector(raw_values, current_mask),
                            "q_mixed": _logged_vector(
                                decision["q_mixed"], current_mask
                            ),
                            "q_mixed_gru_reset": _logged_vector(
                                reset["q_mixed"], current_mask
                            ),
                            "q_mixed_pixels_hidden": _logged_vector(
                                hidden_pixels["q_mixed"], current_mask
                            ),
                            "action_mask": json.dumps(
                                current_mask[0].tolist(), separators=(",", ":")
                            ),
                            "planning_weight": decision["planning_weight"],
                            "planning_weight_world_off": 0.0,
                            "planning_weight_gru_reset": reset["planning_weight"],
                            "planning_weight_pixels_hidden": hidden_pixels[
                                "planning_weight"
                            ],
                            "context_reset": decision["context_reset"],
                            "context_reset_gru_reset": reset["context_reset"],
                            "context_reset_pixels_hidden": hidden_pixels[
                                "context_reset"
                            ],
                            "next_context_boundary": next_boundary,
                        }
                    )

            action_tensor = torch.tensor([action], dtype=torch.long)
            if record_world_predictions:
                # Freeze chosen-action forecasts before the simulator advances.
                with torch.inference_mode():
                    predicted_hidden, predicted_reward, done_logit = network.model(
                        decision["hidden"], action_tensor
                    )
                # Defensive copies also prevent mutable output buffers from
                # changing a forecast while the environment advances.
                predicted_hidden = predicted_hidden.detach().clone()
                predicted_reward = predicted_reward.detach().clone()
                done_logit = done_logit.detach().clone()
            next_observation, reward, terminated, truncated, next_info = env.step(
                action
            )
            # Account for the actual interaction even if its result is invalid
            # or the episode is subsequently interrupted by a resource limit.
            if after_step is not None:
                after_step()
            if (
                isinstance(reward, (bool, np.bool_))
                or not isinstance(reward, (int, float, np.integer, np.floating))
                or not math.isfinite(float(reward))
                or type(terminated) is not bool
                or type(truncated) is not bool
            ):
                raise ValueError("nonfinite reward or invalid termination flags")
            next_pixels = _pixels(next_observation)
            next_mask = _mask(next_info)
            if not torch.equal(next_mask, mechanical_mask):
                raise ValueError(
                    "action mask changed with state; mechanical availability must be constant"
                )
            clipped_reward = max(-1.0, min(1.0, float(reward)))
            if record_world_predictions:
                with torch.inference_mode():
                    _, target_next_hidden, _ = network.step(
                        next_pixels,
                        action_tensor,
                        torch.tensor([clipped_reward], dtype=torch.float32),
                        decision["hidden"],
                    )
                    values = (
                        predicted_hidden,
                        predicted_reward,
                        done_logit,
                        target_next_hidden,
                    )
                    if not all(bool(torch.isfinite(value).all()) for value in values):
                        raise ValueError("nonfinite world-model prediction/target")
                    terminal_probability = float(torch.sigmoid(done_logit)[0].item())
                    reward_prediction = float(predicted_reward[0].item())
                    model_error = float(
                        (predicted_hidden - target_next_hidden).square().mean().item()
                    )
                    copy_error = float(
                        (decision["hidden"] - target_next_hidden).square().mean().item()
                    )
                predictions.append(
                    {
                        "model_seed": model_seed,
                        "stage": stage,
                        "policy": policy,
                        "game": game,
                        "environment_seed": environment_seed,
                        "transition_id": f"{stage}:{policy}:{game}:{model_seed}:{environment_seed}:{step}",
                        "step": step,
                        "action": action,
                        "target_mode": TARGET_MODE,
                        "next_context_boundary": next_boundary,
                        "context_reset": decision["context_reset"],
                        "planning_weight": decision["planning_weight"],
                        "latent_mse_model": model_error,
                        "latent_mse_copy": copy_error,
                        "reward_target_clipped": clipped_reward,
                        "reward_prediction": reward_prediction,
                        "reward_mae_model": abs(reward_prediction - clipped_reward),
                        "reward_mae_baseline": abs(
                            float(baseline["reward_mean"]) - clipped_reward
                        ),
                        "terminal_probability": terminal_probability,
                        "terminal_brier_model": (
                            terminal_probability - float(terminated)
                        )
                        ** 2,
                        "terminal_brier_baseline": (
                            float(baseline["terminal_rate"]) - float(terminated)
                        )
                        ** 2,
                        "terminated": int(terminated),
                        "truncated": int(truncated),
                        "baseline_source": baseline["source"],
                        "baseline_support": baseline["support"],
                        "baseline_reward_mean": float(baseline["reward_mean"]),
                        "baseline_terminal_rate": float(baseline["terminal_rate"]),
                    }
                )

            if record_trajectory:
                frame = {
                    "model_seed": model_seed,
                    "stage": stage,
                    "policy": policy,
                    "game": game,
                    "environment_seed": environment_seed,
                    "step": step,
                    "action": action,
                    "reward": float(reward),
                    "reward_clipped": clipped_reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "rgb_sha256": hashlib.sha256(current_rgb.tobytes()).hexdigest(),
                    "next_rgb_sha256": hashlib.sha256(
                        next_observation.tobytes()
                    ).hexdigest(),
                    "action_mask": current_mask[0].tolist(),
                    "q_direct": (
                        None
                        if decision is None
                        else _logged_vector(decision["q_direct"], current_mask)
                    ),
                    "model_values": _logged_vector(raw_values, current_mask),
                    "q_mixed": (
                        None
                        if decision is None
                        else _logged_vector(decision["q_mixed"], current_mask)
                    ),
                    "planning_weight": (
                        0.0 if decision is None else decision["planning_weight"]
                    ),
                    "context_reset": (
                        False if decision is None else decision["context_reset"]
                    ),
                    "next_context_boundary": next_boundary,
                }
                if include_frames:
                    frame.update(rgb=current_rgb, next_rgb=next_observation.copy())
                trajectory.append(frame)
            raw_return += float(reward)
            clipped_return += clipped_reward
            observation, final_info, current_mask = (
                next_observation,
                next_info,
                next_mask,
            )
            previous_action, previous_reward = action, clipped_reward
            if terminated or truncated:
                break

        row = {
            "evaluation_schema": EVALUATION_SCHEMA,
            "model_seed": model_seed,
            "stage": stage,
            "policy": policy,
            "game": game,
            "environment_seed": environment_seed,
            "episode": 1,
            "steps": step,
            "return": raw_return,
            "return_clipped": clipped_return,
            "terminated": int(terminated),
            "truncated": int(truncated),
            "environment_id": final_info.get(
                "environment_id", getattr(env, "identity", "unknown")
            ),
            "network_sha256": before_weights or "none-random",
            "policy_revision": None if actor is None else actor.policy_revision,
            "context_length": None if actor is None else actor.context_length,
        }
        for name, value in final_info.items():
            if name == "action_mask" or name in row:
                continue
            if isinstance(value, np.generic):
                value = value.item()
            # Output-only telemetry is never passed to the controller.
            row[name] = (
                json.dumps(
                    value, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                if isinstance(value, (list, dict))
                else value
            )
        if "score" not in row:
            raise ValueError("environment did not supply its raw task score")
        return row, counterfactuals, predictions, trajectory
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            mode_changed = any(
                module.training != training for module, training in modes
            )
            for module, training in modes:
                module.training = training
            rng_changed = (
                random.getstate() != before_python
                or not _numpy_state_equal(np.random.get_state(), before_numpy)
                or not torch.equal(torch.random.get_rng_state(), before_torch)
            )
            # A misbehaving factory/forward must not contaminate later cells.
            if rng_changed:
                random.setstate(before_python)
                np.random.set_state(before_numpy)
                torch.random.set_rng_state(before_torch)
            if network is not None and _fingerprint(network) != before_weights:
                raise RuntimeError("evaluation mutated network parameters/buffers")
            if mode_changed:
                raise RuntimeError("evaluation changed frozen network modes")
            if rng_changed:
                raise RuntimeError("evaluation consumed/mutated a global RNG stream")
