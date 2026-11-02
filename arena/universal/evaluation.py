"""Real inference-only episodes for the new visual lineage, never a campaign.

Calling ``evaluate`` does not establish approval to use reserved scientific
seeds. The finite campaign/authorization gate belongs to the caller. This
module has no optimizer, replay insertion, checkpoint loading or expert policy.
"""
from __future__ import annotations

import hashlib
import json
import math
import random

import numpy as np
import torch

from .envs import GAMES, make_game

POLICIES = frozenset(("shared", "scratch", "world_off", "gru_reset",
                      "pixels_hidden", "untrained", "random"))
GAMMA = .98
WORLD_WEIGHT = .20


def _fingerprint(network) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(network.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _numpy_state_equal(first, second):
    return (first[0] == second[0] and np.array_equal(first[1], second[1])
            and first[2:] == second[2:])


def _pixels(frame):
    if not isinstance(frame, np.ndarray) or frame.shape != (32, 32, 3) or frame.dtype != np.uint8:
        raise ValueError("policy receives only RGB uint8 32x32 frames")
    return torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1))).unsqueeze(0)


def _mask(info):
    value = np.asarray(info.get("action_mask"))
    if value.dtype != np.bool_ or value.shape != (10,) or not value.any():
        raise ValueError("mechanical action mask must be bool[10] with a legal input")
    return torch.from_numpy(value.copy()).unsqueeze(0)


def _vector(values, mask):
    numbers = values.detach().cpu()[0].tolist()
    available = mask[0].tolist()
    return json.dumps([float(value) if legal else None for value, legal in zip(numbers, available)],
                      separators=(",", ":"), allow_nan=False)


def _decide(network, q, hidden, mask, *, world_on=True):
    direct = q.masked_fill(~mask, -torch.inf)
    if not bool(torch.isfinite(direct[mask]).all()):
        raise ValueError("nonfinite direct Q value")
    if world_on:
        model_values = network.plan_values(hidden, mask, gamma=GAMMA)
        if not bool(torch.isfinite(model_values[mask]).all()):
            raise ValueError("nonfinite learned model value")
        mixed = ((1-WORLD_WEIGHT)*direct+WORLD_WEIGHT*model_values).masked_fill(~mask, -torch.inf)
    else:
        model_values, mixed = None, direct
    # Torch argmax fixes ties by the first joypad index; no per-game controller.
    return int(mixed.argmax(dim=-1).item()), direct, model_values, mixed


def _training_baseline(training_baselines, game):
    if not isinstance(training_baselines, dict) or not isinstance(training_baselines.get(game), dict):
        raise ValueError("world prediction evaluation requires baselines from training data")
    baseline = training_baselines[game]
    if baseline.get("source") != "training" or type(baseline.get("support")) is not int or baseline["support"] <= 0:
        raise ValueError("baseline provenance/support must describe training data")
    reward, terminal = baseline.get("reward_mean"), baseline.get("terminal_rate")
    if (isinstance(reward, bool) or not isinstance(reward, (int, float))
            or not math.isfinite(reward) or not -1 <= reward <= 1):
        raise ValueError("training reward_mean must be clipped-target mean in [-1,1]")
    if (isinstance(terminal, bool) or not isinstance(terminal, (int, float))
            or not math.isfinite(terminal) or not 0 <= terminal <= 1):
        raise ValueError("training terminal_rate must be a probability")
    return baseline


def _local_rng(model_seed, game, environment_seed, stage, policy):
    identity = ["NNA-MULTIJOGOS-002-evaluation-v1", model_seed, game,
                environment_seed, stage, policy]
    digest = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def evaluate(network, model_seed: int, stage: str, policy: str, game: str,
             environment_seed: int, max_steps: int = 400, *,
             training_baselines=None, record_world_predictions: bool | None = None,
             record_trajectory: bool = False, include_frames: bool = False,
             difficulty: float = 1.0, scenario: str = "standard",
             env_factory=None):
    """Return ``episode, action_pairs, world_predictions, trajectory``.

    The caller supplies an already loaded network (cached between episodes).
    Only ``after_adaptation/shared`` produces same-state counterfactual action
    pairs, at EVERY executed step. Separate ablation episodes have their own
    histories and cannot substitute for those pairs. Hidden/pixels interventions
    alter inference only; environment physics, reward and initial seed stay real.

    Predicted memory targets are ``step(actual_next_rgb, action, clipped_reward,
    current_hidden).hidden``. Persistence baseline copies current_hidden.
    Reward targets and previous reward inputs are clipped to [-1,1]. Termination
    is true environmental termination; a time-limit truncation is not death.
    Baselines are never estimated on evaluation trajectories. Predictions use
    shared-after final trajectories, disjoint from TRAINING, with no extra
    interactions. Default recording is on only for shared-after.
    """
    if policy not in POLICIES or game not in GAMES:
        raise ValueError("unknown evaluation policy or prepared game")
    if type(model_seed) is not int or model_seed < 0 or type(environment_seed) is not int or environment_seed < 0:
        raise ValueError("model/environment seeds must be nonnegative integers")
    if type(max_steps) is not int or not 1 <= max_steps <= 400:
        raise ValueError("evaluation episode max_steps must be 1..400")
    if not isinstance(stage, str) or not stage:
        raise ValueError("evaluation stage must be a nonempty audit label")
    shared_after = stage == "after_adaptation" and policy == "shared"
    if record_world_predictions is None:
        record_world_predictions = shared_after
    if record_world_predictions is True and not shared_after:
        raise ValueError("world prediction rows use only final shared-after trajectories")
    for value in (record_world_predictions, record_trajectory, include_frames):
        if type(value) is not bool:
            raise ValueError("recording flags must be boolean")
    if include_frames and not record_trajectory:
        raise ValueError("frames require trajectory recording")
    if policy == "random":
        if model_seed != 0 or network is not None:
            raise ValueError("random control uses model_seed=0 and no neural network")
        baseline = None
    else:
        if network is None or model_seed == 0:
            raise ValueError("neural policies need their independent model seed and loaded network")
        if any(parameter.device.type != "cpu" for parameter in network.parameters()):
            raise ValueError("this finite protocol uses CPU networks only")
        baseline = _training_baseline(training_baselines, game) if record_world_predictions else None

    before_python = random.getstate()
    before_numpy = np.random.get_state()
    before_torch = torch.random.get_rng_state().clone()
    before_weights = _fingerprint(network) if network is not None else None
    modes = [(module, module.training) for module in network.modules()] if network is not None else []
    generator = _local_rng(model_seed, game, environment_seed, stage, policy)
    counterfactuals, predictions, trajectory = [], [], []
    env = None
    try:
        if network is not None:
            network.eval()
        factory = make_game if env_factory is None else env_factory
        env = factory(game, max_steps=max_steps, difficulty=difficulty, scenario=scenario)
        observation, info = env.reset(seed=environment_seed)
        mechanical_mask = _mask(info)
        # The simulator may log private telemetry; it is never passed to a net.
        current_mask = mechanical_mask
        hidden = network.initial(1) if network is not None else None
        previous_action = torch.zeros(1, dtype=torch.long)
        previous_reward = torch.zeros(1, dtype=torch.float32)
        raw_return = clipped_return = 0.
        final_info = info
        with torch.inference_mode():
            for step in range(1, max_steps+1):
                real_pixels = _pixels(observation)
                direct = model_values = mixed = None
                if policy == "random":
                    action = int(generator.choice(np.flatnonzero(current_mask.numpy()[0])))
                    current_hidden = None
                else:
                    actual_history = hidden
                    incoming_hidden = network.initial(1) if policy == "gru_reset" else actual_history
                    seen_pixels = torch.zeros_like(real_pixels) if policy == "pixels_hidden" else real_pixels
                    q, current_hidden, _ = network.step(seen_pixels, previous_action,
                                                        previous_reward, incoming_hidden)
                    action, direct, model_values, mixed = _decide(network, q, current_hidden,
                                                  current_mask, world_on=policy != "world_off")
                    if stage == "after_adaptation" and policy == "shared":
                        reset_q, reset_hidden, _ = network.step(real_pixels, previous_action,
                                                               previous_reward, network.initial(1))
                        pixels_q, pixels_hidden, _ = network.step(torch.zeros_like(real_pixels),
                                                        previous_action, previous_reward, actual_history)
                        reset_action = _decide(network, reset_q, reset_hidden, current_mask)[0]
                        hidden_pixels_action = _decide(network, pixels_q, pixels_hidden, current_mask)[0]
                        counterfactuals.append({"model_seed": model_seed, "game": game,
                            "environment_seed": environment_seed, "step": step,
                            "action_default": action,
                            "action_world_off": int(direct.argmax(dim=-1).item()),
                            "action_gru_reset": reset_action,
                            "action_pixels_hidden": hidden_pixels_action,
                            "q_direct": _vector(direct, current_mask),
                            "model_values": _vector(model_values, current_mask),
                            "q_mixed": _vector(mixed, current_mask),
                            "action_mask": json.dumps(current_mask[0].tolist(), separators=(",", ":"))})

                next_observation, reward, terminated, truncated, next_info = env.step(action)
                if not math.isfinite(float(reward)) or type(terminated) is not bool or type(truncated) is not bool:
                    raise ValueError("nonfinite reward or invalid termination flags")
                next_mask = _mask(next_info)
                if not torch.equal(next_mask, mechanical_mask):
                    raise ValueError("action mask changed with state; mechanical availability must be constant")
                clipped_reward = float(np.clip(reward, -1., 1.))
                action_tensor = torch.tensor([action], dtype=torch.long)
                reward_tensor = torch.tensor([clipped_reward], dtype=torch.float32)
                if network is not None and record_world_predictions:
                    predicted_hidden, predicted_reward, done_logit = network.model(current_hidden, action_tensor)
                    _, target_next_hidden, _ = network.step(_pixels(next_observation), action_tensor,
                                                           reward_tensor, current_hidden)
                    terminal_probability = float(torch.sigmoid(done_logit)[0].item())
                    values = (predicted_hidden, predicted_reward, done_logit, target_next_hidden)
                    if not all(bool(torch.isfinite(value).all()) for value in values):
                        raise ValueError("nonfinite world-model prediction/target")
                    transition_id = f"{stage}:{policy}:{game}:{model_seed}:{environment_seed}:{step}"
                    predictions.append({"model_seed": model_seed, "stage": stage, "policy": policy,
                        "game": game, "environment_seed": environment_seed,
                        "transition_id": transition_id, "step": step, "action": action,
                        "latent_mse_model": float((predicted_hidden-target_next_hidden).square().mean().item()),
                        "latent_mse_copy": float((current_hidden-target_next_hidden).square().mean().item()),
                        "reward_target_clipped": clipped_reward,
                        "reward_prediction": float(predicted_reward[0].item()),
                        "reward_mae_model": abs(float(predicted_reward[0].item())-clipped_reward),
                        "reward_mae_baseline": abs(float(baseline["reward_mean"])-clipped_reward),
                        "terminal_probability": terminal_probability,
                        "terminal_brier_model": (terminal_probability-float(terminated))**2,
                        "terminal_brier_baseline": (float(baseline["terminal_rate"])-float(terminated))**2,
                        "terminated": int(terminated), "truncated": int(truncated),
                        "baseline_source": baseline["source"], "baseline_support": baseline["support"],
                        "baseline_reward_mean": float(baseline["reward_mean"]),
                        "baseline_terminal_rate": float(baseline["terminal_rate"])})

                if record_trajectory:
                    frame = {"model_seed": model_seed, "stage": stage, "policy": policy,
                             "game": game, "environment_seed": environment_seed, "step": step,
                             "action": action, "reward": float(reward), "reward_clipped": clipped_reward,
                             "terminated": terminated, "truncated": truncated,
                             "rgb_sha256": hashlib.sha256(observation.tobytes()).hexdigest(),
                             "next_rgb_sha256": hashlib.sha256(next_observation.tobytes()).hexdigest(),
                             "action_mask": current_mask[0].tolist(),
                             "q_direct": _vector(direct, current_mask) if direct is not None else None,
                             "model_values": _vector(model_values, current_mask) if model_values is not None else None,
                             "q_mixed": _vector(mixed, current_mask) if mixed is not None else None}
                    if include_frames:
                        frame["rgb"] = observation.copy()
                        frame["next_rgb"] = next_observation.copy()
                    trajectory.append(frame)
                raw_return += float(reward)
                clipped_return += clipped_reward
                observation, final_info, current_mask = next_observation, next_info, next_mask
                hidden = current_hidden
                previous_action, previous_reward = action_tensor, reward_tensor
                if terminated or truncated:
                    break

        row = {"model_seed": model_seed, "stage": stage, "policy": policy, "game": game,
               "environment_seed": environment_seed, "episode": 1, "steps": step,
               "return": raw_return, "return_clipped": clipped_return,
               "terminated": int(terminated), "truncated": int(truncated),
               "environment_id": final_info.get("environment_id", env.identity),
               "network_sha256": before_weights or "none-random"}
        for name, value in final_info.items():
            if name == "action_mask" or name in row:
                continue
            if isinstance(value, np.generic):
                value = value.item()
            # Telemetry is output only; nested metrics are JSON for CSV safety.
            row[name] = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) if isinstance(value, (list, dict)) else value
        if "score" not in row:
            raise ValueError("environment did not supply its raw task score")
        return row, counterfactuals, predictions, trajectory
    finally:
        if env is not None:
            env.close()
        for module, training in modes:
            module.training = training
        if network is not None and _fingerprint(network) != before_weights:
            raise RuntimeError("evaluation mutated network parameters/buffers")
        if (random.getstate() != before_python or not _numpy_state_equal(np.random.get_state(), before_numpy)
                or not torch.equal(torch.random.get_rng_state(), before_torch)):
            raise RuntimeError("evaluation consumed/mutated a global RNG stream")
