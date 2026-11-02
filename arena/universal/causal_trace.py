"""Reconstruct learned action lookahead from authenticated, ALREADY recorded RGB.

This is offline inference, not another episode: no simulator, trainer, replay,
optimizer, environment step or true future observation is used for a decision.
It does not grant execution approval or supply new learning evidence.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import random
import re

import numpy as np
import torch

from . import viewer

TOLERANCE = 1e-6


class CausalTraceMismatch(ValueError):
    pass


def network_sha256(network):
    """Same tensor fingerprint convention as the recorded evaluator."""
    digest = hashlib.sha256()
    for name, tensor in sorted(network.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _rng_state():
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()


def _same_rng(before):
    now = np.random.get_state()
    return (random.getstate() == before[0] and now[0] == before[1][0]
            and np.array_equal(now[1], before[1][1]) and now[2:] == before[1][2:]
            and torch.equal(torch.random.get_rng_state(), before[2]))


def _masked_list(values, mask):
    values = values.detach().cpu().reshape(-1).tolist()
    return [float(value) if available else None for value, available in zip(values, mask)]


def _compare(values, recorded, mask, name, step):
    expected = viewer._q_vector(recorded, mask, name)
    actual = values.detach().cpu().reshape(-1).tolist()
    errors = [abs(float(actual[i])-float(expected[i])) for i in range(10) if mask[i]]
    error = max(errors)
    if not math.isfinite(error) or error > TOLERANCE:
        raise CausalTraceMismatch(f"step {step}: {name} differs by {error}; tolerance {TOLERANCE}")
    return error


def reconstruct_trace(network, frames, steps, *, provenance):
    """Reconstruct one recorded sequence and return complete per-action evidence.

    ``frames[t]`` is the CURRENT RGB at decision t. There is deliberately no
    next-RGB argument. Reward of step t is read only after reconstructing action
    t, then becomes the previous reward for t+1. The recurrent state advances
    through real recorded CURRENT frames, never through model-predicted frames.
    The core accepts synthetic fixtures; only the export wrapper accepts the
    five authenticated final demonstration identities.
    """
    if (not isinstance(frames, np.ndarray) or frames.dtype != np.uint8 or frames.ndim != 4
            or frames.shape[1:] != (32, 32, 3) or not 1 <= len(frames) <= 400
            or not isinstance(steps, list) or len(steps) != len(frames)):
        raise ValueError("bounded current RGB frames and matching recorded steps are required")
    if any(parameter.device.type != "cpu" for parameter in network.parameters()):
        raise ValueError("causal reconstruction uses CPU inference only")
    if not isinstance(provenance, dict):
        raise ValueError("explicit recorded provenance is required")
    for name, length in (("source_revision", 40), ("protocol_sha256", 64), ("network_sha256", 64)):
        if not re.fullmatch(rf"[0-9a-f]{{{length}}}", str(provenance.get(name, ""))):
            raise ValueError(f"invalid provenance {name}")
    before = network_sha256(network)
    if before != provenance["network_sha256"]:
        raise CausalTraceMismatch("loaded weights do not match the recorded network SHA-256")
    if any(not bool(torch.isfinite(value).all()) for value in network.state_dict().values()):
        raise ValueError("nonfinite reconstruction weights")
    rng = _rng_state()
    modes = [(module, module.training) for module in network.modules()]
    result = []
    try:
        network.eval()
        with torch.inference_mode():
            history = network.initial(1)
            previous_action = torch.zeros(1, dtype=torch.long)
            previous_reward = torch.zeros(1, dtype=torch.float32)
            for index, (frame, recorded) in enumerate(zip(frames, steps)):
                step = index+1
                if type(recorded.get("step")) is not int or recorded["step"] != step:
                    raise ValueError("recorded steps must be contiguous from one")
                mask = recorded.get("action_mask")
                if (not isinstance(mask, list) or len(mask) != 10 or not any(mask)
                        or any(type(value) is not bool for value in mask)):
                    raise ValueError("recorded mechanical mask must be bool[10]")
                if index and mask != steps[0]["action_mask"]:
                    raise ValueError("mechanical input availability must stay constant")
                if recorded.get("rgb_sha256") != hashlib.sha256(frame.tobytes()).hexdigest():
                    raise CausalTraceMismatch(f"step {step}: current RGB differs from its recorded hash")
                pixels = torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1))).unsqueeze(0)
                hidden_before = history
                direct, current_hidden, _ = network.step(pixels, previous_action, previous_reward, hidden_before)
                if tuple(current_hidden.shape) != (1, 64):
                    raise ValueError("the frozen causal trace requires the shared 64-unit recurrent state")
                # ALL ten primitive inputs, including currently unbound inputs,
                # are predicted in a single batch. The mask only restricts choice.
                actions = torch.arange(10, dtype=torch.long)
                predicted_hidden, reward_predictions, done_logits = network.model(current_hidden.expand(10, -1), actions)
                next_q = network.q_head(predicted_hidden)
                expected_shapes = ((10, 64), (10,), (10,), (10, 10), (1, 10))
                tensors = (predicted_hidden, reward_predictions, done_logits, next_q, direct)
                if any(tuple(value.shape) != shape or not bool(torch.isfinite(value).all())
                       for value, shape in zip(tensors, expected_shapes)):
                    raise ValueError("invalid/nonfinite action-conditioned model reconstruction")
                legal = torch.tensor(mask, dtype=torch.bool)
                done_probabilities = torch.sigmoid(done_logits)
                next_value = next_q.masked_fill(~legal[None, :], -torch.inf).max(dim=-1).values
                plan = reward_predictions+.98*(1-done_probabilities)*next_value
                final = .8*direct[0]+.2*plan
                choice = int(final.masked_fill(~legal, -torch.inf).argmax().item())
                errors = {name: _compare(values, recorded.get(name), mask, name, step)
                          for name, values in (("q_direct", direct[0]), ("model_values", plan), ("q_mixed", final))}
                if type(recorded.get("action")) is not int or recorded["action"] != choice:
                    raise CausalTraceMismatch(f"step {step}: reconstructed action differs from the recorded action")
                result.append({**provenance, "step": step,
                    "current_rgb_sha256": recorded["rgb_sha256"],
                    "previous_action": int(previous_action.item()), "previous_reward": float(previous_reward.item()),
                    "hidden_before_current_rgb": hidden_before[0].tolist(),
                    "hidden_after_current_rgb": current_hidden[0].tolist(),
                    "model_action_indices": list(range(10)), "pred_next_hidden": predicted_hidden.tolist(),
                    "predicted_reward": reward_predictions.tolist(), "predicted_done_logit": done_logits.tolist(),
                    "predicted_done_probability": done_probabilities.tolist(), "q_at_pred_next_hidden": next_q.tolist(),
                    "q_direct_all_actions": direct[0].tolist(), "q_plan_all_actions": plan.tolist(),
                    "q_final_all_actions": final.tolist(), "q_direct": _masked_list(direct, mask),
                    "q_plan": _masked_list(plan, mask), "q_final": _masked_list(final, mask),
                    "action_mask": mask, "action": choice, "recorded_max_abs_error": errors,
                    "absolute_tolerance": TOLERANCE, "gradient_enabled": False,
                    "future_observation_used_for_decision": False})
                # The action has now been reconstructed and checked. Only now is
                # its observed consequence allowed into the NEXT decision history.
                reward = viewer._number(recorded.get("reward"), "reward")
                clipped = viewer._number(recorded.get("reward_clipped"), "reward_clipped")
                if not math.isclose(clipped, min(1., max(-1., reward)), abs_tol=TOLERANCE):
                    raise ValueError("recorded reward clipping mismatch")
                previous_action = torch.tensor([choice], dtype=torch.long)
                previous_reward = torch.tensor([clipped], dtype=torch.float32)
                history = current_hidden
    finally:
        for module, training in modes:
            module.training = training
        if network_sha256(network) != before:
            raise RuntimeError("offline reconstruction mutated network parameters/buffers")
        if not _same_rng(rng):
            raise RuntimeError("offline reconstruction mutated a global RNG stream")
    return result


def _payload(path):
    text = path.read_text(encoding="utf-8")
    match = re.search(r'<script type="application/json" id="data">(.*?)</script>', text, re.S)
    if match is None:
        raise ValueError("authenticated viewer did not produce its recorded payload")
    return json.loads(match.group(1))


def export_causal_trace(evaluation_directory, checkpoint_directory, output_directory, *,
                        expected_evaluation_manifest_sha256, expected_checkpoint_manifest_sha256,
                        source_revision, protocol_sha256):
    """Authenticate existing clips and checkpoint, then export five offline traces.

    Expected hashes/provenance must come from the independently trusted catalog.
    No campaign permit is requested: this operation cannot collect or train.
    A generated HTML is retained to document the exact validated input clips.
    """
    from .campaign import _RealBackend, _manifest
    evaluation, checkpoint, output = map(Path, (evaluation_directory, checkpoint_directory, output_directory))
    if output.exists():
        raise FileExistsError(output)
    if output.resolve().is_relative_to(evaluation.resolve()) or output.resolve().is_relative_to(checkpoint.resolve()):
        raise ValueError("causal output must be outside the authenticated inputs")
    output.mkdir(parents=True, exist_ok=False)
    rng, failure, count, summaries = _rng_state(), None, 0, []
    try:
        receipt = viewer.build_viewer(evaluation, output/"authenticated-recordings.html",
                                      expected_manifest_sha256=expected_evaluation_manifest_sha256)
        clips = _payload(Path(receipt["path"]))
        backend = _RealBackend()
        network = backend.load_policy(checkpoint.parent,
            {"path": checkpoint.name, "manifest_sha256": expected_checkpoint_manifest_sha256,
             "model_seed": 2207, "stage": "after_adaptation"},
            {"source_revision": source_revision, "protocol_sha256": protocol_sha256})
        for clip in clips:
            steps = clip["steps"]
            # Ignore the final next-observation frame entirely for decisions.
            frames = np.stack([np.frombuffer(base64.b64decode(value, validate=True), dtype=np.uint8).reshape(32, 32, 3)
                               for value in clip["frames"][:len(steps)]])
            provenance = {**clip["cell"], "network_sha256": clip["network_sha256"],
                "environment_id": clip["environment_id"], "source_revision": source_revision,
                "protocol_sha256": protocol_sha256, "evaluation_manifest_sha256": expected_evaluation_manifest_sha256,
                "checkpoint_manifest_sha256": expected_checkpoint_manifest_sha256,
                "architecture": network.architecture_version, "offline_reconstruction": True}
            rows = reconstruct_trace(network, frames, steps, provenance=provenance)
            path = output/f"causal-{clip['game']}.jsonl"
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n")
            count += len(rows)
            summaries.append({"game": clip["game"], "steps": len(rows), "network_sha256": clip["network_sha256"],
                              "maximum_abs_error": max(max(row["recorded_max_abs_error"].values()) for row in rows)})
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if not _same_rng(rng):
            failure = {"type": "RNGMutation", "message": "offline export changed a global RNG stream"}
        status = {"status": "STOPPED" if failure else "VERIFIED", "failure": failure,
            "games": summaries, "verified_steps": count, "absolute_tolerance": TOLERANCE,
            "source_revision": source_revision, "protocol_sha256": protocol_sha256,
            "evaluation_manifest_sha256": expected_evaluation_manifest_sha256,
            "checkpoint_manifest_sha256": expected_checkpoint_manifest_sha256,
            "new_environment_interactions": 0, "new_episodes": 0, "training_started": False,
            "scientific_conclusion": "OFFLINE_REPRODUCTION_ONLY_NOT_NEW_LEARNING_EVIDENCE"}
        with (output/"CAUSAL_TRACE_STATUS.json").open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(status, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
        _manifest(output)
    if failure:
        raise CausalTraceMismatch(f"offline causal trace stopped: {failure['message']}")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-evaluation-manifest-sha256", required=True)
    parser.add_argument("--expected-checkpoint-manifest-sha256", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(export_causal_trace(args.evaluation, args.checkpoint, args.output,
        expected_evaluation_manifest_sha256=args.expected_evaluation_manifest_sha256,
        expected_checkpoint_manifest_sha256=args.expected_checkpoint_manifest_sha256,
        source_revision=args.source_revision, protocol_sha256=args.protocol_sha256), indent=2))


if __name__ == "__main__":
    main()
