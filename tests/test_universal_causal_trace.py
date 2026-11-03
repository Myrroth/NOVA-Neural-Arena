"""Synthetic pixels/tensor traces only, never any game, rollout or learning.

The integration fixture contains 2207/920001 as required FILE METADATA only.
The random network is initialized with 810301; no final seed is executed.
Every score/frame/checkpoint in this file is explicitly an invented fixture.
"""
import copy
import csv
import hashlib
import json
import random
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from arena.universal import causal_trace as ct
from arena.universal import protocol as p
from arena.universal.envs import ACTION_NAMES
from arena.universal.network import VisualQNetwork


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def network():
    with torch.random.fork_rng():
        torch.manual_seed(810301)
        return VisualQNetwork()


def recorded_tensors(network, *, game="synthetic_fixture", metadata_seed=810302):
    """Independently record aggregate plan_values, without a simulator."""
    frames = np.stack([np.full((32, 32, 3), 20+i, dtype=np.uint8) for i in range(4)])
    mask = [True]*5+[False]*5
    mask_tensor = torch.tensor([mask])
    pa, pr, hidden = torch.zeros(1, dtype=torch.long), torch.zeros(1), network.initial(1)
    steps = []
    with torch.inference_mode():
        for i, reward in enumerate((.7, -.9, .3)):
            pixels = torch.from_numpy(frames[i].copy()).permute(2, 0, 1).unsqueeze(0)
            q, hidden, _ = network.step(pixels, pa, pr, hidden)
            plan = network.plan_values(hidden, mask_tensor)
            mixed = (.8*q+.2*plan).masked_fill(~mask_tensor, -torch.inf)
            action = int(mixed.argmax().item())
            vector = lambda x: json.dumps([float(x[0, j]) if mask[j] else None for j in range(10)])
            steps.append({"model_seed": 2207 if metadata_seed == 920001 else 1103,
                "stage": "after_adaptation", "policy": "shared", "game": game,
                "environment_seed": metadata_seed, "step": i+1, "action": action,
                "reward": reward, "reward_clipped": reward, "terminated": i == 2, "truncated": False,
                "action_mask": mask.copy(), "rgb_sha256": hashlib.sha256(frames[i].tobytes()).hexdigest(),
                "next_rgb_sha256": hashlib.sha256(frames[i+1].tobytes()).hexdigest(),
                "q_direct": vector(q), "model_values": vector(plan), "q_mixed": vector(mixed),
                "synthetic_unit_fixture": True})
            pa, pr = torch.tensor([action]), torch.tensor([reward])
    return frames, steps


def provenance(network):
    return {"source_revision": "a"*40, "protocol_sha256": "b"*64,
            "network_sha256": ct.network_sha256(network), "game": "synthetic_fixture",
            "model_seed": 1103, "environment_seed": 810302, "synthetic_unit_fixture": True}


def test_all_ten_model_consequences_reproduce_recorded_planning_and_history(network, monkeypatch):
    frames, steps = recorded_tensors(network)
    calls, original_model = [], network.model
    def capture_model(hidden, actions):
        assert not torch.is_grad_enabled()
        assert hidden.shape == (10, 64) and actions.tolist() == list(range(10))
        calls.append(True)
        return original_model(hidden, actions)
    monkeypatch.setattr(network, "model", capture_model)
    monkeypatch.setattr(network, "plan_values", lambda *a, **k: pytest.fail("core must expose full model output, not reuse aggregate planning"))
    rows = ct.reconstruct_trace(network, frames[:3], steps, provenance=provenance(network))
    assert len(rows) == len(calls) == 3
    for row, recorded in zip(rows, steps):
        assert np.asarray(row["pred_next_hidden"]).shape == (10, 64)
        assert np.asarray(row["q_at_pred_next_hidden"]).shape == (10, 10)
        assert len(row["predicted_reward"]) == len(row["predicted_done_logit"]) == 10
        assert row["action"] == recorded["action"]
        assert max(row["recorded_max_abs_error"].values()) <= 1e-6
        assert row["future_observation_used_for_decision"] is False
    assert rows[0]["previous_reward"] == 0.
    assert rows[1]["previous_reward"] == pytest.approx(.7)
    assert rows[2]["previous_reward"] == pytest.approx(-.9)
    assert rows[1]["previous_action"] == rows[0]["action"]
    assert rows[1]["hidden_before_current_rgb"] == rows[0]["hidden_after_current_rgb"]


def test_future_reward_is_read_only_after_action_comparison_and_next_rgb_is_never_read(network):
    frames, steps = recorded_tensors(network)
    class OrderedRecord(dict):
        action_checked = False
        def get(self, name, default=None):
            if name == "action":
                self.action_checked = True
            if name in ("reward", "reward_clipped"):
                assert self.action_checked, "future outcome read before reconstructed decision"
            if name in ("next_rgb", "next_observation", "future_reward"):
                pytest.fail("future input must never be queried")
            return super().get(name, default)
    ordered = [OrderedRecord({**row, "next_rgb": object(), "future_reward": object()}) for row in steps]
    ct.reconstruct_trace(network, frames[:3], ordered, provenance=provenance(network))
    assert all(row.action_checked for row in ordered)


def test_offline_reconstruction_preserves_weights_rng_modes_and_gradients(network):
    frames, steps = recorded_tensors(network)
    network.train()
    network.encoder.eval()
    modes = [module.training for module in network.modules()]
    weights = {name: value.clone() for name, value in network.state_dict().items()}
    rng = (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone())
    ct.reconstruct_trace(network, frames[:3], steps, provenance=provenance(network))
    assert modes == [module.training for module in network.modules()]
    assert all(torch.equal(value, weights[name]) for name, value in network.state_dict().items())
    assert all(parameter.grad is None for parameter in network.parameters())
    assert random.getstate() == rng[0]
    assert np.array_equal(np.random.get_state()[1], rng[1][1])
    assert torch.equal(torch.random.get_rng_state(), rng[2])


@pytest.mark.parametrize("field", ["q_direct", "model_values", "q_mixed"])
def test_aggregated_record_mismatch_is_not_silently_accepted(network, field):
    frames, steps = recorded_tensors(network)
    values = json.loads(steps[1][field])
    values[0] += 1e-3
    steps[1][field] = json.dumps(values)
    with pytest.raises(ct.CausalTraceMismatch, match=field):
        ct.reconstruct_trace(network, frames[:3], steps, provenance=provenance(network))


def test_wrong_checkpoint_hash_or_recorded_action_fails_closed(network):
    frames, steps = recorded_tensors(network)
    wrong = provenance(network) | {"network_sha256": "0"*64}
    with pytest.raises(ct.CausalTraceMismatch, match="loaded weights"):
        ct.reconstruct_trace(network, frames[:3], steps, provenance=wrong)
    steps[1]["action"] = (steps[1]["action"]+1) % 5
    with pytest.raises(ct.CausalTraceMismatch, match="recorded action"):
        ct.reconstruct_trace(network, frames[:3], steps, provenance=provenance(network))


def seal(root):
    entries = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}"
               for path in sorted(root.rglob("*")) if path.is_file() and path != root/"SHA256SUMS"]
    raw = ("\n".join(entries)+"\n").encode()
    (root/"SHA256SUMS").write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def authenticated_files(tmp_path, network):
    evaluation, checkpoint = tmp_path/"SYNTHETIC_EVALUATION", tmp_path/"SYNTHETIC_CHECKPOINT"
    evaluation.mkdir(); checkpoint.mkdir()
    digest = ct.network_sha256(network)
    metadata = {"source_revision": "a"*40, "protocol_sha256": "b"*64,
        "model_seed": 2207, "stage": "adaptation", "action_names": list(ACTION_NAMES),
        "observation_shape": [32, 32, 3], "architecture": network.architecture_version,
        "config": {"hidden_size": 64, "latent_size": 64}, "synthetic_unit_fixture": True}
    (checkpoint/"metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    torch.save({"online": network.state_dict()}, checkpoint/"state.pt")
    checkpoint_sha = seal(checkpoint)
    for game in p.GAMES:
        frames, steps = recorded_tensors(network, game=game, metadata_seed=920001)
        directory = evaluation/"demonstrations"/game
        directory.mkdir(parents=True)
        np.savez_compressed(directory/"frames.npz", rgb=frames[:3], next_rgb=frames[1:])
        cell = {name: steps[0][name] for name in ("stage", "policy", "game", "model_seed", "environment_seed")}
        (directory/"trajectory.json").write_text(json.dumps({"cell": cell, "steps": steps,
            "network_sha256": digest, "environment_id": f"visual-{game}-v1",
            "synthetic_unit_fixture": True}), encoding="utf-8")
    rows = []
    for cell in p.evaluation_matrix():
        fixed = cell.stage == "after_adaptation" and cell.policy == "shared" and cell.model_seed == 2207 and cell.environment_seed == 920001
        rows.append({**vars(cell), "episode": 1, "steps": 3 if fixed else 1, "return": .1 if fixed else 0.,
            "return_clipped": .1 if fixed else 0., "score": 0., "terminated": 1, "truncated": 0,
            "network_sha256": digest, "environment_id": f"visual-{cell.game}-v1", "synthetic_unit_fixture": True})
    with (evaluation/"episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (evaluation/"EVALUATION_STATUS.json").write_text(json.dumps({"status": "SUCCESS", "test_only": False,
        "completed_episodes": 1735, "synthetic_unit_fixture": True}), encoding="utf-8")
    return evaluation, checkpoint, seal(evaluation), checkpoint_sha


def test_export_authenticates_real_file_formats_and_safe_loads_only_fixture_weights(authenticated_files, tmp_path, monkeypatch):
    from arena.universal import envs
    evaluation, checkpoint, eval_sha, checkpoint_sha = authenticated_files
    monkeypatch.setattr(envs, "make_game", lambda *a, **k: pytest.fail("no simulator may be constructed"))
    monkeypatch.setitem(sys.modules, "arena.universal.trainer", None)
    monkeypatch.setattr(torch.optim.Adam, "step", lambda *a, **k: pytest.fail("no optimizer may be used"))
    output = tmp_path/"SYNTHETIC_OFFLINE_TRACE"
    status = ct.export_causal_trace(evaluation, checkpoint, output,
        expected_evaluation_manifest_sha256=eval_sha, expected_checkpoint_manifest_sha256=checkpoint_sha,
        source_revision="a"*40, protocol_sha256="b"*64)
    assert status["status"] == "VERIFIED" and status["verified_steps"] == 15
    assert len(status["games"]) == 5
    assert status["new_environment_interactions"] == status["new_episodes"] == 0
    assert status["training_started"] is False
    from arena.shared.integrity import verify_checksum_manifest
    assert verify_checksum_manifest(output)
    assert hashlib.sha256((evaluation/"SHA256SUMS").read_bytes()).hexdigest() == eval_sha
    assert hashlib.sha256((checkpoint/"SHA256SUMS").read_bytes()).hexdigest() == checkpoint_sha
    for game in p.GAMES:
        rows = [json.loads(line) for line in (output/f"causal-{game}.jsonl").read_text().splitlines()]
        assert len(rows) == 3 and np.asarray(rows[0]["pred_next_hidden"]).shape == (10, 64)
    with pytest.raises(FileExistsError):
        ct.export_causal_trace(evaluation, checkpoint, output,
            expected_evaluation_manifest_sha256=eval_sha, expected_checkpoint_manifest_sha256=checkpoint_sha,
            source_revision="a"*40, protocol_sha256="b"*64)


def test_tampered_authenticated_input_stops_export_before_safe_loading(authenticated_files, tmp_path, monkeypatch):
    from arena.universal import campaign
    evaluation, checkpoint, eval_sha, checkpoint_sha = authenticated_files
    (evaluation/"demonstrations"/"snake"/"frames.npz").write_bytes(b"TAMPERED SYNTHETIC FIXTURE")
    monkeypatch.setattr(campaign._RealBackend, "load_policy", lambda *a, **k: pytest.fail("tampered input must fail before checkpoint loading"))
    output = tmp_path/"STOPPED_OFFLINE_TRACE"
    with pytest.raises(ct.CausalTraceMismatch, match="SHA-256 mismatch"):
        ct.export_causal_trace(evaluation, checkpoint, output,
            expected_evaluation_manifest_sha256=eval_sha, expected_checkpoint_manifest_sha256=checkpoint_sha,
            source_revision="a"*40, protocol_sha256="b"*64)
    status = json.loads((output/"CAUSAL_TRACE_STATUS.json").read_text())
    assert status["status"] == "STOPPED" and status["verified_steps"] == 0
