"""Fast infrastructure tests with synthetic untrained checkpoint fixtures only.

These tests DO NOT evaluate the three official checkpoints or run 50 seeds.
"""
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch

from arena.agents.dqn import DQNConfig, QNetwork
from arena.evaluation.frozen_phase2 import (
    APPROVED_COMMIT, HASHED_FILES, TRAIN_SEEDS, ENVIRONMENT_SEEDS,
    evaluate_episode, evaluate_frozen_models, file_sha256, greedy_action,
    load_verified_model, untrained_control, verify_artifact_directory,
    weights_fingerprint,
)


def make_fake_pilot(root: Path, seed: int) -> str:
    base = root / f"seed_{seed}"
    base.mkdir(parents=True)
    config = asdict(DQNConfig())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        network = QNetwork(DQNConfig(**config))
    torch.save({"config": config, "online": network.state_dict(),
                "target": network.state_dict(), "steps": 10, "updates": 2}, base / "model.pt")
    (base / "run.json").write_text(json.dumps({
        "seed": seed, "episodes": 80, "eval_episodes": 20,
        "width": 10, "height": 10, "max_steps": 200, "config": config,
        "training_seeds": [seed, seed + 79],
        "evaluation_seeds": list(range(seed + 100000, seed + 100020)),
        "versions": {"torch": "2.10.0+cpu", "gymnasium": "1.4.0"}
    }), encoding="utf-8")
    (base / "provenance.json").write_text(json.dumps({
        "approved_commit": APPROVED_COMMIT, "seed": seed,
        "github_run_id": "41350959549", "github_run_attempt": "1",
        "pull_request_number": 1
    }), encoding="utf-8")
    (base / "commit.txt").write_text(APPROVED_COMMIT + "\n", encoding="utf-8")
    (base / "pip-freeze.txt").write_text("torch==2.10.0+cpu\ngymnasium==1.4.0\n", encoding="utf-8")
    (base / "episodes.csv").write_text("episode,seed,score\n", encoding="utf-8")
    (base / "evaluation.json").write_text("{}\n", encoding="utf-8")
    (base / "SHA256SUMS").write_text(
        "".join(f"{file_sha256(base / name)}  {name}\n" for name in sorted(HASHED_FILES)),
        encoding="utf-8")
    return file_sha256(base / "model.pt")


@pytest.fixture
def synthetic_artifacts(tmp_path):
    root = tmp_path / "pilot"
    hashes = {seed: make_fake_pilot(root, seed) for seed in TRAIN_SEEDS}
    return root, hashes


def test_protocol_is_held_out_and_exact():
    assert ENVIRONMENT_SEEDS == tuple(range(200001, 200051))
    assert TRAIN_SEEDS == (11, 42, 73)
    for seed in TRAIN_SEEDS:
        assert not set(ENVIRONMENT_SEEDS) & set(range(seed, seed + 80))
        assert not set(ENVIRONMENT_SEEDS) & set(range(seed + 100000, seed + 100020))


def test_official_hash_required_by_default(synthetic_artifacts):
    root, hashes = synthetic_artifacts
    with pytest.raises(ValueError, match="differs from approved"):
        verify_artifact_directory(root / "seed_11", seed=11)
    verified = verify_artifact_directory(root / "seed_11", seed=11,
                                         expected_model_hash=hashes[11])
    assert verified["seed"] == 11
    assert verified["source_commit"] == APPROVED_COMMIT


def test_integrity_detects_modification_even_if_model_file_unchanged(synthetic_artifacts):
    root, hashes = synthetic_artifacts
    (root / "seed_11" / "run.json").write_text('{"seed": 11}', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_artifact_directory(root / "seed_11", seed=11,
                                  expected_model_hash=hashes[11])


def test_integrity_detects_missing_and_duplicate_checksums(synthetic_artifacts):
    root, hashes = synthetic_artifacts
    base = root / "seed_42"
    lines = (base / "SHA256SUMS").read_text().splitlines()
    (base / "SHA256SUMS").write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="incomplete"):
        verify_artifact_directory(base, seed=42, expected_model_hash=hashes[42])
    (base / "SHA256SUMS").write_text("\n".join(lines + [lines[0]]) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        verify_artifact_directory(base, seed=42, expected_model_hash=hashes[42])


def test_integrity_detects_wrong_provenance_with_updated_manifest(synthetic_artifacts):
    root, hashes = synthetic_artifacts
    base = root / "seed_73"
    prov = json.loads((base / "provenance.json").read_text())
    prov["approved_commit"] = "0" * 40
    (base / "provenance.json").write_text(json.dumps(prov))
    lines = (base / "SHA256SUMS").read_text().splitlines()
    fixed = [f"{file_sha256(base / 'provenance.json')}  provenance.json"
             if line.endswith("provenance.json") else line for line in lines]
    (base / "SHA256SUMS").write_text("\n".join(fixed) + "\n")
    with pytest.raises(ValueError, match="provenance commit"):
        verify_artifact_directory(base, seed=73, expected_model_hash=hashes[73])


def test_verified_checkpoint_and_untrained_same_architecture(synthetic_artifacts):
    root, hashes = synthetic_artifacts
    checked = verify_artifact_directory(root / "seed_11", seed=11,
                                        expected_model_hash=hashes[11])
    trained = load_verified_model(root / "seed_11", seed=11, verified=checked)
    control = untrained_control(checked["config"], seed=11)
    assert trained.training is False and control.training is False
    assert all(not p.requires_grad for p in trained.parameters())
    assert all(not p.requires_grad for p in control.parameters())
    assert weights_fingerprint(trained) == weights_fingerprint(control)
    assert greedy_action(trained, torch.zeros(11).numpy()) in (0, 1, 2)
    assert all(p.grad is None for p in trained.parameters())


def test_episode_records_explicit_terminal_reason():
    # Always turn straight, eventually reaching the wall.
    row = evaluate_episode(lambda obs: 0, environment_seed=200001)
    assert row["terminated"] is True
    assert row["truncated"] is False
    assert row["terminal_reason"] == "collision"
    assert row["steps"] <= 200
    # One-step limit is independent of reward outcome.
    row2 = evaluate_episode(lambda obs: 0, environment_seed=200001, max_steps=1)
    assert row2["truncated"] is True
    assert row2["terminal_reason"] == "step_limit"


def test_integration_two_fixture_seeds_does_not_train(tmp_path, synthetic_artifacts, monkeypatch):
    root, hashes = synthetic_artifacts
    def forbidden_step(*args, **kwargs):
        raise AssertionError("Optimizer.step() is forbidden in evaluation")
    monkeypatch.setattr(torch.optim.Optimizer, "step", forbidden_step)
    before = {seed: file_sha256(root / f"seed_{seed}" / "model.pt") for seed in TRAIN_SEEDS}
    out1, out2 = tmp_path / "eval1", tmp_path / "eval2"
    for out in (out1, out2):
        result = evaluate_frozen_models(root, out, environmental_seeds=(200001, 200002),
                                        _model_hash_overrides=hashes)
        assert result["rows"] == 14
        assert result["frozen_weights_verified"] is True
    assert (out1 / "episodes.csv").read_bytes() == (out2 / "episodes.csv").read_bytes()
    rows = list(csv.DictReader((out1 / "episodes.csv").open()))
    assert set(r["policy_id"] for r in rows) == {
        "trained-11", "trained-42", "trained-73",
        "untrained-11", "untrained-42", "untrained-73", "random",
    }
    for policy_id in set(r["policy_id"] for r in rows):
        assert [int(r["environment_seed"]) for r in rows if r["policy_id"] == policy_id] == [200001, 200002]
    assert all(r["terminated"] in ("True", "False") for r in rows)
    assert all(r["terminal_reason"] in ("collision", "board_filled", "step_limit") for r in rows)
    for seed in TRAIN_SEEDS:
        assert file_sha256(root / f"seed_{seed}" / "model.pt") == before[seed]


def test_rejects_unapproved_seed_range_without_running(synthetic_artifacts, tmp_path):
    root, hashes = synthetic_artifacts
    with pytest.raises(ValueError, match="only approved held-out"):
        evaluate_frozen_models(root, tmp_path / "out", environmental_seeds=(1,),
                               _model_hash_overrides=hashes)
    assert not (tmp_path / "out").exists()
