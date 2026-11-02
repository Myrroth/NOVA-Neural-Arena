"""Phase 3 safety tests: synthetic checkpoints and tiny scenarios ONLY.

No official model runs a game; no 50-seed cross-size campaign is initiated.
"""
from dataclasses import asdict
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from arena.agents.dqn import DQNConfig, QNetwork
from arena.evaluation.frozen_phase2 import (
    APPROVED_COMMIT, HASHED_FILES, TRAIN_SEEDS, file_sha256,
    untrained_control, verify_artifact_directory, weights_fingerprint,
)
from arena.evaluation import phase3_generalization as p3


def create_fixture(root: Path, seed: int) -> str:
    folder = root / f"seed_{seed}"
    folder.mkdir(parents=True)
    config = asdict(DQNConfig())
    net = untrained_control(config, seed)
    torch.save({"config": config, "online": net.state_dict(),
                "target": net.state_dict(), "steps": 0, "updates": 0},
               folder / "model.pt")
    (folder / "run.json").write_text(json.dumps({
        "seed": seed, "episodes": 80, "eval_episodes": 20, "width": 10,
        "height": 10, "max_steps": 200, "config": config,
        "training_seeds": [seed, seed + 79],
        "evaluation_seeds": list(range(seed + 100000, seed + 100020)),
        "versions": {"torch": "2.10.0+cpu", "gymnasium": "1.4.0"}
    }), encoding="utf-8")
    (folder / "provenance.json").write_text(json.dumps({
        "seed": seed, "approved_commit": APPROVED_COMMIT,
        "github_run_id": "41350959549", "github_run_attempt": "1",
        "pull_request_number": 1,
    }), encoding="utf-8")
    (folder / "commit.txt").write_text(APPROVED_COMMIT + "\n", encoding="utf-8")
    (folder / "pip-freeze.txt").write_text(
        "gymnasium==1.4.0\ntorch==2.10.0+cpu\n", encoding="utf-8")
    (folder / "episodes.csv").write_text("episode,score\n", encoding="utf-8")
    (folder / "evaluation.json").write_text("{}\n", encoding="utf-8")
    (folder / "SHA256SUMS").write_text(
        "".join(f"{file_sha256(folder / name)}  {name}\n" for name in sorted(HASHED_FILES)),
        encoding="utf-8")
    return file_sha256(folder / "model.pt")


@pytest.fixture
def fixtures(tmp_path):
    root = tmp_path / "synthetic"
    hashes = {seed: create_fixture(root, seed) for seed in TRAIN_SEEDS}
    return root, hashes


def test_expected_matrix_full_without_running_a_game(monkeypatch):
    monkeypatch.setattr(p3, "evaluate_episode",
                        lambda *args, **kwargs: pytest.fail("No episode allowed"))
    expected = p3.expected_scenarios()
    assert len(expected) == 1050
    assert set(x[0] for x in expected) == {8, 10, 12}
    assert set(x[1] for x in expected) == set(range(300001, 300051))
    assert set(x[2] for x in expected) == set(p3.POLICY_IDS)
    assert p3.MAX_STEPS == 200
    assert p3.POLICY_IDS == (
        "trained-11", "untrained-11", "trained-42", "untrained-42",
        "trained-73", "untrained-73", "random",
    )


@pytest.mark.parametrize("sizes,seeds", [
    ((), (300001,)), ((8, 8), (300001,)),
    ((7,), (300001,)), ((8,), ()),
    ((8,), (300001, 300001)), ((8,), (200001,)),
])
def test_invalid_or_repeated_scenarios_rejected_before_simulation(sizes, seeds):
    with pytest.raises(ValueError):
        p3.expected_scenarios(board_sizes=sizes, environmental_seeds=seeds)


def test_first_food_step_efficiency_and_explicit_truncation(monkeypatch):
    class ThreeStepEnv:
        food = (1, 1)
        def __init__(self, **kwargs):
            assert kwargs["width"] == 8 and kwargs["height"] == 8
            assert kwargs["max_steps"] == 200
            self.count = 0
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def reset(self, *, seed):
            assert seed == 300001
            return np.zeros(11, dtype=np.float32), {}
        def step(self, action):
            assert action == 0
            self.count += 1
            return (
                np.zeros(11, dtype=np.float32),
                1.0 if self.count == 2 else -0.01,
                False, self.count == 3,
                {"score": int(self.count >= 2), "steps": self.count},
            )
    monkeypatch.setattr(p3, "SnakeEnv", ThreeStepEnv)
    row = p3.evaluate_episode(lambda obs: 0, board_size=8, environment_seed=300001)
    assert row["score"] == 1
    assert row["steps"] == 3
    assert row["first_food_step"] == 2
    assert row["food_per_100_steps"] == pytest.approx(33.333333, abs=1e-6)
    assert row["reward"] == 0.98
    assert row["truncated"] is True and row["terminated"] is False
    assert row["terminal_reason"] == "step_limit"


def test_synthetic_checkpoints_integrity_and_untrained_control(fixtures):
    root, hashes = fixtures
    for seed in TRAIN_SEEDS:
        with pytest.raises(ValueError, match="differs from approved"):
            verify_artifact_directory(root / f"seed_{seed}", seed=seed)
        verified = verify_artifact_directory(root / f"seed_{seed}", seed=seed,
                                             expected_model_hash=hashes[seed])
        model = untrained_control(verified["config"], seed)
        same = untrained_control(verified["config"], seed)
        assert weights_fingerprint(model) == weights_fingerprint(same)
        assert model.training is False and all(not p.requires_grad for p in model.parameters())


def test_checkpoint_tampering_cannot_be_evaluated(fixtures, tmp_path):
    root, hashes = fixtures
    with (root / "seed_42" / "model.pt").open("ab") as handle:
        handle.write(b"TAMPER")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        p3.evaluate_generalization(root, tmp_path / "result",
                                   board_sizes=(8,), environmental_seeds=(300001,),
                                   _model_hash_overrides=hashes)
    assert not (tmp_path / "result").exists()


def test_tiny_cross_size_fixture_matrix_repeatable_and_frozen(fixtures, tmp_path, monkeypatch):
    root, hashes = fixtures
    before = {seed: file_sha256(root / f"seed_{seed}" / "model.pt") for seed in TRAIN_SEEDS}

    def never_optimize(*args, **kwargs):
        raise AssertionError("No optimizer step is allowed in Phase 3")
    monkeypatch.setattr(torch.optim.Optimizer, "step", never_optimize)

    dirs = (tmp_path / "a", tmp_path / "b")
    for out in dirs:
        result = p3.evaluate_generalization(
            root, out, board_sizes=(8, 10, 12),
            environmental_seeds=(300001,),
            _model_hash_overrides=hashes,
        )
        assert result["episodes"] == 21
        assert result["frozen_weights_verified"] is True
    assert (dirs[0] / "episodes.csv").read_bytes() == (dirs[1] / "episodes.csv").read_bytes()
    assert (dirs[0] / "summary.json").read_bytes() == (dirs[1] / "summary.json").read_bytes()
    rows = list(csv.DictReader((dirs[0] / "episodes.csv").open(encoding="utf-8")))
    pairs = {(int(row["board_width"]), int(row["environment_seed"]), row["policy_id"])
             for row in rows}
    assert pairs == p3.expected_scenarios(board_sizes=(8, 10, 12),
                                          environmental_seeds=(300001,))
    assert all(int(row["board_width"]) == int(row["board_height"]) for row in rows)
    assert all(1 <= int(row["steps"]) <= 200 for row in rows)
    assert all(float(row["food_per_100_steps"]) ==
               pytest.approx(100 * int(row["score"]) / int(row["steps"]), abs=1e-6)
               for row in rows)
    assert all(row["terminal_reason"] in ("collision", "step_limit", "board_filled") for row in rows)
    config = json.loads((dirs[0] / "config.json").read_text())
    assert config["total_episodes"] == 21
    assert config["board_sizes"] == [8, 10, 12]
    assert config["frozen_weights_verified"] and config["no_training"]
    summary = json.loads((dirs[0] / "summary.json").read_text())
    assert set(summary) == {"8", "10", "12"}
    assert all(set(summary[str(size)]) == set(p3.POLICY_IDS) for size in (8, 10, 12))
    assert all(summary[str(size)][pid]["n"] == 1
               for size in (8, 10, 12) for pid in p3.POLICY_IDS)
    for seed in TRAIN_SEEDS:
        assert before[seed] == file_sha256(root / f"seed_{seed}" / "model.pt")
    checksums = (dirs[0] / "SHA256SUMS").read_text().splitlines()
    assert len(checksums) == 3
    for line in checksums:
        expected, name = line.split()
        assert file_sha256(dirs[0] / name) == expected


def test_cli_requires_new_authorization(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv",
                        ["phase3", "--artifacts", str(tmp_path), "--out", str(tmp_path / "out")])
    with pytest.raises(SystemExit) as result:
        p3.main()
    assert result.value.code == 2
    assert not (tmp_path / "out").exists()
