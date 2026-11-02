"""Phase 4: small synthetic/no-training checks. NEVER run the 4,200 official episodes."""
from dataclasses import asdict
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from arena.agents.dqn import DQNConfig
from arena.envs.snake import DIRECTIONS, SnakeEnv
from arena.evaluation.frozen_phase2 import (
    APPROVED_COMMIT, HASHED_FILES, TRAIN_SEEDS, file_sha256,
    untrained_control, verify_artifact_directory,
)
from arena.evaluation import phase4_startstates as p4


def fake_artifact(root: Path, seed: int) -> str:
    folder = root / f"seed_{seed}"
    folder.mkdir(parents=True)
    conf = asdict(DQNConfig())
    model = untrained_control(conf, seed=seed)
    torch.save({
        "config": conf, "online": model.state_dict(),
        "target": model.state_dict(), "steps": 0, "updates": 0,
    }, folder / "model.pt")
    (folder / "run.json").write_text(json.dumps({
        "seed": seed, "episodes": 80, "eval_episodes": 20, "width": 10,
        "height": 10, "max_steps": 200, "config": conf,
        "training_seeds": [seed, seed + 79],
        "evaluation_seeds": list(range(seed + 100000, seed + 100020)),
        "versions": {"torch": "2.10.0+cpu", "gymnasium": "1.4.0"},
    }), encoding="utf-8")
    (folder / "provenance.json").write_text(json.dumps({
        "seed": seed, "approved_commit": APPROVED_COMMIT,
        "github_run_id": "41350959549", "github_run_attempt": "1",
        "pull_request_number": 1,
    }), encoding="utf-8")
    (folder / "commit.txt").write_text(APPROVED_COMMIT + "\n", encoding="utf-8")
    (folder / "pip-freeze.txt").write_text(
        "torch==2.10.0+cpu\ngymnasium==1.4.0\n", encoding="utf-8")
    (folder / "episodes.csv").write_text("episode,seed,score\n", encoding="utf-8")
    (folder / "evaluation.json").write_text("{}\n", encoding="utf-8")
    (folder / "SHA256SUMS").write_text(
        "".join(f"{file_sha256(folder / name)}  {name}\n"
                for name in sorted(HASHED_FILES)), encoding="utf-8")
    return file_sha256(folder / "model.pt")


@pytest.fixture
def synthetic(tmp_path):
    root = tmp_path / "synthetic"
    hashes = {seed: fake_artifact(root, seed) for seed in TRAIN_SEEDS}
    return root, hashes


def test_full_matrix_enumeration_only_no_episodes(monkeypatch):
    monkeypatch.setattr(p4, "evaluate_episode",
                        lambda *args, **kwargs: pytest.fail("no evaluation allowed"))
    expected = p4.expected_scenarios()
    assert len(expected) == 4200
    assert len(p4.START_STATES) == 12
    assert p4.HEAD_POSITIONS == ((5, 5), (3, 3), (6, 6))
    assert p4.DIRECTION_NAMES == ("east", "south", "west", "north")
    assert p4.ENVIRONMENT_SEEDS == tuple(range(400001, 400051))
    assert p4.MAX_STEPS == 200 and p4.BOARD_SIZE == 10
    assert p4.POLICY_IDS == (
        "trained-11", "untrained-11", "trained-42", "untrained-42",
        "trained-73", "untrained-73", "random",
    )
    assert all(sum((x, y, d, seed, pid) in expected for seed in p4.ENVIRONMENT_SEEDS)
               == 50 for (x, y), d in p4.START_STATES for pid in p4.POLICY_IDS)


@pytest.mark.parametrize("head,direction", p4.START_STATES)
def test_all_12_initial_bodies_food_and_repeatability(head, direction):
    ix = p4.DIRECTION_NAMES.index(direction)
    dx, dy = DIRECTIONS[ix]
    body = [(head[0] - n * dx, head[1] - n * dy) for n in range(3)]
    assert len(set(body)) == 3
    assert all(0 <= x < 10 and 0 <= y < 10 for x, y in body)
    snapshots = []
    for seed in (400001, 400002):
        with p4.StartStateSnakeEnv(initial_head=head, initial_direction=direction) as env:
            obs, info = env.reset(seed=seed)
            assert env.direction == ix
            assert list(env.snake) == body
            assert env.score == env.steps == 0
            assert env.food not in env.snake
            assert 0 <= env.food[0] < 10 and 0 <= env.food[1] < 10
            assert info["initial_food"] == env.food
            assert info["initial_head"] == head
            assert info["initial_direction"] == direction
            assert env.observation_space.contains(obs)
            assert len(obs) == 11
            snapshots.append((env.food, obs.copy()))
        with p4.StartStateSnakeEnv(initial_head=head, initial_direction=direction) as env:
            second_obs, second_info = env.reset(seed=seed)
            assert second_info["initial_food"] == snapshots[-1][0]
            np.testing.assert_array_equal(second_obs, snapshots[-1][1])


def test_default_initialization_reproduces_canonical_snake_exactly():
    for seed in (400001, 400002, 400050):
        with SnakeEnv(width=10, height=10, max_steps=200) as canonical:
            original, original_info = canonical.reset(seed=seed)
            with p4.StartStateSnakeEnv(initial_head=(5, 5),
                                       initial_direction="east") as adapted:
                current, current_info = adapted.reset(seed=seed)
                assert list(adapted.snake) == list(canonical.snake)
                assert adapted.food == canonical.food
                assert adapted.direction == canonical.direction
                np.testing.assert_array_equal(original, current)
                assert canonical.score == adapted.score
                # Same seed and same action must produce the same next state.
                for action in (0, 1, 2, 0):
                    c_next = canonical.step(action)
                    a_next = adapted.step(action)
                    np.testing.assert_array_equal(c_next[0], a_next[0])
                    assert c_next[1:] == a_next[1:]
                    if c_next[2] or c_next[3]:
                        break


@pytest.mark.parametrize("head,direction", [
    ((0, 0), "east"), ((5, 5), "northeast"),
    ((5, 5), "EAST"), ((5, 5), "northwest"),
    ((5, 4), "east"), ((5, 5), ""),
])
def test_non_protocol_initial_states_are_rejected(head, direction):
    with pytest.raises(ValueError, match="not in the approved"):
        p4.StartStateSnakeEnv(initial_head=head, initial_direction=direction)


def test_reset_options_cannot_force_food_or_leak_across_episodes():
    with p4.StartStateSnakeEnv(initial_head=(3, 3), initial_direction="north") as env:
        with pytest.raises(ValueError, match="does not permit"):
            env.reset(seed=400001, options={"food": (4, 4)})
        _, info = env.reset(seed=400001)
        assert info["initial_food"] not in env.snake
        for _ in range(2):
            env.step(1)
        _, repeated = env.reset(seed=400001)
        assert repeated["initial_food"] == info["initial_food"]
        assert env.score == 0 and env.steps == 0


@pytest.mark.parametrize("states,seeds", [
    ((), (400001,)),
    (((5, 5), "east"), (400001,)),  # wrong structure: a single state tuple, not a sequence
    ((((5, 5), "east"), ((5, 5), "east")), (400001,)),
    ((((1, 1), "east"),), (400001,)),
    ((((5, 5), "east"),), ()),
    ((((5, 5), "east"),), (400001, 400001)),
    ((((5, 5), "east"),), (300001,)),
])
def test_unknown_or_duplicate_conditions_and_seeds_rejected(states, seeds):
    with pytest.raises((ValueError, TypeError)):
        p4.expected_scenarios(start_states=states, environmental_seeds=seeds)


def test_first_food_efficiency_and_initial_food_are_recorded(monkeypatch):
    class FakeEnv:
        def __init__(self, *, initial_head, initial_direction):
            assert initial_head == (6, 6) and initial_direction == "west"
            self.counter = 0
            self.food = (2, 2)
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def reset(self, *, seed):
            assert seed == 400001
            return np.zeros(11, dtype=np.float32), {"initial_food": (8, 8)}
        def step(self, action):
            assert action == 0
            self.counter += 1
            return (np.zeros(11, dtype=np.float32),
                    1.0 if self.counter == 2 else -0.01,
                    False, self.counter == 3,
                    {"score": int(self.counter >= 2), "steps": self.counter})
    monkeypatch.setattr(p4, "StartStateSnakeEnv", FakeEnv)
    row = p4.evaluate_episode(lambda o: 0, initial_head=(6, 6),
                              initial_direction="west", environment_seed=400001)
    assert row["first_food_step"] == 2
    assert row["initial_food_x"] == row["initial_food_y"] == 8
    assert row["score"] == 1 and row["steps"] == 3
    assert row["reward"] == 0.98
    assert row["food_per_100_steps"] == pytest.approx(33.333333, abs=1e-6)
    assert row["terminal_reason"] == "step_limit"
    assert row["truncated"] and not row["terminated"]


def test_synthetic_fixture_rejected_by_official_digest(synthetic):
    root, hashes = synthetic
    for seed in TRAIN_SEEDS:
        with pytest.raises(ValueError, match="differs from approved"):
            verify_artifact_directory(root / f"seed_{seed}", seed=seed)
        verified = verify_artifact_directory(root / f"seed_{seed}", seed=seed,
                                             expected_model_hash=hashes[seed])
        assert verified["source_commit"] == APPROVED_COMMIT


def test_tamper_fails_before_creating_results(synthetic, tmp_path):
    root, hashes = synthetic
    with (root / "seed_11" / "model.pt").open("ab") as f:
        f.write(b"tamper")
    out = tmp_path / "bad"
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        p4.evaluate_startstates(root, out,
                                start_states=(((5, 5), "east"),),
                                environmental_seeds=(400001,),
                                _model_hash_overrides=hashes)
    assert not out.exists()


def test_tiny_factorial_is_deterministic_read_only_and_not_mixed(synthetic, tmp_path, monkeypatch):
    root, hashes = synthetic
    before = {seed: file_sha256(root / f"seed_{seed}" / "model.pt") for seed in TRAIN_SEEDS}
    def forbidden_step(*_args, **_kwargs):
        raise AssertionError("optimizer.step must never execute")
    monkeypatch.setattr(torch.optim.Optimizer, "step", forbidden_step)
    conditions = (((5, 5), "east"), ((3, 3), "north"))
    outs = (tmp_path / "a", tmp_path / "b")
    for output in outs:
        result = p4.evaluate_startstates(
            root, output, start_states=conditions,
            environmental_seeds=(400001,),
            _model_hash_overrides=hashes,
        )
        assert result["episodes"] == 14 and result["conditions"] == 2
        assert result["frozen_weights_verified"]
    assert (outs[0] / "episodes.csv").read_bytes() == (outs[1] / "episodes.csv").read_bytes()
    assert (outs[0] / "summary.json").read_bytes() == (outs[1] / "summary.json").read_bytes()
    with (outs[0] / "episodes.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    seen = {(int(r["head_x"]), int(r["head_y"]), r["initial_direction"],
             int(r["environment_seed"]), r["policy_id"]) for r in rows}
    assert len(seen) == 14
    assert seen == p4.expected_scenarios(start_states=conditions,
                                          environmental_seeds=(400001,))
    assert all(int(r["board_width"]) == 10 and int(r["board_height"]) == 10
               and int(r["max_steps"]) == 200 for r in rows)
    assert all(1 <= int(r["steps"]) <= 200 and r["terminal_reason"] in
               ("collision", "step_limit", "board_filled") for r in rows)
    assert all(float(r["food_per_100_steps"]) == pytest.approx(
               100 * int(r["score"]) / int(r["steps"]), abs=1e-6) for r in rows)
    # Food must match within a condition/seed across all seven policies.
    for condition in conditions:
        food = {(r["initial_food_x"], r["initial_food_y"]) for r in rows
                if (int(r["head_x"]), int(r["head_y"]), r["initial_direction"]) ==
                   (*condition[0], condition[1])}
        assert len(food) == 1
    summary = json.loads((outs[0] / "summary.json").read_text())
    assert set(summary) == {"5,5|east", "3,3|north"}
    assert all(set(section) == set(p4.POLICY_IDS) for section in summary.values())
    conf = json.loads((outs[0] / "config.json").read_text())
    assert conf["total_episodes"] == 14
    assert conf["no_training"] and conf["frozen_weights_verified"]
    assert conf["environmental_seeds"] == [400001]
    for seed in TRAIN_SEEDS:
        assert file_sha256(root / f"seed_{seed}" / "model.pt") == before[seed]
    for line in (outs[0] / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        assert file_sha256(outs[0] / name) == digest


def test_cli_requires_approval_and_has_no_seed_or_position_flags(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "phase4", "--artifacts", str(tmp_path), "--out", str(tmp_path / "out"),
    ])
    with pytest.raises(SystemExit) as exc:
        p4.main()
    assert exc.value.code == 2
    assert not (tmp_path / "out").exists()
    monkeypatch.setattr(sys, "argv", [
        "phase4", "--artifacts", str(tmp_path), "--out", str(tmp_path / "out"),
        "--seed", "400001", "--authorized",
    ])
    with pytest.raises(SystemExit) as exc:
        p4.main()
    assert exc.value.code == 2
