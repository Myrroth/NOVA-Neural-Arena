"""Multigame storage, identity and readback checks without training."""
from dataclasses import asdict
import json
import pytest

from arena.envs.pong import PongEnv
from arena.envs.snake import SnakeEnv
from arena.shared.multigame import (
    check_checkpoint_contract, describe_environment, verify_demo, write_demo,
)


def test_pong_and_snake_get_distinct_versioned_contracts():
    with PongEnv(max_steps=30) as pong, SnakeEnv(max_steps=30) as snake:
        p = describe_environment(pong)
        s = describe_environment(snake)
        assert p.environment_id == "pong-local-v2" and p.observation_shape == (8,)
        assert s.environment_id == "snake-v1" and s.observation_shape == (11,)
        assert p.reward_version != s.reward_version
        assert p.action_count == s.action_count == 3


def test_checkpoint_contract_fails_closed_for_cross_game_or_reward_change():
    with PongEnv() as pong, SnakeEnv() as snake:
        p, s = describe_environment(pong), describe_environment(snake)
        check_checkpoint_contract(asdict(p), p)
        old_pong = {**asdict(p), "environment_id": "pong-local-v1"}
        with pytest.raises(ValueError, match="environment_id"):
            check_checkpoint_contract(old_pong, p)
        with pytest.raises(ValueError, match="environment_id"):
            check_checkpoint_contract(asdict(s), p)
        wrong = asdict(p)
        wrong["reward_version"] = "altered"
        with pytest.raises(ValueError, match="reward_version"):
            check_checkpoint_contract(wrong, p)
        wrong = asdict(p)
        wrong["observation_shape"] = [11]
        with pytest.raises(ValueError, match="observation_shape"):
            check_checkpoint_contract(wrong, p)
        with pytest.raises(ValueError):
            check_checkpoint_contract({}, p)


def test_random_demo_exclusive_recording_and_return_verification(tmp_path):
    path = tmp_path / "pong-demo"
    meta = write_demo(path, seed=7, max_steps=25)
    assert meta["kind"] == "RANDOM_POLICY_TECHNICAL_DEMO"
    assert meta["no_training"] and not meta["neural_memory"]
    assert 1 <= meta["steps"] <= 25
    assert verify_demo(path) == meta
    lines = (path / "trajectory.jsonl").read_text().splitlines()
    assert len(lines) == meta["steps"]
    for index, line in enumerate(lines, start=1):
        row = json.loads(line)
        assert row["step"] == index
        assert len(row["observation"]) == len(row["next_observation"]) == 8
        assert row["action"] in (0, 1, 2)
    with pytest.raises(FileExistsError):
        write_demo(path, seed=7, max_steps=25)


def test_random_demo_reproducible_and_tamper_rejected(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_demo(a, seed=42, max_steps=20)
    write_demo(b, seed=42, max_steps=20)
    assert (a / "trajectory.jsonl").read_bytes() == (b / "trajectory.jsonl").read_bytes()
    assert (a / "report.json").read_bytes() == (b / "report.json").read_bytes()
    with (b / "trajectory.jsonl").open("a") as f:
        f.write('{"injected":true}\n')
    with pytest.raises(ValueError, match="SHA-256"):
        verify_demo(b)


def test_invalid_demo_budget_and_missing_metadata(tmp_path):
    for steps in (0, 501):
        with pytest.raises(ValueError):
            write_demo(tmp_path / ("bad" + str(steps)), max_steps=steps)
    with PongEnv() as env:
        spec = describe_environment(env)
        bad = asdict(spec)
        bad.pop("environment_id")
        with pytest.raises(ValueError):
            check_checkpoint_contract(bad, spec)
