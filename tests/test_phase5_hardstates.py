"""Small synthetic checks only; no scientific campaign or official-model game."""
import csv
import hashlib
import json
from dataclasses import replace
import sys

import numpy as np
import pytest
import torch

from arena.evaluation import phase5_hardstates as p5


def test_design_is_only_enumerated_and_geometrically_valid(monkeypatch):
    monkeypatch.setattr(p5, "_evaluate_episode", lambda *a, **kw: pytest.fail("no games"))
    manifest = p5.protocol_manifest()
    scenarios = p5.candidate_scenarios()
    assert manifest["status"] == "UNAPPROVED"
    assert manifest["total_episodes"] == 32200
    assert manifest["config"]["environmental_seeds"] == tuple(range(500001, 500051))
    assert len(scenarios) == 92
    spawns = {(s.head, s.direction) for s in scenarios}
    assert len(spawns) == 24
    assert len({s.id for s in scenarios}) == 92
    for s in scenarios:
        s.validate(p5.ExperimentConfig())
        assert len(set(s.body(p5.ExperimentConfig()))) == 3
    assert len([s for s in scenarios if s.food_mode == "physical"]) == 48
    assert len([s for s in scenarios if s.food_mode == "relative"]) == 44
    assert len(manifest["protocol_sha256"]) == 64
    assert p5.protocol_manifest() == manifest


def test_physical_and_relative_controls_are_explicit():
    scenarios = p5.candidate_scenarios()
    for key in ("4,4", "6,6"):
        physical = [s for s in scenarios if s.food_mode == "physical" and s.food_key == key]
        assert len(physical) == 24
        assert len({s.food for s in physical}) == 1
    for s in scenarios:
        if s.food_mode == "relative" and s.region != "center":
            reference = [c for c in scenarios if c.region == "center"
                         and c.direction == s.direction and c.food_mode == "relative"
                         and c.food_key == s.food_key]
            assert len(reference) == 1


@pytest.mark.parametrize("kwargs", [
    {"environmental_seeds": ()}, {"environmental_seeds": (500001, 500001)},
    {"environmental_seeds": (True,)}, {"environmental_seeds": (400001,)},
    {"max_steps": 201}, {"max_steps": False}, {"width": 11},
])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        p5.ExperimentConfig(**kwargs)


@pytest.mark.parametrize("changes", [
    {"head": (0, 0)}, {"head": (5.0, 5)}, {"direction": "EAST"},
    {"food": (5, 5)}, {"food": (-1, 0)}, {"food": (4.1, 4)},
    {"food_key": "0,0"}, {"food_mode": "random"}, {"region": "wall"},
])
def test_invalid_body_food_and_metadata_rejected(changes):
    scenario = p5.candidate_scenarios()[0]
    with pytest.raises(ValueError):
        replace(scenario, **changes).validate(p5.ExperimentConfig())


def test_all_controlled_resets_draw_no_rng_and_repeat_exactly():
    for scenario in p5.candidate_scenarios():
        with p5.HardStateSnakeEnv(scenario) as env:
            observation, info = env.reset(seed=500001)
            expected_rng = np.random.default_rng(500001)
            assert env.np_random.bit_generator.state == expected_rng.bit_generator.state
            assert env.food == scenario.food == info["initial_food"]
            assert env.observation_space.contains(observation)
            available = [(x, y) for y in range(10) for x in range(10)
                         if (x, y) not in env.snake]
            assert env._spawn_food() == available[int(expected_rng.integers(len(available)))]
            observation2, _ = env.reset(seed=500001)
            np.testing.assert_array_equal(observation2, observation)
            with pytest.raises(ValueError, match="options"):
                env.reset(seed=500001, options={"food": (1, 1)})


def test_corner_outward_orientation_is_valid_and_high_risk():
    scenario = next(s for s in p5.candidate_scenarios()
                    if s.head == (0, 0) and s.direction == "west")
    with p5.HardStateSnakeEnv(scenario) as env:
        observation, _ = env.reset(seed=500001)
        assert observation[0] == 1  # immediate forward collision, geometry itself valid
        _, reward, terminated, truncated, info = env.step(0)
        assert reward == -1 and terminated and not truncated and info["steps"] == 1


def test_repeat_detection_uses_body_direction_food_and_does_not_stop_early():
    scenario = p5.Scenario("cycle", (5, 5), "east", (0, 0), "physical", "0,0", "center")
    config = p5.ExperimentConfig(max_steps=12, environmental_seeds=(500001,))
    row = p5._evaluate_episode(lambda _: 1, scenario, 500001, config)
    assert row["steps"] == 12 and row["truncated"]
    assert row["first_repeat_step"] == 5 and row["first_repeat_period"] == 4
    assert row["repeated_state_visits"] == 8
    assert row["first_food_step"] is None and row["first_food_censored"]
    assert row["first_food_observation_steps"] == 12
    assert row["max_steps_between_food_events"] == 12
    with p5.HardStateSnakeEnv(scenario, config) as env:
        env.reset(seed=500001)
        key = p5.full_state(env)
        env.food = (0, 1)
        assert p5.full_state(env) != key
        env.food = (0, 0)
        env.direction = 1
        assert p5.full_state(env) != key
        env.direction = 0
        env.snake[-1] = (3, 4)
        assert p5.full_state(env) != key


def test_collision_is_not_misclassified_as_cycle():
    scenario = next(s for s in p5.candidate_scenarios()
                    if s.head == (0, 0) and s.direction == "west")
    row = p5._evaluate_episode(lambda _: 0, scenario, 500001, p5.ExperimentConfig())
    assert row["terminal_reason"] == "collision"
    assert row["repeated_state_visits"] == 0


def test_first_food_reward_efficiency_and_censoring(monkeypatch):
    monkeypatch.setattr(p5.HardStateSnakeEnv, "_spawn_food", lambda self: (0, 0))
    scenario = p5.Scenario("eat", (5, 5), "east", (6, 5), "relative", "1,0", "center")
    config = p5.ExperimentConfig(max_steps=3, environmental_seeds=(500001,))
    row = p5._evaluate_episode(lambda _: 0, scenario, 500001, config)
    assert row["first_food_step"] == 1 and not row["first_food_censored"]
    assert row["score"] == 1 and row["steps"] == 3
    assert row["reward"] == pytest.approx(0.98)
    assert row["food_per_100_steps"] == pytest.approx(100 / 3, abs=1e-6)
    assert row["max_steps_between_food_events"] == 2


def test_synthetic_smoke_is_small_frozen_reproducible_and_exclusive(tmp_path, monkeypatch):
    monkeypatch.setattr(p5, "verify_official_policies", lambda *_: pytest.fail("no official models"))
    def no_optimizer(*args, **kwargs):
        pytest.fail("no optimizer")
    monkeypatch.setattr(torch.optim.Optimizer, "step", no_optimizer)
    first, second = tmp_path / "a", tmp_path / "b"
    for path in (first, second):
        result = p5.synthetic_smoke(path)
        assert result["episodes"] == 14
        assert result["status"] == "SYNTHETIC_TECHNICAL_VALIDATION"
        provenance = json.loads((path / "provenance.json").read_text())
        assert not provenance["official_checkpoints_used"] and provenance["no_training"]
        with (path / "episodes.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 14 and all(int(r["steps"]) <= 12 for r in rows)
        for line in (path / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split()
            assert hashlib.sha256((path / name).read_bytes()).hexdigest() == digest
    assert (first / "episodes.csv").read_bytes() == (second / "episodes.csv").read_bytes()
    before = (first / "episodes.csv").read_bytes()
    with pytest.raises(FileExistsError):
        p5.synthetic_smoke(first)
    assert (first / "episodes.csv").read_bytes() == before


def test_bootstrap_pairs_by_seed_not_row_order_and_rejects_incomplete_pairs():
    rows = [{"scenario_id": "same", "policy_id": policy, "environment_seed": seed,
             "score": value + (3 if policy == "trained" else 0)}
            for policy in ("trained", "random") for seed, value in ((500002, 4), (500001, 0), (500003, 2))]
    interval = p5.paired_score_interval(rows[::-1], ("same", "trained"), ("same", "random"), resamples=100)
    assert interval["mean_difference"] == interval["lower_95"] == interval["upper_95"] == 3
    assert interval["pairing_seeds"] == [500001, 500002, 500003]
    with pytest.raises(ValueError, match="complete"):
        p5.paired_score_interval(rows[:-1], ("same", "trained"), ("same", "random"))
    with pytest.raises(ValueError, match="duplicate"):
        p5.paired_score_interval(rows + [rows[0]], ("same", "trained"), ("same", "random"))
    rows[0]["score"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        p5.paired_score_interval(rows, ("same", "trained"), ("same", "random"))


def test_holm_adjusts_a_complete_declared_family():
    assert p5.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03}) == {"a": 0.03, "c": 0.06, "b": 0.06}
    assert p5.holm_adjust({}) == {}
    for p in (-0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            p5.holm_adjust({"a": p})


def test_cli_is_validation_only_even_with_campaign_flag(monkeypatch, capsys):
    monkeypatch.setattr(p5, "_evaluate_episode", lambda *a, **kw: pytest.fail("no games"))
    for args in ([], ["--validate-only"]):
        monkeypatch.setattr(sys, "argv", ["phase5", *args])
        p5.main()
        assert json.loads(capsys.readouterr().out)["status"] == "UNAPPROVED"
    for args in (["--campaign"], ["--authorized"], ["--validate-only", "--campaign"]):
        monkeypatch.setattr(sys, "argv", ["phase5", *args])
        with pytest.raises(SystemExit) as exc:
            p5.main()
        assert exc.value.code == 2
