"""Unapproved Phase 5 infrastructure. The CLI only validates; no campaign runs.

Historical environment, rewards and evaluators are imported unchanged. A future
review must add an immutable evaluator pin and Chat 1 approval before execution.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import torch

from arena.envs.snake import DIRECTIONS, SnakeEnv
from arena.evaluation.frozen_phase2 import (
    APPROVED_COMMIT, ARTIFACTS, TRAIN_SEEDS, greedy_action, untrained_control,
    verify_artifact_directory, weights_fingerprint,
)
from arena.evaluation.phase4_startstates import DIRECTION_NAMES, POLICY_IDS

STATUS = "UNAPPROVED"
HEADS = ((5, 5), (0, 5), (9, 5), (5, 0), (5, 9),
         (0, 0), (0, 9), (9, 0), (9, 9))
PHYSICAL_FOOD = ((4, 4), (6, 6))
RELATIVE_FOOD = ((1, 0), (0, 1), (0, -1))  # coordinates: (forward distance, right distance)


def _cell(value: tuple[int, int], label: str) -> tuple[int, int]:
    if (not isinstance(value, tuple) or len(value) != 2
            or any(type(v) is not int for v in value)):
        raise ValueError(f"{label} must be a pair of integers")
    return value


@dataclass(frozen=True)
class ExperimentConfig:
    """The candidate design, not an authorization record."""
    width: int = 10
    height: int = 10
    max_steps: int = 200
    environmental_seeds: tuple[int, ...] = tuple(range(500001, 500051))
    protocol: str = "phase5-hardstates-candidate-v1"

    def __post_init__(self):
        if (any(type(v) is not int for v in (self.width, self.height, self.max_steps))
                or self.width != 10 or self.height != 10
                or not 1 <= self.max_steps <= 200):
            raise ValueError("candidate protocol requires 10x10 and 1..200 steps")
        if (not isinstance(self.environmental_seeds, tuple)
                or not self.environmental_seeds
                or any(type(s) is not int or not 500001 <= s <= 500050
                       for s in self.environmental_seeds)
                or len(set(self.environmental_seeds)) != len(self.environmental_seeds)):
            raise ValueError("environmental seeds must be unique integers in 500001..500050")
        if self.protocol != "phase5-hardstates-candidate-v1":
            raise ValueError("unknown protocol")


@dataclass(frozen=True)
class Scenario:
    id: str
    head: tuple[int, int]
    direction: str
    food: tuple[int, int]
    food_mode: str
    food_key: str
    region: str

    def body(self, config: ExperimentConfig) -> tuple[tuple[int, int], ...]:
        _cell(self.head, "head")
        if self.direction not in DIRECTION_NAMES:
            raise ValueError("unknown direction")
        dx, dy = DIRECTIONS[DIRECTION_NAMES.index(self.direction)]
        body = tuple((self.head[0] - n * dx, self.head[1] - n * dy) for n in range(3))
        if len(set(body)) != 3 or any(
            not (0 <= x < config.width and 0 <= y < config.height) for x, y in body
        ):
            raise ValueError("initial three-segment body is out of bounds or overlapping")
        return body

    def validate(self, config: ExperimentConfig) -> None:
        body = self.body(config)
        fx, fy = _cell(self.food, "food")
        if not (0 <= fx < config.width and 0 <= fy < config.height) or self.food in body:
            raise ValueError("initial food must be a free in-bounds cell")
        if self.food_mode not in ("physical", "relative"):
            raise ValueError("unknown food equivalence mode")
        if self.region not in ("center", "wall", "corner") or not self.id or not self.food_key:
            raise ValueError("invalid scenario metadata")
        region = ("center", "wall", "corner")[
            int(self.head[0] in (0, config.width - 1))
            + int(self.head[1] in (0, config.height - 1))]
        if self.region != region:
            raise ValueError("region metadata differs from geometry")
        try:
            a, b = (int(v) for v in self.food_key.split(","))
        except (ValueError, AttributeError):
            raise ValueError("food_key must contain two integer coordinates") from None
        dx, dy = DIRECTIONS[DIRECTION_NAMES.index(self.direction)]
        expected_food = (a, b) if self.food_mode == "physical" else (
            self.head[0] + a * dx - b * dy, self.head[1] + a * dy + b * dx)
        if expected_food != self.food:
            raise ValueError("food equivalence metadata differs from geometry")


def candidate_scenarios(config: ExperimentConfig = ExperimentConfig()) -> tuple[Scenario, ...]:
    """Generate 92 geometrically valid scenarios without resetting or playing."""
    scenarios = []
    for head in HEADS:
        boundary_axes = int(head[0] in (0, 9)) + int(head[1] in (0, 9))
        region = ("center", "wall", "corner")[boundary_axes]
        for direction, (dx, dy) in zip(DIRECTION_NAMES, DIRECTIONS):
            stem = f"{head[0]},{head[1]}-{direction}"
            probe = Scenario(stem, head, direction, (4, 4), "physical", "4,4", region)
            try:
                probe.body(config)
            except ValueError:
                continue  # invalid body is excluded, never silently repaired
            for mode, foods in (("physical", PHYSICAL_FOOD), ("relative", RELATIVE_FOOD)):
                for a, b in foods:
                    food = (a, b) if mode == "physical" else (
                        head[0] + a * dx - b * dy, head[1] + a * dy + b * dx)
                    key = f"{a},{b}"
                    scenario = Scenario(f"{stem}-{mode}-{key}", head, direction,
                                        food, mode, key, region)
                    try:
                        scenario.validate(config)
                    except ValueError:
                        continue
                    scenarios.append(scenario)
    if len({s.id for s in scenarios}) != len(scenarios):
        raise RuntimeError("duplicate scenario ids")
    return tuple(scenarios)


def protocol_manifest(config: ExperimentConfig = ExperimentConfig()) -> dict:
    scenarios = candidate_scenarios(config)
    payload = {
        "status": STATUS, "config": asdict(config),
        "scenarios": [asdict(s) for s in scenarios],
        "policies": list(POLICY_IDS), "training_seeds": list(TRAIN_SEEDS),
        "official_model_sha256": {str(s): ARTIFACTS[s][1] for s in TRAIN_SEEDS},
        "official_training_commit": APPROVED_COMMIT,
        "total_episodes": len(scenarios) * len(config.environmental_seeds) * len(POLICY_IDS),
        "initial_rng_draws": 0,
        "subsequent_food": "canonical row-major free-cell sampling; trajectories may diverge",
        "no_training": True,
    }
    payload["protocol_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


class HardStateSnakeEnv(SnakeEnv):
    """Only initialization differs. Forced first food consumes ZERO random draws."""

    def __init__(self, scenario: Scenario, config: ExperimentConfig = ExperimentConfig()):
        scenario.validate(config)
        super().__init__(width=config.width, height=config.height, max_steps=config.max_steps)
        self.scenario, self.config = scenario, config

    def reset(self, *, seed=None, options=None):
        if options is not None:
            raise ValueError("scenario owns initial food; reset options are not allowed")
        gym.Env.reset(self, seed=seed)
        self.snake = deque(self.scenario.body(self.config))
        self.direction = DIRECTION_NAMES.index(self.scenario.direction)
        self.score, self.steps, self.finished = 0, 0, False
        self.food = self.scenario.food
        return self._observation(), {**self._info(), "initial_food": self.food}


def full_state(env: SnakeEnv) -> tuple:
    """A repeated 11D observation alone is NOT a repeated game state."""
    return tuple(env.snake), env.direction, env.food


class EpisodeMetrics:
    def __init__(self, env: SnakeEnv):
        self.seen = {full_state(env): 0}
        self.previous_score = 0
        self.last_food_step = 0
        self.first_food_step = None
        self.first_repeat_step = None
        self.first_repeat_period = None
        self.repeated_state_visits = 0
        self.max_no_food_steps = 0
        self.reward = 0.0

    def observe(self, env: SnakeEnv, reward: float, *, terminated: bool):
        self.reward += reward
        # The eating step belongs to the completed interval until that food.
        self.max_no_food_steps = max(self.max_no_food_steps, env.steps - self.last_food_step)
        if env.score > self.previous_score:
            if self.first_food_step is None:
                self.first_food_step = env.steps
            self.last_food_step = env.steps
        self.previous_score = env.score
        # Collision leaves the body unchanged; do not count that as a loop.
        if not terminated:
            key = full_state(env)
            if key in self.seen:
                self.repeated_state_visits += 1
                if self.first_repeat_step is None:
                    self.first_repeat_step = env.steps
                    self.first_repeat_period = env.steps - self.seen[key]
            self.seen[key] = env.steps

    def result(self, env: SnakeEnv, terminated: bool, truncated: bool) -> dict:
        return {
            "score": env.score, "reward": round(self.reward, 6), "steps": env.steps,
            "terminated": terminated, "truncated": truncated,
            "terminal_reason": ("board_filled" if terminated and env.food is None
                                else "collision" if terminated else "step_limit"),
            "first_food_step": self.first_food_step,
            "first_food_censored": self.first_food_step is None,
            "first_food_observation_steps": self.first_food_step or env.steps,
            "food_per_100_steps": round(100 * env.score / env.steps, 6),
            "repeated_state_visits": self.repeated_state_visits,
            "first_repeat_step": self.first_repeat_step,
            "first_repeat_period": self.first_repeat_period,
            "max_steps_between_food_events": self.max_no_food_steps,
        }


def _evaluate_episode(policy: Callable, scenario: Scenario, seed: int,
                      config: ExperimentConfig) -> dict:
    """Internal engine. This revision exposes no official campaign caller."""
    with HardStateSnakeEnv(scenario, config) as env:
        observation, _ = env.reset(seed=seed)
        metrics = EpisodeMetrics(env)
        while True:
            with torch.inference_mode():
                action = policy(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            metrics.observe(env, reward, terminated=terminated)
            if terminated or truncated:
                return metrics.result(env, terminated, truncated)


@dataclass
class FrozenPolicies:
    networks: dict
    metadata: dict
    fingerprints: dict

    def assert_frozen(self):
        for name, network in self.networks.items():
            if (network.training or any(p.requires_grad or p.grad is not None
                                        for p in network.parameters())
                    or weights_fingerprint(network) != self.fingerprints[name]):
                raise RuntimeError(f"frozen policy integrity failure: {name}")


def verify_official_policies(artifact_root: str | Path) -> FrozenPolicies:
    """Prepare all seven policies, checking official models WITHOUT any episode."""
    from arena.shared.checkpoints import load_official_checkpoint
    networks, metadata = {}, {}
    for seed in TRAIN_SEEDS:
        folder = Path(artifact_root) / f"seed_{seed}"
        verified = verify_artifact_directory(folder, seed=seed)
        trained = load_official_checkpoint(folder, seed=seed)
        control = untrained_control(verified["config"], seed)
        for kind, network in (("trained", trained), ("untrained", control)):
            name = f"{kind}-{seed}"
            networks[name] = network
            metadata[name] = {
                "kind": kind, "seed": seed,
                "checkpoint_sha256": verified["model_sha256"] if kind == "trained" else None,
                "artifact_id": ARTIFACTS[seed][0] if kind == "trained" else None,
            }
    metadata["random"] = {"kind": "random", "seed_formula": "90000000 + environment_seed"}
    bundle = FrozenPolicies(networks, metadata,
                            {name: weights_fingerprint(net) for name, net in networks.items()})
    bundle.assert_frozen()
    return bundle


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_score_interval(rows: list[dict], left: tuple[str, str], right: tuple[str, str],
                          *, resamples: int = 10000, seed: int = 510000) -> dict:
    """Percentile 95% paired bootstrap of mean score differences over seeds.

    Pair keys are (scenario_id, policy_id). The caller declares the comparison;
    this function cannot establish physical comparability of different scenarios.
    Intervals are conditional on those checkpoints, not future training runs.
    """
    if type(resamples) is not int or not 100 <= resamples <= 100000:
        raise ValueError("resamples must be an integer in 100..100000")
    if type(seed) is not int or seed < 0:
        raise ValueError("bootstrap seed must be a nonnegative integer")
    pairs = []
    for key in (left, right):
        values = {}
        for row in rows:
            if (row["scenario_id"], row["policy_id"]) != key:
                continue
            env_seed, score = row["environment_seed"], float(row["score"])
            if type(env_seed) is not int or env_seed in values or not math.isfinite(score):
                raise ValueError("duplicate/noninteger seed or nonfinite score")
            values[env_seed] = score
        pairs.append(values)
    if len(pairs[0]) < 2 or set(pairs[0]) != set(pairs[1]):
        raise ValueError("paired intervals require >=2 identical, complete seed sets")
    seeds = sorted(pairs[0])
    differences = np.asarray([pairs[0][s] - pairs[1][s] for s in seeds])
    rng = np.random.default_rng(seed)
    # Chunking keeps memory bounded even for a larger future seed count.
    estimates = []
    for offset in range(0, resamples, 1000):
        indices = rng.integers(len(seeds), size=(min(1000, resamples - offset), len(seeds)))
        estimates.extend(np.mean(differences[indices], axis=1).tolist())
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {
        "n_pairs": len(seeds), "mean_difference": float(np.mean(differences)),
        "lower_95": float(lower), "upper_95": float(upper),
        "method": "paired percentile bootstrap, conditional on frozen weights",
        "resamples": resamples, "bootstrap_seed": seed,
        "pairing_seeds": seeds,
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Adjust a declared family of externally computed valid p-values.

    Does not invent a null test or turn a descriptive bootstrap interval into a
    p-value. The future protocol must predeclare the test and complete family.
    """
    for name, p in p_values.items():
        if (not isinstance(name, str) or not name or type(p) not in (int, float)
                or not math.isfinite(p) or not 0 <= p <= 1):
            raise ValueError("Holm requires named finite p-values in [0,1]")
    ordered = sorted(p_values, key=lambda name: (p_values[name], name))
    result, previous = {}, 0.0
    for index, name in enumerate(ordered):
        previous = max(previous, min(1.0, (len(ordered) - index) * p_values[name]))
        result[name] = previous
    return result


def write_results(output: str | Path, rows: list[dict], provenance: dict) -> dict:
    """Create a new result directory exclusively; never replace an earlier run."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "episodes.csv").open("x", encoding="utf-8", newline="") as handle:
        fields = list(rows[0]) if rows else ["scenario_id", "policy_id", "environment_seed"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    source_root = Path(__file__).parents[2]
    source_paths = (
        "arena/evaluation/phase5_hardstates.py", "arena/envs/snake.py",
        "arena/agents/dqn.py", "arena/evaluation/frozen_phase2.py",
        "arena/evaluation/phase4_startstates.py", "arena/shared/checkpoints.py",
        "arena/shared/integrity.py",
    )
    metadata = {**provenance, "versions": {
        "python": platform.python_version(), "numpy": np.__version__,
        "gymnasium": gym.__version__, "torch": torch.__version__,
    }, "source_files_sha256": {name: _sha(source_root / name) for name in source_paths},
        "torch_execution": {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                            "num_threads": torch.get_num_threads(),
                            "num_interop_threads": torch.get_num_interop_threads()}}
    with (output / "provenance.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for name in ("episodes.csv", "provenance.json"):
            handle.write(f"{_sha(output / name)}  {name}\n")
    return {"output": str(output), "episodes": len(rows), "status": metadata["status"]}


def synthetic_smoke(output: str | Path) -> dict:
    """Exactly 14 tiny synthetic evaluations: NO official weights or artifact path.

    Labels synthetic-a/b are separately constructed same-seed fixtures, never
    scientific trained-model results. Nothing here is a Phase 5 campaign.
    """
    from arena.agents.dqn import DQNConfig
    if Path(output).exists():
        raise FileExistsError(output)
    config = ExperimentConfig(max_steps=12, environmental_seeds=(500001,))
    candidates = candidate_scenarios(config)
    scenarios = (candidates[0], next(s for s in candidates if s.region == "corner"))
    networks = {
        f"synthetic-{kind}-{seed}": untrained_control(asdict(DQNConfig()), seed)
        for seed in TRAIN_SEEDS for kind in ("a", "b")
    }
    before = {name: weights_fingerprint(net) for name, net in networks.items()}
    bundle = FrozenPolicies(networks, {}, before)
    rows = []
    for scenario in scenarios:
        for name in (*networks, "random"):
            rng = random.Random(90_000_000 + 500001)
            policy = (lambda obs: rng.randrange(3)) if name == "random" else (
                lambda obs, network=networks[name]: greedy_action(network, obs))
            result = _evaluate_episode(policy, scenario, 500001, config)
            rows.append({"scenario_id": scenario.id, "policy_id": name,
                         "environment_seed": 500001, "head_x": scenario.head[0],
                         "head_y": scenario.head[1], "direction": scenario.direction,
                         "food_x": scenario.food[0], "food_y": scenario.food[1],
                         "food_mode": scenario.food_mode, **result})
    bundle.assert_frozen()
    return write_results(output, rows, {
        "status": "SYNTHETIC_TECHNICAL_VALIDATION", "no_training": True,
        "official_checkpoints_used": False, "frozen_weights_verified": True,
        "config": asdict(config), "scenarios": [asdict(s) for s in scenarios],
        "weights_before_and_after": before,
        "phase5_protocol_sha256": protocol_manifest()["protocol_sha256"],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true",
                        help="enumerate and validate candidate design; default")
    parser.add_argument("--campaign", action="store_true",
                        help="always blocked in this unapproved revision")
    args = parser.parse_args()
    if args.campaign:
        parser.error("UNAPPROVED: Chat 1 must approve protocol and immutable evaluator SHA; campaign disabled")
    print(json.dumps(protocol_manifest(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
