"""Reusable multigame specification, read-only demo and on-disk episode memory.

The recorded trajectories are data, NOT neural weights or cross-game learning.
Historical Snake evaluators/checkpoints are untouched.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from arena.envs.pong import PongEnv


@dataclass(frozen=True)
class EnvironmentSpec:
    environment_id: str
    observation_shape: tuple[int, ...]
    action_count: int
    reward_version: str
    max_steps: int

    def __post_init__(self):
        if (not self.environment_id or not self.reward_version or
                not self.observation_shape or
                any(type(x) is not int or x < 1 for x in self.observation_shape) or
                type(self.action_count) is not int or self.action_count < 1 or
                type(self.max_steps) is not int or self.max_steps < 1):
            raise ValueError("invalid environment specification")


def describe_environment(env) -> EnvironmentSpec:
    name = getattr(env, "environment_id", None)
    version = getattr(env, "reward_version", None)
    if name is None and type(env).__name__ == "SnakeEnv":
        name, version = "snake-v1", "snake-food-step-v1"
    if not name or not version:
        raise ValueError("unregistered environment; provide a versioned contract")
    if not hasattr(env.action_space, "n") or env.observation_space.shape is None:
        raise ValueError("only fixed-vector observations and Discrete actions supported")
    return EnvironmentSpec(
        environment_id=name,
        observation_shape=tuple(int(x) for x in env.observation_space.shape),
        action_count=int(env.action_space.n),
        reward_version=version,
        max_steps=int(env.max_steps),
    )


def check_checkpoint_contract(metadata: dict[str, Any], spec: EnvironmentSpec) -> None:
    """Fail closed before loading weights from a different game/schema."""
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be an object")
    expected = asdict(spec)
    for field in ("environment_id", "observation_shape", "action_count", "reward_version"):
        actual = metadata.get(field)
        if field == "observation_shape" and isinstance(actual, list):
            actual = tuple(actual)
        if actual != expected[field]:
            raise ValueError(f"incompatible checkpoint {field}: expected {expected[field]!r}")
    # A checkpoint may have a different episode horizon, but not a different
    # observation, action space, reward definition or environment identity.


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_demo(output: str | Path, *, seed: int = 123,
               max_steps: int = 100, target_score: int = 3) -> dict:
    """Small random-policy demonstration. Creates an exclusive directory."""
    if type(seed) is not int or type(max_steps) is not int or not 1 <= max_steps <= 500:
        raise ValueError("seed must be int; max_steps must be 1..500")
    output = Path(output)
    # Reserve BEFORE evaluating; retries cannot overwrite data.
    output.mkdir(parents=True, exist_ok=False)
    trajectory = []
    with PongEnv(max_steps=max_steps, target_score=target_score) as env:
        spec = describe_environment(env)
        obs, info = env.reset(seed=seed)
        rng = random.Random(seed + 77831)
        while True:
            action = rng.randrange(spec.action_count)
            next_obs, reward, terminated, truncated, info = env.step(action)
            trajectory.append({
                "step": info["steps"], "observation": obs.tolist(),
                "action": action, "reward": reward,
                "next_observation": next_obs.tolist(),
                "terminated": terminated, "truncated": truncated,
                "left_score": info["left_score"], "right_score": info["right_score"],
            })
            obs = next_obs
            if terminated or truncated:
                break
        metadata = {
            "kind": "RANDOM_POLICY_TECHNICAL_DEMO",
            "no_training": True,
            "neural_memory": False,
            "environment": asdict(spec),
            "seed": seed,
            "policy": "random.Random(seed + 77831).randrange(3)",
            "steps": len(trajectory),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "final_info": info,
        }
    with (output / "trajectory.jsonl").open("x", encoding="utf-8") as f:
        for event in trajectory:
            f.write(json.dumps(event, sort_keys=True) + "\n")
    with (output / "report.json").open("x", encoding="utf-8") as f:
        json.dump(metadata, f, sort_keys=True, indent=2)
        f.write("\n")
    with (output / "summary.csv").open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "environment_id", "seed", "steps", "left_score", "right_score",
            "terminated", "truncated", "no_training",
        ])
        writer.writeheader()
        writer.writerow({
            "environment_id": spec.environment_id,
            "seed": seed, "steps": len(trajectory),
            "left_score": info["left_score"], "right_score": info["right_score"],
            "terminated": terminated, "truncated": truncated, "no_training": True,
        })
    names = ("trajectory.jsonl", "report.json", "summary.csv")
    with (output / "SHA256SUMS").open("x", encoding="utf-8") as f:
        for name in names:
            f.write(f"{_sha256(output / name)}  {name}\n")
    return metadata


def verify_demo(path: str | Path) -> dict:
    root = Path(path)
    expected_names = ("trajectory.jsonl", "report.json", "summary.csv")
    manifest = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    if len(manifest) != len(expected_names):
        raise ValueError("unexpected manifest length")
    for line, expected_name in zip(manifest, expected_names):
        parts = line.split()
        if len(parts) != 2 or parts[1] != expected_name or parts[0] != _sha256(root / expected_name):
            raise ValueError("demo SHA-256 mismatch")
    metadata = json.loads((root / "report.json").read_text(encoding="utf-8"))
    if metadata.get("kind") != "RANDOM_POLICY_TECHNICAL_DEMO" or metadata.get("no_training") is not True:
        raise ValueError("demo identity invalid")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Pong technical demonstration, never trains")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(write_demo(args.output, seed=args.seed,
                                max_steps=args.max_steps), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
