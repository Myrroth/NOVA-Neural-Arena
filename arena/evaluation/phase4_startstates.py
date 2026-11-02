"""Phase 4: frozen evaluation of alternative Snake initial heads and directions.

Never trains a model. Does not modify the original SnakeEnv, DQN, or weights.
Full 4,200-episode campaign requires separate, explicit approval and --authorized.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import deque
from pathlib import Path
from statistics import mean, median, stdev
from typing import Callable

import gymnasium as gym
import numpy as np
import torch

from arena.envs.snake import DIRECTIONS, SnakeEnv
from arena.evaluation.frozen_phase2 import (
    APPROVED_COMMIT, ARTIFACTS, PILOT_RUN_ID, TRAIN_SEEDS, file_sha256,
    greedy_action, load_verified_model, untrained_control,
    verify_artifact_directory, weights_fingerprint,
)

HEAD_POSITIONS = ((5, 5), (3, 3), (6, 6))
DIRECTION_NAMES = ("east", "south", "west", "north")
START_STATES = tuple((head, direction) for head in HEAD_POSITIONS
                     for direction in DIRECTION_NAMES)
ENVIRONMENT_SEEDS = tuple(range(400001, 400051))
BOARD_SIZE = 10
MAX_STEPS = 200
POLICY_IDS = tuple(
    pid for seed in TRAIN_SEEDS for pid in (f"trained-{seed}", f"untrained-{seed}")
) + ("random",)

RESULT_FIELDS = (
    "policy_id", "policy_kind", "training_seed", "initialization_seed",
    "environment_seed", "head_x", "head_y", "initial_direction",
    "initial_food_x", "initial_food_y", "board_width", "board_height",
    "max_steps", "score", "reward", "steps", "terminated", "truncated",
    "terminal_reason", "first_food_step", "food_per_100_steps",
    "model_sha256", "artifact_id",
)


class StartStateSnakeEnv(SnakeEnv):
    """Evaluation-only subclass: canonical Snake step, observation, rewards.

    The initial body has 3 cells directly behind the head in the inverse
    facing direction; food is sampled AFTER placing that body.
    """

    def __init__(self, *, initial_head: tuple[int, int],
                 initial_direction: str):
        if (initial_head, initial_direction) not in START_STATES:
            raise ValueError("initial head/direction not in the approved protocol")
        super().__init__(width=BOARD_SIZE, height=BOARD_SIZE, max_steps=MAX_STEPS)
        self.initial_head = initial_head
        self.initial_direction = initial_direction

    def reset(self, *, seed: int | None = None,
              options: dict | None = None):
        # Seed the official Gymnasium generator without calling SnakeEnv.reset:
        # that would spawn food before moving the body, consuming one RNG draw.
        gym.Env.reset(self, seed=seed)
        if options is not None:
            raise ValueError("Phase 4 does not permit reset options / forced food")
        direction_index = DIRECTION_NAMES.index(self.initial_direction)
        dx, dy = DIRECTIONS[direction_index]
        x, y = self.initial_head
        body = [(x - offset * dx, y - offset * dy) for offset in range(3)]
        if len(set(body)) != 3 or not all(self._inside(cell) for cell in body):
            raise ValueError("initial snake body would overlap or be out of bounds")
        self.snake = deque(body)
        self.direction = direction_index
        self.score, self.steps, self.finished = 0, 0, False
        self.food = self._spawn_food()
        if self.food is None or self.food in self.snake:
            raise RuntimeError("initial food must be free and in bounds")
        info = self._info()
        info.update({
            "initial_head": self.initial_head,
            "initial_direction": self.initial_direction,
            "initial_food": self.food,
        })
        return self._observation(), info


def expected_scenarios(
    *, start_states: tuple[tuple[tuple[int, int], str], ...] = START_STATES,
    environmental_seeds: tuple[int, ...] = ENVIRONMENT_SEEDS,
) -> set[tuple[int, int, str, int, str]]:
    """Enumerate experimental design WITHOUT running any game."""
    if (not start_states or len(start_states) != len(set(start_states))
            or not set(start_states).issubset(set(START_STATES))):
        raise ValueError("unapproved or duplicated starting states")
    if (not environmental_seeds or
            len(environmental_seeds) != len(set(environmental_seeds)) or
            not set(environmental_seeds).issubset(set(ENVIRONMENT_SEEDS))):
        raise ValueError("unapproved or duplicated environmental seeds")
    return {
        (head[0], head[1], direction, env_seed, policy_id)
        for head, direction in start_states
        for env_seed in environmental_seeds
        for policy_id in POLICY_IDS
    }


def evaluate_episode(
    policy: Callable[[np.ndarray], int], *, initial_head: tuple[int, int],
    initial_direction: str, environment_seed: int,
) -> dict:
    """Single evaluation episode. Scores/observations come from original Snake."""
    with StartStateSnakeEnv(initial_head=initial_head,
                            initial_direction=initial_direction) as env:
        observation, reset_info = env.reset(seed=environment_seed)
        initial_food = reset_info["initial_food"]
        total_reward = 0.0
        previous_score = 0
        first_food_step = None
        while True:
            with torch.inference_mode():
                action = policy(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if info["score"] > previous_score and first_food_step is None:
                first_food_step = info["steps"]
            previous_score = info["score"]
            if terminated or truncated:
                score = int(info["score"])
                steps = int(info["steps"])
                if not 1 <= steps <= MAX_STEPS:
                    raise RuntimeError("episode exceeds the approved step budget")
                return {
                    "initial_food_x": initial_food[0],
                    "initial_food_y": initial_food[1],
                    "score": score, "reward": round(total_reward, 6),
                    "steps": steps, "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "terminal_reason": ("board_filled" if terminated and env.food is None
                                        else "collision" if terminated else "step_limit"),
                    "first_food_step": first_food_step,
                    "food_per_100_steps": round(100.0 * score / steps, 6),
                }


def _summary(rows: list[dict]) -> dict:
    stats: dict = {"n": len(rows)}
    for name in ("score", "reward", "steps", "food_per_100_steps"):
        vals = [float(r[name]) for r in rows]
        stats[name] = {
            "mean": mean(vals), "median": median(vals),
            "std": stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals), "max": max(vals),
        }
    first_food = [r["first_food_step"] for r in rows
                  if r["first_food_step"] is not None]
    stats["food_collected_episodes"] = sum(r["score"] > 0 for r in rows)
    stats["zero_food_episodes"] = sum(r["score"] == 0 for r in rows)
    stats["first_food_step_mean_if_observed"] = (
        mean(first_food) if first_food else None
    )
    stats["terminal_reasons"] = {
        reason: sum(r["terminal_reason"] == reason for r in rows)
        for reason in ("collision", "step_limit", "board_filled")
    }
    return stats


def evaluate_startstates(
    artifact_root: str | Path, output: str | Path,
    *, start_states: tuple[tuple[tuple[int, int], str], ...] = START_STATES,
    environmental_seeds: tuple[int, ...] = ENVIRONMENT_SEEDS,
    _model_hash_overrides: dict[int, str] | None = None,
) -> dict:
    """Run read-only evaluation; small subsets/checkpoint overrides for tests only.

    Command-line entry has no subset/hash override options, so its protocol is fixed.
    """
    scenarios = expected_scenarios(start_states=start_states,
                                   environmental_seeds=environmental_seeds)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    artifact_root, output = Path(artifact_root), Path(output)
    networks, policies, integrity = {}, {}, {}
    for seed in TRAIN_SEEDS:
        folder = artifact_root / f"seed_{seed}"
        verified = verify_artifact_directory(
            folder, seed=seed,
            expected_model_hash=(_model_hash_overrides or {}).get(seed),
        )
        trained = load_verified_model(folder, seed=seed, verified=verified)
        control = untrained_control(verified["config"], seed=seed)
        for kind, network in (("trained", trained), ("untrained", control)):
            policy_id = f"{kind}-{seed}"
            networks[policy_id] = network
            policies[policy_id] = {
                "policy_kind": kind,
                "training_seed": seed if kind == "trained" else "",
                "initialization_seed": seed if kind == "untrained" else "",
                "model_sha256": verified["model_sha256"] if kind == "trained"
                                else weights_fingerprint(network),
                "artifact_id": ARTIFACTS[seed][0] if kind == "trained" else "",
            }
        integrity[str(seed)] = verified
    policies["random"] = {
        "policy_kind": "random", "training_seed": "", "initialization_seed": "",
        "model_sha256": "", "artifact_id": "",
    }
    fingerprints = {pid: weights_fingerprint(net) for pid, net in networks.items()}
    rows = []
    for head, direction in start_states:
        for env_seed in environmental_seeds:
            for policy_id in POLICY_IDS:
                if policy_id == "random":
                    rng = random.Random(90_000_000 + env_seed)
                    policy = lambda _obs, rng=rng: rng.randrange(3)
                else:
                    network = networks[policy_id]
                    policy = lambda obs, network=network: greedy_action(network, obs)
                result = evaluate_episode(
                    policy, initial_head=head,
                    initial_direction=direction, environment_seed=env_seed,
                )
                rows.append({
                    "policy_id": policy_id, **policies[policy_id],
                    "environment_seed": env_seed, "head_x": head[0],
                    "head_y": head[1], "initial_direction": direction,
                    "board_width": BOARD_SIZE, "board_height": BOARD_SIZE,
                    "max_steps": MAX_STEPS, **result,
                })

    observed = [(r["head_x"], r["head_y"], r["initial_direction"],
                 r["environment_seed"], r["policy_id"]) for r in rows]
    if (len(observed) != len(scenarios) or len(set(observed)) != len(observed)
            or set(observed) != scenarios):
        raise RuntimeError("incomplete, duplicate or mixed start-state scenario matrix")
    for policy_id, net in networks.items():
        if weights_fingerprint(net) != fingerprints[policy_id]:
            raise RuntimeError("evaluating a model changed its weights: " + policy_id)
        if (net.training or
                any(p.requires_grad or p.grad is not None for p in net.parameters())):
            raise RuntimeError("model not frozen or gradient produced: " + policy_id)
    summary = {
        f"{head[0]},{head[1]}|{direction}": {
            policy_id: _summary([
                r for r in rows
                if (r["head_x"], r["head_y"], r["initial_direction"], r["policy_id"])
                == (*head, direction, policy_id)
            ]) for policy_id in POLICY_IDS
        } for head, direction in start_states
    }
    config = {
        "protocol": "phase4_frozen_startstate_robustness",
        "official_training_commit": APPROVED_COMMIT,
        "official_pilot_run_id": PILOT_RUN_ID,
        "training_seeds": list(TRAIN_SEEDS),
        "environment": {
            "name": "StartStateSnakeEnv", "original_env": "SnakeEnv",
            "board_width": BOARD_SIZE, "board_height": BOARD_SIZE,
            "max_steps": MAX_STEPS, "observation_dimensions": 11,
            "relative_actions": 3,
        },
        "head_positions": [list(h) for h in HEAD_POSITIONS],
        "initial_directions": list(DIRECTION_NAMES),
        "start_states": [{"head": list(h), "direction": d} for h, d in start_states],
        "environmental_seeds": list(environmental_seeds),
        "total_episodes": len(scenarios),
        "policy_ids": list(POLICY_IDS),
        "policies": policies,
        "artifact_integrity": integrity,
        "versions": {"gymnasium": gym.__version__, "torch": torch.__version__,
                     "numpy": np.__version__},
        "frozen_weights_verified": True, "no_training": True,
        "efficiency_formula": "100 * score / steps (episode)",
        "first_food_step_definition": "1-based first step with increased score; null if never",
        "random_action_seed_formula": "90000000 + environment_seed",
        "seed_comparability_caveat": (
            "Same numeric seed across initial body configurations does not imply "
            "identical food position or trajectories. Exact initial food is logged."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for filename, payload in (("summary.json", summary), ("config.json", config)):
        (output / filename).write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8",
        )
    (output / "SHA256SUMS").write_text(
        "".join(f"{file_sha256(output / filename)}  {filename}\n"
                for filename in ("episodes.csv", "summary.json", "config.json")),
        encoding="utf-8",
    )
    return {"episodes": len(rows), "conditions": len(start_states),
            "policies": list(POLICY_IDS), "frozen_weights_verified": True,
            "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--authorized", action="store_true",
                        help="requires separate explicit Chat 1 authorization")
    args = parser.parse_args()
    if not args.authorized:
        parser.error("Phase 4 full evaluation locked pending Chat 1 authorization")
    print(json.dumps(evaluate_startstates(args.artifacts, args.out), indent=2))


if __name__ == "__main__":
    main()
