"""Phase 3: frozen cross-size Snake evaluation; NEVER trains any model.

The official pilot checkpoints and their SHA-256 manifest are validated by
the previously audited Phase 2 loader before any weight deserialization.
Actual 50-seed/3-size evaluation requires explicit CLI authorization.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean, median, stdev
from typing import Callable

import gymnasium
import numpy as np
import torch

from arena.envs.snake import SnakeEnv
from arena.evaluation.frozen_phase2 import (
    APPROVED_COMMIT, ARTIFACTS, PILOT_RUN_ID, TRAIN_SEEDS,
    file_sha256, greedy_action, load_verified_model, untrained_control,
    verify_artifact_directory, weights_fingerprint,
)

BOARD_SIZES = (8, 10, 12)
ENVIRONMENT_SEEDS = tuple(range(300001, 300051))
MAX_STEPS = 200
POLICY_IDS = tuple(
    pid for seed in TRAIN_SEEDS for pid in (f"trained-{seed}", f"untrained-{seed}")
) + ("random",)
RESULT_FIELDS = (
    "policy_id", "policy_kind", "training_seed", "initialization_seed",
    "environment_seed", "board_width", "board_height", "max_steps",
    "score", "reward", "steps", "terminated", "truncated", "terminal_reason",
    "first_food_step", "food_per_100_steps", "model_sha256", "artifact_id",
)


def expected_scenarios(
    *, board_sizes: tuple[int, ...] = BOARD_SIZES,
    environmental_seeds: tuple[int, ...] = ENVIRONMENT_SEEDS,
) -> set[tuple[int, int, str]]:
    """Compute the full design without running a single Snake episode."""
    if (not board_sizes or len(set(board_sizes)) != len(board_sizes)
            or not set(board_sizes).issubset(BOARD_SIZES)):
        raise ValueError("invalid or repeated approved board sizes")
    if (not environmental_seeds or len(set(environmental_seeds)) != len(environmental_seeds)
            or not set(environmental_seeds).issubset(ENVIRONMENT_SEEDS)):
        raise ValueError("invalid or repeated approved environmental seeds")
    return {(size, env_seed, pid) for size in board_sizes
            for env_seed in environmental_seeds for pid in POLICY_IDS}


def evaluate_episode(
    policy: Callable[[np.ndarray], int], *, board_size: int,
    environment_seed: int, max_steps: int = MAX_STEPS,
) -> dict:
    """Same unmodified SnakeEnv, default initial state and reward rules.

    first_food_step is 1-indexed and absent (None) if the snake never eats.
    food_per_100_steps is score / number_of_steps * 100, not survival rate.
    """
    if board_size not in BOARD_SIZES or max_steps != MAX_STEPS:
        raise ValueError("unapproved board size or maximum steps")
    with SnakeEnv(width=board_size, height=board_size, max_steps=max_steps) as env:
        observation, _ = env.reset(seed=environment_seed)
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
                steps = int(info["steps"])
                score = int(info["score"])
                if not 1 <= steps <= max_steps:
                    raise RuntimeError("episode length outside protocol bounds")
                return {
                    "score": score, "reward": round(total_reward, 6),
                    "steps": steps, "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "terminal_reason": ("board_filled" if terminated and env.food is None
                                        else "collision" if terminated else "step_limit"),
                    "first_food_step": first_food_step,
                    "food_per_100_steps": round(100.0 * score / steps, 6),
                }


def _statistics(rows: list[dict]) -> dict:
    stats: dict = {"n": len(rows)}
    for metric in ("score", "reward", "steps", "food_per_100_steps"):
        values = [float(row[metric]) for row in rows]
        stats[metric] = {
            "mean": mean(values), "median": median(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values),
        }
    first_steps = [row["first_food_step"] for row in rows
                   if row["first_food_step"] is not None]
    stats["first_food_observed"] = len(first_steps)
    stats["first_food_step_mean_if_observed"] = mean(first_steps) if first_steps else None
    stats["zero_food_episodes"] = sum(row["score"] == 0 for row in rows)
    stats["terminal_reasons"] = {
        reason: sum(row["terminal_reason"] == reason for row in rows)
        for reason in ("collision", "step_limit", "board_filled")
    }
    return stats


def evaluate_generalization(
    artifact_root: str | Path, output: str | Path,
    *, board_sizes: tuple[int, ...] = BOARD_SIZES,
    environmental_seeds: tuple[int, ...] = ENVIRONMENT_SEEDS,
    _model_hash_overrides: dict[int, str] | None = None,
) -> dict:
    """Run frozen, paired-size design. Subsets/overrides only used in unit tests.

    CLI never accepts size/seed overrides and always evaluates the full design.
    """
    scenarios = expected_scenarios(board_sizes=board_sizes, environmental_seeds=environmental_seeds)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root, output = Path(artifact_root), Path(output)
    networks = {}
    policies: dict[str, dict] = {}
    integrity: dict[str, dict] = {}
    for seed in TRAIN_SEEDS:
        folder = root / f"seed_{seed}"
        verified = verify_artifact_directory(
            folder, seed=seed,
            expected_model_hash=(_model_hash_overrides or {}).get(seed),
        )
        trained = load_verified_model(folder, seed=seed, verified=verified)
        untrained = untrained_control(verified["config"], seed=seed)
        for kind, model in (("trained", trained), ("untrained", untrained)):
            pid = f"{kind}-{seed}"
            networks[pid] = model
            policies[pid] = {
                "policy_kind": kind,
                "training_seed": seed if kind == "trained" else "",
                "initialization_seed": seed if kind == "untrained" else "",
                "model_sha256": verified["model_sha256"] if kind == "trained"
                                else weights_fingerprint(model),
                "artifact_id": ARTIFACTS[seed][0] if kind == "trained" else "",
            }
        integrity[str(seed)] = verified
    policies["random"] = {
        "policy_kind": "random", "training_seed": "", "initialization_seed": "",
        "model_sha256": "", "artifact_id": "",
    }
    fingerprints = {pid: weights_fingerprint(net) for pid, net in networks.items()}
    rows = []
    for size in board_sizes:
        for env_seed in environmental_seeds:
            for pid in POLICY_IDS:
                if pid == "random":
                    # Identical policy definition and random seed schedule across sizes.
                    rng = random.Random(90_000_000 + env_seed)
                    policy = lambda _obs, rng=rng: rng.randrange(3)
                else:
                    network = networks[pid]
                    policy = lambda obs, network=network: greedy_action(network, obs)
                record = evaluate_episode(
                    policy, board_size=size, environment_seed=env_seed,
                )
                rows.append({
                    "policy_id": pid, **policies[pid],
                    "environment_seed": env_seed, "board_width": size,
                    "board_height": size, "max_steps": MAX_STEPS, **record,
                })

    observed = [(r["board_width"], r["environment_seed"], r["policy_id"]) for r in rows]
    if len(observed) != len(scenarios) or len(set(observed)) != len(observed) or set(observed) != scenarios:
        raise RuntimeError("incomplete, duplicate or mixed-size evaluation matrix")
    for pid, model in networks.items():
        if weights_fingerprint(model) != fingerprints[pid]:
            raise RuntimeError(f"frozen model changed weights: {pid}")
        if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise RuntimeError(f"model lost frozen, no-gradient state: {pid}")

    summary = {
        str(size): {
            pid: _statistics([row for row in rows
                              if row["board_width"] == size and row["policy_id"] == pid])
            for pid in POLICY_IDS
        } for size in board_sizes
    }
    metadata = {
        "protocol": "phase3_cross_size_frozen_evaluation",
        "official_training_commit": APPROVED_COMMIT,
        "official_pilot_run_id": PILOT_RUN_ID,
        "original_training_seeds": list(TRAIN_SEEDS),
        "environmental_seeds": list(environmental_seeds),
        "board_sizes": list(board_sizes),
        "max_steps": MAX_STEPS,
        "total_episodes": len(scenarios),
        "policy_ids": list(POLICY_IDS),
        "policies": policies,
        "artifact_integrity": integrity,
        "versions": {"gymnasium": gymnasium.__version__,
                     "torch": torch.__version__, "numpy": np.__version__},
        "frozen_weights_verified": True,
        "random_action_seed_formula": "90000000 + environment_seed",
        "efficiency_formula": "100 * score / steps, per episode",
        "first_food_step_definition": "1-indexed first step with increased score; null if never",
        "same_seed_cross_size_caveat": (
            "Same numeric environment seed does not imply identical physical food "
            "coordinates or trajectories across board sizes."
        ),
        "fixed_initial_state": True,
        "no_training": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for filename, data in (("summary.json", summary), ("config.json", metadata)):
        (output / filename).write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
    filenames = ("episodes.csv", "summary.json", "config.json")
    (output / "SHA256SUMS").write_text(
        "".join(f"{file_sha256(output / name)}  {name}\n" for name in filenames),
        encoding="utf-8",
    )
    return {"episodes": len(rows), "board_sizes": list(board_sizes),
            "policies": list(POLICY_IDS), "frozen_weights_verified": True,
            "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--authorized", action="store_true",
                        help="only after specific, new approval by Chat 1")
    args = parser.parse_args()
    if not args.authorized:
        parser.error("Phase 3 full evaluation locked until new Chat 1 authorization")
    print(json.dumps(evaluate_generalization(args.artifacts, args.out), indent=2))


if __name__ == "__main__":
    main()
