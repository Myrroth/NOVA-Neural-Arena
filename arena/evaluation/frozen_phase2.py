"""Frozen, paired-seed Snake evaluation. NO training or optimizer operations.

Only official pilot artifact files are accepted. Always verify SHA256SUMS,
fixed model hashes and provenance before deserializing checkpoint weights.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Callable

import gymnasium
import numpy as np
import torch

from arena.agents.dqn import DQNConfig, QNetwork
from arena.envs.snake import SnakeEnv

APPROVED_COMMIT = "24bd38e9e78a14e894f32d7d1c5b5b6d6d724463"
PILOT_RUN_ID = 41350959549
ENVIRONMENT_SEEDS = tuple(range(200001, 200051))
TRAIN_SEEDS = (11, 42, 73)
ARTIFACTS = {
    11: (13101471957, "ff132bc23aaf5b56d00679a07961d23c2de30bafa7bbaa4e99ff966b61a49130"),
    42: (13101072100, "0623256abf3b068f212bacaa5c46f8bbcc6aabf2aeaa0f7ef5ecdc0ec8120cbe"),
    73: (13100988601, "2899df84afe4512ca127bf3367490a5bc42c88f3ad7868daf8195ed419ecb8fe"),
}
HASHED_FILES = frozenset({
    "model.pt", "run.json", "episodes.csv", "evaluation.json",
    "provenance.json", "commit.txt", "pip-freeze.txt",
})
RESULT_FIELDS = (
    "policy_id", "policy_kind", "training_seed", "initialization_seed",
    "environment_seed", "score", "reward", "steps", "terminated",
    "truncated", "terminal_reason", "model_sha256", "artifact_id",
)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def weights_fingerprint(network: QNetwork) -> str:
    """Stable fingerprint independent of torch.save container metadata."""
    h = hashlib.sha256()
    for name, value in sorted(network.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("ascii"))
        h.update(str(tensor.dtype).encode("ascii"))
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def _file(base: Path, name: str) -> Path:
    path = base / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing / unsafe artifact file: {path}")
    return path


def _read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def verify_artifact_directory(
    directory: str | Path, *, seed: int,
    expected_model_hash: str | None = None,
    expected_artifact_id: int | None = None,
) -> dict:
    """Fail closed on missing/tampered files before torch.load.

    Model hash overrides are only for synthetic unit-test fixtures. The CLI
    uses only the immutable official mappings above.
    """
    base = Path(directory)
    if seed not in ARTIFACTS:
        raise ValueError("seed is not an approved official training seed")
    artifact_id, official_hash = ARTIFACTS[seed]
    if expected_artifact_id is None:
        expected_artifact_id = artifact_id
    if expected_artifact_id != artifact_id:
        raise ValueError("unexpected artifact ID")
    if expected_model_hash is None:
        expected_model_hash = official_hash

    lines = _file(base, "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    listed: dict[str, str] = {}
    for line in lines:
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+([A-Za-z0-9_.-]+)", line)
        if not match:
            raise ValueError("invalid SHA256SUMS entry")
        digest, name = match.groups()
        if name in listed:
            raise ValueError("duplicate checksum entry")
        listed[name] = digest.lower()
    if set(listed) != HASHED_FILES:
        raise ValueError(f"incomplete or unexpected checksums: {set(listed) ^ HASHED_FILES}")
    for name, expected_hash in listed.items():
        if file_sha256(_file(base, name)) != expected_hash:
            raise ValueError(f"SHA256 mismatch: {name}")
    if listed["model.pt"] != expected_model_hash:
        raise ValueError(f"model.pt differs from approved seed {seed} checkpoint")

    run = _read_json(_file(base, "run.json"))
    provenance = _read_json(_file(base, "provenance.json"))
    if _file(base, "commit.txt").read_text(encoding="utf-8").strip() != APPROVED_COMMIT:
        raise ValueError("commit.txt is not the approved training commit")
    if provenance.get("approved_commit") != APPROVED_COMMIT:
        raise ValueError("provenance commit mismatch")
    if (provenance.get("seed"), provenance.get("github_run_id"),
        provenance.get("github_run_attempt"), provenance.get("pull_request_number")) != (
            seed, str(PILOT_RUN_ID), "1", 1
        ):
        # GitHub identifiers may be encoded as integers in documented metadata.
        if (provenance.get("seed") != seed or
            str(provenance.get("github_run_id")) != str(PILOT_RUN_ID) or
            str(provenance.get("github_run_attempt")) != "1" or
            provenance.get("pull_request_number") != 1):
            raise ValueError("provenance run or seed mismatch")
    if (run.get("seed"), run.get("episodes"), run.get("eval_episodes"),
        run.get("width"), run.get("height"), run.get("max_steps")) != (
            seed, 80, 20, 10, 10, 200):
        raise ValueError("run.json pilot configuration mismatch")
    if run.get("training_seeds") != [seed, seed + 79]:
        raise ValueError("training seeds mismatch")
    if run.get("evaluation_seeds") != list(range(seed + 100000, seed + 100020)):
        raise ValueError("pilot evaluation seeds mismatch")
    versions = run.get("versions", {})
    if versions.get("gymnasium") != "1.4.0" or not str(versions.get("torch", "")).startswith("2.10.0"):
        raise ValueError("pilot did not use the approved official libraries")
    freeze = _file(base, "pip-freeze.txt").read_text(encoding="utf-8").lower()
    if "gymnasium==1.4.0" not in freeze or "torch==2.10.0" not in freeze:
        raise ValueError("missing official dependency versions in pip freeze")
    return {"seed": seed, "artifact_id": artifact_id, "model_sha256": listed["model.pt"],
            "source_run_id": PILOT_RUN_ID, "source_commit": APPROVED_COMMIT,
            "config": run["config"]}


def _frozen(network: QNetwork) -> QNetwork:
    network.eval()
    network.requires_grad_(False)
    return network


def load_verified_model(directory: str | Path, *, seed: int, verified: dict | None = None) -> QNetwork:
    verified = verified or verify_artifact_directory(directory, seed=seed)
    data = torch.load(Path(directory) / "model.pt", map_location="cpu", weights_only=True)
    if data["config"] != verified["config"]:
        raise ValueError("checkpoint config differs from verified run config")
    config = DQNConfig(**data["config"])
    if config.observation_dim != 11 or config.actions != 3:
        raise ValueError("checkpoint dimensions incompatible with Snake")
    network = QNetwork(config)
    network.load_state_dict(data["online"], strict=True)
    return _frozen(network)


def untrained_control(config: dict, seed: int) -> QNetwork:
    """Initialize the very same first QNetwork instantiated by DQNAgent(seed)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        network = QNetwork(DQNConfig(**config))
    return _frozen(network)


def greedy_action(network: QNetwork, observation: np.ndarray) -> int:
    with torch.inference_mode():
        x = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        return int(network(x).argmax(dim=1).item())


def evaluate_episode(policy: Callable[[np.ndarray], int], *, environment_seed: int,
                     width: int = 10, height: int = 10, max_steps: int = 200) -> dict:
    """One held-out episode; terminal reason recorded without changing Snake."""
    with SnakeEnv(width=width, height=height, max_steps=max_steps) as env:
        observation, _ = env.reset(seed=environment_seed)
        total_reward = 0.0
        while True:
            action = policy(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                return {
                    "score": info["score"], "reward": round(total_reward, 6),
                    "steps": info["steps"], "terminated": terminated, "truncated": truncated,
                    "terminal_reason": ("board_filled" if terminated and env.food is None
                                        else "collision" if terminated else "step_limit"),
                }


def _statistics(rows: list[dict]) -> dict:
    out = {}
    for metric in ("score", "reward", "steps"):
        vals = [float(row[metric]) for row in rows]
        out[metric] = {"mean": mean(vals), "median": median(vals),
                       "std": stdev(vals) if len(vals) > 1 else 0.0,
                       "min": min(vals), "max": max(vals)}
    return out


def evaluate_frozen_models(
    artifact_root: str | Path, output: str | Path,
    *, environmental_seeds: tuple[int, ...] = ENVIRONMENT_SEEDS,
    _model_hash_overrides: dict[int, str] | None = None,
) -> dict:
    """Read-only evaluation; tests may pass a *smaller* seed subset and fixture hashes."""
    if not environmental_seeds or len(set(environmental_seeds)) != len(environmental_seeds):
        raise ValueError("environment seeds must be nonempty and unique")
    if not set(environmental_seeds).issubset(set(ENVIRONMENT_SEEDS)):
        raise ValueError("only approved held-out environmental seeds are allowed")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root, output = Path(artifact_root), Path(output)
    networks: dict[str, QNetwork] = {}
    policies: dict[str, dict] = {}
    integrity: dict[str, dict] = {}
    for seed in TRAIN_SEEDS:
        directory = root / f"seed_{seed}"
        verified = verify_artifact_directory(
            directory, seed=seed,
            expected_model_hash=(_model_hash_overrides or {}).get(seed),
        )
        trained = load_verified_model(directory, seed=seed, verified=verified)
        untrained = untrained_control(verified["config"], seed)
        for kind, model in (("trained", trained), ("untrained", untrained)):
            policy_id = f"{kind}-{seed}"
            networks[policy_id] = model
            policies[policy_id] = {
                "policy_kind": kind, "training_seed": seed if kind == "trained" else "",
                "initialization_seed": seed if kind == "untrained" else "",
                "model_sha256": verified["model_sha256"] if kind == "trained" else weights_fingerprint(model),
                "artifact_id": verified["artifact_id"] if kind == "trained" else "",
            }
        integrity[str(seed)] = verified
    # One random policy is evaluated once per environment seed, shared across comparisons.
    fingerprints = {name: weights_fingerprint(model) for name, model in networks.items()}
    policies["random"] = {"policy_kind": "random", "training_seed": "",
                          "initialization_seed": "", "model_sha256": "", "artifact_id": ""}
    rows: list[dict] = []
    for env_seed in environmental_seeds:
        for policy_id, meta in policies.items():
            if policy_id == "random":
                rng = random.Random(90_000_000 + env_seed)
                policy = lambda _obs, rng=rng: rng.randrange(3)
            else:
                model = networks[policy_id]
                policy = lambda obs, model=model: greedy_action(model, obs)
            result = evaluate_episode(policy, environment_seed=env_seed)
            rows.append({"policy_id": policy_id, **meta,
                         "environment_seed": env_seed, **result})
    for name, model in networks.items():
        if weights_fingerprint(model) != fingerprints[name]:
            raise RuntimeError(f"INVARIANT FAILED: weights changed during evaluation: {name}")
        if any(p.grad is not None for p in model.parameters()):
            raise RuntimeError(f"INVARIANT FAILED: gradients created during evaluation: {name}")
    assert len(rows) == len(environmental_seeds) * 7
    summary = {name: _statistics([row for row in rows if row["policy_id"] == name])
               for name in policies}
    output.mkdir(parents=True, exist_ok=True)
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    config = {"approved_training_commit": APPROVED_COMMIT, "pilot_run_id": PILOT_RUN_ID,
              "evaluation_seeds": list(environmental_seeds), "environment": {
                  "name": "SnakeEnv", "width": 10, "height": 10, "max_steps": 200},
              "policies": policies, "official_artifacts": integrity,
              "versions": {"torch": torch.__version__, "gymnasium": gymnasium.__version__,
                           "numpy": np.__version__},
              "frozen_weights_verified": True, "episodes_per_policy": len(environmental_seeds)}
    for filename, payload in (("summary.json", summary), ("config.json", config)):
        (output / filename).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                                       encoding="utf-8")
    (output / "SHA256SUMS").write_text(
        "".join(f"{file_sha256(output / name)}  {name}\n"
                for name in ("episodes.csv", "summary.json", "config.json")), encoding="utf-8")
    return {"rows": len(rows), "policies": list(policies), "frozen_weights_verified": True,
            "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True,
                        help="directory with verified extracted pilot artifact seed_11/42/73")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--authorized", action="store_true",
                        help="required for the full 50-seed evaluation; only after Chat 1 approval")
    args = parser.parse_args()
    if not args.authorized:
        parser.error("Evaluation locked: obtain new Chat 1 authorization before --authorized")
    print(json.dumps(evaluate_frozen_models(args.artifacts, args.out), indent=2))


if __name__ == "__main__":
    main()
