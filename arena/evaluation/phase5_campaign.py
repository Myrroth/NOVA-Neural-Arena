"""NNA-FECHAMENTO-001: single authorized read-only Phase 5 campaign.

No train/optimizer/model save. Historical candidate, Snake and DQN are unchanged.
Full campaign is gated by exact GitHub approval, run identity and one-time reservation.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys

import torch

from arena.evaluation.frozen_phase2 import TRAIN_SEEDS, greedy_action
from arena.evaluation.phase5_hardstates import (
    ExperimentConfig, candidate_scenarios, protocol_manifest, POLICY_IDS,
    _evaluate_episode, verify_official_policies,
)

SOURCE_CANDIDATE_SHA = "00f71a9ebaedf54c31bf876298e31b99e7e2b913789fa0d4ccd18e8b5fcb1a65"
PROTOCOL_VERSION = "phase5-hardstates-closure-approved-v1"
REPO = "Myrroth/NOVA-Neural-Arena"
APPROVAL_ISSUE = 6
APPROVAL_LABEL = "neural-phase5-closure-approved"
TOTAL_EPISODES = 32200
MAX_TOTAL_STEPS = 6_440_000
SEEDS = tuple(range(500001, 500051))
FIELDS = (
    "scenario_id", "region", "food_mode", "food_key",
    "head_x", "head_y", "direction", "food_x", "food_y",
    "environment_seed", "policy_id", "policy_kind", "training_seed",
    "checkpoint_sha256", "artifact_id", "score", "reward", "steps",
    "terminated", "truncated", "terminal_reason", "first_food_step",
    "first_food_censored", "first_food_observation_steps",
    "food_per_100_steps", "repeated_state_visits", "first_repeat_step",
    "first_repeat_period", "max_steps_between_food_events",
)


def _canonical_sha(document: dict) -> str:
    raw = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def approved_manifest() -> dict:
    candidate = protocol_manifest(ExperimentConfig())
    if candidate["protocol_sha256"] != SOURCE_CANDIDATE_SHA:
        raise RuntimeError("Phase 5 candidate matrix or serialized contract changed")
    if (candidate["total_episodes"] != TOTAL_EPISODES or
            len(candidate["scenarios"]) != 92 or
            len(candidate["policies"]) != 7 or
            tuple(candidate["config"]["environmental_seeds"]) != SEEDS):
        raise RuntimeError("Phase 5 matrix exceeds or differs from approved candidate")
    final = {key: value for key, value in candidate.items() if key != "protocol_sha256"}
    final.update({
        "status": "PRE_REGISTERED_FOR_CHAT1_FINAL_APPROVAL",
        "protocol_version": PROTOCOL_VERSION,
        "candidate_protocol_sha256": SOURCE_CANDIDATE_SHA,
        "approval_issue": APPROVAL_ISSUE,
        "max_total_steps": MAX_TOTAL_STEPS,
        "primary_endpoint": "score",
        "primary_contrasts": [
            {"trained": f"trained-{seed}", "control": control,
             "estimand": "balanced mean per environmental seed across all 92 scenarios"}
            for seed in TRAIN_SEEDS for control in (f"untrained-{seed}", "random")
        ],
        "uncertainty": {
            "method": "paired seed-block percentile bootstrap",
            "confidence": 0.95, "resamples": 10000, "bootstrap_seed": 510000,
            "note": "Descriptive conditional intervals; no hypothesis tests.",
        },
        "secondary_contrasts": "552 matched control pairs by scenario and 216 versus matched center",
        "hypothesis_tests": False,
        "noninferiority_or_equivalence": False,
        "note": ("Initial food fixed, pre-first-food trajectories may duplicate across "
                 "seeds; same seed does not guarantee identical future food positions."),
    })
    final["approved_protocol_sha256"] = _canonical_sha(final)
    return final


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json_x(path: Path, obj: dict) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, sort_keys=True, indent=2)
        f.write("\n")


def _validate_execution_identity(manifest: dict) -> dict:
    values = {
        "github_run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),
        "github_repository": os.environ.get("GITHUB_REPOSITORY", ""),
        "evaluator_sha": os.environ.get("NNA_PHASE5_EVALUATOR_SHA", ""),
        "approved_protocol_sha": os.environ.get("NNA_PHASE5_APPROVED_HASH", ""),
        "reserved_issue_comment_id": os.environ.get("NNA_PHASE5_RESERVATION_ID", ""),
    }
    if (values["github_repository"] != REPO or
            values["github_run_attempt"] != "1" or
            not values["github_run_id"].isdigit() or
            not values["reserved_issue_comment_id"].isdigit() or
            values["approved_protocol_sha"] != manifest["approved_protocol_sha256"]):
        raise RuntimeError("Missing exact GitHub authorization/run/unique reservation")
    checkout = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if len(values["evaluator_sha"]) != 40 or checkout != values["evaluator_sha"]:
        raise RuntimeError("unapproved or mutable evaluator code")
    return values


def campaign(artifact_root: Path, output: Path) -> dict:
    manifest = approved_manifest()
    identity = _validate_execution_identity(manifest)
    config = ExperimentConfig()
    scenarios = candidate_scenarios(config)
    if output.exists():
        raise FileExistsError("Phase 5 execution destination already exists")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    bundle = verify_official_policies(artifact_root)
    bundle.assert_frozen()
    # The directory is reserved only after all checkpoint provenance checks.
    output.mkdir(parents=True, exist_ok=False)
    _write_json_x(output / "protocol.json", manifest)
    _write_json_x(output / "execution.json", {
        **identity, "approval_issue": APPROVAL_ISSUE,
        "approval_label": APPROVAL_LABEL, "no_training": True,
        "frozen_weights_before": bundle.fingerprints,
        "policies": bundle.metadata,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    })
    with (output / "pip-freeze.txt").open("x", encoding="utf-8") as f:
        subprocess.run([sys.executable, "-m", "pip", "freeze"],
                       stdout=f, check=True)
    done, steps_done = 0, 0
    success = False
    try:
        with (output / "episodes.csv").open("x", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            for scenario in scenarios:
                for seed in SEEDS:
                    for policy_id in POLICY_IDS:
                        if policy_id == "random":
                            rng = random.Random(90_000_000 + seed)
                            policy = lambda obs, rng=rng: rng.randrange(3)
                        else:
                            model = bundle.networks[policy_id]
                            policy = lambda obs, model=model: greedy_action(model, obs)
                        episode = _evaluate_episode(policy, scenario, seed, config)
                        steps = int(episode["steps"])
                        if not 1 <= steps <= 200:
                            raise RuntimeError("individual step budget exceeded")
                        steps_done += steps
                        if steps_done > MAX_TOTAL_STEPS:
                            raise RuntimeError("total step budget exceeded")
                        metadata = bundle.metadata[policy_id]
                        writer.writerow({
                            "scenario_id": scenario.id, "region": scenario.region,
                            "food_mode": scenario.food_mode, "food_key": scenario.food_key,
                            "head_x": scenario.head[0], "head_y": scenario.head[1],
                            "direction": scenario.direction,
                            "food_x": scenario.food[0], "food_y": scenario.food[1],
                            "environment_seed": seed, "policy_id": policy_id,
                            "policy_kind": metadata["kind"],
                            "training_seed": metadata.get("seed", ""),
                            "checkpoint_sha256": metadata.get("checkpoint_sha256") or "",
                            "artifact_id": metadata.get("artifact_id") or "",
                            **episode,
                        })
                        done += 1
                # Persist a complete scenario after every batch of 350 episodes.
                f.flush()
        if done != TOTAL_EPISODES:
            raise RuntimeError("incomplete official Phase 5 factorial matrix")
        bundle.assert_frozen()
        success = True
    finally:
        _write_json_x(output / "status.json", {
            "complete": success, "episodes_written": done,
            "expected_episodes": TOTAL_EPISODES, "steps_used": steps_done,
            "max_total_steps": MAX_TOTAL_STEPS, "no_training": True,
            "weights_frozen_checked_at_end": success,
            "frozen_weights_reference": bundle.fingerprints,
        })
        names = ("protocol.json", "execution.json", "pip-freeze.txt",
                 "episodes.csv", "status.json")
        with (output / "SHA256SUMS").open("x", encoding="utf-8") as f:
            for name in names:
                f.write(f"{_sha(output / name)}  {name}\n")
    return {"complete": True, "episodes": done, "steps": steps_done,
            "protocol_sha256": manifest["approved_protocol_sha256"],
            "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--campaign", action="store_true")
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not args.campaign:
        print(json.dumps(approved_manifest(), indent=2, sort_keys=True))
        return
    if args.validate_only or not args.artifacts or not args.out:
        parser.error("campaign requires --artifacts and --out, without --validate-only")
    print(json.dumps(campaign(args.artifacts, args.out), indent=2))


if __name__ == "__main__":
    main()
