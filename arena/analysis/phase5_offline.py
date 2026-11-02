"""Independent Phase 5 scientific OFFLINE analysis; never loads models or games.

Reads previously completed, SHA-verified official result files, checks the exact
approved experimental matrix, and estimates the six preregistered score
contrasts using 10,000 paired bootstrap resamples of environmental-seed
BLOCKS. Does not calculate p-values, train models, or invoke an evaluator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import random
import statistics

APPROVED_PROTOCOL_SHA = "ceb65ad7a9529d091611a7e0f60cc3b937abfc76d00cc8fd766b07557ed5160a"
APPROVED_PROTOCOL_VERSION = "phase5-hardstates-closure-approved-v1"
APPROVED_EVALUATOR_SHA = "39461168db30edb90a25385fb3a1eaca8dfdde33"
EPISODES = 32200
POLICIES = ("trained-11", "trained-42", "trained-73",
            "untrained-11", "untrained-42", "untrained-73", "random")
SEEDS = tuple(range(500001, 500051))
OFFICIAL_INPUTS = ("protocol.json", "execution.json", "pip-freeze.txt",
                   "episodes.csv", "status.json")
REQUIRED_COLUMNS = (
    "scenario_id", "region", "food_mode", "food_key", "head_x", "head_y",
    "direction", "food_x", "food_y", "environment_seed", "policy_id",
    "score", "steps", "terminated", "truncated", "terminal_reason",
    "first_food_step", "first_food_censored", "repeated_state_visits",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{path.name}: expected JSON object")
    return obj


def check_source(root: Path) -> tuple[dict, dict, dict, dict]:
    """Verify all official files using recorded hashes and known provenance."""
    if not root.is_dir():
        raise ValueError("input must be an extracted official result directory")
    checks = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    if len(checks) != len(OFFICIAL_INPUTS):
        raise ValueError("SHA256SUMS must contain five official files")
    digests = {}
    for entry in checks:
        parts = entry.split()
        if len(parts) != 2 or parts[1] not in OFFICIAL_INPUTS or parts[1] in digests:
            raise ValueError("unexpected/duplicate hash manifest member")
        name = parts[1]
        file = root / name
        if file.is_symlink() or not file.is_file() or _sha(file) != parts[0]:
            raise ValueError(f"source integrity failure: {name}")
        digests[name] = parts[0]
    if set(digests) != set(OFFICIAL_INPUTS):
        raise ValueError("missing official result file")
    protocol = _json(root / "protocol.json")
    execution = _json(root / "execution.json")
    status = _json(root / "status.json")
    if (protocol.get("approved_protocol_sha256") != APPROVED_PROTOCOL_SHA or
            protocol.get("protocol_version") != APPROVED_PROTOCOL_VERSION or
            execution.get("approved_protocol_sha") != APPROVED_PROTOCOL_SHA or
            execution.get("evaluator_sha") != APPROVED_EVALUATOR_SHA):
        raise ValueError("wrong preregistered protocol/evaluator identity")
    if (protocol.get("total_episodes") != EPISODES or
            protocol.get("max_total_steps") != 6440000 or
            len(protocol.get("scenarios", [])) != 92 or
            tuple(protocol.get("policies", [])) != POLICIES or
            tuple(protocol.get("config", {}).get("environmental_seeds", [])) != SEEDS or
            protocol.get("hypothesis_tests") is not False or
            protocol.get("noninferiority_or_equivalence") is not False):
        raise ValueError("scientific matrix does not match preregistration")
    if (status.get("complete") is not True or
            status.get("episodes_written") != EPISODES or
            status.get("no_training") is not True or
            status.get("weights_frozen_checked_at_end") is not True or
            not 1 <= status.get("steps_used", 0) <= 6440000 or
            execution.get("no_training") is not True):
        raise ValueError("result incomplete/unverified; no full-campaign inference permitted")
    if (not str(execution.get("github_run_id", "")).isdigit() or
            str(execution.get("github_run_attempt")) != "1" or
            not str(execution.get("reserved_issue_comment_id", "")).isdigit()):
        raise ValueError("official run identity or reservation receipt missing")
    return protocol, execution, status, digests


def read_matrix(root: Path, protocol: dict) -> tuple[dict, dict]:
    scenarios = {s["id"]: s for s in protocol["scenarios"]}
    if len(scenarios) != 92:
        raise ValueError("scenario identifiers not unique")
    rows = {}
    score_values = {}
    with (root / "episodes.csv").open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if (reader.fieldnames is None or
                len(reader.fieldnames) != len(set(reader.fieldnames)) or
                not set(REQUIRED_COLUMNS).issubset(reader.fieldnames)):
            raise ValueError("invalid or duplicate CSV headers")
        for r in reader:
            scenario_id, policy_id = r["scenario_id"], r["policy_id"]
            if scenario_id not in scenarios or policy_id not in POLICIES:
                raise ValueError("unknown scenario or policy")
            try:
                seed = int(r["environment_seed"])
                score = float(r["score"])
                steps = int(r["steps"])
                repeated = int(r["repeated_state_visits"])
            except (TypeError, ValueError) as exc:
                raise ValueError("non-numeric official outcome") from exc
            if (str(seed) != r["environment_seed"] or seed not in SEEDS or
                    not math.isfinite(score) or not score.is_integer() or score < 0 or
                    not 1 <= steps <= 200 or repeated < 0):
                raise ValueError("invalid official per-episode values")
            key = (scenario_id, seed, policy_id)
            if key in rows:
                raise ValueError("duplicate episode key")
            scenario = scenarios[scenario_id]
            for col, val in (
                ("region", scenario["region"]),
                ("food_mode", scenario["food_mode"]),
                ("food_key", scenario["food_key"]),
                ("direction", scenario["direction"]),
                ("head_x", scenario["head"][0]),
                ("head_y", scenario["head"][1]),
                ("food_x", scenario["food"][0]),
                ("food_y", scenario["food"][1]),
            ):
                if r[col] != str(val):
                    raise ValueError(f"{col}: scenario metadata mismatch")
            if r["terminated"] not in ("True", "False") or r["truncated"] not in ("True", "False"):
                raise ValueError("invalid terminal flags")
            terminated, truncated = r["terminated"] == "True", r["truncated"] == "True"
            if terminated == truncated or r["terminal_reason"] not in ("collision", "board_filled", "step_limit"):
                raise ValueError("terminal state not valid")
            if truncated != (r["terminal_reason"] == "step_limit"):
                raise ValueError("truncation and reason disagree")
            if r["first_food_censored"] not in ("True", "False"):
                raise ValueError("invalid first-food censorship")
            censored = r["first_food_censored"] == "True"
            first = r["first_food_step"]
            if (censored and (first != "" or score != 0) or
                    not censored and (not first.isdecimal() or not 1 <= int(first) <= steps or score == 0)):
                raise ValueError("first-food outcome/censoring inconsistent")
            rows[key] = {"score": score, "steps": steps, "censored": censored,
                         "repeated": repeated, "collision": r["terminal_reason"] == "collision",
                         "truncated": truncated}
            score_values[key] = score
    expected = {(s, seed, policy) for s in scenarios for seed in SEEDS for policy in POLICIES}
    if len(rows) != EPISODES or set(rows) != expected:
        raise ValueError("official matrix incomplete or contains unexpected episodes")
    return rows, scenarios


def quantile(data: list[float], q: float) -> float:
    if not data or not 0 <= q <= 1:
        raise ValueError("invalid percentile request")
    items = sorted(data)
    p = (len(items) - 1) * q
    i = math.floor(p)
    j = math.ceil(p)
    return items[i] * (j - p) + items[j] * (p - i) if i != j else items[i]


def primary_contrasts(rows: dict, scenarios: dict,
                      *, resamples: int = 10000, bootstrap_seed: int = 510000) -> list[dict]:
    """Six differences. A bootstrap sample resamples full environmental-seed blocks."""
    if not isinstance(resamples, int) or not 100 <= resamples <= 100000:
        raise ValueError("invalid bootstrap count")
    names = [(f"trained-{s}", f"untrained-{s}") for s in (11, 42, 73)]
    names += [(f"trained-{s}", "random") for s in (11, 42, 73)]
    blocks = []
    scenario_ids = sorted(scenarios)
    for seed in SEEDS:
        blocks.append(tuple(statistics.fmean(
            rows[(scenario, seed, trained)]["score"] -
            rows[(scenario, seed, control)]["score"]
            for scenario in scenario_ids
        ) for trained, control in names))
    rng = random.Random(bootstrap_seed)
    sampled = [[] for _ in names]
    for _ in range(resamples):
        sampled_indices = [rng.randrange(len(SEEDS)) for _ in SEEDS]
        for col in range(len(names)):
            sampled[col].append(statistics.fmean(blocks[i][col] for i in sampled_indices))
    return [
        {"trained": trained, "control": control,
         "mean_score_difference": statistics.fmean(block[col] for block in blocks),
         "ci95_lower": quantile(sampled[col], .025),
         "ci95_upper": quantile(sampled[col], .975),
         "units": "foods per episode",
         "bootstrap": "paired full-environmental-seed-block percentile",
         "resamples": resamples, "seed": bootstrap_seed,
         "checkpoints_independent": 3}
        for col, (trained, control) in enumerate(names)
    ]


def scenario_descriptives(rows: dict, scenarios: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """552 matched policy differences, 216 eligible difficult-vs-center contrasts."""
    results = []
    difficult = []
    diag = []
    center_lookup = {
        (s["direction"], s["food_mode"], s["food_key"]): sid
        for sid, s in scenarios.items() if s["region"] == "center"
    }
    for sid, scenario in sorted(scenarios.items()):
        for trained_seed in (11, 42, 73):
            trained = f"trained-{trained_seed}"
            per_trained = [rows[(sid, seed, trained)] for seed in SEEDS]
            diag.append({
                "scenario_id": sid, "region": scenario["region"],
                "food_mode": scenario["food_mode"], "policy": trained,
                "mean_score": statistics.fmean(x["score"] for x in per_trained),
                "collected_at_least_one_rate": sum(not x["censored"] for x in per_trained)/len(SEEDS),
                "first_food_censored": sum(x["censored"] for x in per_trained),
                "collision_rate": sum(x["collision"] for x in per_trained)/len(SEEDS),
                "step_limit_rate": sum(x["truncated"] for x in per_trained)/len(SEEDS),
                "mean_repeated_full_state_visits": statistics.fmean(x["repeated"] for x in per_trained),
                "distinct_scores": len({x["score"] for x in per_trained}),
                "NOTE": "distinct_scores is NOT distinct trajectories",
            })
            for control in (f"untrained-{trained_seed}", "random"):
                results.append({
                    "scenario_id": sid, "region": scenario["region"],
                    "food_mode": scenario["food_mode"], "trained": trained,
                    "control": control, "n_paired_seeds": len(SEEDS),
                    "mean_score_difference": statistics.fmean(
                        rows[(sid, seed, trained)]["score"] -
                        rows[(sid, seed, control)]["score"] for seed in SEEDS),
                })
            if scenario["region"] != "center":
                center_sid = center_lookup.get((scenario["direction"],
                                                scenario["food_mode"],
                                                scenario["food_key"]))
                if center_sid is None:
                    raise ValueError("missing matched center control for a difficult scenario")
                difficult.append({
                    "difficult_scenario": sid, "center_scenario": center_sid,
                    "region": scenario["region"], "food_mode": scenario["food_mode"],
                    "trained": trained, "n_paired_seeds": len(SEEDS),
                    "mean_score_difference_difficult_minus_center": statistics.fmean(
                        rows[(sid, seed, trained)]["score"] -
                        rows[(center_sid, seed, trained)]["score"]
                        for seed in SEEDS),
                })
    if len(results) != 552 or len(difficult) != 216 or len(diag) != 276:
        raise ValueError("preregistered secondary contrast count mismatch")
    return results, difficult, diag


def report(source: Path, output: Path) -> dict:
    protocol, execution, status, hashes = check_source(source)
    rows, scenarios = read_matrix(source, protocol)
    estimates = primary_contrasts(rows, scenarios)
    scenario_pairs, center_pairs, diagnostics = scenario_descriptives(rows, scenarios)
    payload = {
        "label": "NNA-FECHAMENTO-001 OFFLINE, FULL MATRIX ONLY",
        "scope": "descriptive inference conditional on 3 fixed checkpoints; no general transfer claim",
        "source_sha256": hashes,
        "protocol_sha256": APPROVED_PROTOCOL_SHA,
        "evaluator_sha": APPROVED_EVALUATOR_SHA,
        "run_id": execution["github_run_id"],
        "run_attempt": execution["github_run_attempt"],
        "unique_episodes": len(rows),
        "primary_endpoint": "score",
        "primary_contrasts": estimates,
        "secondary_scenario_pairs_count": len(scenario_pairs),
        "secondary_center_pairs_count": len(center_pairs),
        "no_hypothesis_tests_or_p_values": True,
        "no_noninferiority_or_equivalence": True,
        "note": ("All 50 seeds within a scenario share forced initial food. "
                 "Observed episode-summary diversity does not prove unique action "
                 "trajectories. First-food failures are censored. Same seeds across "
                 "scenarios do not guarantee identical future food positions."),
    }
    if output.exists():
        raise FileExistsError("offline analysis destination must be a new directory")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "analysis.json").open("x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    for name, data in (
        ("policy-vs-controls.csv", scenario_pairs),
        ("difficult-vs-center.csv", center_pairs),
        ("trained-diagnostics.csv", diagnostics),
    ):
        with (output / name).open("x", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    with (output / "SHA256SUMS").open("x", encoding="utf-8") as f:
        for name in ("analysis.json", "policy-vs-controls.csv",
                     "difficult-vs-center.csv", "trained-diagnostics.csv"):
            f.write(f"{_sha(output / name)}  {name}\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="EXISTING official Phase 5 extracted artifact directory")
    parser.add_argument("--out", type=Path, required=True,
                        help="NEW directory for OFFLINE reports only")
    args = parser.parse_args()
    result = report(args.source, args.out)
    print(json.dumps({"offline_analysis": True, "unique_episodes": result["unique_episodes"],
                      "run_id": result["run_id"],
                      "output": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
