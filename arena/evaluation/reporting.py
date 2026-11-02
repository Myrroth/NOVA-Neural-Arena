"""Descriptive reports from verified existing CSVs; never runs games or models."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from io import StringIO
import json
import math
from pathlib import Path
import platform
import re
from statistics import mean, median, stdev

from arena.shared.integrity import (
    read_verified_bytes, safe_relative_path, sha256_file,
    validate_sha256, verify_checksum_manifest, write_new_json,
)

REQUIRED = {"policy_id", "environment_seed", "score", "reward", "steps",
            "terminated", "truncated", "terminal_reason"}


def _integer(value: str, name: str, *, minimum: int = 0) -> int:
    if not re.fullmatch(r"-?\d+", value or "") or int(value) < minimum:
        raise ValueError(f"invalid integer: {name}")
    return int(value)


def _number(value: str, name: str) -> float:
    try:
        number = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"invalid numeric value: {name}") from None
    if not math.isfinite(number):
        raise ValueError(f"nonfinite value: {name}")
    return number


def _boolean(value: str, name: str) -> bool:
    if value not in ("True", "False", "true", "false"):
        raise ValueError(f"invalid boolean: {name}")
    return value.lower() == "true"


def _scenario(row: dict) -> str:
    if "scenario_id" in row:
        if not row["scenario_id"].strip():
            raise ValueError("empty scenario_id")
        return row["scenario_id"]
    fields = ("board_width", "board_height", "head_x", "head_y", "initial_direction")
    return "|".join(f"{key}={row[key]}" for key in fields if key in row) or "unspecified"


def load_rows(source: str | Path, *, expected_csv_sha256: str | None = None) -> tuple[list[dict], dict]:
    """Verify every listed source member, then parse the exact verified CSV bytes.

    An internal manifest detects corruption; independent archive/catalog hashes
    are still required to establish scientific authenticity.
    """
    source = Path(source)
    members = verify_checksum_manifest(source)
    if "episodes.csv" not in members:
        raise ValueError("episodes.csv is not authenticated by SHA256SUMS")
    if expected_csv_sha256 is not None and validate_sha256(expected_csv_sha256) != members["episodes.csv"]:
        raise ValueError("CSV differs from independently expected SHA-256")
    data = read_verified_bytes(safe_relative_path(source, "episodes.csv"),
                               expected_sha256=members["episodes.csv"])
    reader = csv.DictReader(StringIO(data.decode("utf-8-sig"), newline=""))
    fields = reader.fieldnames or []
    if len(set(fields)) != len(fields) or not REQUIRED.issubset(fields):
        raise ValueError("missing or duplicate CSV schema fields")
    for group in (("board_width", "board_height"), ("head_x", "head_y", "initial_direction")):
        if set(fields).intersection(group) and not set(group).issubset(fields):
            # Synthetic Phase 5 has scenario_id and uses 'direction'.
            if not (group[0] == "head_x" and "scenario_id" in fields and "direction" in fields
                    and {"head_x", "head_y"}.issubset(fields)):
                raise ValueError("incomplete scenario geometry fields")
    rows, seen = [], set()
    for raw in reader:
        if None in raw or any(value is None for value in raw.values()):
            raise ValueError("malformed CSV row length")
        scenario, policy = _scenario(raw), raw["policy_id"].strip()
        if not policy:
            raise ValueError("empty policy_id")
        seed = _integer(raw["environment_seed"], "environment_seed")
        key = scenario, policy, seed
        if key in seen:
            raise ValueError("duplicate scenario/policy/environment_seed key")
        seen.add(key)
        steps = _integer(raw["steps"], "steps", minimum=1)
        score = _integer(raw["score"], "score")
        if score > steps:
            raise ValueError("score exceeds steps")
        for name in ("board_width", "board_height", "max_steps"):
            if name in raw:
                value = _integer(raw[name], name, minimum=1)
                if name == "max_steps" and steps > value:
                    raise ValueError("steps exceed max_steps")
        for name, dimension in (("head_x", "board_width"), ("head_y", "board_height")):
            if name in raw:
                coordinate = _integer(raw[name], name)
                if dimension in raw and coordinate >= int(raw[dimension]):
                    raise ValueError("head is out of bounds")
        for name in ("initial_direction", "direction"):
            if name in raw and raw[name] not in ("east", "south", "west", "north"):
                raise ValueError("invalid initial direction")
        terminated = _boolean(raw["terminated"], "terminated")
        truncated = _boolean(raw["truncated"], "truncated")
        reason = raw["terminal_reason"]
        if (terminated == truncated or reason not in ("collision", "board_filled", "step_limit")
                or (reason == "step_limit") != truncated):
            raise ValueError("inconsistent terminal outcome")
        if reason == "step_limit" and "max_steps" in raw and steps != int(raw["max_steps"]):
            raise ValueError("step-limit outcome differs from max_steps")
        efficiency = 100.0 * score / steps
        if "food_per_100_steps" in raw and abs(
            _number(raw["food_per_100_steps"], "food_per_100_steps") - efficiency
        ) > 0.0000011:
            raise ValueError("efficiency differs from score/steps")
        first, censored, repeated = None, None, None
        if "first_food_step" in raw:
            first = _integer(raw["first_food_step"], "first_food_step", minimum=1) if raw["first_food_step"] else None
            if (first is None) != (score == 0) or (first is not None and first > steps):
                raise ValueError("first food is inconsistent with score/steps")
            censored = first is None
        if "first_food_censored" in raw and _boolean(raw["first_food_censored"], "first_food_censored") != censored:
            raise ValueError("first-food censoring differs from first_food_step")
        if "repeated_state_visits" in raw:
            repeated = _integer(raw["repeated_state_visits"], "repeated_state_visits")
            if repeated > steps:
                raise ValueError("repeated-state count exceeds steps")
        rows.append({"scenario_id": scenario, "policy_id": policy, "environment_seed": seed,
                     "score": score, "reward": _number(raw["reward"], "reward"), "steps": steps,
                     "food_per_100_steps": efficiency, "first_food_step": first,
                     "first_food_censored": censored, "repeated_state_visits": repeated})
    if not rows:
        raise ValueError("empty episode table")
    return rows, members


def _stats(values: list[float]) -> dict:
    return {"mean": mean(values), "median": median(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values)}


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["scenario_id"], row["policy_id"]].append(row)
    result = []
    for (scenario, policy), group in sorted(groups.items()):
        first = [r["first_food_step"] for r in group if r["first_food_step"] is not None]
        result.append({
            "scenario_id": scenario, "policy_id": policy, "n": len(group),
            **{name: _stats([r[name] for r in group])
               for name in ("score", "reward", "steps", "food_per_100_steps")},
            "at_least_one_food_rate": mean(r["score"] > 0 for r in group),
            "first_food_step_mean_if_observed": mean(first) if first else None,
            "first_food_censoring_rate": (mean(r["first_food_censored"] for r in group)
                if all(r["first_food_censored"] is not None for r in group) else None),
            "repeated_state_episode_rate": (mean(r["repeated_state_visits"] > 0 for r in group)
                if all(r["repeated_state_visits"] is not None for r in group) else None),
        })
    return result


def _md(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("`", "\\`").replace("\n", " ").replace("\r", " ")


def create_report(source: str | Path, output: str | Path, *, plots: bool = False,
                  expected_csv_sha256: str | None = None) -> dict:
    source, output = Path(source), Path(output)
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError("report destination must be outside the source directory")
    if output.exists():
        raise FileExistsError(output)
    rows, members = load_rows(source, expected_csv_sha256=expected_csv_sha256)
    summary = summarize(rows)
    if plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
        except ImportError:
            raise RuntimeError("Optional plots require: python -m pip install matplotlib") from None
    safe_relative_path(output.parent, output.name)
    output.mkdir(parents=True, exist_ok=False)
    write_new_json(output / "summary.json", {"episodes": len(rows), "groups": summary})
    text = ["# Descriptive evaluation report", "",
            "Derived from existing verified CSV data. No games, model loading, or inferential tests.", "",
            "Missing instrumentation is reported as unavailable (n/a), never as zero.", "",
            "| Scenario | Policy | n | Score mean / median / SD | Reward mean | Steps mean | Food/100 steps | >=1 food | First-food censoring | Repeated states |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group in summary:
        rate = lambda name: "n/a" if group[name] is None else f"{100 * group[name]:.2f}%"
        score = group["score"]
        text.append(f"| {_md(group['scenario_id'])} | {_md(group['policy_id'])} | {group['n']} | "
                    f"{score['mean']:.3f} / {score['median']:.3f} / {score['std']:.3f} | "
                    f"{group['reward']['mean']:.3f} | {group['steps']['mean']:.3f} | "
                    f"{group['food_per_100_steps']['mean']:.3f} | {rate('at_least_one_food_rate')} | "
                    f"{rate('first_food_censoring_rate')} | {rate('repeated_state_episode_rate')} |")
    text.extend(["", "SD is sample standard deviation (0 for a single observation).",
                 "Food efficiency is averaged per episode, not a ratio of aggregate means.",
                 "Repeated states are observations of recurrence, not proof of a deterministic cycle.",
                 "Statistics are conditional on the evaluated checkpoints and correlated scenarios."])
    with (output / "report.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(text) + "\n")
    products = ["summary.json", "report.md", "provenance.json"]
    if plots:
        by_policy = defaultdict(list)
        for row in rows:
            by_policy[row["policy_id"]].append(row["score"])
        policies = sorted(by_policy)
        figure, axis = plt.subplots(figsize=(max(8, min(16, len(policies))), 5))
        axis.bar(range(len(policies)), [mean(by_policy[p]) for p in policies])
        axis.set_xticks(range(len(policies)), policies, rotation=35, ha="right")
        axis.set_ylabel("Mean score (foods)")
        axis.set_title("Mean score by policy across observed conditions")
        figure.text(.5, .01, f"Descriptive aggregation of {len(rows)} episodes / {len(summary)} scenario-policy groups; no inferential intervals.", ha="center", fontsize=8)
        figure.tight_layout(rect=(0, .06, 1, 1))
        figure.savefig(output / "score-means.png", dpi=150)
        with matplotlib.rc_context({"svg.fonttype": "none"}):
            figure.savefig(output / "score-means.svg")
        plt.close(figure)
        products.extend(["score-means.png", "score-means.svg"])
    write_new_json(output / "provenance.json", {
        "operation": "descriptive_reanalysis_existing_csv", "no_new_episodes": True,
        "no_model_deserialization": True, "python": platform.python_version(),
        "source_members_sha256": members, "reporter_sha256": sha256_file(Path(__file__)),
        "integrity_module_sha256": sha256_file(Path(__file__).parents[1] / "shared" / "integrity.py"),
        "source_authenticity_caveat": "Internal SHA256SUMS alone does not establish independent authenticity",
        "independently_expected_csv_sha256": expected_csv_sha256,
        "plots": plots,
        "matplotlib": ({"version": matplotlib.__version__, "backend": matplotlib.get_backend()}
                       if plots else None),
    })
    with (output / "SHA256SUMS").open("x", encoding="utf-8", newline="\n") as handle:
        handle.writelines(f"{sha256_file(output / name)}  {name}\n" for name in products)
    verify_checksum_manifest(output, expected_files=set(products))
    return {"episodes": len(rows), "groups": len(summary), "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--expected-csv-sha256")
    args = parser.parse_args()
    print(json.dumps(create_report(args.source, args.out, plots=args.plots,
                                  expected_csv_sha256=args.expected_csv_sha256), indent=2))


if __name__ == "__main__":
    main()
