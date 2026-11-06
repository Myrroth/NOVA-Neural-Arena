"""Pure synthetic/offline tests. DO NOT import Snake, Torch, or run episodes."""
import csv
import hashlib
import json
from pathlib import Path
import pytest

from arena.analysis.phase5_offline import (
    APPROVED_EVALUATOR_SHA, APPROVED_PROTOCOL_SHA, APPROVED_PROTOCOL_VERSION,
    OFFICIAL_INPUTS, POLICIES, SEEDS, EPISODES,
    check_source, quantile, primary_contrasts, report, scenario_descriptives,
)


def _scenarios():
    scenarios = {}
    for direction in ("east", "south", "west", "north"):
        for mode, key in (("physical", "4,4"), ("physical", "6,6"),
                          ("relative", "1,0"), ("relative", "0,1"), ("relative", "0,-1")):
            name = f"center-{direction}-{mode}-{key}"
            scenarios[name] = {"id": name, "head": [5, 5], "direction": direction,
                               "food": [4, 4], "region": "center",
                               "food_mode": mode, "food_key": key}
    centers = list(scenarios.values())
    for i in range(72):
        center = centers[i % len(centers)]
        name = f"challenging-{i:02d}"
        scenarios[name] = {
            **center, "id": name, "region": "wall" if i < 40 else "corner",
        }
    assert len(scenarios) == 92
    return scenarios


def _scores(scenarios):
    scores = {"trained-11": 5, "trained-42": 10, "trained-73": 8,
              "untrained-11": 0, "untrained-42": 0, "untrained-73": 0,
              "random": 1}
    return {
        (sid, seed, policy): {
            "score": float(scores[policy]), "steps": 20,
            "censored": scores[policy] == 0, "repeated": 0,
            "collision": True, "truncated": False,
        }
        for sid in scenarios for seed in SEEDS for policy in POLICIES
    }


def test_interpolated_quantiles():
    assert quantile([0, 10], .025) == .25
    assert quantile([10, 0], .975) == 9.75
    with pytest.raises(ValueError):
        quantile([], .5)


def test_six_seed_block_bootstrap_primary_contrasts():
    rows = _scores({"s1": {}, "s2": {}})
    contrasts = primary_contrasts(rows, {"s1": {}, "s2": {}},
                                  resamples=100, bootstrap_seed=510000)
    assert len(contrasts) == 6
    assert [(x["trained"], x["control"]) for x in contrasts] == [
        ("trained-11", "untrained-11"), ("trained-42", "untrained-42"),
        ("trained-73", "untrained-73"), ("trained-11", "random"),
        ("trained-42", "random"), ("trained-73", "random"),
    ]
    assert [x["mean_score_difference"] for x in contrasts] == [5, 10, 8, 4, 9, 7]
    assert all(x["ci95_lower"] == x["ci95_upper"] == x["mean_score_difference"]
               for x in contrasts)


def test_secondary_contrast_counts_and_center_matching():
    scenarios = _scenarios()
    rows = _scores(scenarios)
    policy_pairs, difficult_pairs, diagnostics = scenario_descriptives(rows, scenarios)
    assert len(policy_pairs) == 552
    assert len(difficult_pairs) == 216
    assert len(diagnostics) == 276
    assert all(x["mean_score_difference"] == 5 for x in policy_pairs
               if x["trained"] == "trained-11" and x["control"] == "untrained-11")
    assert all(x["mean_score_difference_difficult_minus_center"] == 0
               for x in difficult_pairs)
    assert all(x["NOTE"].endswith("NOT distinct trajectories") for x in diagnostics)


def _synthetic_full_fixture(root: Path) -> str:
    """Fabricated complete 32200-row dataset. Absolutely NO games are played."""
    root.mkdir()
    scenarios = _scenarios()
    scores = {"trained-11": 5, "trained-42": 10, "trained-73": 8,
              "untrained-11": 0, "untrained-42": 0, "untrained-73": 0,
              "random": 1}
    protocol = {
        "approved_protocol_sha256": APPROVED_PROTOCOL_SHA,
        "protocol_version": APPROVED_PROTOCOL_VERSION,
        "total_episodes": EPISODES,
        "max_total_steps": 6440000,
        "scenarios": list(scenarios.values()),
        "policies": list(POLICIES),
        "config": {"environmental_seeds": list(SEEDS)},
        "hypothesis_tests": False, "noninferiority_or_equivalence": False,
    }
    execution = {
        "approved_protocol_sha": APPROVED_PROTOCOL_SHA,
        "evaluator_sha": APPROVED_EVALUATOR_SHA,
        "github_run_id": "123456",
        "github_run_attempt": "1",
        "reserved_issue_comment_id": "987654",
        "no_training": True,
    }
    status = {
        "complete": True, "episodes_written": EPISODES, "steps_used": EPISODES * 20,
        "no_training": True, "weights_frozen_checked_at_end": True,
    }
    for filename, value in (("protocol.json", protocol), ("execution.json", execution),
                            ("status.json", status)):
        (root / filename).write_text(json.dumps(value) + "\n", encoding="utf-8")
    (root / "pip-freeze.txt").write_text("synthetic-only-no-packages\n", encoding="utf-8")
    header = [
        "scenario_id", "region", "food_mode", "food_key", "head_x", "head_y",
        "direction", "food_x", "food_y", "environment_seed", "policy_id",
        "score", "steps", "terminated", "truncated", "terminal_reason",
        "first_food_step", "first_food_censored", "repeated_state_visits",
    ]
    with (root / "episodes.csv").open("x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for sid, scenario in scenarios.items():
            for seed in SEEDS:
                for policy in POLICIES:
                    score = scores[policy]
                    w.writerow({
                        "scenario_id": sid, "region": scenario["region"],
                        "food_mode": scenario["food_mode"],
                        "food_key": scenario["food_key"],
                        "head_x": scenario["head"][0], "head_y": scenario["head"][1],
                        "direction": scenario["direction"],
                        "food_x": scenario["food"][0], "food_y": scenario["food"][1],
                        "environment_seed": seed, "policy_id": policy,
                        "score": score, "steps": 20, "terminated": True,
                        "truncated": False, "terminal_reason": "collision",
                        "first_food_step": "1" if score else "",
                        "first_food_censored": not bool(score),
                        "repeated_state_visits": 0,
                    })
    digest = lambda name: hashlib.sha256((root / name).read_bytes()).hexdigest()
    (root / "SHA256SUMS").write_text(
        "".join(f"{digest(name)}  {name}\n" for name in OFFICIAL_INPUTS),
        encoding="utf-8",
    )
    return digest("episodes.csv")


def test_full_synthetic_fixture_offline_no_games(tmp_path):
    source, output = tmp_path / "source-fixture", tmp_path / "analysis-output"
    expected = _synthetic_full_fixture(source)
    manifest = report(source, output, expected_csv_sha256=expected)
    assert manifest["unique_episodes"] == 32200
    assert manifest["secondary_scenario_pairs_count"] == 552
    assert manifest["secondary_center_pairs_count"] == 216
    assert [v["mean_score_difference"] for v in manifest["primary_contrasts"]] == [
        5, 10, 8, 4, 9, 7
    ]
    assert manifest["no_hypothesis_tests_or_p_values"] is True
    assert "NNA-FECHAMENTO-001 OFFLINE" in manifest["label"]
    with pytest.raises(FileExistsError):
        report(source, output, expected_csv_sha256=expected)
    with pytest.raises(ValueError, match="independent"):
        check_source(source, expected_csv_sha256="0" * 64)


def test_input_tampering_is_detected(tmp_path):
    source = tmp_path / "source-fixture"
    expected = _synthetic_full_fixture(source)
    with (source / "episodes.csv").open("a", encoding="utf-8") as f:
        f.write("injected row\n")
    with pytest.raises(ValueError, match="integrity"):
        check_source(source, expected_csv_sha256=expected)


def test_no_checkpoints_imported_or_simulation_started():
    source = (Path(__file__).resolve().parents[1] /
              "arena/analysis/phase5_offline.py").read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import gymnasium" not in source
    assert "SnakeEnv" not in source
    assert "run_episode(" not in source
