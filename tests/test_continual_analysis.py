"""Synthetic arithmetic/validation fixtures ONLY: no games or trained models run."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from arena.continual import analysis as a
from arena.continual import protocol as p


def test_analyzer_import_does_not_load_models_environments_or_checkpoint_code():
    code = """
import importlib.abc
import sys
class BlockML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'gymnasium') or fullname in (
            'arena.continual.evaluation', 'arena.continual.trainer',
            'arena.continual.persistence', 'arena.universal.envs'):
            raise RuntimeError('Analyzer imported model/environment/checkpoint code: '+fullname)
sys.meta_path.insert(0, BlockML())
from arena.continual import analysis
assert analysis.analyze([])['scientific_execution_performed_by_analyzer'] is False
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


def fixture_rows():
    """Invented numbers exercise gates; they are explicitly not experiment data."""
    rows = []
    for index, cell in enumerate(p.evaluation_matrix()):
        baseline = cell.policy in ("random", "untrained")
        learned = not baseline
        rows.append(
            {
                **vars(cell),
                "evaluation_schema": a.EVALUATION_SCHEMA,
                "episode": index + 1,
                "steps": 100,
                "return": float(learned),
                "score": float(learned),
                "wall_collision": 0,
                "body_collision": 0,
                "foods": 4.0 if learned else 0.0,
                "points": 3.0 if learned else 0.0,
                "conceded": 0.0 if learned else 3.0,
                "hits": 8.0 if learned else 2.0,
                "incoming": 10.0,
                "prey_eaten": 1.0 if learned else 0.0,
                "eaten": int(baseline),
                "large_near_steps": 20,
                "small_near_steps": 20,
                "pipes_passed": 3.0 if learned else 0.0,
                "flaps": 10,
                "narrow_pipes_passed": 1.0 if learned else 0.0,
                "wide_pipes_passed": 2.0 if learned else 0.0,
                "progress_normalized": 0.75 if learned else 0.05,
                "lap_completed": int(learned),
                "distance": 100.0,
                "collision": int(baseline),
                "spawn_context": str(cell.environment_seed % 2),
                "gap_stratum": str(cell.environment_seed % 2),
            }
        )
    return rows


def test_full_matrix_validates_and_duplicates_or_missing_fail():
    rows = fixture_rows()
    _, report = a.validate_episode_matrix(rows)
    assert report["status"] == a.PASS
    assert report["steps"] == 173_500
    _, report = a.validate_episode_matrix(rows[:-1])
    assert report["status"] == a.FAIL and report["missing_rows"] == 1
    _, report = a.validate_episode_matrix(rows + [rows[0]])
    assert report["status"] == a.FAIL and "duplicate" in report["errors"][0]


@pytest.mark.parametrize(
    "field,value",
    [("steps", 401), ("steps", 0), ("return", "nan"), ("model_seed", 12345)],
)
def test_invalid_final_data_fail_closed(field, value):
    rows = fixture_rows()
    rows[0][field] = value
    _, report = a.validate_episode_matrix(rows)
    assert report["status"] == a.FAIL


def test_three_models_not_sixty_independent_agents_and_bootstrap_repeatable():
    treatment = np.ones((3, 20, 2))
    treatment[..., 0] = np.array([2.0, 3.0, 4.0])[:, None]
    control = np.ones((3, 20, 2))
    control[..., 0] = 0.0
    result = a.paired_contrast(treatment, control, margin=1.0, label="synthetic-only")
    assert result == a.paired_contrast(
        treatment, control, margin=1.0, label="synthetic-only"
    )
    assert result["status"] == a.PASS
    assert result["independent_models"] == 3
    assert result["gain"] == 3.0
    assert result["model_seed_gains"] == [2.0, 3.0, 4.0]


def test_pool_opportunities_not_unweighted_mean_of_episode_hit_rates():
    values = np.array([[[1.0, 1.0], [0.0, 9.0]]] * 3)
    assert a._evaluate(values).tolist() == [0.1, 0.1, 0.1]
    values[..., 1] = 0.0
    with pytest.raises(a.MissingEvidence, match="denominator"):
        a._evaluate(values)


def test_one_good_training_seed_cannot_win_by_many_episodes():
    t, c = np.ones((3, 20, 2)), np.ones((3, 20, 2))
    t[..., 0] = np.array([100.0, -1.0, -1.0])[:, None]
    c[..., 0] = 0.0
    result = a.paired_contrast(t, c, margin=1.0, label="one-outlier-fixture")
    assert result["gain"] > 1 and result["favorable_models"] == 1
    assert result["status"] == a.FAIL


def test_impl_and_score_only_never_prove_complete_mission():
    result = a.analyze([])
    assert result["overall"] == "NOT_CONCLUDED"
    assert result["execution_authorized"] is False
    assert result["requirements"]["9"]["status"] == a.MISSING
    assert result["requirements"]["8"]["status"] == a.MISSING


def test_complete_synthetic_score_matrix_still_requires_attribution_and_real_receipts():
    result = a.analyze(fixture_rows())
    assert result["matrix"]["status"] == a.PASS
    assert result["requirements"]["1"]["status"] == a.PASS
    assert result["requirements"]["2"]["status"] == a.PASS
    assert result["requirements"]["4"]["status"] == a.PASS
    assert result["requirements"]["3"]["status"] == a.MISSING
    assert result["requirements"]["6"]["status"] == a.MISSING
    assert result["overall"] == "NOT_CONCLUDED"
    assert (
        result["after_adaptation_descriptive_metrics"]["snake"]["wall_collision"][
            "mean"
        ]
        == 0
    )


def test_missing_lap_telemetry_not_fabricated_from_distance():
    indexed, _ = a.validate_episode_matrix(fixture_rows())
    cell = p.EvaluationCell(
        "after_adaptation", "shared", "racing", 4103, p.TEST_SEEDS[0]
    )
    del indexed[cell]["progress_normalized"]
    with pytest.raises(a.MissingEvidence, match="progress_normalized"):
        a._contrast(indexed, "racing", "progress_normalized", 0.15)


def test_action_attribution_cannot_cherrypick_one_state_per_seed():
    indexed, _ = a.validate_episode_matrix(fixture_rows())
    pairs = [
        {
            "model_seed": seed,
            "game": game,
            "environment_seed": env_seed,
            "step": 1,
            "stage": "after_adaptation",
            "policy": "shared",
            "action_default": 0,
            "action_world_off": 1,
            "action_gru_reset": 0,
            "action_pixels_hidden": 0,
            "planning_weight": 0.2,
            "planning_weight_world_off": 0.0,
            "planning_weight_gru_reset": 0.2,
            "planning_weight_pixels_hidden": 0.2,
            "context_reset": True,
            "context_reset_gru_reset": True,
            "context_reset_pixels_hidden": True,
            "next_context_boundary": False,
        }
        for game in p.GAMES
        for seed in p.MODEL_SEEDS
        for env_seed in p.TEST_SEEDS
    ]
    with pytest.raises(a.MissingEvidence, match="every shared rollout step"):
        a._action_attribution(indexed, pairs, "world_off", 0.01)


def test_world_holdout_reuses_final_shared_transitions_without_extra_episode_namespace():
    rows = fixture_rows()
    for row in rows:
        if row["stage"] == "after_adaptation" and row["policy"] == "shared":
            row["steps"] = 1
    indexed, _ = a.validate_episode_matrix(rows)
    predictions = []
    for cell in p.evaluation_matrix():
        if cell.stage == "after_adaptation" and cell.policy == "shared":
            predictions.append(
                {
                    **vars(cell),
                    "transition_id": str(cell),
                    "step": 1,
                    "baseline_source": "training",
                    "baseline_support": 100,
                    "target_mode": a.TARGET_MODE,
                    "next_context_boundary": False,
                    "context_reset": True,
                    "planning_weight": 0.0,
                    "baseline_reward_mean": 0.0,
                    "baseline_terminal_rate": 0.1,
                    "reward_target_clipped": 0.0,
                    "reward_prediction": 0.1,
                    "terminal_probability": 0.2,
                    "truncated": 0,
                    "latent_mse_model": 0.01,
                    "latent_mse_copy": 0.1,
                    "reward_mae_model": 0.1,
                    "reward_mae_baseline": 0.2,
                    "terminal_brier_model": 0.1,
                    "terminal_brier_baseline": 0.2,
                    "terminated": 0,
                }
            )
    assert a._world_predictions(predictions, indexed)["status"] == a.PASS
    with pytest.raises(a.MissingEvidence, match="all existing"):
        a._world_predictions(predictions[:-1], indexed)
    predictions[0]["baseline_source"] = "evaluation"
    with pytest.raises(a.MissingEvidence, match="training provenance"):
        a._world_predictions(predictions, indexed)


def test_relative_retention_not_estimable_when_source_never_learned():
    rows = fixture_rows()
    for row in rows:
        if row["stage"] == "before_adaptation" and row["game"] == "snake":
            row["foods"] = 0.0
    indexed, _ = a.validate_episode_matrix(rows)
    retention = a._retention(indexed)
    assert retention["components"]["snake"]["status"] == a.MISSING


def test_auc_uses_fixed_target_steps_not_episode_count():
    rows = []
    for seed in p.MODEL_SEEDS:
        for stage, scores in (("adaptation", (0.5, 0.8)), ("from_scratch", (0.1, 0.2))):
            for step, score in zip((10_000, 60_000), scores):
                rows.append(
                    {
                        "model_seed": seed,
                        "stage": stage,
                        "game": "racing",
                        "training_step": step,
                        "progress_normalized": score,
                        "episode_complete": True,
                    }
                )
    report = a._training_auc(rows)
    assert report["status"] == a.PASS
    assert report["mean_relative_gain"] == pytest.approx(
        (0.5 / 6 + 0.8 * 5 / 6) / (0.1 / 6 + 0.2 * 5 / 6) - 1
    )
    with pytest.raises(a.MissingEvidence, match="60,000"):
        a._training_auc(rows[:-1])


def test_claimed_engineering_pass_needs_actual_untampered_receipt(tmp_path):
    evidence = {
        "shared_policy": {
            "status": "PASS",
            "receipts": [],
            "mission": p.MISSION,
            "protocol_sha256": p.protocol_sha256(),
            "source_revision": "a" * 40,
        }
    }
    with pytest.raises(a.MissingEvidence):
        a._receipt(evidence, "shared_policy", tmp_path)
    receipt = tmp_path / "synthetic-receipt.txt"
    receipt.write_text("TEST FIXTURE, NOT A REAL AUDIT", encoding="utf-8")
    evidence["shared_policy"]["receipts"] = [
        {
            "path": receipt.name,
            "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        }
    ]
    assert a._receipt(evidence, "shared_policy", tmp_path)["status"] == a.PASS
    receipt.write_text("tampered", encoding="utf-8")
    with pytest.raises(a.MissingEvidence, match="hash mismatch"):
        a._receipt(evidence, "shared_policy", tmp_path)


def test_analysis_directory_empty_is_offline_and_json_finite(tmp_path):
    result = a.analyze_directory(tmp_path)
    assert json.loads(json.dumps(result, allow_nan=False))["overall"] == "NOT_CONCLUDED"


def test_analysis_directory_default_preserves_local_training_curve_input(
    tmp_path, monkeypatch
):
    raw = b"synthetic_fixture\nlocal-training-input\n"
    curve = tmp_path / "training_curve.csv"
    curve.write_bytes(raw)
    captured = {}

    def inspect_only(rows, **kwargs):
        captured.update(kwargs)
        return {"synthetic_fixture": True}

    monkeypatch.setattr(a, "analyze", inspect_only)
    assert a.analyze_directory(tmp_path) == {"synthetic_fixture": True}
    assert captured["training_curves"] == [
        {"synthetic_fixture": "local-training-input"}
    ]
    assert curve.read_bytes() == raw


def test_analysis_directory_accepts_explicit_external_curve_with_exact_byte_hash(
    tmp_path, monkeypatch
):
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    curve = tmp_path / "training_curve.csv"
    raw = b"synthetic_fixture\r\nexternal-training-input\r\n"
    curve.write_bytes(raw)
    (evaluation / "training_curve.csv").write_text(
        "synthetic_fixture\nunused-local-input\n", encoding="utf-8"
    )
    captured = {}

    def inspect_only(rows, **kwargs):
        assert rows == []
        captured.update(kwargs)
        return {"synthetic_fixture": True}

    monkeypatch.setattr(a, "analyze", inspect_only)
    assert a.analyze_directory(
        evaluation,
        training_curve_path=curve,
        training_curve_sha256=hashlib.sha256(raw).hexdigest(),
    ) == {"synthetic_fixture": True}
    assert captured["training_curves"] == [
        {"synthetic_fixture": "external-training-input"}
    ]
    assert captured["evidence_root"] == evaluation
    assert curve.read_bytes() == raw


@pytest.mark.parametrize(
    "case, message",
    [
        ("path_only", "both path and SHA256"),
        ("hash_only", "both path and SHA256"),
        ("invalid_hash", "valid SHA256"),
        ("different_hash", "SHA256 mismatch"),
        ("missing_file", "missing or unreadable"),
        ("tampered_file", "SHA256 mismatch"),
    ],
)
def test_analysis_directory_rejects_unverified_explicit_curve_before_analysis(
    tmp_path, monkeypatch, case, message
):
    curve = tmp_path / "training_curve.csv"
    raw = b"synthetic_fixture\noriginal-input\n"
    curve.write_bytes(raw)
    arguments = {
        "training_curve_path": curve,
        "training_curve_sha256": hashlib.sha256(raw).hexdigest(),
    }
    if case == "path_only":
        del arguments["training_curve_sha256"]
    elif case == "hash_only":
        del arguments["training_curve_path"]
    elif case == "invalid_hash":
        arguments["training_curve_sha256"] = "g" * 64
    elif case == "different_hash":
        arguments["training_curve_sha256"] = "0" * 64
    elif case == "missing_file":
        arguments["training_curve_path"] = tmp_path / "absent.csv"
    elif case == "tampered_file":
        curve.write_bytes(b"synthetic_fixture\naltered-input\n")
    monkeypatch.setattr(
        a,
        "analyze",
        lambda *args, **kwargs: pytest.fail("unverified input reached analysis"),
    )
    with pytest.raises(a.MissingEvidence, match=message):
        a.analyze_directory(tmp_path, **arguments)


def test_analysis_cli_forwards_explicit_curve_and_hash(tmp_path, monkeypatch, capsys):
    curve, digest = tmp_path / "training_curve.csv", "a" * 64
    captured = {}

    def inspect_only(directory, **kwargs):
        captured.update(directory=directory, **kwargs)
        return {"synthetic_fixture": True}

    monkeypatch.setattr(a, "analyze_directory", inspect_only)
    monkeypatch.setattr(
        "sys.argv",
        [
            "analysis",
            str(tmp_path / "evaluation"),
            "--training-curve",
            str(curve),
            "--training-curve-sha256",
            digest,
        ],
    )
    a.main()
    assert captured == {
        "directory": tmp_path / "evaluation",
        "training_curve_path": curve,
        "training_curve_sha256": digest,
    }
    assert json.loads(capsys.readouterr().out) == {"synthetic_fixture": True}


def short_index(steps=1):
    rows = fixture_rows()
    for row in rows:
        if row["stage"] == "after_adaptation" and row["policy"] == "shared":
            row["steps"] = steps
    indexed, report = a.validate_episode_matrix(rows)
    assert report["status"] == a.PASS
    return indexed


def audit_fields(step):
    boundary = step % a.CONTEXT_LENGTH == 0
    reset = (step - 1) % a.CONTEXT_LENGTH == 0
    return {
        "planning_weight": 0.0 if boundary else 0.2,
        "planning_weight_world_off": 0.0,
        "planning_weight_gru_reset": 0.0 if boundary else 0.2,
        "planning_weight_pixels_hidden": 0.0 if boundary else 0.2,
        "context_reset": reset,
        "context_reset_gru_reset": True,
        "context_reset_pixels_hidden": reset,
        "next_context_boundary": boundary,
    }


def pair_rows(indexed):
    rows = []
    for cell, episode in indexed.items():
        if cell.stage == "after_adaptation" and cell.policy == "shared":
            for step in range(1, int(episode["steps"]) + 1):
                rows.append(
                    {
                        **vars(cell),
                        "step": step,
                        **audit_fields(step),
                        "action_default": 0,
                        "action_world_off": 0,
                        "action_gru_reset": 0,
                        "action_pixels_hidden": 0,
                    }
                )
    return rows


def prediction_rows(indexed):
    rows = []
    for cell, episode in indexed.items():
        if cell.stage == "after_adaptation" and cell.policy == "shared":
            for step in range(1, int(episode["steps"]) + 1):
                rows.append(
                    {
                        **vars(cell),
                        "step": step,
                        "transition_id": str((cell, step)),
                        **audit_fields(step),
                        "target_mode": a.TARGET_MODE,
                        "baseline_source": "training",
                        "baseline_support": 100,
                        "baseline_reward_mean": 0.0,
                        "baseline_terminal_rate": 0.1,
                        "latent_mse_model": 0.01,
                        "latent_mse_copy": 0.1,
                        "reward_mae_model": 0.1,
                        "reward_mae_baseline": 0.2,
                        "terminal_brier_model": 0.1,
                        "terminal_brier_baseline": 0.2,
                        "terminated": 0,
                        "truncated": 0,
                    }
                )
    return rows


def test_old_schema_or_seed_namespace_cannot_be_relabelled_as_v2_evidence():
    rows = fixture_rows()
    del rows[0]["evaluation_schema"]
    _, report = a.validate_episode_matrix(rows)
    assert report["status"] == a.FAIL and "schema" in report["errors"][0]
    rows = fixture_rows()
    rows[0]["environment_seed"] = 920001
    _, report = a.validate_episode_matrix(rows)
    assert report["status"] == a.FAIL and "unexpected" in report["errors"][0]


@pytest.mark.parametrize("value", [1, "yes", "0", None, 0.0])
def test_audit_flags_never_use_python_truthiness(value):
    with pytest.raises(a.MissingEvidence, match="nonboolean"):
        a.boolean({"flag": value}, "flag")


def test_csv_boolean_spellings_are_explicitly_supported():
    assert a.boolean({"flag": "True"}, "flag") is True
    assert a.boolean({"flag": "false"}, "flag") is False


def test_same_state_rows_reject_unexpected_stage_duplicate_or_extra_transition():
    indexed = short_index()
    pairs = pair_rows(indexed)
    assert a._validate_action_pairs(indexed, pairs)["status"] == a.PASS
    for mutation in ("stage", "duplicate", "extra"):
        changed = copy.deepcopy(pairs)
        if mutation == "stage":
            changed[0]["stage"] = "ablation"
        elif mutation == "duplicate":
            changed.append(changed[0])
        else:
            changed[0]["step"] = 2
        with pytest.raises(a.MissingEvidence):
            a._validate_action_pairs(indexed, changed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("planning_weight", float("nan")),
        ("planning_weight", 0.20001),
        ("planning_weight", -0.01),
        ("planning_weight_world_off", 0.1),
        ("context_reset", False),
        ("context_reset_gru_reset", False),
        ("context_reset_pixels_hidden", False),
    ],
)
def test_invalid_same_state_gate_or_context_audit_is_missing_evidence(field, value):
    indexed = short_index()
    pairs = pair_rows(indexed)
    pairs[0][field] = value
    with pytest.raises(a.MissingEvidence):
        a._validate_action_pairs(indexed, pairs)


def test_periodic_next_boundary_forces_all_counterfactual_planning_off():
    indexed = short_index(steps=a.CONTEXT_LENGTH)
    pairs = pair_rows(indexed)
    assert a._validate_action_pairs(indexed, pairs)["status"] == a.PASS
    boundary = next(row for row in pairs if row["next_context_boundary"])
    boundary["planning_weight_gru_reset"] = 0.2
    with pytest.raises(a.MissingEvidence, match="boundary"):
        a._validate_action_pairs(indexed, pairs)


def test_zero_gate_is_not_world_influence_or_forecast_success():
    indexed = short_index()
    pairs = pair_rows(indexed)
    for row in pairs:
        row.update(
            planning_weight=0.0,
            planning_weight_gru_reset=0.0,
            planning_weight_pixels_hidden=0.0,
        )
    report = a._action_attribution(indexed, pairs, "world_off", 0.01)
    assert report["status"] == a.FAIL and report["games_above_threshold"] == 0
    predictions = prediction_rows(indexed)
    for row in predictions:
        row.update(planning_weight=0.0, latent_mse_model=0.2)
    report = a._world_predictions(predictions, indexed)
    assert report["status"] == a.FAIL
    assert all(game["status"] == a.FAIL for game in report["components"].values())


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_mode", "reset_control_state"),
        ("target_mode", None),
        ("next_context_boundary", 0),
        ("planning_weight", float("inf")),
        ("baseline_support", 0),
        ("baseline_source", "final_evaluation"),
        ("baseline_reward_mean", 2.0),
        ("baseline_terminal_rate", -0.1),
        ("latent_mse_model", -0.01),
        ("truncated", 2),
    ],
)
def test_world_audit_rejects_wrong_target_flags_gate_or_training_baseline(field, value):
    indexed = short_index()
    predictions = prediction_rows(indexed)
    predictions[0][field] = value
    with pytest.raises(a.MissingEvidence):
        a._world_predictions(predictions, indexed)


def test_world_training_baseline_is_frozen_across_final_episodes():
    indexed = short_index()
    predictions = prediction_rows(indexed)
    same = [
        r
        for r in predictions
        if r["game"] == "snake" and r["model_seed"] == p.MODEL_SEEDS[0]
    ]
    assert len(same) == len(p.TEST_SEEDS)
    same[1]["baseline_reward_mean"] = 0.2
    with pytest.raises(a.MissingEvidence, match="changed across"):
        a._world_predictions(predictions, indexed)


def test_historical_receipt_integrity_does_not_transfer_scientific_acceptance(tmp_path):
    receipt = tmp_path / "historical.txt"
    receipt.write_text("SYNTHETIC HISTORICAL RECEIPT", encoding="utf-8")
    record = {
        "status": a.PASS,
        "mission": "NNA-MULTIJOGOS-002",
        "protocol_sha256": "0" * 64,
        "source_revision": "a" * 40,
        "receipts": [
            {
                "path": receipt.name,
                "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
            }
        ],
    }
    with pytest.raises(a.MissingEvidence, match="historical receipts"):
        a._receipt({"shared_policy": record}, "shared_policy", tmp_path)


def test_partial_final_auc_endpoint_is_explicit_and_nonfinal_partial_is_rejected():
    rows = []
    for seed in p.MODEL_SEEDS:
        for stage, score in (("adaptation", 0.8), ("from_scratch", 0.2)):
            rows.extend(
                [
                    {
                        "model_seed": seed,
                        "stage": stage,
                        "game": "racing",
                        "training_step": 30000,
                        "progress_normalized": score,
                        "episode_complete": True,
                    },
                    {
                        "model_seed": seed,
                        "stage": stage,
                        "game": "racing",
                        "training_step": 60000,
                        "progress_normalized": score,
                        "episode_complete": False,
                    },
                ]
            )
    assert a._training_auc(rows)["status"] == a.PASS
    assert a._training_auc(rows)["compute_matched"] is False
    rows[0]["episode_complete"] = False
    with pytest.raises(a.MissingEvidence, match="partial target episode"):
        a._training_auc(rows)
    del rows[0]["episode_complete"]
    with pytest.raises(a.MissingEvidence, match="episode_complete"):
        a._training_auc(rows)


def test_report_declares_environment_budget_scope_and_never_compute_matching():
    result = a.analyze([])
    assert result["mission"] == "NNA-CONTINUAL-003"
    assert result["transfer_comparison_scope"]["compute_matched"] is False
    assert result["transfer_comparison_scope"]["physical_target_steps_per_arm"] == 60000
    assert result["overall"] == "NOT_CONCLUDED"
