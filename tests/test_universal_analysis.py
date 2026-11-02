"""Synthetic arithmetic/validation fixtures ONLY: no games or trained models run."""
import copy
import hashlib
import json

import numpy as np
import pytest

from arena.universal import analysis as a
from arena.universal import protocol as p


def fixture_rows():
    """Invented numbers exercise gates; they are explicitly not experiment data."""
    rows = []
    for index, cell in enumerate(p.evaluation_matrix()):
        baseline = cell.policy in ("random", "untrained")
        learned = not baseline
        rows.append({**vars(cell), "episode": index+1, "steps": 100,
                     "return": float(learned), "score": float(learned),
                     "wall_collision": 0, "body_collision": 0,
                     "foods": 4. if learned else 0., "points": 3. if learned else 0.,
                     "conceded": 0. if learned else 3.,
                     "hits": 8. if learned else 2., "incoming": 10.,
                     "prey_eaten": 1. if learned else 0., "eaten": int(baseline),
                     "large_near_steps": 20, "small_near_steps": 20,
                     "pipes_passed": 3. if learned else 0., "flaps": 10,
                     "narrow_pipes_passed": 1. if learned else 0.,
                     "wide_pipes_passed": 2. if learned else 0.,
                     "progress_normalized": .75 if learned else .05,
                     "lap_completed": int(learned), "distance": 100.,
                     "collision": int(baseline),
                     "spawn_context": str(cell.environment_seed % 2),
                     "gap_stratum": str(cell.environment_seed % 2)})
    return rows


def test_candidate_hash_matrix_and_exact_budget_without_execution():
    matrix = p.evaluation_matrix()
    assert len(matrix) == len(set(matrix)) == 1735
    assert len(matrix)*p.MAX_EPISODE_STEPS == 694_000
    assert p.MAX_TRAINING_STEPS == 3*(4+1+1)*60_000
    config = p.protocol_config()
    assert config["execution_authorized"] is False
    assert config["status"] == "CANDIDATE_NOT_FROZEN"
    assert len(config["unresolved_before_freeze"]) > 0
    assert p.protocol_sha256() == hashlib.sha256(p.canonical_json(config).encode()).hexdigest()
    config["architecture"]["gru_units"] = 999
    assert p.protocol_config()["architecture"]["gru_units"] == 64


def test_seed_streams_exclude_test_and_pair_target_environments():
    pre = p.derive_training_seed(1103, "snake", 0)
    assert pre not in p.TEST_SEEDS
    assert pre == p.derive_training_seed(1103, "snake", 0)
    assert pre != p.derive_training_seed(1103, "snake", 1)
    assert p.derive_training_seed(1103, "racing", 3, stage="adaptation") == p.derive_training_seed(
        1103, "racing", 3, stage="from_scratch")
    with pytest.raises(ValueError, match="excluded"):
        p.derive_training_seed(1103, "racing", 0)
    with pytest.raises(ValueError):
        p.derive_training_seed(True, "snake", 0)


def test_full_matrix_validates_and_duplicates_or_missing_fail():
    rows = fixture_rows()
    _, report = a.validate_episode_matrix(rows)
    assert report["status"] == a.PASS
    assert report["steps"] == 173_500
    _, report = a.validate_episode_matrix(rows[:-1])
    assert report["status"] == a.FAIL and report["missing_rows"] == 1
    _, report = a.validate_episode_matrix(rows+[rows[0]])
    assert report["status"] == a.FAIL and "duplicate" in report["errors"][0]


@pytest.mark.parametrize("field,value", [("steps", 401), ("steps", 0),
                                         ("return", "nan"), ("model_seed", 12345)])
def test_invalid_final_data_fail_closed(field, value):
    rows = fixture_rows()
    rows[0][field] = value
    _, report = a.validate_episode_matrix(rows)
    assert report["status"] == a.FAIL


def test_three_models_not_sixty_independent_agents_and_bootstrap_repeatable():
    treatment = np.ones((3, 20, 2))
    treatment[..., 0] = np.array([2., 3., 4.])[:, None]
    control = np.ones((3, 20, 2))
    control[..., 0] = 0.
    result = a.paired_contrast(treatment, control, margin=1., label="synthetic-only")
    assert result == a.paired_contrast(treatment, control, margin=1., label="synthetic-only")
    assert result["status"] == a.PASS
    assert result["independent_models"] == 3
    assert result["gain"] == 3.
    assert result["model_seed_gains"] == [2., 3., 4.]


def test_pool_opportunities_not_unweighted_mean_of_episode_hit_rates():
    values = np.array([[[1., 1.], [0., 9.]]]*3)
    assert a._evaluate(values).tolist() == [.1, .1, .1]
    values[..., 1] = 0.
    with pytest.raises(a.MissingEvidence, match="denominator"):
        a._evaluate(values)


def test_one_good_training_seed_cannot_win_by_many_episodes():
    t, c = np.ones((3, 20, 2)), np.ones((3, 20, 2))
    t[..., 0] = np.array([100., -1., -1.])[:, None]
    c[..., 0] = 0.
    result = a.paired_contrast(t, c, margin=1., label="one-outlier-fixture")
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
    assert result["after_adaptation_descriptive_metrics"]["snake"]["wall_collision"]["mean"] == 0


def test_missing_lap_telemetry_not_fabricated_from_distance():
    indexed, _ = a.validate_episode_matrix(fixture_rows())
    cell = p.EvaluationCell("after_adaptation", "shared", "racing", 1103, p.TEST_SEEDS[0])
    del indexed[cell]["progress_normalized"]
    with pytest.raises(a.MissingEvidence, match="progress_normalized"):
        a._contrast(indexed, "racing", "progress_normalized", .15)


def test_action_attribution_cannot_cherrypick_one_state_per_seed():
    indexed, _ = a.validate_episode_matrix(fixture_rows())
    pairs = [{"model_seed": seed, "game": game, "environment_seed": env_seed,
              "step": 1, "action_default": 0, "action_world_off": 1}
             for game in p.GAMES for seed in p.MODEL_SEEDS for env_seed in p.TEST_SEEDS]
    with pytest.raises(a.MissingEvidence, match="every shared rollout step"):
        a._action_attribution(indexed, pairs, "world_off", .01)


def test_world_holdout_reuses_final_shared_transitions_without_extra_episode_namespace():
    rows = fixture_rows()
    for row in rows:
        if row["stage"] == "after_adaptation" and row["policy"] == "shared":
            row["steps"] = 1
    indexed, _ = a.validate_episode_matrix(rows)
    predictions = []
    for cell in p.evaluation_matrix():
        if cell.stage == "after_adaptation" and cell.policy == "shared":
            predictions.append({**vars(cell), "transition_id": str(cell), "step": 1,
                "baseline_source": "training", "baseline_support": 100,
                "latent_mse_model": .01, "latent_mse_copy": .1,
                "reward_mae_model": .1, "reward_mae_baseline": .2,
                "terminal_brier_model": .1, "terminal_brier_baseline": .2, "terminated": 0})
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
            row["foods"] = 0.
    indexed, _ = a.validate_episode_matrix(rows)
    retention = a._retention(indexed)
    assert retention["components"]["snake"]["status"] == a.MISSING


def test_auc_uses_fixed_target_steps_not_episode_count():
    rows = []
    for seed in p.MODEL_SEEDS:
        for stage, scores in (("adaptation", (.5, .8)), ("from_scratch", (.1, .2))):
            for step, score in zip((10_000, 60_000), scores):
                rows.append({"model_seed": seed, "stage": stage, "game": "racing",
                             "training_step": step, "progress_normalized": score})
    report = a._training_auc(rows)
    assert report["status"] == a.PASS
    assert report["mean_relative_gain"] == pytest.approx((.5/6+.8*5/6)/(.1/6+.2*5/6)-1)
    with pytest.raises(a.MissingEvidence, match="60,000"):
        a._training_auc(rows[:-1])


def test_claimed_engineering_pass_needs_actual_untampered_receipt(tmp_path):
    evidence = {"shared_policy": {"status": "PASS", "receipts": []}}
    with pytest.raises(a.MissingEvidence):
        a._receipt(evidence, "shared_policy", tmp_path)
    receipt = tmp_path/"synthetic-receipt.txt"
    receipt.write_text("TEST FIXTURE, NOT A REAL AUDIT", encoding="utf-8")
    evidence["shared_policy"]["receipts"] = [{"path": receipt.name,
                                              "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest()}]
    assert a._receipt(evidence, "shared_policy", tmp_path)["status"] == a.PASS
    receipt.write_text("tampered", encoding="utf-8")
    with pytest.raises(a.MissingEvidence, match="hash mismatch"):
        a._receipt(evidence, "shared_policy", tmp_path)


def test_analysis_directory_empty_is_offline_and_json_finite(tmp_path):
    result = a.analyze_directory(tmp_path)
    assert json.loads(json.dumps(result, allow_nan=False))["overall"] == "NOT_CONCLUDED"
