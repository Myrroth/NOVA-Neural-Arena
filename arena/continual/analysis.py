"""Offline, fail-closed analysis for the independent continual-v2 candidate.

No environments, checkpoints or learning code are imported. Synthetic test data
exercise arithmetic only; they are never evidence of learned game competence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from io import StringIO
import json
import math
from pathlib import Path
import re
from typing import Callable

import numpy as np

from . import protocol as p

PASS = "PASS"
FAIL = "FAIL"
MISSING = "NOT_EVIDENCED"
EVALUATION_SCHEMA = "continual-v2-evaluation-v1"
TARGET_MODE = "continuous_gru_one_step"
CONTEXT_LENGTH = p.protocol_config()["learner_config"]["context_length"]


class MissingEvidence(ValueError):
    pass


def number(row: dict, field: str) -> float:
    try:
        value = row[field]
        if isinstance(value, bool) or value is None or value == "":
            raise ValueError
        value = float(value)
        if not math.isfinite(value):
            raise ValueError
        return value
    except (KeyError, TypeError, ValueError) as exc:
        raise MissingEvidence(f"missing/nonfinite numeric field: {field}") from exc


def integer(row: dict, field: str) -> int:
    value = number(row, field)
    if value != int(value):
        raise MissingEvidence(f"noninteger field: {field}")
    return int(value)


def boolean(row: dict, field: str) -> bool:
    """Accept actual booleans and their JSON/CSV spellings, never truthiness."""
    value = row.get(field)
    if type(value) is bool:
        return value
    if isinstance(value, str) and value in ("True", "False", "true", "false"):
        return value.lower() == "true"
    raise MissingEvidence(f"missing/nonboolean audit field: {field}")


def _control_audit(row: dict, *, paired=False):
    """Validate the recorded gate and periodic phase, not a confidence claim."""
    step = integer(row, "step")
    context = CONTEXT_LENGTH
    if step < 1:
        raise MissingEvidence("invalid control audit step")
    boundary = boolean(row, "next_context_boundary")
    reset = boolean(row, "context_reset")
    if boundary != (step % context == 0) or reset != ((step - 1) % context == 0):
        raise MissingEvidence(
            "recorded context phase differs from fixed periodic context"
        )
    weights = ["planning_weight"]
    if paired:
        weights += [
            "planning_weight_world_off",
            "planning_weight_gru_reset",
            "planning_weight_pixels_hidden",
        ]
        if not boolean(row, "context_reset_gru_reset"):
            raise MissingEvidence(
                "GRU-reset counterfactual must reset its incoming memory"
            )
        if boolean(row, "context_reset_pixels_hidden") != reset:
            raise MissingEvidence("pixel ablation changed periodic context phase")
    for field in weights:
        weight = number(row, field)
        if not 0 <= weight <= 0.20 or (boundary and weight != 0):
            raise MissingEvidence(
                "planning weight outside [0,.2] or enabled at next context boundary"
            )
    if paired and number(row, "planning_weight_world_off") != 0:
        raise MissingEvidence("world-off counterfactual must have zero planning weight")


def _validate_action_pairs(indexed, rows):
    expected = {
        (cell.model_seed, cell.game, cell.environment_seed, step)
        for cell, episode in indexed.items()
        if cell.stage == "after_adaptation" and cell.policy == "shared"
        for step in range(1, integer(episode, "steps") + 1)
    }
    seen = set()
    for row in rows:
        if row.get("stage") != "after_adaptation" or row.get("policy") != "shared":
            raise MissingEvidence(
                "same-state action pairs must use shared-after trajectories"
            )
        key = (
            integer(row, "model_seed"),
            str(row.get("game")),
            integer(row, "environment_seed"),
            integer(row, "step"),
        )
        if key in seen or key not in expected:
            raise MissingEvidence("duplicate/unexpected same-state action pair")
        seen.add(key)
        _control_audit(row, paired=True)
        for name in (
            "action_default",
            "action_world_off",
            "action_gru_reset",
            "action_pixels_hidden",
        ):
            if not 0 <= integer(row, name) < 10:
                raise MissingEvidence("action pair outside Discrete(10)")
    if not expected or seen != expected:
        raise MissingEvidence("same-state pairs must cover every shared rollout step")
    return {
        "status": PASS,
        "transitions": len(seen),
        "limitation": "gate validity is not action influence or objective improvement",
    }


def cell_key(row: dict) -> p.EvaluationCell:
    return p.EvaluationCell(
        str(row.get("stage", "")),
        str(row.get("policy", "")),
        str(row.get("game", "")),
        integer(row, "model_seed"),
        integer(row, "environment_seed"),
    )


def validate_episode_matrix(rows: list[dict]) -> tuple[dict, dict]:
    expected = set(p.evaluation_matrix())
    indexed, errors = {}, []
    for index, row in enumerate(rows):
        try:
            if row.get("evaluation_schema") != EVALUATION_SCHEMA:
                raise MissingEvidence(
                    "episode is not from the continual-v2 evaluation schema"
                )
            key = cell_key(row)
            if key not in expected:
                raise MissingEvidence(f"unexpected evaluation cell: {key}")
            if key in indexed:
                raise MissingEvidence(f"duplicate evaluation cell: {key}")
            steps = integer(row, "steps")
            if not 1 <= steps <= p.MAX_EPISODE_STEPS:
                raise MissingEvidence("episode steps outside 1..400")
            if integer(row, "episode") < 1:
                raise MissingEvidence("episode must be positive")
            number(row, "return")
            number(row, "score")
            indexed[key] = row
        except MissingEvidence as exc:
            errors.append(f"row {index+1}: {exc}")
    missing = expected - set(indexed)
    total_steps = sum(integer(row, "steps") for row in indexed.values())
    valid = not errors and not missing and len(rows) == len(expected)
    report = {
        "status": PASS if valid else FAIL,
        "expected_rows": len(expected),
        "observed_rows": len(rows),
        "unique_valid_rows": len(indexed),
        "missing_rows": len(missing),
        "steps": total_steps,
        "maximum_steps": p.MAX_EVALUATION_STEPS,
        "errors": errors,
        "missing_examples": [vars(cell) for cell in sorted(missing)[:10]],
    }
    return indexed, report


def _metric(row: dict, name: str) -> tuple[float, float]:
    """Numerator and denominator; rates pool opportunities, not episode ratios."""
    if name == "net_points":
        return number(row, "points") - number(row, "conceded"), 1.0
    if name == "interception":
        hits, incoming = number(row, "hits"), number(row, "incoming")
        if incoming < hits or hits < 0:
            raise MissingEvidence("invalid hit/opportunity counts")
        return hits, incoming
    if name == "survival":
        eaten = number(row, "eaten")
        if eaten not in (0.0, 1.0):
            raise MissingEvidence("eaten must be binary")
        return 1.0 - eaten, 1.0
    if name == "collision_rate":
        distance = number(row, "distance")
        collision = number(row, "collision")
        if distance < 0 or collision not in (0.0, 1.0):
            raise MissingEvidence("invalid racing collision/distance")
        return collision, distance
    value = number(row, name)
    if name in ("progress_normalized", "lap_completed") and not 0 <= value <= 1:
        raise MissingEvidence(f"{name} must be within [0,1]")
    if name == "lap_completed" and value not in (0.0, 1.0):
        raise MissingEvidence("lap_completed must be binary")
    return value, 1.0


def _evaluate(values: np.ndarray) -> np.ndarray:
    denominator = values[..., 1].sum(axis=-1)
    if np.any(denominator <= 0):
        raise MissingEvidence(
            "zero opportunity/exposure denominator; rate not estimable"
        )
    return values[..., 0].sum(axis=-1) / denominator


def _paired_arrays(
    indexed: dict,
    game: str,
    metric: str,
    treatment: tuple[str, str],
    control: tuple[str, str],
    seeds=p.TEST_SEEDS,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = []
    for stage, policy in (treatment, control):
        data = []
        for model_seed in p.MODEL_SEEDS:
            row_seed = 0 if policy == "random" else model_seed
            data.append(
                [
                    _metric(
                        indexed[
                            p.EvaluationCell(stage, policy, game, row_seed, env_seed)
                        ],
                        metric,
                    )
                    for env_seed in seeds
                ]
            )
        arrays.append(np.asarray(data, dtype=np.float64))
    return tuple(arrays)


def paired_contrast(
    treatment: np.ndarray,
    control: np.ndarray,
    *,
    margin: float,
    label: str,
    repetitions=p.BOOTSTRAP_REPETITIONS,
) -> dict:
    """Pair within model/environment; resample models first, environments second.

    This percentile interval has only THREE independent fitted agents. It is
    descriptive evidence under the preregistration, not a universal guarantee.
    """
    if (
        treatment.shape != control.shape
        or treatment.ndim != 3
        or treatment.shape[0] != 3
    ):
        raise MissingEvidence("paired arrays must be 3 model seeds x environments x 2")
    if (
        treatment.shape[-1] != 2
        or not np.isfinite(treatment).all()
        or not np.isfinite(control).all()
    ):
        raise MissingEvidence("invalid paired metric values")
    per_model = _evaluate(treatment) - _evaluate(control)
    salt = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(p.BOOTSTRAP_SEED ^ salt)
    models = rng.integers(0, 3, size=(repetitions, 3))
    environments = rng.integers(
        0, treatment.shape[1], size=(repetitions, 3, treatment.shape[1])
    )
    sampled_t = treatment[models[..., None], environments]
    sampled_c = control[models[..., None], environments]
    differences = (_evaluate(sampled_t) - _evaluate(sampled_c)).mean(axis=-1)
    low, high = np.quantile(differences, [0.025, 0.975])
    gain, favorable = float(per_model.mean()), int(np.count_nonzero(per_model > 0))
    return {
        "status": PASS if gain >= margin and low > 0 and favorable >= 2 else FAIL,
        "gain": gain,
        "minimum_gain": margin,
        "ci95": [float(low), float(high)],
        "favorable_models": favorable,
        "model_seed_gains": per_model.tolist(),
        "independent_models": 3,
        "environment_seeds_per_model": treatment.shape[1],
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": p.BOOTSTRAP_SEED,
        "treatment_means": _evaluate(treatment).tolist(),
        "control_means": _evaluate(control).tolist(),
    }


def _attempt(operation: Callable[[], dict]) -> dict:
    try:
        return operation()
    except (MissingEvidence, KeyError) as exc:
        return {"status": MISSING, "reason": str(exc)}


def _all(parts: dict) -> dict:
    statuses = [part["status"] for part in parts.values()]
    status = (
        FAIL
        if FAIL in statuses
        else MISSING if MISSING in statuses or not parts else PASS
    )
    return {"status": status, "components": parts}


def _contrast(
    indexed,
    game,
    metric,
    margin,
    *,
    treatment=("after_adaptation", "shared"),
    control=("baseline", "random"),
    seeds=p.TEST_SEEDS,
):
    label = f"{game}:{metric}:{treatment}:{control}:{tuple(seeds)}"
    return paired_contrast(
        *_paired_arrays(indexed, game, metric, treatment, control, seeds),
        margin=margin,
        label=label,
    )


def _learning(indexed, game, metrics):
    comparisons = {}
    for metric, margin in metrics:
        for policy in ("random", "untrained"):
            # Chat1 specifies absolute margins versus random. Untrained is a
            # separate positive-gain control, pending explicit final recertification.
            threshold = margin if policy == "random" else 0.0
            comparisons[f"{metric}_vs_{policy}"] = _attempt(
                lambda m=metric, b=policy, t=threshold: _contrast(
                    indexed, game, m, t, control=("baseline", b)
                )
            )
    return _all(comparisons)


def _context_coverage(indexed, game: str, field: str, metric: str) -> dict:
    strata = {}
    for cell, row in indexed.items():
        if (
            cell.game == game
            and cell.stage == "after_adaptation"
            and cell.policy == "shared"
        ):
            value = row.get(field)
            if value is None or str(value) == "":
                raise MissingEvidence(f"required scenario telemetry absent: {field}")
            strata.setdefault(str(value), []).append(number(row, metric))
    positive = [name for name, values in strata.items() if np.mean(values) > 0]
    return {
        "status": PASS if len(positive) >= 2 else FAIL,
        "positive_contexts": positive,
        "stratum_counts": {k: len(v) for k, v in strata.items()},
    }


def _flappy_gap_coverage(indexed):
    # The upcoming gap at episode end is not necessarily a gap actually passed.
    # Use the physical crossing counters, never relabel total score by final gap.
    return _all(
        {
            stratum: _attempt(
                lambda name=f"{stratum}_pipes_passed": _contrast(
                    indexed, "flappy", name, 0.0
                )
            )
            for stratum in ("narrow", "wide")
        }
    )


def _agar_coverage(indexed):
    per_model = {}
    for seed in p.MODEL_SEEDS:
        rows = [
            indexed[p.EvaluationCell("after_adaptation", "shared", "agar", seed, env)]
            for env in p.TEST_SEEDS
        ]
        large = sum(number(r, "large_near_steps") for r in rows)
        small = sum(number(r, "small_near_steps") for r in rows)
        per_model[str(seed)] = {"large_near_steps": large, "small_near_steps": small}
    # These fields prove exposure only. They do not count discrete escapes or
    # establish causality of chasing; reviewer must inspect scenario/trajectory.
    passed = all(
        v["large_near_steps"] > 0 and v["small_near_steps"] > 0
        for v in per_model.values()
    )
    return {
        "status": PASS if passed else FAIL,
        "exposure": per_model,
        "limitation": "proximity exposure is not an escape-opportunity denominator",
    }


def _telemetry(indexed, game, fields):
    selected = [
        row
        for cell, row in indexed.items()
        if cell.game == game
        and cell.stage == "after_adaptation"
        and cell.policy == "shared"
    ]
    if len(selected) != len(p.MODEL_SEEDS) * len(p.TEST_SEEDS):
        raise MissingEvidence("raw outcome telemetry needs all final shared episodes")
    for row in selected:
        for field in fields:
            number(row, field)
    return {"status": PASS, "required_fields": list(fields), "episodes": len(selected)}


def _retention(indexed):
    metrics = {
        "snake": "foods",
        "pong": "net_points",
        "agar": "foods",
        "flappy": "pipes_passed",
    }
    reports = {}
    for game, metric in metrics.items():
        before, random = _paired_arrays(
            indexed,
            game,
            metric,
            ("before_adaptation", "shared"),
            ("baseline", "random"),
        )
        after, _ = _paired_arrays(
            indexed,
            game,
            metric,
            ("after_adaptation", "shared"),
            ("baseline", "random"),
        )
        baseline = _evaluate(random)
        pre_gain, post_gain = _evaluate(before) - baseline, _evaluate(after) - baseline
        if np.any(pre_gain <= 0):
            reports[game] = {
                "status": MISSING,
                "reason": "nonpositive pre-adaptation gain in at least one replica",
                "before_gains": pre_gain.tolist(),
                "after_gains": post_gain.tolist(),
            }
        else:
            ratio = post_gain / pre_gain
            reports[game] = {
                "status": PASS if np.all(ratio >= 0.80) else FAIL,
                "retained_gain_by_model": ratio.tolist(),
                "before_gains": pre_gain.tolist(),
                "after_gains": post_gain.tolist(),
                "minimum_fraction": 0.80,
            }
    return _all(reports)


def _training_auc(rows):
    by_branch = {}
    for model_seed in p.MODEL_SEEDS:
        for stage in ("adaptation", "from_scratch"):
            selected = [
                r
                for r in rows
                if r.get("stage") == stage
                and integer(r, "model_seed") == model_seed
                and r.get("game") == "racing"
            ]
            selected.sort(key=lambda r: integer(r, "training_step"))
            previous, total = 0, 0.0
            for row in selected:
                end = integer(row, "training_step")
                progress = number(row, "progress_normalized")
                complete = boolean(row, "episode_complete")
                if end <= previous or end > p.STEPS_PER_GAME or not 0 <= progress <= 1:
                    raise MissingEvidence(
                        "invalid/duplicate target training-curve endpoint"
                    )
                if not complete and end != p.STEPS_PER_GAME:
                    raise MissingEvidence(
                        "partial target episode may appear only at the exact final stage endpoint"
                    )
                total += progress * (end - previous)
                previous = end
            if previous != p.STEPS_PER_GAME:
                raise MissingEvidence(
                    "target training curve must cover exactly 60,000 steps"
                )
            by_branch[model_seed, stage] = total / p.STEPS_PER_GAME
    gains = []
    for seed in p.MODEL_SEEDS:
        transfer, scratch = (
            by_branch[seed, "adaptation"],
            by_branch[seed, "from_scratch"],
        )
        if scratch <= 0:
            raise MissingEvidence(
                "zero scratch AUC: relative transfer advantage not estimable"
            )
        gains.append(transfer / scratch - 1.0)
    return {
        "status": (
            PASS
            if float(np.mean(gains)) >= 0.10 and sum(g > 0 for g in gains) >= 2
            else FAIL
        ),
        "relative_auc_gain_by_model": gains,
        "mean_relative_gain": float(np.mean(gains)),
        "minimum_relative_gain": 0.10,
        "definition": "step-weighted episode progress, right endpoint including explicit partial final episode; exploratory training policy",
        "compute_matched": False,
    }


def _action_attribution(indexed, action_pairs, ablation: str, threshold: float):
    _validate_action_pairs(indexed, action_pairs)
    results = {}
    for game in p.GAMES:
        per_model = []
        seeds = p.PIXEL_TEST_SEEDS if ablation == "pixels_hidden" else p.TEST_SEEDS
        for model_seed in p.MODEL_SEEDS:
            selected = [
                r
                for r in action_pairs
                if r.get("game") == game
                and integer(r, "model_seed") == model_seed
                and integer(r, "environment_seed") in seeds
            ]
            expected = {
                (env_seed, step)
                for env_seed in seeds
                for step in range(
                    1,
                    integer(
                        indexed[
                            p.EvaluationCell(
                                "after_adaptation", "shared", game, model_seed, env_seed
                            )
                        ],
                        "steps",
                    )
                    + 1,
                )
            }
            seen, changed = set(), 0
            for row in selected:
                key = (integer(row, "environment_seed"), integer(row, "step"))
                if key in seen or key[1] < 1:
                    raise MissingEvidence("duplicate/invalid same-state action pair")
                seen.add(key)
                original, alternative = integer(row, "action_default"), integer(
                    row, f"action_{ablation}"
                )
                if not 0 <= original < 10 or not 0 <= alternative < 10:
                    raise MissingEvidence("action pair outside Discrete(10)")
                changed += original != alternative
            if seen != expected:
                raise MissingEvidence(
                    f"same-state pairs must cover every shared rollout step: {game}/{model_seed}"
                )
            per_model.append(changed / len(selected))
        results[game] = {
            "changed_fraction": float(np.mean(per_model)),
            "changed_fraction_by_model": per_model,
        }
    count = sum(r["changed_fraction"] >= threshold for r in results.values())
    return {
        "status": PASS if count >= 4 else FAIL,
        "games_above_threshold": count,
        "minimum_fraction": threshold,
        "by_game": results,
    }


def _ablation(indexed, action_pairs, ablation, threshold):
    action = _attempt(
        lambda: _action_attribution(indexed, action_pairs, ablation, threshold)
    )
    objectives = {
        "snake": "foods",
        "pong": "net_points",
        "agar": "foods",
        "flappy": "pipes_passed",
        "racing": "progress_normalized",
    }
    seeds = p.PIXEL_TEST_SEEDS if ablation == "pixels_hidden" else p.TEST_SEEDS
    comparisons = {
        game: _attempt(
            lambda g=game, m=metric: _contrast(
                indexed, g, m, 0.0, control=("ablation", ablation), seeds=seeds
            )
        )
        for game, metric in objectives.items()
    }
    if any(r["status"] == MISSING for r in comparisons.values()):
        performance = {"status": MISSING, "by_game": comparisons}
    else:
        count = sum(r["status"] == PASS for r in comparisons.values())
        performance = {
            "status": PASS if count >= 2 else FAIL,
            "improved_games": count,
            "by_game": comparisons,
        }
    return _all(
        {"same_state_action_effect": action, "objective_improvement": performance}
    )


def _world_predictions(rows, indexed):
    results, seen = {}, set()
    observed_steps = set()
    baselines = {}
    for row in rows:
        key = (
            integer(row, "model_seed"),
            str(row.get("game")),
            str(row.get("transition_id")),
        )
        if key in seen or not key[2] or key[2] == "None":
            raise MissingEvidence("duplicate/missing world transition identity")
        seen.add(key)
        step_key = (
            integer(row, "model_seed"),
            str(row.get("game")),
            integer(row, "environment_seed"),
            integer(row, "step"),
        )
        if step_key in observed_steps:
            raise MissingEvidence(
                "duplicate world prediction at the same final transition"
            )
        observed_steps.add(step_key)
        if row.get("stage") != "after_adaptation" or row.get("policy") != "shared":
            raise MissingEvidence(
                "world predictions must use the existing final shared-after trajectories"
            )
        if row.get("target_mode") != TARGET_MODE:
            raise MissingEvidence(
                "world target must be explicit continuous_gru_one_step, not reset control memory"
            )
        _control_audit(row)
        if (
            row.get("baseline_source") != "training"
            or integer(row, "baseline_support") < 1
        ):
            raise MissingEvidence(
                "world prediction baselines require training provenance/support"
            )
        reward_mean = number(row, "baseline_reward_mean")
        terminal_rate = number(row, "baseline_terminal_rate")
        if not -1 <= reward_mean <= 1 or not 0 <= terminal_rate <= 1:
            raise MissingEvidence(
                "invalid clipped-reward/termination training baseline"
            )
        baseline_key = (integer(row, "model_seed"), str(row.get("game")))
        baseline = (integer(row, "baseline_support"), reward_mean, terminal_rate)
        if baselines.setdefault(baseline_key, baseline) != baseline:
            raise MissingEvidence(
                "training baseline changed across heldout predictions"
            )
        for field in (
            "latent_mse_model",
            "latent_mse_copy",
            "reward_mae_model",
            "reward_mae_baseline",
            "terminal_brier_model",
            "terminal_brier_baseline",
        ):
            if number(row, field) < 0:
                raise MissingEvidence("negative world prediction loss")
        terminal = integer(row, "terminated")
        truncated = integer(row, "truncated")
        if terminal not in (0, 1) or truncated not in (0, 1):
            raise MissingEvidence("invalid world terminal/truncation labels")
    expected_steps = {
        (cell.model_seed, cell.game, cell.environment_seed, step)
        for cell, row in indexed.items()
        if cell.stage == "after_adaptation" and cell.policy == "shared"
        for step in range(1, integer(row, "steps") + 1)
    }
    if not expected_steps or observed_steps != expected_steps:
        raise MissingEvidence(
            "world predictions must cover all existing shared-after final transitions"
        )
    for game in p.GAMES:
        by_model = {}
        for seed in p.MODEL_SEEDS:
            selected = [
                r
                for r in rows
                if r.get("game") == game and integer(r, "model_seed") == seed
            ]
            if not selected:
                raise MissingEvidence(f"world holdout missing: {game}/{seed}")
            means = {
                field: float(np.mean([number(r, field) for r in selected]))
                for field in (
                    "latent_mse_model",
                    "latent_mse_copy",
                    "reward_mae_model",
                    "reward_mae_baseline",
                    "terminal_brier_model",
                    "terminal_brier_baseline",
                )
            }
            if (
                any(value < 0 for value in means.values())
                or means["latent_mse_copy"] <= 0
            ):
                raise MissingEvidence("invalid world prediction loss/baseline")
            terminals = [integer(r, "terminated") for r in selected]
            if any(t not in (0, 1) for t in terminals):
                raise MissingEvidence("invalid world terminal labels")
            by_model[str(seed)] = {
                **means,
                "support": len(selected),
                "terminal_events": sum(terminals),
                "latent_ratio": means["latent_mse_model"] / means["latent_mse_copy"],
            }
        passed = all(r["latent_ratio"] <= 0.95 for r in by_model.values())
        results[game] = {"status": PASS if passed else FAIL, "by_model": by_model}
    return _all(results)


def _receipt(evidence, name: str, root: Path | None):
    record = evidence.get(name)
    if not isinstance(record, dict) or record.get("status") != PASS:
        raise MissingEvidence(f"reviewed engineering receipt absent: {name}")
    if (
        record.get("mission") != p.MISSION
        or record.get("protocol_sha256") != p.protocol_sha256()
        or not re.fullmatch(r"[0-9a-f]{40}", str(record.get("source_revision", "")))
    ):
        raise MissingEvidence(
            "engineering receipt needs new mission/protocol/source binding; historical receipts cannot transfer"
        )
    receipts = record.get("receipts")
    if not receipts or root is None:
        raise MissingEvidence(f"{name} needs actual hashed receipt files")
    verified = []
    for item in receipts:
        relative, digest = item.get("path"), item.get("sha256", "")
        if not isinstance(relative, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise MissingEvidence("invalid receipt path/hash")
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise MissingEvidence("receipt file missing or outside evidence directory")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise MissingEvidence("engineering receipt hash mismatch")
        verified.append({"path": relative, "sha256": actual})
    return {
        "status": PASS,
        "receipts": verified,
        "limitation": "hash verifies receipt integrity; content remains subject to independent review",
    }


def _descriptive_metrics(indexed):
    """Publish raw endpoints without treating absent fields as zero."""
    fields = {
        "snake": ("foods", "wall_collision", "body_collision"),
        "pong": ("points", "conceded", "hits", "incoming"),
        "agar": (
            "foods",
            "prey_eaten",
            "eaten",
            "mass",
            "large_near_steps",
            "large_distance_change",
            "small_near_steps",
            "small_distance_change",
        ),
        "flappy": ("pipes_passed", "collision", "flaps"),
        "racing": (
            "distance",
            "speed",
            "collision",
            "accelerations",
            "brakes",
            "turns",
            "curved_steps",
            "curved_distance",
            "track_length",
            "lap_completed",
            "progress_normalized",
        ),
    }
    summaries = {}
    for game, names in fields.items():
        selected = [
            row
            for cell, row in indexed.items()
            if cell.game == game
            and cell.stage == "after_adaptation"
            and cell.policy == "shared"
        ]
        report = {}
        for field in names:
            values, missing = [], 0
            for row in selected:
                try:
                    values.append(number(row, field))
                except MissingEvidence:
                    missing += 1
            report[field] = {
                "support": len(values),
                "missing": missing,
                "mean": float(np.mean(values)) if values else None,
                "sum": float(np.sum(values)) if values else None,
            }
        summaries[game] = report
    return summaries


def analyze(
    rows: list[dict],
    *,
    world_predictions=None,
    action_pairs=None,
    training_curves=None,
    evidence=None,
    evidence_root=None,
) -> dict:
    """Compute preregistered gates from supplied records; missing proof never passes."""
    indexed, matrix = validate_episode_matrix(rows)
    evidence, action_pairs = evidence or {}, action_pairs or []
    root = Path(evidence_root) if evidence_root is not None else None
    requirements = {}
    for number_, game, metrics in (
        (1, "snake", (("foods", 1.0),)),
        (2, "pong", (("net_points", 0.5), ("interception", 0.10))),
        (3, "agar", (("foods", 2.0), ("prey_eaten", 0.3), ("survival", 0.10))),
        (4, "flappy", (("pipes_passed", 1.0),)),
        (5, "racing", (("progress_normalized", 0.15), ("lap_completed", 0.10))),
    ):
        components = {"learning": _learning(indexed, game, metrics)}
        if game == "snake":
            components["outcome_telemetry"] = _attempt(
                lambda: _telemetry(
                    indexed, game, ("foods", "wall_collision", "body_collision")
                )
            )
            components["spawn_coverage"] = _attempt(
                lambda: _context_coverage(indexed, game, "spawn_context", "foods")
            )
        if game == "flappy":
            components["outcome_telemetry"] = _attempt(
                lambda: _telemetry(
                    indexed, game, ("pipes_passed", "collision", "flaps")
                )
            )
            components["gap_coverage"] = _flappy_gap_coverage(indexed)
        if game == "agar":
            components["outcome_telemetry"] = _attempt(
                lambda: _telemetry(
                    indexed,
                    game,
                    (
                        "foods",
                        "prey_eaten",
                        "eaten",
                        "mass",
                        "large_near_steps",
                        "large_distance_change",
                        "small_near_steps",
                        "small_distance_change",
                    ),
                )
            )
            components["opportunity_exposure"] = _attempt(
                lambda: _agar_coverage(indexed)
            )
            components["scenario_audit"] = _attempt(
                lambda: _receipt(evidence, "agar_scenario_audit", root)
            )
        if game == "racing":
            components["outcome_telemetry"] = _attempt(
                lambda: _telemetry(
                    indexed,
                    game,
                    (
                        "track_length",
                        "distance",
                        "accelerations",
                        "brakes",
                        "turns",
                        "curved_steps",
                        "curved_distance",
                    ),
                )
            )

            def collision_noninferiority():
                t, c = _paired_arrays(
                    indexed,
                    "racing",
                    "collision_rate",
                    ("after_adaptation", "shared"),
                    ("baseline", "random"),
                )
                per_model = _evaluate(t) - _evaluate(c)
                return {
                    "status": PASS if np.all(per_model <= 0) else FAIL,
                    "collision_rate_difference_by_model": per_model.tolist(),
                }

            components["collision_per_distance"] = _attempt(collision_noninferiority)
            components["curve_throttle_brake_audit"] = _attempt(
                lambda: _receipt(evidence, "racing_control_audit", root)
            )
        requirements[str(number_)] = _all(components)
    requirements["6"] = _attempt(lambda: _receipt(evidence, "shared_policy", root))
    requirements["7"] = _all(
        {
            "perception": _ablation(indexed, action_pairs, "pixels_hidden", 0.05),
            "temporal_memory": _ablation(indexed, action_pairs, "gru_reset", 0.01),
            "observation_audit": _attempt(
                lambda: _receipt(evidence, "observation_audit", root)
            ),
        }
    )
    requirements["8"] = _all(
        {
            "prediction": _attempt(
                lambda: _world_predictions(world_predictions or [], indexed)
            ),
            "causal_control": _ablation(indexed, action_pairs, "world_off", 0.01),
            "heldout_provenance": _attempt(
                lambda: _receipt(evidence, "world_holdout_provenance", root)
            ),
        }
    )
    requirements["9"] = _attempt(
        lambda: _receipt(evidence, "checkpoint_resume_25_updates", root)
    )
    requirements["10"] = _all(
        {
            "target_final_transfer": _attempt(
                lambda: _contrast(
                    indexed,
                    "racing",
                    "progress_normalized",
                    0.05,
                    control=("from_scratch", "scratch"),
                )
            ),
            "target_training_auc": _attempt(
                lambda: _training_auc(training_curves or [])
            ),
            "source_retention": _attempt(lambda: _retention(indexed)),
            "holdout_and_physical_budget": _attempt(
                lambda: _receipt(evidence, "holdout_physical_budget", root)
            ),
        }
    )
    requirements["11"] = _all(
        {
            "matrix": matrix,
            "separation_and_limits": _attempt(
                lambda: _receipt(evidence, "separation_and_limits", root)
            ),
        }
    )
    requirements["12"] = _attempt(
        lambda: _receipt(evidence, "demonstration_recovery_delivery", root)
    )
    requirements["13"] = _attempt(
        lambda: _receipt(evidence, "coordinated_acceptance", root)
    )
    all_pass = matrix["status"] == PASS and all(
        r["status"] == PASS for r in requirements.values()
    )
    return {
        "mission": p.MISSION,
        "candidate_protocol_sha256": p.protocol_sha256(),
        "execution_authorized": False,
        "matrix": matrix,
        "requirements": requirements,
        "after_adaptation_descriptive_metrics": _descriptive_metrics(indexed),
        "overall": "ALL_REQUIRED_EVIDENCE_REPORTED" if all_pass else "NOT_CONCLUDED",
        "scientific_execution_performed_by_analyzer": False,
        "transfer_comparison_scope": {
            "physical_target_steps_per_arm": p.STEPS_PER_GAME,
            "scheduled_update_opportunities": "same config and16/32-step intervals",
            "realized_optimizer_updates": "may differ with reserved calibration and insufficient context; report actual arm deltas",
            "compute_matched": False,
            "interpretation": "sample efficiency at equal environment budget; no compute-matched transfer claim",
        },
        "limitations": [
            "Three independent trained agents only; hierarchical intervals are exploratory and uncertain.",
            "Historical001/002 results, approvals and receipts do not establish003 competence.",
            "Training arms match physical steps and scheduled opportunities; realized update counts may differ and must be disclosed.",
            "Random baseline is shared across model replicas, not three independent random campaigns.",
            "Only the five versioned prepared local simulators are in scope.",
            "Candidate metric/receipt definitions need Chat1 recertification before scientific execution.",
            "Hashed engineering receipts require independent content review; analysis cannot issue mission acceptance.",
            "Missing telemetry, incomplete final matrix, unsupported opportunities and absent receipts never count as PASS.",
        ],
    }


def _read_csv(path: Path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_explicit_training_curve(path: str | Path, digest: str) -> list[dict]:
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise MissingEvidence("explicit training curve needs a valid SHA256")
    try:
        with Path(path).open("rb") as handle:
            raw = handle.read(64 * 1024 * 1024 + 1)
    except OSError as exc:
        raise MissingEvidence(
            "explicit training curve file is missing or unreadable"
        ) from exc
    if len(raw) > 64 * 1024 * 1024:
        raise MissingEvidence("explicit training curve exceeds 64 MiB")
    if hashlib.sha256(raw).hexdigest() != digest.lower():
        raise MissingEvidence("explicit training curve SHA256 mismatch")
    try:
        return list(csv.DictReader(StringIO(raw.decode("utf-8"), newline="")))
    except UnicodeDecodeError as exc:
        raise MissingEvidence("explicit training curve must be UTF-8 CSV") from exc


def analyze_directory(
    directory: str | Path,
    *,
    training_curve_path: str | Path | None = None,
    training_curve_sha256: str | None = None,
) -> dict:
    root = Path(directory)
    if (training_curve_path is None) != (training_curve_sha256 is None):
        raise MissingEvidence("explicit training curve requires both path and SHA256")
    training_curves = (
        _read_csv(root / "training_curve.csv")
        if training_curve_path is None
        else _read_explicit_training_curve(training_curve_path, training_curve_sha256)
    )
    receipt = root / "evidence.json"
    evidence = (
        json.loads(receipt.read_text(encoding="utf-8")) if receipt.is_file() else {}
    )
    return analyze(
        _read_csv(root / "episodes.csv"),
        world_predictions=_read_csv(root / "model_predictions.csv"),
        action_pairs=_read_csv(root / "action_pairs.csv"),
        training_curves=training_curves,
        evidence=evidence,
        evidence_root=root,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--training-curve",
        type=Path,
        help="explicit training CSV, requiring its SHA256",
    )
    parser.add_argument(
        "--training-curve-sha256",
        help="SHA256 of the exact explicit training CSV bytes",
    )
    args = parser.parse_args()
    try:
        result = analyze_directory(
            args.directory,
            training_curve_path=args.training_curve,
            training_curve_sha256=args.training_curve_sha256,
        )
    except MissingEvidence as exc:
        parser.exit(2, f"STOP: {exc}\n")
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
