"""Pure planning checks: no ML imports, checkpoints, games or experiments."""

import ast
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from arena.continual import protocol as p
from arena.universal import protocol as historical


ROOT = Path(__file__).resolve().parents[1]


def test_matrix_is_unique_complete_and_preserves_the_historical_allocation():
    cells = p.evaluation_matrix()
    assert len(cells) == len(set(cells)) == 1735
    expected = {
        ("before_adaptation", "shared"): 300,
        ("after_adaptation", "shared"): 300,
        ("from_scratch", "scratch"): 60,
        ("baseline", "untrained"): 300,
        ("baseline", "random"): 100,
        ("ablation", "world_off"): 300,
        ("ablation", "gru_reset"): 300,
        ("ablation", "pixels_hidden"): 75,
    }
    assert Counter((c.stage, c.policy) for c in cells) == expected
    assert (
        Counter((c.stage, c.policy) for c in historical.evaluation_matrix()) == expected
    )
    assert {c.environment_seed for c in cells} == set(range(930001, 930021))
    assert set(p.TEST_SEEDS).isdisjoint(historical.TEST_SEEDS)
    assert p.MODEL_SEEDS == (4103, 5207, 6301)
    assert set(p.MODEL_SEEDS).isdisjoint(historical.MODEL_SEEDS)
    for cell in cells:
        assert cell.game in p.GAMES
        assert cell.model_seed in ((0,) if cell.policy == "random" else p.MODEL_SEEDS)
        if cell.policy == "scratch":
            assert cell.game == "racing"
        if cell.policy == "pixels_hidden":
            assert cell.environment_seed in p.TEST_SEEDS[:5]


def test_physical_and_optimizer_ceilings_cover_only_the_fixed_design():
    config = p.protocol_config()
    train = config["training"]
    assert train["source_steps_per_model"] == 4 * p.STEPS_PER_GAME == 240000
    assert (
        train["steps_per_model"]
        == train["source_steps_per_model"] + 2 * p.STEPS_PER_GAME
    )
    assert (
        p.MAX_TRAINING_STEPS == len(p.MODEL_SEEDS) * train["steps_per_model"] == 1080000
    )
    assert (
        p.MAX_EVALUATION_STEPS
        == len(p.evaluation_matrix()) * p.MAX_EPISODE_STEPS
        == 694000
    )
    assert config["limits"]["scheduled_maximum_steps"] == 1774000
    assert p.MAX_TOTAL_STEPS == 1800000
    assert p.MAX_TRAINING_STEPS + p.MAX_EVALUATION_STEPS <= p.MAX_TOTAL_STEPS
    assert p.MAX_Q_UPDATES == p.MAX_TRAINING_STEPS // 16 == 67500
    assert p.MAX_WORLD_UPDATES == p.MAX_TRAINING_STEPS // 32 == 33750
    assert config["limits"]["combined_updates"] == 101250
    assert config["limits"]["campaign_attempts"] == 1
    assert config["limits"]["automatic_retry"] is False
    assert config["limits"]["paid_resources"] is False


def test_learner_literal_covers_every_current_config_field_without_importing_trainer():
    # AST-only inspection checks the asdict contract without importing Torch or
    # constructing even an untrained network, let alone an environment.
    tree = ast.parse((ROOT / "arena/continual/trainer.py").read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ContinualConfig"
    )
    defaults = {
        node.target.id: ast.literal_eval(node.value)
        for node in cls.body
        if isinstance(node, ast.AnnAssign) and node.value is not None
    }
    locked = p.protocol_config()["learner_config"]
    assert locked == defaults
    assert locked["context_length"] == 16 and locked["n_step"] == 3
    assert locked["source_distillation_weight"] == 1.0
    assert locked["adaptation_target_fraction"] == 0.5
    assert locked["calibration_every_episodes"] == 10
    assert locked["planning_fraction"] == 0.2
    assert locked["world_learning_rate"] != locked["learning_rate"]


def test_seed_plan_matches_the_current_pure_training_derivation():
    tree = ast.parse((ROOT / "arena/continual/trainer.py").read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "training_seed"
    )
    # Execute only the extracted pure hash function, not the learner module.
    scope = {"hashlib": hashlib}
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]), "pure_training_seed", "exec"
        ),
        scope,
    )
    trainer_seed = scope["training_seed"]
    for model in p.MODEL_SEEDS:
        for stage, game in (
            ("pretrain", "snake"),
            ("adaptation", "racing"),
            ("from_scratch", "racing"),
        ):
            for calibration in (False, True):
                expected = trainer_seed(model, game, 3, stage, calibration)
                assert (
                    p.derive_training_seed(
                        model, game, 3, stage=stage, calibration=calibration
                    )
                    == expected
                )
    assert p.derive_training_seed(
        4103, "snake", 3, stage="source"
    ) == p.derive_training_seed(4103, "snake", 3, stage="pretrain")


def test_fit_calibration_and_final_namespaces_are_structurally_disjoint_and_target_paired():
    fit, calibration = set(), set()
    for model in p.MODEL_SEEDS:
        for episode in range(20):
            fit.add(p.derive_training_seed(model, "snake", episode))
            calibration.add(
                p.derive_training_seed(model, "snake", episode, calibration=True)
            )
            for heldout in (False, True):
                adapted = p.derive_training_seed(
                    model, "racing", episode, stage="adaptation", calibration=heldout
                )
                scratch = p.derive_training_seed(
                    model, "racing", episode, stage="from_scratch", calibration=heldout
                )
                assert adapted == scratch
    assert all(seed >= 2000000 and seed % 2 == 0 for seed in fit)
    assert all(seed >= 2000000 and seed % 2 == 1 for seed in calibration)
    assert fit.isdisjoint(calibration)
    assert (fit | calibration).isdisjoint(p.TEST_SEEDS)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_seed": True},
        {"model_seed": 1103},
        {"game": "unknown"},
        {"game": "racing"},
        {"episode": -1},
        {"episode": True},
        {"stage": "adaptation"},
        {"stage": "unknown"},
        {"calibration": 1},
    ],
)
def test_invalid_seed_plan_is_rejected_without_runtime_dependencies(kwargs):
    args = {"model_seed": 4103, "game": "snake", "episode": 0}
    args.update(kwargs)
    with pytest.raises(ValueError):
        p.derive_training_seed(**args)


def test_scientific_margins_and_analysis_requirements_are_not_relaxed():
    current = p.protocol_config()
    old = historical.protocol_config()
    assert current["gates"] == old["gates"]
    for key in (
        "bootstrap_repetitions",
        "interval",
        "minimum_favorable_model_seeds",
        "retained_positive_gain_fraction",
        "transfer_final_margin",
        "transfer_auc_relative_margin",
        "random_seed_shared_across_models",
    ):
        assert current["analysis"][key] == old["analysis"][key]
    assert current["analysis"]["bootstrap_repetitions"] == 10000
    assert current["analysis"]["bootstrap_seed"] != old["analysis"]["bootstrap_seed"]
    assert current["calibration"]["mse_ratio_maximum"] == 0.90
    assert current["gates"]["world_mse_ratio_maximum"] == 0.95
    assert current["architecture"]["world_target_mode"] == "continuous_gru_one_step"
    assert current["architecture"]["world_boundary_flag"] == "next_context_boundary"
    assert current["analysis"]["retention_nonpositive_before"].startswith("MISSING")
    assert (
        "not compute-matched"
        in current["training"]["update_fairness"]["transfer_claim_scope"]
    )


def test_fixed_demonstrations_are_five_existing_cells_without_selection_or_extra_rollouts():
    config = p.protocol_config()
    demos = config["evaluation"]["demonstrations"]
    cells = {tuple(asdict(cell).items()) for cell in p.evaluation_matrix()}
    assert len(demos) == 5
    assert {demo["game"] for demo in demos} == set(p.GAMES)
    assert all(tuple(demo.items()) in cells for demo in demos)
    assert all(
        demo["model_seed"] == 5207 and demo["environment_seed"] == 930001
        for demo in demos
    )


def test_candidate_hash_is_deterministic_detached_and_never_grants_authority():
    first = p.protocol_config()
    expected = hashlib.sha256(p.canonical_json(first).encode("utf-8")).hexdigest()
    assert p.protocol_sha256() == expected
    first["execution_authorized"] = True
    first["learner_config"]["context_length"] = 64
    first["evaluation_matrix"].clear()
    first["review_status"]["human_consent"] = "not actual permission"
    current = p.protocol_config()
    assert p.protocol_sha256() == expected
    assert current["mission"] == "NNA-CONTINUAL-003"
    assert current["status"] == "CANDIDATE_NOT_AUTHORIZED"
    assert current["frozen"] is current["execution_authorized"] is False
    assert current["learner_config"]["context_length"] == 16
    assert len(current["evaluation_matrix"]) == 1735
    assert current["review_status"]["human_consent"] is None
    assert current["scientific_status"]["source_retention"] == "NOT_EVALUATED"
    assert current["scientific_status"]["transfer_benefit"] == "NOT_EVALUATED"


def test_candidate_cost_ceiling_is_explicitly_unaccepted_with_all_required_blockers():
    config = p.protocol_config()
    assert config["limits"]["cpu_minutes"] == 300
    assert config["limits"]["internal_stop_minutes"] == 285
    assert config["limits"]["rss_bytes"] == 3 * 1024**3
    assert config["limits"]["cpu_cost_status"] == "CANDIDATE_UNMEASURED_NOT_ACCEPTED"
    unresolved = " ".join(config["unresolved_before_freeze"]).lower()
    assert all(
        word in unresolved
        for word in ("synthetic", "review", "runner", "workflow", "human", "consent")
    )
    assert config["review_status"]["runner_source"] is None
    assert config["review_status"]["workflow_source"] is None


def test_import_hash_matrix_and_cli_plan_cannot_import_ml_or_simulators():
    program = r"""
import contextlib, importlib.abc, io, json, runpy, sys
sys.path.insert(0, sys.argv[1])
class BlockML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'gymnasium', 'gym', 'matplotlib'}:
            raise AssertionError('planning imported ML: ' + fullname)
        if fullname in {'arena.continual.trainer', 'arena.continual.network',
                         'arena.continual.policy', 'arena.continual.evaluation',
                         'arena.universal.envs'}:
            raise AssertionError('planning imported runtime: ' + fullname)
sys.meta_path.insert(0, BlockML())
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    runpy.run_module('arena.continual.protocol', run_name='__main__')
payload = json.loads(buffer.getvalue())
from arena.continual import protocol as p
assert payload['sha256'] == p.protocol_sha256()
assert len(p.evaluation_matrix()) == 1735
assert payload['protocol']['execution_authorized'] is False
print(json.dumps({'status': payload['protocol']['status'], 'count': 1735}))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", program, str(ROOT)],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(result.stdout) == {
        "status": "CANDIDATE_NOT_AUTHORIZED",
        "count": 1735,
    }
