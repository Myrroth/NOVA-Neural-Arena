"""Fixed NNA-CONTINUAL-003 content for review; importing/planning grants no permit.

Only standard-library imports are allowed here. Models, environments and the
runner are deliberately absent. Historical 001/002 protocols remain unchanged.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json


MISSION = "NNA-CONTINUAL-003"
VERSION = "continual003-v0.2-fixed-for-approval"
STATUS = "FROZEN_APPROVAL_REQUIRED"
SOURCE_GAMES = ("snake", "pong", "agar", "flappy")
TARGET_GAME = "racing"
GAMES = SOURCE_GAMES + (TARGET_GAME,)
MODEL_SEEDS = (4103, 5207, 6301)
TEST_SEEDS = tuple(range(930001, 930021))
PIXEL_TEST_SEEDS = TEST_SEEDS[:5]
BOOTSTRAP_SEED = 530003
BOOTSTRAP_REPETITIONS = 10_000
MAX_EPISODE_STEPS = 400
STEPS_PER_GAME = 60_000
SOURCE_STEPS_PER_MODEL = 240_000
MAX_TRAINING_STEPS = 1_080_000
MAX_EVALUATION_STEPS = 694_000
MAX_TOTAL_STEPS = 1_800_000
MAX_Q_UPDATES = 67_500
MAX_WORLD_UPDATES = 33_750
# Revised finite proposal after the single measured Linux cost assessment.
# CPU is cumulative process time; two threads can consume two CPU seconds per
# wall second. These are separate ceilings, never a runtime guarantee/permit.
MAX_WALL_MINUTES = MAX_MINUTES = 360
MAX_CPU_MINUTES = 720
INTERNAL_STOP_MINUTES = 330
INTERNAL_STOP_CPU_MINUTES = 660
MAX_RSS_BYTES = 3 * 1024**3
TRAINING_SEED_NAMESPACE = "NNA-learning-corrections-v2"


@dataclass(frozen=True, order=True)
class EvaluationCell:
    stage: str
    policy: str
    game: str
    model_seed: int
    environment_seed: int


def evaluation_matrix() -> tuple[EvaluationCell, ...]:
    """Describe exactly 1,735 episodes; never construct or execute any of them."""
    regimes = (
        ("before_adaptation", "shared", GAMES, MODEL_SEEDS, TEST_SEEDS),
        ("after_adaptation", "shared", GAMES, MODEL_SEEDS, TEST_SEEDS),
        ("from_scratch", "scratch", (TARGET_GAME,), MODEL_SEEDS, TEST_SEEDS),
        ("baseline", "untrained", GAMES, MODEL_SEEDS, TEST_SEEDS),
        ("baseline", "random", GAMES, (0,), TEST_SEEDS),
        ("ablation", "world_off", GAMES, MODEL_SEEDS, TEST_SEEDS),
        ("ablation", "gru_reset", GAMES, MODEL_SEEDS, TEST_SEEDS),
        ("ablation", "pixels_hidden", GAMES, MODEL_SEEDS, PIXEL_TEST_SEEDS),
    )
    return tuple(
        EvaluationCell(stage, policy, game, model_seed, environment_seed)
        for stage, policy, games, models, environments in regimes
        for game in games
        for model_seed in models
        for environment_seed in environments
    )


def canonical_json(value) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def derive_training_seed(
    model_seed: int,
    game: str,
    episode: int,
    *,
    stage: str = "pretrain",
    calibration: bool = False,
) -> int:
    """Pure plan-time equivalent of the current v2 trainer's environment seed.

    Fit is even, training-calibration is odd, and both are >=2,000,000.
    Final holdout seeds are below 1,000,000. Adaptation/scratch intentionally
    share the target environment stream. Hash labels do not prove collision
    freedom within a stream; the parity/range separations are structural.
    """
    if type(model_seed) is not int or model_seed not in MODEL_SEEDS:
        raise ValueError("unknown independent model seed")
    if game not in GAMES or type(episode) is not int or episode < 0:
        raise ValueError("invalid game or training episode")
    if stage not in ("source", "pretrain", "adaptation", "from_scratch"):
        raise ValueError("unknown training stage")
    stage = "pretrain" if stage == "source" else stage
    if (stage == "pretrain") != (game in SOURCE_GAMES):
        raise ValueError("Racing must be excluded from source pretraining")
    if type(calibration) is not bool:
        raise ValueError("calibration must be boolean")
    payload = [
        TRAINING_SEED_NAMESPACE,
        model_seed,
        game,
        episode,
        "target" if stage in ("adaptation", "from_scratch") else stage,
        "calibration" if calibration else "fit",
    ]
    slots = (2**31 - 1 - 2_000_000) // 2
    return (
        2_000_000
        + 2
        * (
            int.from_bytes(hashlib.sha256(str(payload).encode()).digest()[:8], "big")
            % slots
        )
        + int(calibration)
    )


# Literal asdict(ContinualConfig()) snapshot. Importing the learner to obtain
# these values would import ML/Gym and undermine a side-effect-free plan.
_LEARNER_CONFIG = {
    "hidden_size": 64,
    "latent_size": 64,
    "context_length": 16,
    "unroll": 8,
    "n_step": 3,
    "batch_size": 16,
    "calibration_batch_size": 32,
    "replay_capacity": 40000,
    "capacity_per_game": 10000,
    "calibration_capacity": 8000,
    "calibration_every_episodes": 10,
    "warmup_steps": 128,
    "q_interval": 16,
    "world_interval": 32,
    "learning_rate": 0.0003,
    "world_learning_rate": 0.001,
    "gamma": 0.98,
    "target_tau": 0.01,
    "source_distillation_weight": 1.0,
    "adaptation_target_fraction": 0.5,
    "planning_fraction": 0.2,
    "epsilon_end": 0.05,
    "episode_steps": 400,
    "threads": 2,
    "max_training_steps": 1080000,
}

# Copied scientific decision thresholds from the immutable 002 protocol.
# Their provenance does not transfer its approvals or imply v2 has passed.
_GATES = {
    "snake_food_gain": 1.0,
    "pong_net_points_gain": 0.5,
    "pong_intercept_gain": 0.10,
    "agar_food_gain": 2.0,
    "agar_prey_gain": 0.3,
    "agar_death_reduction": 0.10,
    "flappy_pipe_gain": 1.0,
    "racing_progress_gain": 0.15,
    "racing_completion_gain": 0.10,
    "world_mse_ratio_maximum": 0.95,
    "pixel_action_change": 0.05,
    "temporal_action_change": 0.01,
    "world_action_change": 0.01,
    "ablation_changed_games": 4,
    "ablation_improved_games": 2,
    "resume_transitions_and_updates": 25,
    "resume_tolerance": 1e-6,
}

_CONFIG = {
    "mission": MISSION,
    "version": VERSION,
    "status": STATUS,
    "frozen": True,
    "execution_authorized": False,
    "freeze_meaning": "fixed pre-data content for exact A/B/P review; neither human consent nor Chat1 acceptance",
    "source_binding": "A and workflow-only descendant B are bound externally by the real Planning review and owner attestation after their commits exist; no self-referential source hash inside P",
    "development_pr": "https://github.com/Myrroth/NOVA-Neural-Arena/pull/14",
    "historical_delivery": "38b4ffb62bec74eead91e80eadc199fcdd7c9471",
    "historical_threshold_source": "arena/universal/protocol.py (NNA-MULTIJOGOS-002)",
    "historical_analysis_source": "arena/universal/analysis.py (unchanged reference formulas)",
    "scope": "the same five prepared local pixel simulators; no universal-game claim",
    "source_games": SOURCE_GAMES,
    "target_game": TARGET_GAME,
    "model_seeds": MODEL_SEEDS,
    "test_seeds": TEST_SEEDS,
    "pixel_test_seeds": PIXEL_TEST_SEEDS,
    "learner_config": _LEARNER_CONFIG,
    "network_version": "continual-drqn-world-residual-v2",
    "architecture": {
        "observation": [32, 32, 3],
        "dtype": "uint8",
        "actions": 10,
        "gru_units": 64,
        "per_game_heads": False,
        "algorithm": "recurrent Double DQN with three-step returns",
        "policy_inputs": [
            "current RGB",
            "own previous action",
            "own clipped previous reward",
            "constant mechanical action mask",
        ],
        "forbidden_policy_inputs": [
            "environment object",
            "game ID",
            "private telemetry",
            "future RGB",
            "winning actions or per-game expert code",
        ],
        "context": "reset BEFORE observation 0,16,32,...; reconstruct current context after control revision",
        "world_target": "detached online recurrent hidden after actual next RGB/action/clipped reward",
        "world_target_mode": "continuous_gru_one_step",
        "world_boundary_flag": "next_context_boundary",
        "world_target_at_boundary": "continuous successor for audit, not the next reset control state",
        "target_network_usage": "DoubleDQN TD bootstrap only; never planning",
        "world_lookahead": "predicted reward + gamma*(1-sigmoid(done_logit))*max legal online Q(predicted hidden)",
        "world_delta": "hidden+2*tanh(delta), clamped[-1,1]; zero delta initialization copies memory",
        "decoder": "inherited, frozen and unused; no reconstruction claim",
    },
    "training": {
        "source_steps_per_game_per_model": STEPS_PER_GAME,
        "source_steps_per_model": SOURCE_STEPS_PER_MODEL,
        "adaptation_steps_per_model": STEPS_PER_GAME,
        "scratch_steps_per_model": STEPS_PER_GAME,
        "steps_per_model": 360000,
        "total_steps": MAX_TRAINING_STEPS,
        "calibration_interactions": "included in the same stage step ceilings, never extra episodes",
        "checkpoint_selection": "fixed final source/adaptation/scratch budgets, never test-best",
        "untrained_control": "initial network for each independent model seed, without learning",
        "source_teacher": "own final source policy frozen before target adaptation, never an expert",
        "retention_replay": "protected own source fit experience; no target insertion into source copy",
        "loss_normalization": "mean over valid transitions per sequence, then mean across sequences",
        "optimizers": "separate Adam control and world; world updates detached current/next memory only",
        "adaptation_fit_sampling": "target .5, sources share .5, plus own-source policy distillation",
        "world_latent_fit": "exclude successor-reset transitions from latent fit; heldout continuous targets still audit all transitions with boundary flags",
        "update_fairness": {
            "design": "equal physical target-interaction budget, not equal realized optimizer updates",
            "physical_steps_per_target_arm": 60000,
            "q_interval": 16,
            "world_interval": 32,
            "warmup_steps_per_stage": 128,
            "same_config": True,
            "eligibility": "fit replay must satisfy warmup and complete-context requirements; reserved calibration episodes can delay availability",
            "required_report": [
                "scheduled opportunities",
                "attempted updates",
                "successful updates",
                "warmup/replay skips",
                "InsufficientContext skips",
                "fit/calibration physical exposure",
                "adaptation-minus-scratch update deltas",
            ],
            "on_different_realized_counts": "disclose differences; neither stop a valid run solely for unequal counts nor add compensating updates",
            "transfer_claim_scope": "sample efficiency at the same environment budget, not compute-matched training",
            "strict_compute_matching": "requires a new reviewed design before freeze; not claimed here",
        },
    },
    "calibration": {
        "every_episodes": 10,
        "fit_exclusion": "entire reserved episode excluded from all fit losses",
        "active_domain_only": True,
        "min_unique_transitions_per_legal_action": 8,
        "unique_support_key": ["episode_id", "absolute_transition_index"],
        "mse_ratio_maximum": 0.90,
        "reward_mae_slack": 0.02,
        "terminal_brier_slack": 0.01,
        "max_mix": 0.20,
        "all_legal_actions_must_pass": True,
        "gate_off": [
            "cold",
            "poor",
            "unsupported",
            "stale revision",
            "new target domain",
            "before world update",
            "before next periodic context reset",
        ],
        "final_holdout_use": "never refresh, select a gate or estimate its baselines on final data",
        "separate_role": "training-local safety gate; not the heldout world-benefit scientific criterion",
    },
    "random_generators": {
        "environment_namespace": TRAINING_SEED_NAMESPACE,
        "environment_derivation": "derive_training_seed; exact current trainer training_seed equivalent",
        "fit": "even seeds >=2000000",
        "calibration": "odd seeds >=2000000",
        "final_holdout": "930001..930020, never used by fit/calibration or engineering tests",
        "structural_disjointness": "fit/calibration parity; all training vs final by nonoverlapping ranges",
        "target_environments": "adaptation and from_scratch use identical per-episode target environment seeds",
        "initial_weights": "torch.manual_seed(model_seed)",
        "exploration": "local NumPy default_rng(model_seed+41), persisted for adaptation",
        "fit_replay": "local Python random.Random(model_seed+71), full state persisted",
        "evaluation": "NNA-CONTINUAL-V2-evaluation-v1 local RNG; preserve all global streams",
        "collision_caveat": "hash labels are not a proof of no collisions within a stream; record all used seeds",
    },
    "environment_revisions": {
        "snake": "visual-snake-v2",
        "pong": "visual-pong-v1",
        "agar": "visual-agar-v2",
        "flappy": "visual-flappy-v2",
        "racing": "visual-racing-v2",
    },
    "environment_config": {"max_steps": 400, "difficulty": 1.0, "scenario": "standard"},
    "pong_scope": {
        "learner_paddle_speed": 1.1,
        "npc_paddle_speed": 0.65,
        "npc_move_every_steps": 2,
        "claim": "only this unchanged asymmetric local opponent",
    },
    "evaluation": {
        "episode_count": 1735,
        "same_state_pairs": "every actual after_adaptation/shared step; fork BEFORE observation",
        "separate_ablations": "own recurrent histories and own executed previous actions/rewards",
        "gru_reset_intervention": "reset incoming memory only; same gate and periodic phase as normal",
        "world_prediction_rows": "all actual after_adaptation/shared transitions; no extra interactions",
        "selected_action_prediction_timing": "freeze chosen-action forecast before env.step",
        "q_mixed": "exact controller output; raw model_values logging must never bypass zero gate",
        "audit_fields": [
            "planning_weight",
            "context_reset",
            "target_mode",
            "next_context_boundary",
        ],
        "world_error_baselines": "copy current hidden; reward mean and terminal rate from TRAINING only",
        "world_reward_error": "MAE on clipped continuous reward, not reward Brier",
        "world_terminal_error": "Brier for true termination; truncation is not death",
        "demonstrations": [
            {
                "stage": "after_adaptation",
                "policy": "shared",
                "game": game,
                "model_seed": 5207,
                "environment_seed": TEST_SEEDS[0],
            }
            for game in GAMES
        ],
        "demonstration_selection": "these five fixed matrix cells, recorded on first evaluation; never best-of-test",
        "missing_demonstration": "preserve missing evidence; no substitute or extra episode",
    },
    "limits": {
        "evaluation_episode_steps": MAX_EPISODE_STEPS,
        "training_steps": MAX_TRAINING_STEPS,
        "evaluation_steps": MAX_EVALUATION_STEPS,
        "scheduled_maximum_steps": MAX_TRAINING_STEPS + MAX_EVALUATION_STEPS,
        "total_steps": MAX_TOTAL_STEPS,
        "q_updates": MAX_Q_UPDATES,
        "world_updates": MAX_WORLD_UPDATES,
        "combined_updates": MAX_Q_UPDATES + MAX_WORLD_UPDATES,
        "wall_minutes": MAX_WALL_MINUTES,
        "cpu_minutes": MAX_CPU_MINUTES,
        "internal_stop_minutes": INTERNAL_STOP_MINUTES,
        "internal_stop_wall_minutes": INTERNAL_STOP_MINUTES,
        "internal_stop_cpu_minutes": INTERNAL_STOP_CPU_MINUTES,
        "rss_bytes": MAX_RSS_BYTES,
        "cpu_cost_status": "MEASURED_LINUX_MODEL_REVIEWED_NOT_GUARANTEED_OR_AUTHORIZED",
        "cost_assessment": "docs/evidence/learning-v2/linux-cost-assessment.json",
        "measured_source": "635cf438d4334c5676341b893b4f2b1c898985da",
        "measured_run_id": 41752386162,
        "time_accounting": "cumulative training+evaluation process CPU and runner wall time; both bounded",
        "paid_resources": False,
        "campaign_attempts": 1,
        "automatic_retry": False,
        "extra_interaction_allowance": "none; global ceiling slack does not authorize more episodes or retries",
        "on_limit_or_failure": "stop once, preserve checkpoints/logs/partial results and report incomplete/FAIL",
    },
    "analysis": {
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "interval": 0.95,
        "unit": "independent training seed; environment seeds nested",
        "metric_aggregation": "per-model sum(numerators)/sum(denominators), not mean episode ratios",
        "resampling": "sample 3 model indices with replacement, then paired environment indices per sampled position",
        "pairing": "identical treatment/control model and environment indices; random rows shared across models",
        "bootstrap_rng": "local NumPy default_rng(bootstrap_seed XOR SHA256(label)[:8])",
        "interval_method": "percentile [.025,.975] of mean paired gains across sampled models",
        "pass_rule": "mean gain >= margin AND lower CI >0 AND at least 2 positive replica gains",
        "minimum_favorable_model_seeds": 2,
        "random_seed_shared_across_models": True,
        "multiple_testing": "all requirements conjunctive; no selected winner or optional stopping",
        "untrained_control": "paired positive gain and positive CI, margin zero",
        "flappy_strata": ["narrow_pipes_passed", "wide_pipes_passed"],
        "flappy_stratum_gate": "positive paired gain vs random in both physical crossing counters",
        "agar_exposure": "each replica encounters larger and smaller opponents; exposure alone is not escape causality",
        "racing_collision_gate": "collisions/distance no greater than random in each replica",
        "retention_gate": "each source game and replica: (after-random)/(before-random)>=.80",
        "retention_nonpositive_before": "MISSING evidence, never PASS",
        "retained_positive_gain_fraction": 0.80,
        "transfer_final_margin": 0.05,
        "transfer_auc_relative_margin": 0.10,
        "transfer_budget_matching": "60000 physical target steps per arm; report unequal realized updates and heldout exposure, no compute-matched claim",
        "training_auc": "step-weighted episode progress at right endpoints, exactly 60000 target steps including partial final episode",
        "auc_pass_rule": "mean(adaptation_AUC/scratch_AUC-1)>=.10 and at least 2 positive replicas",
        "auc_zero_scratch": "MISSING evidence; relative gain undefined",
        "world_gate": "model/copy latent MSE<=.95 in every game and replica; zero copy MSE is MISSING",
        "world_reward_and_terminal_metrics": "report MAE/Brier against training baselines; no new relaxed competence criterion",
        "ablation_gate": "action-change threshold in >=4 games AND significant objective gain in >=2 games",
        "missing_evidence": "never PASS; preserve missing/failed gates and all negative results",
        "three_replica_limit": "descriptive evidence under preregistration, not a universal guarantee",
    },
    "gates": _GATES,
    "scientific_status": {
        "competence": "NOT_EVALUATED",
        "source_retention": "NOT_EVALUATED",
        "transfer_benefit": "NOT_EVALUATED",
        "world_benefit": "NOT_EVALUATED",
        "memory_and_pixel_benefit": "NOT_EVALUATED",
        "engineering_tests": "mechanism checks do not establish learning or scientific PASS",
        "historical_results": "001/002 remain separate and unchanged; never relabel their outcomes as 003",
    },
    "review_status": {
        "chat1_recertification": "NOT_RECEIVED; latest request has no reviewed response",
        "independent_review": "Engineering and offline analysis/cost/transport reviews completed; not scientific competence or Chat1 approval",
        "measured_cost_acceptance": "Independent modeled-cost review; human consent remains absent",
        "runner_source": None,
        "workflow_source": None,
        "human_consent": None,
    },
    "unresolved_before_freeze": [],
    "required_before_execution": [
        "All cost/review/source/workflow issues resolved and a new immutable final protocol reviewed and frozen.",
        "Distinct explicit human consent to that exact finite 003 campaign, recorded with a new reference.",
        "Literal technical recertification bound to final source/protocol/workflow hashes, without assuming an unread Chat1 request is approval.",
        "One fresh remote reservation under non-cancelling global concurrency; no prior 003 attempt, automatic repeat, paid resource or merge.",
    ],
}


def protocol_config() -> dict:
    """Return the detached fixed plan; authorization is external and absent here."""
    config = copy.deepcopy(_CONFIG)
    config["evaluation_matrix"] = [asdict(cell) for cell in evaluation_matrix()]
    return config


def protocol_sha256() -> str:
    """Hash the fixed content for review; a hash is not consent."""
    return hashlib.sha256(canonical_json(protocol_config()).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    print(canonical_json({"sha256": protocol_sha256(), "protocol": protocol_config()}))
