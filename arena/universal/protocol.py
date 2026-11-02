"""Finite candidate preregistration for NNA-MULTIJOGOS-002; never a runner permit.

The coordinating Chat 1 comment supersedes the earlier Chat 3/4 proposals.
Importing this module does not construct an environment or run an experiment.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json

MISSION = "NNA-MULTIJOGOS-002"
VERSION = "mg002-plano-v0.2-pre-data-clarifications"
PRIMARY_PROTOCOL_URL = (
    "https://github.com/Myrroth/NOVA-Neural-Arena/issues/12#issuecomment-6583974240"
)
HISTORICAL_BASE = "bce5db99f38466119a0818e36312b1f82072615d"
SOURCE_GAMES = ("snake", "pong", "agar", "flappy")
TARGET_GAME = "racing"
GAMES = SOURCE_GAMES + (TARGET_GAME,)
MODEL_SEEDS = (1103, 2207, 3301)
TEST_SEEDS = tuple(range(920001, 920021))
PIXEL_TEST_SEEDS = TEST_SEEDS[:5]
BOOTSTRAP_SEED = 510002
BOOTSTRAP_REPETITIONS = 10_000
MAX_EPISODE_STEPS = 400
STEPS_PER_GAME = 60_000
MAX_TRAINING_STEPS = 1_080_000
MAX_EVALUATION_STEPS = 694_000
MAX_TOTAL_STEPS = 1_800_000
MAX_Q_UPDATES = 67_500
MAX_WORLD_UPDATES = 33_750
MAX_MINUTES = 300


@dataclass(frozen=True, order=True)
class EvaluationCell:
    stage: str
    policy: str
    game: str
    model_seed: int
    environment_seed: int


def evaluation_matrix() -> tuple[EvaluationCell, ...]:
    """Exactly 1,735 final episodes, each capped at 400 interactions."""
    cells = []
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
    for stage, policy, games, models, environments in regimes:
        for game in games:
            for model_seed in models:
                for environment_seed in environments:
                    cells.append(EvaluationCell(stage, policy, game,
                                                model_seed, environment_seed))
    return tuple(cells)


def derive_training_seed(model_seed: int, game: str, episode: int,
                         *, stage: str = "source", stream: str = "environment") -> int:
    """Versioned SHA256 derivation, disjoint from the reserved test namespace.

    The tuple is JSON-encoded rather than ambiguously concatenated. Scratch and
    adaptation environment streams intentionally match; exploration/replay do not.
    """
    if model_seed not in MODEL_SEEDS or type(model_seed) is not int:
        raise ValueError("unknown independent model seed")
    if game not in GAMES or type(episode) is not int or episode < 0:
        raise ValueError("invalid game or training episode")
    if stage not in ("source", "adaptation", "from_scratch"):
        raise ValueError("unknown training stage")
    if stream not in ("environment", "exploration", "replay"):
        raise ValueError("unknown random stream")
    if (stage == "source") != (game in SOURCE_GAMES):
        raise ValueError("Racing must be excluded from source pretraining")
    seed_stage = "target" if stream == "environment" and stage != "source" else stage
    payload = [MISSION, VERSION, model_seed, game, episode, "train", seed_stage, stream]
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).digest()
    # Keep training outside the entire six-digit evaluation namespace.
    return 1_000_000 + int.from_bytes(digest[:8], "big") % (2**31-1-1_000_000)


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


_CONFIG = {
    "mission": MISSION, "version": VERSION,
    "primary_protocol": PRIMARY_PROTOCOL_URL,
    "pre_data_amendment": "https://github.com/Myrroth/NOVA-Neural-Arena/issues/12#issuecomment-6590543972",
    "historical_base": HISTORICAL_BASE,
    "status": "CANDIDATE_NOT_FROZEN",
    "execution_authorized": False,
    "scope": "five prepared local pixel simulators; no universal-game claim",
    "source_games": SOURCE_GAMES, "target_game": TARGET_GAME,
    "model_seeds": MODEL_SEEDS, "test_seeds": TEST_SEEDS,
    "pixel_test_seeds": PIXEL_TEST_SEEDS,
    "architecture": {"observation": [32, 32, 3], "dtype": "uint8",
                     "actions": 10, "gru_units": 64, "per_game_heads": False,
                     "algorithm": "recurrent Double DQN", "gamma": 0.98,
                     "world_weight": 0.20, "burn_in": 4, "unroll": 8,
                     "batch_size": 16, "replay_capacity": 40_000,
                     "q_update_every": 16, "world_update_every": 32,
                     "adaptation_target_replay_fraction": 0.80,
                     "reward_clipping": [-1, 1],
                     "world_target": "detached ONLINE recurrent hidden after observing actual next RGB; same coordinates in control and held-out audit",
                     "world_lookahead": "predicted_reward_ONLINE + gamma*(1-sigmoid(done_logit_ONLINE))*max_masked_Q_ONLINE(predicted_next_hidden_ONLINE)",
                     "target_network_usage": "DoubleDQN TD bootstrap only; never planning"},
    "training": {"source_steps_per_game_per_model": STEPS_PER_GAME,
                 "adaptation_steps_per_model": STEPS_PER_GAME,
                 "scratch_steps_per_model": STEPS_PER_GAME,
                 "total_steps": MAX_TRAINING_STEPS,
                 "checkpoint_selection": "fixed final budget, never test-best"},
    "learner_config": {"hidden_size": 64, "latent_size": 64, "gamma": .98,
                       "learning_rate": .0003, "batch_size": 16, "burn_in": 4,
                       "unroll": 8, "replay_capacity": 40000, "capacity_per_game": 10000,
                       "warmup_steps": 128, "q_interval": 16, "world_interval": 32,
                       "target_tau": .01, "planning_fraction": .20, "epsilon_end": .05,
                       "episode_steps": 400, "threads": 2, "max_training_steps": 1080000},
    "network_version": "visual-drqn-world-residual-v1",
    "world_loss": {"next_hidden_mse": 1., "reward_huber": .5,
                   "terminal_bce": .1, "encoder_thumbnail_mse": .1,
                   "next_hidden_target_gradient": False,
                   "delta_initialization": "zero; hidden+.1*tanh(delta), clamped[-1,1]"},
    "optimizer": {"type": "Adam", "lr": .0003, "gradient_clip_norm": 10.,
                  "shared_across_games": True, "continue_for_adaptation": True,
                  "target_ema_for": "DoubleDQN only"},
    "exploration": {"start_epsilon": 1., "end_epsilon": .05,
                    "decay": "linear over first .8*stage_budget; reset schedule for target and scratch",
                    "unbound_buttons": "environment no-op; excluded by constant mechanical mask"},
    "random_generators": {"initial_weights": "torch.manual_seed(model_seed)",
                          "exploration": "NumPy default_rng(model_seed+41), continued across adaptation",
                          "replay": "Python random.Random(model_seed+71), full getstate persisted across adaptation",
                          "environment": "SHA256 derivation in derive_training_seed; Gymnasium PCG64",
                          "target_environments": "same per-episode seeds for adaptation and from_scratch"},
    "environment_revisions": {"snake": "visual-snake-v2", "pong": "visual-pong-v1",
                              "agar": "visual-agar-v2", "flappy": "visual-flappy-v2",
                              "racing": "visual-racing-v2"},
    "environment_config": {"max_steps": 400, "difficulty": 1., "scenario": "standard"},
    "pong_scope": {"learner_paddle_speed": 1.1, "npc_paddle_speed": .65,
                   "npc_move_every_steps": 2,
                   "claim": "only this frozen asymmetric local opponent; no general Pong mastery"},
    "causal_trace": {"source": "five fixed already-recorded final demonstration episodes",
                     "method": "offline current-RGB recurrent reconstruction, no new rollout or future observation",
                     "all_action_predictions": 10, "next_hidden_shape": [10, 64],
                     "max_absolute_reconstruction_error": 1e-6,
                     "weights": "same after_adaptation model2207 fingerprint as recording",
                     "selected_prediction_timing": "computed before env.step; next RGB/reward only for error target"},
    "limits": {"evaluation_episode_steps": MAX_EPISODE_STEPS,
               "evaluation_steps": MAX_EVALUATION_STEPS,
               "total_steps": MAX_TOTAL_STEPS, "q_updates": MAX_Q_UPDATES,
               "world_updates": MAX_WORLD_UPDATES, "combined_updates": 101_250,
               "cpu_minutes": MAX_MINUTES, "rss_bytes": 3*1024**3,
               "time_accounting": "both runner wall time and process CPU; cumulative train+eval; internal STOP285min, job timeout300min",
               "paid_resources": False, "campaign_attempts": 1,
               "automatic_retry": False,
               "preflight_random_steps": 2_000,
               "preflight_synthetic_updates": 100},
    "analysis": {"bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
                 "bootstrap_seed": BOOTSTRAP_SEED, "interval": 0.95,
                 "unit": "independent training seed; environment seeds nested",
                 "minimum_favorable_model_seeds": 2,
                 "random_seed_shared_across_models": True,
                 "multiple_testing": "all requirements conjunctive; no selected winner",
                 "untrained_control": "report paired gain and require positive gain/CI",
                 "flappy_strata": ["narrow_pipes_passed", "wide_pipes_passed"],
                 "flappy_stratum_gate": "positive paired gain vs random in both physical crossing counters",
                 "world_prediction_rows": "all existing after_adaptation/shared final transitions; no extra episodes",
                 "world_holdout_exclusion": "training data, not other final evaluation metrics",
                 "world_reward_error": "MAE of clipped continuous reward; no reward Brier",
                 "world_terminal_error": "Brier vs train-only terminal base rate; truncation is not terminal",
                 "training_auc": "step-weighted episode progress at right endpoints, exactly 60000 target steps",
                 "retention_gate": "each source game and model seed retains positive pre-adaptation gain",
                 "retained_positive_gain_fraction": 0.80,
                 "transfer_final_margin": 0.05, "transfer_auc_relative_margin": 0.10},
    "gates": {"snake_food_gain": 1.0, "pong_net_points_gain": 0.5,
              "pong_intercept_gain": 0.10, "agar_food_gain": 2.0,
              "agar_prey_gain": 0.3, "agar_death_reduction": 0.10,
              "flappy_pipe_gain": 1.0, "racing_progress_gain": 0.15,
              "racing_completion_gain": 0.10,
              "world_mse_ratio_maximum": 0.95,
              "pixel_action_change": 0.05, "temporal_action_change": 0.01,
              "world_action_change": 0.01,
              "ablation_changed_games": 4, "ablation_improved_games": 2,
              "resume_transitions_and_updates": 25, "resume_tolerance": 1e-6},
    "unresolved_before_freeze": [
        "Chat2/3/4 independent final source review and Chat1 recertification are pending.",
        "Final source/CI and offline causal reconstruction validation must be bound before approval.",
        "G1 cost equivalence for moving the same prediction forward before env.step needs final Chat1 recertification; no new games or updates.",
    ],
}


def protocol_config() -> dict:
    config = copy.deepcopy(_CONFIG)
    config["evaluation_matrix"] = [asdict(cell) for cell in evaluation_matrix()]
    return config


def protocol_sha256() -> str:
    """Candidate content hash; hashing is not scientific execution approval."""
    return hashlib.sha256(canonical_json(protocol_config()).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    print(canonical_json({"sha256": protocol_sha256(), "protocol": protocol_config()}))
