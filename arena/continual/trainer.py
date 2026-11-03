"""Finite continual learner v2; no experiment starts by importing this module.

Fits pixels and own consequences, never a game-specific controller. Historical
002 models/source are deliberately not accepted as v2 continuation snapshots.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import math
import random

import numpy as np
import torch
from torch.nn import functional as F

from arena.universal.envs import make_game, _json_value
from arena.universal.replay import SequenceReplay
from .network import ContinualQNetwork
from .calibration import GateCalibration
from .objectives import masked_sequence_mean, policy_distillation, n_step_targets
from .replay import sample_context_sequences, InsufficientContext

SOURCE_GAMES = ("snake", "pong", "agar", "flappy")
TARGET_GAME = "racing"


@dataclass(frozen=True)
class ContinualConfig:
    hidden_size: int = 64
    latent_size: int = 64
    context_length: int = 16
    unroll: int = 8
    n_step: int = 3
    batch_size: int = 16
    calibration_batch_size: int = 32
    replay_capacity: int = 40000
    capacity_per_game: int = 10000
    calibration_capacity: int = 8000
    calibration_every_episodes: int = 10
    warmup_steps: int = 128
    q_interval: int = 16
    world_interval: int = 32
    learning_rate: float = 0.0003
    world_learning_rate: float = 0.001
    gamma: float = 0.98
    target_tau: float = 0.01
    source_distillation_weight: float = 1.0
    adaptation_target_fraction: float = 0.5
    planning_fraction: float = 0.2
    epsilon_end: float = 0.05
    episode_steps: int = 400
    threads: int = 2
    max_training_steps: int = 1080000

    def __post_init__(self):
        ints = (
            "hidden_size",
            "latent_size",
            "context_length",
            "unroll",
            "n_step",
            "batch_size",
            "calibration_batch_size",
            "replay_capacity",
            "capacity_per_game",
            "calibration_capacity",
            "warmup_steps",
            "q_interval",
            "world_interval",
            "episode_steps",
            "max_training_steps",
        )
        if any(type(getattr(self, k)) is not int or getattr(self, k) < 1 for k in ints):
            raise ValueError("positive integer learner parameters required")
        if (
            not 2 <= self.context_length <= 64
            or self.unroll > 64
            or self.n_step > self.unroll
            or max(self.batch_size, self.calibration_batch_size) > 128
            or self.hidden_size > 512
            or self.latent_size > 512
            or self.replay_capacity > 40000
            or self.calibration_capacity > 8000
            or self.capacity_per_game > self.replay_capacity
            or self.max_training_steps > 1080000
            or self.episode_steps > 400
            or type(self.calibration_every_episodes) is not int
            or self.calibration_every_episodes < 2
            or type(self.threads) is not int
            or self.threads not in (1, 2)
        ):
            raise ValueError("finite technical bounds exceeded")
        floats = (
            "learning_rate",
            "world_learning_rate",
            "gamma",
            "target_tau",
            "source_distillation_weight",
            "adaptation_target_fraction",
            "planning_fraction",
            "epsilon_end",
        )
        if any(
            type(getattr(self, k)) not in (int, float)
            or not math.isfinite(getattr(self, k))
            for k in floats
        ):
            raise ValueError("finite real learner parameters required")
        if (
            min(self.learning_rate, self.world_learning_rate) <= 0
            or not 0 < self.gamma < 1
            or not 0 < self.target_tau <= 1
            or self.source_distillation_weight < 0
            or not 0 < self.adaptation_target_fraction < 1
            or not 0 <= self.planning_fraction <= 0.2
            or not 0 <= self.epsilon_end <= 1
        ):
            raise ValueError("invalid learner parameters")


def training_seed(model_seed, game, episode, stage, calibration):
    """Separate lineage and fit/calibration namespaces; never final test seeds."""
    payload = [
        "NNA-learning-corrections-v2",
        int(model_seed),
        game,
        int(episode),
        "target" if stage in ("adaptation", "from_scratch") else stage,
        "calibration" if calibration else "fit",
    ]
    # Distinct hash labels alone cannot rule out collisions. Even/odd streams
    # make fit/calibration disjoint by construction within the 31-bit range.
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


class ContinualTrainer:
    def __init__(self, model_seed: int, config: ContinualConfig | None = None):
        if type(model_seed) is not int or not 0 <= model_seed < 2**31:
            raise ValueError("model_seed must be a nonnegative 31-bit integer")
        self.model_seed, self.config = model_seed, config or ContinualConfig()
        torch.set_num_threads(self.config.threads)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(model_seed)
        random.seed(model_seed)
        np.random.seed(model_seed)
        self.rng = np.random.default_rng(model_seed + 41)
        self.online = ContinualQNetwork(
            hidden_size=self.config.hidden_size, latent_size=self.config.latent_size
        )
        self.target = copy.deepcopy(self.online).requires_grad_(False)
        self.source_teacher = None
        self.policy_optimizer = torch.optim.Adam(
            self.online.control_parameters(), lr=self.config.learning_rate
        )
        self.world_optimizer = torch.optim.Adam(
            self.online.world_parameters(), lr=self.config.world_learning_rate
        )
        self.replay = SequenceReplay(
            self.config.capacity_per_game,
            model_seed + 71,
            capacity_total=self.config.replay_capacity,
        )
        self.calibration_replay = SequenceReplay(
            min(self.config.capacity_per_game, self.config.calibration_capacity),
            model_seed + 73,
            capacity_total=self.config.calibration_capacity,
        )
        self.source_replay = None
        self.calibration = GateCalibration(max_mix=self.config.planning_fraction)
        self.stage = "unconfigured"
        self.stage_budget = self.stage_steps = self.global_steps = 0
        self.q_updates = self.world_updates = self.policy_revision = 0
        self.fit_steps = self.calibration_steps = 0
        self.last_q_due = self.last_world_due = self.next_actor = 0
        self.games = ()
        self.actors = {}
        self.episode_counters = {}
        self.training_curve = []
        self.loss_history = []
        self.last_update = {}

    @property
    def burn_in(self):
        return self.config.context_length - 1

    def _is_calibration(self, episode):
        return (episode + 1) % self.config.calibration_every_episodes == 0

    def _new_actor(self, game):
        episode = self.episode_counters[game]
        calibration = self._is_calibration(episode)
        seed = training_seed(self.model_seed, game, episode, self.stage, calibration)
        env = make_game(game, max_steps=self.config.episode_steps)
        obs, _ = env.reset(seed=seed)
        return {
            "env": env,
            "observation": obs,
            "previous_action": 0,
            "previous_reward": 0.0,
            "episode_steps": 0,
            "episode_return": 0.0,
            "environment_seed": seed,
            "calibration_episode": calibration,
            "history": [],
        }

    def configure_stage(self, stage, budget):
        if (
            stage not in ("pretrain", "adaptation", "from_scratch")
            or type(budget) is not int
            or budget < 1
        ):
            raise ValueError("explicit finite stage and positive budget required")
        if self.global_steps + budget > self.config.max_training_steps:
            raise ValueError("finite total training cap exceeded")
        allowed = (
            self.stage == "unconfigured"
            and stage in ("pretrain", "from_scratch")
            or self.stage == "pretrain"
            and self.stage_steps == self.stage_budget
            and stage == "adaptation"
        )
        if not allowed:
            raise ValueError("cannot repeat or abandon a finite stage")
        if stage == "pretrain" and budget % len(SOURCE_GAMES):
            raise ValueError("source budget must balance four games")
        for game, actor in self.actors.items():
            self.replay.truncate_active(game)
            self.calibration_replay.truncate_active(game)
            actor["env"].close()
        if stage == "adaptation":
            self.source_teacher = (
                copy.deepcopy(self.online).requires_grad_(False).eval()
            )
            self.source_replay = copy.deepcopy(self.replay)
        self.stage, self.stage_budget, self.stage_steps = stage, budget, 0
        self.last_q_due = self.last_world_due = self.next_actor = 0
        self.games = SOURCE_GAMES if stage == "pretrain" else (TARGET_GAME,)
        self.episode_counters = {g: 0 for g in self.games}
        self.actors = {g: self._new_actor(g) for g in self.games}
        self.calibration = GateCalibration(max_mix=self.config.planning_fraction)

    def epsilon(self):
        f = min(1.0, self.stage_steps / max(1, 0.8 * self.stage_budget))
        return 1 + f * (self.config.epsilon_end - 1)

    def _observe_actor(self, actor):
        # A cache is valid only for the exact control-parameter revision that
        # produced it. Reconstruct past context after an optimizer update. The
        # periodic boundaries agree with replay and bound reconstruction cost.
        boundary = len(actor["history"]) == self.config.context_length
        past = [] if boundary else actor["history"]
        history = [
            *past,
            (
                actor["observation"].copy(),
                actor["previous_action"],
                actor["previous_reward"],
            ),
        ]
        pixels = torch.from_numpy(actor["observation"]).permute(2, 0, 1)[None]
        pa = torch.tensor([actor["previous_action"]], dtype=torch.long)
        pr = torch.tensor([actor["previous_reward"]], dtype=torch.float32)
        with torch.inference_mode():
            hidden = self.online.initial(1)
            if past and actor.get("_cached_revision") == self.policy_revision:
                hidden = actor["_cached_hidden"]
            elif past:
                frames = torch.from_numpy(np.stack([x[0] for x in past])).permute(
                    0, 3, 1, 2
                )[None]
                previous = torch.tensor([[x[1] for x in past]], dtype=torch.long)
                rewards = torch.tensor([[x[2] for x in past]], dtype=torch.float32)
                _, states, _ = self.online.sequence(frames, previous, rewards)
                hidden = states[:, -1]
            q, h, _ = self.online.step(pixels, pa, pr, hidden)
            mask = torch.from_numpy(actor["env"].action_mask())[None]
            mix = self.calibration.weights(mask, revision=self.policy_revision)
            # A scheduled controller-memory reset is outside the ordinary
            # model transition; never ask its lookahead to bridge that boundary.
            if len(history) == self.config.context_length:
                mix = torch.zeros_like(mix)
            values = q.clone()
            if bool((mix > 0).any()):
                plan = self.online.plan_values(h, mask, self.config.gamma)
                legal = mask & (mix[:, None] > 0)
                values[legal] = (
                    (1 - mix[:, None]) * q + mix[:, None] * plan.masked_fill(~mask, 0)
                )[legal]
            values = values.masked_fill(~mask, -torch.inf)
        return values[0], history, h

    def collect(self, transitions, *, learn=False):
        """Explicit bounded collection. No workflow/import invokes it implicitly.

        Scientific use needs its own new human-approved finite protocol; the
        campaign002 permission and workflows cannot execute this lineage.
        """
        if type(transitions) is not int or transitions < 0 or not self.actors:
            raise ValueError("configure actors and supply nonnegative count")
        if (
            transitions > self.stage_budget - self.stage_steps
            or self.global_steps + transitions > self.config.max_training_steps
        ):
            raise ValueError("requested collection exceeds finite stage/total budget")
        for _ in range(transitions):
            game = self.games[self.next_actor]
            actor = self.actors[game]
            values, history, next_hidden = self._observe_actor(actor)
            mask = actor["env"].action_mask()
            action = (
                int(self.rng.choice(np.flatnonzero(mask)))
                if self.rng.random() < self.epsilon()
                else int(values.argmax())
            )
            after, reward, term, trunc, info = actor["env"].step(action)
            replay = (
                self.calibration_replay if actor["calibration_episode"] else self.replay
            )
            replay.append(
                game,
                actor["observation"],
                action,
                reward,
                after,
                term,
                trunc,
                mask,
                actor["env"].action_mask(),
                prev_action=actor["previous_action"],
                prev_reward=actor["previous_reward"],
            )
            self.calibration_steps += int(actor["calibration_episode"])
            self.fit_steps += int(not actor["calibration_episode"])
            actor.update(
                observation=after,
                previous_action=action,
                previous_reward=float(np.clip(reward, -1, 1)),
                history=history,
                _cached_hidden=next_hidden,
                _cached_revision=self.policy_revision,
            )
            actor["episode_steps"] += 1
            actor["episode_return"] += reward
            self.global_steps += 1
            self.stage_steps += 1
            if term or trunc:
                self.training_curve.append(
                    self._episode_record(game, actor, info, True)
                )
                actor["env"].close()
                self.episode_counters[game] += 1
                self.actors[game] = self._new_actor(game)
            self.next_actor = (self.next_actor + 1) % len(self.games)
            if learn:
                self._due_updates()
        if transitions and self.stage_steps == self.stage_budget:
            for game, actor in self.actors.items():
                if actor["episode_steps"]:
                    self.training_curve.append(
                        self._episode_record(game, actor, actor["env"].metrics(), False)
                    )
        return {
            "global_steps": self.global_steps,
            "fit_steps": self.fit_steps,
            "calibration_steps": self.calibration_steps,
            "q_updates": self.q_updates,
            "world_updates": self.world_updates,
        }

    def _episode_record(self, game, actor, info, complete):
        return {
            "model_seed": self.model_seed,
            "stage": self.stage,
            "game": game,
            "environment_seed": actor["environment_seed"],
            "episode": self.episode_counters[game],
            "calibration_episode": actor["calibration_episode"],
            "steps": actor["episode_steps"],
            "return": actor["episode_return"],
            "global_step": self.global_steps,
            "training_step": self.stage_steps,
            "episode_complete": complete,
            **{k: v for k, v in info.items() if k not in ("action_mask", "steps")},
        }

    def _due_updates(self):
        if (
            self.stage_steps < self.config.warmup_steps
            or len(self.replay) < self.config.warmup_steps
        ):
            return
        q_due = self.stage_steps // self.config.q_interval
        world_due = self.stage_steps // self.config.world_interval
        if q_due > self.last_q_due:
            self.last_q_due = q_due
            try:
                self.learn_batch(world=False)
            except InsufficientContext:
                pass  # no valid context yet; never invent a hidden state
        if world_due > self.last_world_due:
            self.last_world_due = world_due
            try:
                self.learn_batch(world=True)
            except InsufficientContext:
                pass

    def _batch(self, replay=None, *, calibration=False, source=False):
        replay = self.replay if replay is None else replay
        weights = None
        if self.stage == "adaptation" and not source and not calibration:
            fraction = self.config.adaptation_target_fraction
            weights = {
                g: (
                    fraction if g == TARGET_GAME else (1 - fraction) / len(SOURCE_GAMES)
                )
                for g in replay.games
            }
        if calibration:
            # A source-only calibration buffer is not evidence about a new
            # target domain, even when its input buttons happen to match.
            weights = {g: 1.0 for g in self.games if g in replay.games}
        data = sample_context_sequences(
            replay,
            (
                self.config.calibration_batch_size
                if calibration
                else self.config.batch_size
            ),
            self.config.context_length,
            self.config.unroll,
            game_weights=weights,
        )
        frames = torch.from_numpy(data["frames"]).permute(0, 1, 4, 2, 3).contiguous()
        batch = {
            k: torch.from_numpy(v)
            for k, v in data.items()
            if isinstance(v, np.ndarray) and k != "frames"
        }
        return frames, batch

    def _sequence(self, network, pixels, batch):
        burn = self.burn_in
        hidden = None
        if burn:
            with torch.no_grad():
                _, states, _ = network.sequence(
                    pixels[:, :burn],
                    batch["prev_actions"][:, :burn],
                    batch["prev_rewards"][:, :burn],
                    resets=batch["resets"][:, :burn],
                )
                hidden = states[:, -1].detach()
        return network.sequence(
            pixels[:, burn:],
            batch["prev_actions"][:, burn:],
            batch["prev_rewards"][:, burn:],
            hidden=hidden,
            resets=batch["resets"][:, burn:],
        )

    def _distillation(self):
        if self.source_teacher is None or self.source_replay is None:
            return None
        pixels, b = self._batch(self.source_replay, source=True)
        student, _, _ = self._sequence(self.online, pixels, b)
        with torch.no_grad():
            teacher, _, _ = self._sequence(self.source_teacher, pixels, b)
        burn = self.burn_in
        return policy_distillation(
            student[:, :-1],
            teacher[:, :-1],
            b["masks"][:, burn:-1],
            b["valid"][:, burn:],
        )

    def learn_batch(self, *, world=False):
        """One update on fit replay; validation episodes never enter this loss."""
        pixels, b = self._batch()
        burn = self.burn_in
        valid = b["valid"][:, burn:]
        rewards = b["rewards"][:, burn:].float().clamp(-1, 1)
        terms = b["terminated"][:, burn:]
        actions = b["actions"][:, burn:].long()
        if world:
            with torch.no_grad():
                _, states, _ = self._sequence(self.online, pixels, b)
            predicted_h, predicted_r, done = self.online.model(
                states[:, :-1].detach().reshape(-1, self.config.hidden_size),
                actions.reshape(-1),
            )
            successor = states[:, 1:].detach()
            latent_valid = valid & ~b["resets"][:, burn + 1 :]
            errors = (predicted_h.reshape_as(successor) - successor).square().mean(-1)
            latent = masked_sequence_mean(errors, latent_valid)
            reward = masked_sequence_mean(
                F.smooth_l1_loss(
                    predicted_r.reshape_as(rewards), rewards, reduction="none"
                ),
                valid,
            )
            terminal = masked_sequence_mean(
                F.binary_cross_entropy_with_logits(
                    done.reshape_as(rewards), terms.float(), reduction="none"
                ),
                valid,
            )
            loss = latent + 0.5 * reward + 0.1 * terminal
            optimizer = self.world_optimizer
        else:
            q, _, _ = self._sequence(self.online, pixels, b)
            with torch.no_grad():
                target_q, _, _ = self._sequence(self.target, pixels, b)
                selected = (
                    q.detach()
                    .masked_fill(~b["masks"][:, burn:], -1e9)
                    .argmax(-1, keepdim=True)
                )
                bootstrap = target_q.gather(-1, selected).squeeze(-1)
                target = n_step_targets(
                    rewards,
                    terms,
                    b["truncated"][:, burn:],
                    valid,
                    bootstrap,
                    self.config.gamma,
                    self.config.n_step,
                )
            chosen = q[:, :-1].gather(-1, actions[..., None]).squeeze(-1)
            td = masked_sequence_mean(
                F.smooth_l1_loss(chosen, target, reduction="none"), valid
            )
            retain = self._distillation()
            loss = (
                td
                if retain is None
                else td + self.config.source_distillation_weight * retain
            )
            optimizer = self.policy_optimizer
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite learning loss")
        # Clear both groups to expose accidental cross-objective gradients.
        self.policy_optimizer.zero_grad(set_to_none=True)
        self.world_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (
                self.online.world_parameters()
                if world
                else self.online.control_parameters()
            ),
            10.0,
        )
        if world:
            # The planner itself changes even if control parameters do not.
            # Fail closed before mutation; a failed/skipped refresh cannot
            # leave an old confidence record attached to a new world model.
            self.calibration = GateCalibration(max_mix=self.config.planning_fraction)
        optimizer.step()
        if world:
            self.world_updates += 1
            if self._calibration_ready():
                self.refresh_calibration()
        else:
            self.q_updates += 1
            self.policy_revision += 1
            with torch.no_grad():
                for target_param, param in zip(
                    self.target.parameters(), self.online.parameters()
                ):
                    target_param.lerp_(param, self.config.target_tau)
        value = float(loss.detach())
        self.last_update = {
            "kind": "world" if world else "q",
            "loss": value,
            "policy_revision": self.policy_revision,
        }
        self.loss_history.append({"global_step": self.global_steps, **self.last_update})
        return value

    def _calibration_ready(self):
        counts = self.calibration_replay.game_counts()
        return (
            bool(self.games)
            and all(counts.get(g, 0) > 0 for g in self.games)
            and sum(counts.get(g, 0) for g in self.games) >= self.config.warmup_steps
        )

    def refresh_calibration(self):
        """Re-evaluate separate training-calibration replay in current coordinates."""
        self.calibration = GateCalibration(max_mix=self.config.planning_fraction)
        if not self._calibration_ready():
            return False
        try:
            pixels, b = self._batch(self.calibration_replay, calibration=True)
        except InsufficientContext:
            return False
        burn = self.burn_in
        with torch.inference_mode():
            _, states, _ = self._sequence(self.online, pixels, b)
            actions = b["actions"][:, burn:]
            hp, rp, dp = self.online.model(
                states[:, :-1].reshape(-1, self.config.hidden_size), actions.reshape(-1)
            )
            valid = b["valid"][:, burn:] & ~b["resets"][:, burn + 1 :]
            # Replacement sampling and overlapping unrolls must not manufacture
            # independent support by counting the same transition repeatedly.
            seen = set()
            for row in range(valid.shape[0]):
                for time in range(valid.shape[1]):
                    if not bool(valid[row, time]):
                        continue
                    identity = (
                        int(b["episode_ids"][row]),
                        int(b["start_indices"][row]) + time,
                    )
                    if identity in seen:
                        valid[row, time] = False
                    else:
                        seen.add(identity)
            self.calibration.refresh(
                hp.reshape_as(states[:, 1:]),
                states[:, 1:],
                states[:, :-1],
                rp.reshape_as(actions),
                b["rewards"][:, burn:].float().clamp(-1, 1),
                dp.reshape_as(actions),
                b["terminated"][:, burn:],
                actions,
                valid,
                revision=self.policy_revision,
            )
        return True

    def snapshot_payload(self):
        metadata = {
            "schema_version": 2,
            "model_seed": self.model_seed,
            "config": asdict(self.config),
            "calibration": self.calibration.state_dict(),
            "generator_rng": self.rng.bit_generator.state,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "games": list(self.games),
            "episode_counters": self.episode_counters,
            "training_curve": self.training_curve,
            "loss_history": self.loss_history,
            "last_update": self.last_update,
            "source_teacher_present": self.source_teacher is not None,
            "environments": {g: a["env"].snapshot() for g, a in self.actors.items()},
            "actors": {
                g: {
                    k: v
                    for k, v in a.items()
                    if k not in ("env", "observation", "history")
                    and not k.startswith("_cached_")
                }
                for g, a in self.actors.items()
            },
            "actor_history": {
                g: [
                    {"previous_action": x[1], "previous_reward": x[2]}
                    for x in a["history"]
                ]
                for g, a in self.actors.items()
            },
        }
        for k in (
            "stage",
            "stage_budget",
            "stage_steps",
            "global_steps",
            "fit_steps",
            "calibration_steps",
            "q_updates",
            "world_updates",
            "policy_revision",
            "last_q_due",
            "last_world_due",
            "next_actor",
        ):
            metadata[k] = getattr(self, k)
        tensors = {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "source_teacher": (
                None
                if self.source_teacher is None
                else self.source_teacher.state_dict()
            ),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "world_optimizer": self.world_optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
        }
        observations = {}
        for g, a in self.actors.items():
            observations["current_" + g] = a["observation"]
            observations["history_" + g] = (
                np.stack([x[0] for x in a["history"]])
                if a["history"]
                else np.zeros((0, 32, 32, 3), dtype=np.uint8)
            )
        return _json_value(metadata), tensors, observations

    @classmethod
    def from_payload(cls, metadata, tensors, observations, replays):
        if metadata.get("schema_version") != 2:
            raise ValueError("only v2 snapshots accepted")
        config = ContinualConfig(**metadata["config"])
        counters = (
            "stage_budget",
            "stage_steps",
            "global_steps",
            "fit_steps",
            "calibration_steps",
            "q_updates",
            "world_updates",
            "policy_revision",
            "last_q_due",
            "last_world_due",
            "next_actor",
        )
        if any(type(metadata.get(k)) is not int or metadata[k] < 0 for k in counters):
            raise ValueError("snapshot counters must be nonnegative integers")
        stage = metadata.get("stage")
        expected_games = (
            SOURCE_GAMES
            if stage == "pretrain"
            else (TARGET_GAME,) if stage in ("adaptation", "from_scratch") else ()
        )
        if (
            stage not in ("unconfigured", "pretrain", "adaptation", "from_scratch")
            or tuple(metadata.get("games", ())) != expected_games
            or not metadata["stage_steps"]
            <= metadata["stage_budget"]
            <= config.max_training_steps
            or not metadata["stage_steps"]
            <= metadata["global_steps"]
            <= config.max_training_steps
            or metadata["fit_steps"] + metadata["calibration_steps"]
            != metadata["global_steps"]
            or metadata["policy_revision"] != metadata["q_updates"]
            or metadata["last_q_due"] > metadata["stage_steps"] // config.q_interval
            or metadata["last_world_due"]
            > metadata["stage_steps"] // config.world_interval
            or metadata["next_actor"] >= max(1, len(expected_games))
            or type(metadata.get("source_teacher_present")) is not bool
            or metadata["source_teacher_present"] != (stage == "adaptation")
        ):
            raise ValueError("inconsistent finite stage/counters/teacher in snapshot")
        if set(metadata["actors"]) != set(expected_games) or set(
            metadata["episode_counters"]
        ) != set(expected_games):
            raise ValueError("snapshot actor catalog differs from stage")
        for game in expected_games:
            a = metadata["actors"][game]
            episode = metadata["episode_counters"][game]
            if (
                type(episode) is not int
                or episode < 0
                or type(a["episode_steps"]) is not int
                or not 0 <= a["episode_steps"] < config.episode_steps
                or type(a["calibration_episode"]) is not bool
                or a["calibration_episode"]
                != ((episode + 1) % config.calibration_every_episodes == 0)
                or a["environment_seed"]
                != training_seed(
                    metadata["model_seed"],
                    game,
                    episode,
                    stage,
                    a["calibration_episode"],
                )
                or len(metadata["actor_history"][game])
                != (
                    (a["episode_steps"] - 1) % config.context_length + 1
                    if a["episode_steps"]
                    else 0
                )
            ):
                raise ValueError("inconsistent actor seed/context/calibration split")
        obj = cls(metadata["model_seed"], config)
        obj.online.load_state_dict(tensors["online"], strict=True)
        obj.target.load_state_dict(tensors["target"], strict=True)
        if metadata["source_teacher_present"]:
            obj.source_teacher = copy.deepcopy(obj.online).requires_grad_(False).eval()
            obj.source_teacher.load_state_dict(tensors["source_teacher"], strict=True)
        elif tensors["source_teacher"] is not None:
            raise ValueError("unexpected source teacher")
        obj.policy_optimizer.load_state_dict(tensors["policy_optimizer"])
        obj.world_optimizer.load_state_dict(tensors["world_optimizer"])
        obj.replay, obj.calibration_replay, obj.source_replay = (
            replays["fit"],
            replays["calibration"],
            replays.get("source"),
        )
        if (obj.source_teacher is None) != (obj.source_replay is None):
            raise ValueError("teacher and protected replay must coexist")
        for slot, replay in (
            ("fit", obj.replay),
            ("calibration", obj.calibration_replay),
            ("source", obj.source_replay),
        ):
            if replay is None:
                continue
            total = (
                config.calibration_capacity
                if slot == "calibration"
                else config.replay_capacity
            )
            per_game = min(config.capacity_per_game, total)
            if replay.capacity_total != total or replay.capacity_per_game != per_game:
                raise ValueError(
                    "replay capacity differs from snapshot learner configuration"
                )
        for k in (
            "stage",
            "stage_budget",
            "stage_steps",
            "global_steps",
            "fit_steps",
            "calibration_steps",
            "q_updates",
            "world_updates",
            "policy_revision",
            "last_q_due",
            "last_world_due",
            "next_actor",
            "episode_counters",
            "training_curve",
            "loss_history",
            "last_update",
        ):
            setattr(obj, k, copy.deepcopy(metadata[k]))
        obj.games = tuple(metadata["games"])
        obj.calibration = GateCalibration.from_state_dict(metadata["calibration"])
        if (
            obj.calibration.revision is not None
            and obj.calibration.revision > obj.policy_revision
        ):
            raise ValueError("calibration comes from a future parameter revision")
        for g in obj.games:
            s = metadata["environments"][g]
            env = make_game(g, **s["config"])
            env.restore(s)
            current = observations["current_" + g]
            if not np.array_equal(current, env.render()):
                raise ValueError("physical state and current pixels differ")
            history = [
                (f.copy(), v["previous_action"], v["previous_reward"])
                for f, v in zip(
                    observations["history_" + g],
                    metadata["actor_history"][g],
                    strict=True,
                )
            ]
            if len(history) > obj.config.context_length:
                raise ValueError("actor context exceeds bound")
            obj.actors[g] = {
                **metadata["actors"][g],
                "env": env,
                "observation": current.copy(),
                "history": history,
            }

        def tuples(x):
            return tuple(tuples(v) for v in x) if isinstance(x, list) else x

        obj.rng.bit_generator.state = copy.deepcopy(metadata["generator_rng"])
        random.setstate(tuples(metadata["python_rng"]))
        n = metadata["numpy_rng"]
        np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
        torch.set_rng_state(tensors["torch_rng"])
        return obj
