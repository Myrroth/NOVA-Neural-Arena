"""One recurrent visual learning algorithm for every prepared game.

No official campaign is started by importing this module. The eventual runner
must enforce its preregistered authorization, source hash and global budget.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import random

import numpy as np
import torch
from torch.nn import functional as F

from arena.universal.envs import make_game
from arena.universal.network import VisualQNetwork
from arena.universal.protocol import SOURCE_GAMES, derive_training_seed
from arena.universal.replay import SequenceReplay


@dataclass(frozen=True)
class LearnerConfig:
    hidden_size: int = 64
    latent_size: int = 64
    gamma: float = .98
    learning_rate: float = .0003
    batch_size: int = 16
    burn_in: int = 4
    unroll: int = 8
    replay_capacity: int = 40000
    capacity_per_game: int = 10000
    warmup_steps: int = 128
    q_interval: int = 16
    world_interval: int = 32
    target_tau: float = .01
    planning_fraction: float = .20
    epsilon_end: float = .05
    episode_steps: int = 400
    threads: int = 2
    max_training_steps: int = 1_080_000

    def __post_init__(self):
        positive = (self.hidden_size, self.latent_size, self.batch_size,
                    self.unroll, self.replay_capacity, self.capacity_per_game,
                    self.warmup_steps, self.q_interval, self.world_interval,
                    self.episode_steps, self.max_training_steps)
        if any(type(x) is not int or x < 1 for x in positive):
            raise ValueError("positive integer learner parameters required")
        if (self.burn_in < 0 or self.threads not in (1, 2) or
                not 0 < self.gamma < 1 or not 0 < self.target_tau <= 1 or
                not 0 <= self.planning_fraction <= 1 or
                not 0 <= self.epsilon_end <= 1 or self.learning_rate <= 0):
            raise ValueError("invalid learner parameter")
        if self.replay_capacity > 40000 or self.max_training_steps > 1_080_000:
            raise ValueError("candidate training/replay limit exceeded")


def _pixels(frames):
    value = torch.as_tensor(np.asarray(frames), dtype=torch.uint8)
    return value.permute(0, 3, 1, 2).contiguous()


class UniversalTrainer:
    """Shared weights, optimizer and replay; game labels only schedule data."""

    def __init__(self, model_seed: int, config: LearnerConfig | None = None):
        self.config = config or LearnerConfig()
        self.model_seed = int(model_seed)
        torch.set_num_threads(self.config.threads)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(self.model_seed)
        random.seed(self.model_seed)
        np.random.seed(self.model_seed)
        self.rng = np.random.default_rng(self.model_seed + 41)
        self.online = VisualQNetwork(hidden_size=self.config.hidden_size,
                                     latent_size=self.config.latent_size)
        self.target = copy.deepcopy(self.online).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(),
                                          lr=self.config.learning_rate)
        self.replay = SequenceReplay(capacity_per_game=self.config.capacity_per_game,
                                     capacity_total=self.config.replay_capacity,
                                     seed=self.model_seed + 71)
        self.stage = "unconfigured"
        self.stage_budget = 0
        self.stage_steps = 0
        self.global_steps = 0
        self.q_updates = 0
        self.world_updates = 0
        self.next_actor = 0
        self.games = ()
        self.actors = {}
        self.episode_counters = {}
        self.training_curve = []
        self.loss_history = []
        self.transition_statistics = {}
        self.last_q_due = self.last_world_due = 0

    def _environment_seed(self, game, episode):
        return derive_training_seed(self.model_seed, game, episode,
                                    stage="source" if self.stage == "pretrain" else self.stage)

    def configure_stage(self, stage: str, budget: int):
        if stage not in ("pretrain", "adaptation", "from_scratch"):
            raise ValueError("unknown preregistered stage")
        if type(budget) is not int or not 0 < budget <= self.config.max_training_steps:
            raise ValueError("invalid finite stage budget")
        if self.stage != "unconfigured" and self.stage_steps != self.stage_budget:
            raise ValueError("cannot abandon a partially executed stage")
        for game in self.games:
            self.replay.truncate_active(game)
        self.stage, self.stage_budget, self.stage_steps = stage, budget, 0
        self.last_q_due = self.last_world_due = 0
        self.next_actor = 0
        self.games = SOURCE_GAMES if stage == "pretrain" else ("racing",)
        self.actors, self.episode_counters = {}, {}
        for game in self.games:
            env = make_game(game, max_steps=self.config.episode_steps)
            seed = self._environment_seed(game, 0)
            obs, _ = env.reset(seed=seed)
            self.episode_counters[game] = 0
            self.actors[game] = {"env": env, "observation": obs,
                                 "hidden": self.online.initial(1),
                                 "previous_action": 0, "previous_reward": 0.,
                                 "episode_return": 0., "episode_steps": 0,
                                 "environment_seed": seed}

    def epsilon(self):
        fraction = min(1., self.stage_steps/max(1, .8*self.stage_budget))
        return 1 + fraction*(self.config.epsilon_end-1)

    def _episode_record(self, game, actor, info, complete):
        return {"model_seed": self.model_seed, "stage": self.stage, "game": game,
                "environment_seed": actor["environment_seed"],
                "episode": self.episode_counters[game],
                "training_step": self.stage_steps, "global_step": self.global_steps,
                "steps": actor["episode_steps"], "return": actor["episode_return"],
                "episode_complete": complete,
                **{key: value for key, value in info.items()
                   if key not in ("action_mask", "environment_id", "steps")}}

    def collect(self, transitions: int, *, learn: bool = True):
        """Bounded collection; never executes beyond the configured stage."""
        if type(transitions) is not int or transitions < 0 or not self.actors:
            raise ValueError("configure stage and provide a nonnegative count")
        if transitions > self.stage_budget-self.stage_steps:
            raise ValueError("requested interactions exceed remaining stage budget")
        if self.global_steps+transitions > self.config.max_training_steps:
            raise ValueError("training limit exceeded")
        if transitions == 0:
            return {"stage_steps": self.stage_steps, "global_steps": self.global_steps,
                    "q_updates": self.q_updates, "world_updates": self.world_updates}
        remaining = transitions
        while remaining:
            count = min(len(self.games), remaining)
            selected = [self.games[(self.next_actor+i) % len(self.games)]
                        for i in range(count)]
            actors = [self.actors[name] for name in selected]
            obs = _pixels([actor["observation"] for actor in actors])
            h = torch.cat([actor["hidden"] for actor in actors])
            pa = torch.tensor([actor["previous_action"] for actor in actors])
            pr = torch.tensor([actor["previous_reward"] for actor in actors], dtype=torch.float32)
            masks = np.stack([actor["env"].action_mask() for actor in actors])
            explores = self.rng.random(count) < self.epsilon()
            with torch.no_grad():
                q, next_h, _ = self.online.step(obs, pa, pr, h)
                values = q.clone()
                greedy_indices = np.flatnonzero(~explores)
                if len(greedy_indices) and self.config.planning_fraction:
                    idx = torch.as_tensor(greedy_indices)
                    plan = self.online.plan_values(next_h[idx], torch.as_tensor(masks[greedy_indices]),
                                                   gamma=self.config.gamma)
                    mix = self.config.planning_fraction
                    values[idx] = (1-mix)*q[idx]+mix*plan
                values.masked_fill_(~torch.as_tensor(masks), -1e9)
                actions = values.argmax(-1).tolist()
            for i, (game, actor) in enumerate(zip(selected, actors)):
                action = int(self.rng.choice(np.flatnonzero(masks[i]))) if explores[i] else actions[i]
                after, reward, terminated, truncated, info = actor["env"].step(action)
                self.replay.append(game, actor["observation"], action, reward, after,
                                   terminated, truncated, masks[i], actor["env"].action_mask(),
                                   prev_action=actor["previous_action"],
                                   prev_reward=actor["previous_reward"])
                stats = self.transition_statistics.setdefault(game, {"support": 0,
                         "clipped_reward_sum": 0., "terminations": 0})
                stats["support"] += 1
                stats["clipped_reward_sum"] += float(np.clip(reward, -1, 1))
                stats["terminations"] += int(terminated)
                actor["observation"] = after
                actor["hidden"] = next_h[i:i+1].clone()
                actor["previous_action"], actor["previous_reward"] = action, float(np.clip(reward, -1, 1))
                actor["episode_steps"] += 1
                actor["episode_return"] += reward
                self.global_steps += 1
                self.stage_steps += 1
                remaining -= 1
                if terminated or truncated:
                    self.training_curve.append(self._episode_record(game, actor, info, True))
                    self.episode_counters[game] += 1
                    seed = self._environment_seed(game, self.episode_counters[game])
                    observation, _ = actor["env"].reset(seed=seed)
                    actor.update(observation=observation, hidden=self.online.initial(1),
                                 previous_action=0, previous_reward=0., episode_return=0.,
                                 episode_steps=0, environment_seed=seed)
                if learn:
                    self._due_updates()
            self.next_actor = (self.next_actor+count) % len(self.games)
        if self.stage_steps == self.stage_budget:
            for game, actor in self.actors.items():
                if actor["episode_steps"]:
                    self.training_curve.append(self._episode_record(
                        game, actor, actor["env"].metrics(), False))
        return {"stage_steps": self.stage_steps, "global_steps": self.global_steps,
                "q_updates": self.q_updates, "world_updates": self.world_updates}

    def training_baselines(self):
        return {game: {"source": "training", "support": values["support"],
                       "reward_mean": values["clipped_reward_sum"]/values["support"],
                       "terminal_rate": values["terminations"]/values["support"]}
                for game, values in self.transition_statistics.items() if values["support"]}

    def _due_updates(self):
        if self.stage_steps < self.config.warmup_steps or len(self.replay) < self.config.warmup_steps:
            return
        q_due = self.stage_steps//self.config.q_interval
        world_due = self.stage_steps//self.config.world_interval
        if q_due > self.last_q_due:
            self.last_q_due = q_due
            self.learn_batch(world=False)
        if world_due > self.last_world_due:
            self.last_world_due = world_due
            self.learn_batch(world=True)

    def _batch(self):
        weights = None
        if self.stage == "adaptation":
            weights = {"racing": .8, **{name: .05 for name in SOURCE_GAMES}}
        data = self.replay.sample(self.config.batch_size, self.config.burn_in,
                                  self.config.unroll, game_weights=weights)
        frames = torch.as_tensor(data["frames"], dtype=torch.uint8).permute(0, 1, 4, 2, 3).contiguous()
        converted = {key: torch.as_tensor(value) for key, value in data.items()
                     if key not in ("frames", "games") and isinstance(value, np.ndarray)}
        return frames, converted

    def _sequence(self, network, pixels, batch):
        burn = self.config.burn_in
        hidden = None
        if burn:
            with torch.no_grad():
                _, states, _ = network.sequence(pixels[:, :burn], batch["prev_actions"][:, :burn],
                                                batch["prev_rewards"][:, :burn],
                                                resets=batch["resets"][:, :burn])
                hidden = states[:, -1].detach()
        return network.sequence(pixels[:, burn:], batch["prev_actions"][:, burn:],
                                batch["prev_rewards"][:, burn:], hidden=hidden,
                                resets=batch["resets"][:, burn:])

    def learn_batch(self, *, world=False):
        """One update on existing sequences; also usable with synthetic fixtures."""
        pixels, b = self._batch()
        burn = self.config.burn_in
        valid = b["valid"][:, burn:].float()
        denominator = valid.sum().clamp_min(1)
        q, states, latents = self._sequence(self.online, pixels, b)
        rewards = b["rewards"][:, burn:].float().clamp(-1, 1)
        terminated = b["terminated"][:, burn:].float()
        actions = b["actions"][:, burn:].long()
        if world:
            h = states[:, :-1].reshape(-1, self.config.hidden_size)
            predicted_h, predicted_r, predicted_done = self.online.model(h, actions.reshape(-1))
            # World control and its held-out audit both use ONLINE recurrent
            # coordinates. Stop gradients through the actual successor target;
            # the EMA target network is reserved for Double-DQN bootstrapping.
            successor = states[:, 1:].detach()
            predicted_h = predicted_h.reshape_as(successor)
            latent_loss = ((predicted_h-successor)**2).mean(-1)
            reward_loss = F.smooth_l1_loss(predicted_r.reshape_as(rewards), rewards, reduction="none")
            done_loss = F.binary_cross_entropy_with_logits(predicted_done.reshape_as(terminated),
                                                            terminated, reduction="none")
            decoded = self.online.decode(latents[:, :-1].reshape(-1, self.config.latent_size))
            target_image = F.avg_pool2d(pixels[:, burn:-1].reshape(-1, 3, 32, 32).float()/255, 4)
            reconstruction = ((decoded-target_image)**2).mean((1, 2, 3)).reshape_as(valid)
            per_step = latent_loss+.5*reward_loss+.1*done_loss+.1*reconstruction
            loss = (per_step*valid).sum()/denominator
        else:
            with torch.no_grad():
                target_q, _, _ = self._sequence(self.target, pixels, b)
            selected_q = q[:, :-1].gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            next_mask = b["masks"][:, burn+1:].bool()
            chosen = q[:, 1:].detach().masked_fill(~next_mask, -1e9).argmax(-1, keepdim=True)
            next_q = target_q[:, 1:].gather(-1, chosen).squeeze(-1)
            target = rewards+self.config.gamma*(1-terminated)*next_q
            loss = (F.smooth_l1_loss(selected_q, target, reduction="none")*valid).sum()/denominator
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite learning loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.)
        self.optimizer.step()
        if world:
            self.world_updates += 1
        else:
            self.q_updates += 1
            with torch.no_grad():
                for target_param, online_param in zip(self.target.parameters(), self.online.parameters()):
                    target_param.lerp_(online_param, self.config.target_tau)
        value = float(loss.detach())
        self.loss_history.append({"global_step": self.global_steps,
                                  "kind": "world" if world else "q", "loss": value})
        return value

    def snapshot_payload(self):
        """Return bounded primitive metadata plus tensors for safe persistence."""
        from arena.universal.envs import _json_value
        metadata = {"schema_version": 1, "model_seed": self.model_seed,
                    "config": asdict(self.config), "stage": self.stage,
                    "stage_budget": self.stage_budget, "stage_steps": self.stage_steps,
                    "global_steps": self.global_steps, "q_updates": self.q_updates,
                    "world_updates": self.world_updates, "next_actor": self.next_actor,
                    "games": list(self.games), "episode_counters": self.episode_counters,
                    "last_q_due": self.last_q_due, "last_world_due": self.last_world_due,
                    "training_curve": self.training_curve, "loss_history": self.loss_history,
                    "transition_statistics": self.transition_statistics,
                    "generator_rng": self.rng.bit_generator.state,
                    "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                    "actors": {game: {key: value for key, value in actor.items()
                                       if key not in ("env", "observation", "hidden")}
                               for game, actor in self.actors.items()},
                    "environments": {game: actor["env"].snapshot()
                                     for game, actor in self.actors.items()}}
        tensors = {"online": self.online.state_dict(), "target": self.target.state_dict(),
                   "optimizer": self.optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                   "hidden": {game: actor["hidden"] for game, actor in self.actors.items()}}
        observations = {game: actor["observation"] for game, actor in self.actors.items()}
        return _json_value(metadata), tensors, observations

    @classmethod
    def from_payload(cls, metadata, tensors, observations, replay):
        if metadata.get("schema_version") != 1:
            raise ValueError("unsupported trainer snapshot")
        obj = cls(metadata["model_seed"], LearnerConfig(**metadata["config"]))
        obj.online.load_state_dict(tensors["online"], strict=True)
        obj.target.load_state_dict(tensors["target"], strict=True)
        obj.optimizer.load_state_dict(tensors["optimizer"])
        obj.replay = replay
        for key in ("stage", "stage_budget", "stage_steps", "global_steps", "q_updates",
                    "world_updates", "next_actor", "episode_counters", "last_q_due",
                    "last_world_due", "training_curve", "loss_history", "transition_statistics"):
            setattr(obj, key, copy.deepcopy(metadata[key]))
        obj.games = tuple(metadata["games"])
        obj.actors = {}
        for game in obj.games:
            env_state = metadata["environments"][game]
            env = make_game(game, **env_state["config"])
            env.restore(env_state)
            if not np.array_equal(observations[game], env.render()):
                raise ValueError("snapshot observation differs from physical environment")
            obj.actors[game] = {**metadata["actors"][game], "env": env,
                                "observation": observations[game].copy(),
                                "hidden": tensors["hidden"][game].clone()}
        def tuples(value):
            return tuple(tuples(x) for x in value) if isinstance(value, list) else value
        obj.rng.bit_generator.state = copy.deepcopy(metadata["generator_rng"])
        random.setstate(tuples(metadata["python_rng"]))
        ns = metadata["numpy_rng"]
        np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
        torch.set_rng_state(tensors["torch_rng"])
        return obj
