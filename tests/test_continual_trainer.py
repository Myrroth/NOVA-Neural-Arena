"""V2 integration on synthetic pixels/actors only; no game learning evidence."""

import copy
from dataclasses import replace
import hashlib

import numpy as np
import pytest
import torch

from arena.continual.trainer import ContinualConfig, ContinualTrainer, training_seed
from arena.continual.persistence import save_checkpoint, load_checkpoint


class SyntheticActor:
    """Random pixel chains, not Snake/Pong/Agar/Flappy/Racing physics."""

    length = 2

    def __init__(self, **config):
        self.config = {
            "max_steps": config.get("max_steps", 400),
            "difficulty": config.get("difficulty", 1.0),
            "scenario": config.get("scenario", "standard"),
        }
        self.steps = 0

    def reset(self, *, seed):
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.frame = self.rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        return self.render(), {}

    def action_mask(self):
        return np.ones(10, dtype=bool)

    def step(self, action):
        self.steps += 1
        self.frame = self.rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        return (
            self.render(),
            float(action % 3 - 1) * 0.1,
            self.steps == self.length,
            False,
            {"steps": self.steps},
        )

    def render(self):
        return self.frame.copy()

    def metrics(self):
        return {"synthetic_progress": self.steps / self.length}

    def close(self):
        pass

    def snapshot(self):
        return {
            "config": self.config,
            "steps": self.steps,
            "frame": self.frame.tolist(),
            "rng": copy.deepcopy(self.rng.bit_generator.state),
        }

    def restore(self, state):
        self.config = state["config"]
        self.steps = state["steps"]
        self.frame = np.asarray(state["frame"], dtype=np.uint8)
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])


@pytest.fixture
def fake_games(monkeypatch):
    monkeypatch.setattr(
        "arena.continual.trainer.make_game",
        lambda game, **kwargs: SyntheticActor(**kwargs),
    )


def compact(**kwargs):
    base = ContinualConfig(
        hidden_size=8,
        latent_size=8,
        context_length=4,
        unroll=3,
        n_step=3,
        batch_size=2,
        calibration_batch_size=2,
        replay_capacity=128,
        capacity_per_game=32,
        calibration_capacity=32,
        calibration_every_episodes=3,
        warmup_steps=4,
        threads=1,
        max_training_steps=200,
    )
    return replace(base, **kwargs)


def weights(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def fingerprint(module):
    h = hashlib.sha256()
    for k, v in module.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def ready(fake_games):
    obj = ContinualTrainer(831003, compact())
    obj.configure_stage("pretrain", 24)
    obj.collect(24, learn=False)
    return obj


def test_reserved_episodes_never_enter_fit_or_protected_replay(fake_games):
    t = ready(fake_games)
    assert t.fit_steps == 16 and t.calibration_steps == 8 and t.global_steps == 24
    fit = {f.tobytes() for e in t.replay._episodes for f in e.frames}
    validation = {f.tobytes() for e in t.calibration_replay._episodes for f in e.frames}
    assert not fit.intersection(validation)
    curves = t.training_curve
    assert len(curves) == 12
    assert all(
        r["calibration_episode"] == ((r["episode"] + 1) % 3 == 0) for r in curves
    )
    assert len({r["environment_seed"] for r in curves}) == 12
    assert all(r["environment_seed"] >= 2_000_000 for r in curves)
    counts = copy.deepcopy(t.replay.game_counts())
    t.configure_stage("adaptation", 64)
    t.collect(6, learn=False)
    assert (
        t.source_replay.game_counts() == counts
        and "racing" not in t.source_replay.games
    )
    assert t.source_teacher is not t.online


def test_world_update_cannot_move_encoder_gru_q_or_policy_optimizer(fake_games):
    t = ready(fake_games)
    before = weights(t.online)
    loss = t.learn_batch(world=True)
    assert np.isfinite(loss) and t.world_updates == 1 and t.q_updates == 0
    assert any(
        not torch.equal(v, t.online.state_dict()[k])
        for k, v in before.items()
        if k.startswith("transition.")
    )
    assert all(
        torch.equal(v, t.online.state_dict()[k])
        for k, v in before.items()
        if not k.startswith("transition.")
    )
    assert all(p.grad is None for p in t.online.control_parameters())
    assert not t.policy_optimizer.state
    assert t.world_optimizer.state


def test_policy_update_does_not_move_world_and_teacher_is_immutable(fake_games):
    t = ready(fake_games)
    t.configure_stage("adaptation", 64)
    t.collect(8, learn=False)
    teacher = weights(t.source_teacher)
    before = weights(t.online)
    loss = t.learn_batch(world=False)
    assert np.isfinite(loss) and t.policy_revision == 1
    assert all(
        torch.equal(v, t.online.state_dict()[k])
        for k, v in before.items()
        if k.startswith("transition.")
    )
    assert any(
        not torch.equal(v, t.online.state_dict()[k])
        for k, v in before.items()
        if k.startswith(("encoder.", "memory.", "q_head."))
    )
    assert all(
        torch.equal(v, t.source_teacher.state_dict()[k]) for k, v in teacher.items()
    )
    assert all(
        not p.requires_grad and p.grad is None for p in t.source_teacher.parameters()
    )


def test_cold_model_never_influences_action_and_cache_rebuilds_after_revision(
    fake_games, monkeypatch
):
    t = ContinualTrainer(831004, compact())
    t.configure_stage("pretrain", 40)
    t.collect(1, learn=False)
    a = t.actors["snake"]

    def forbidden(*args, **kwargs):
        raise AssertionError("cold unvalidated planner called")

    monkeypatch.setattr(t.online, "plan_values", forbidden)
    with torch.no_grad():
        t.online.memory.bias_ih.add_(0.25)
    t.policy_revision += 1
    actual, history, _ = t._observe_actor(a)
    frames = torch.from_numpy(np.stack([x[0] for x in history])).permute(0, 3, 1, 2)[
        None
    ]
    pa = torch.tensor([[x[1] for x in history]])
    pr = torch.tensor([[x[2] for x in history]])
    with torch.inference_mode():
        expected, _, _ = t.online.sequence(frames, pa, pr)
    torch.testing.assert_close(actual, expected[0, -1], rtol=0, atol=1e-6)
    assert a["_cached_revision"] != t.policy_revision


def test_replay_and_actor_have_same_periodic_memory_boundaries(fake_games, monkeypatch):
    monkeypatch.setattr(SyntheticActor, "length", 11)
    t = ContinualTrainer(831005, compact(batch_size=8))
    t.configure_stage("pretrain", 40)
    t.collect(40, learn=False)
    pixels, b = t._batch()
    q, _, _ = t._sequence(t.online, pixels, b)
    for row, anchor in enumerate(b["start_indices"].tolist()):
        episode = next(
            e for e in t.replay._episodes if e.identifier == b["episode_ids"][row]
        )
        begin = anchor - anchor % t.config.context_length
        frames = torch.from_numpy(np.stack(episode.frames[begin : anchor + 1])).permute(
            0, 3, 1, 2
        )[None]
        pa = torch.tensor([episode.prev_actions[begin : anchor + 1]])
        pr = torch.tensor([episode.prev_rewards[begin : anchor + 1]])
        with torch.no_grad():
            expected, _, _ = t.online.sequence(frames, pa, pr)
        torch.testing.assert_close(q[row, 0], expected[0, -1], rtol=0, atol=1e-6)


def test_partial_episode_records_cover_exact_budget_once(fake_games, monkeypatch):
    monkeypatch.setattr(SyntheticActor, "length", 11)
    t = ContinualTrainer(831007, compact())
    t.configure_stage("pretrain", 40)
    t.collect(16)
    t.collect(24)
    assert len(t.training_curve) == 4
    assert sum(r["steps"] for r in t.training_curve) == 40
    assert all(
        r["training_step"] == 40 and not r["episode_complete"] for r in t.training_curve
    )
    before = copy.deepcopy(t.training_curve)
    t.collect(0)
    assert t.training_curve == before


def test_complete_snapshot_reproduces_25_synthetic_future_updates(fake_games, tmp_path):
    t = ready(fake_games)
    t.configure_stage("adaptation", 64)
    t.collect(8, learn=False)
    t.learn_batch(world=False)
    t.learn_batch(world=True)
    source = "a" * 40
    protocol = "b" * 64
    receipt = save_checkpoint(
        t, tmp_path / "v2", source_revision=source, protocol_sha256=protocol
    )
    expected = []
    for i in range(25):
        t.collect(1, learn=False)
        loss = t.learn_batch(world=bool(i % 2))
        expected.append(
            (
                loss,
                fingerprint(t.online),
                fingerprint(t.source_teacher),
                t.snapshot_payload()[0],
                t.calibration.state_dict(),
            )
        )
    restored = load_checkpoint(
        tmp_path / "v2",
        expected_manifest_sha256=receipt["manifest_sha256"],
        expected_source_revision=source,
        expected_protocol_sha256=protocol,
    )
    for i, (loss, online, teacher, meta, calibration) in enumerate(expected):
        restored.collect(1, learn=False)
        assert restored.learn_batch(world=bool(i % 2)) == pytest.approx(loss, abs=1e-6)
        assert (
            fingerprint(restored.online) == online
            and fingerprint(restored.source_teacher) == teacher
        )
        assert restored.snapshot_payload()[0] == meta
        assert restored.calibration.state_dict() == calibration


def test_finite_stage_caps_and_zero_collection_do_not_repeat(fake_games):
    t = ContinualTrainer(831006, compact())
    t.configure_stage("pretrain", 8)
    with pytest.raises(ValueError):
        t.configure_stage("adaptation", 8)
    with pytest.raises(ValueError):
        t.collect(9)
    assert t.collect(0)["global_steps"] == 0
    t.collect(8)
    t.configure_stage("adaptation", 8)
    t.collect(8)
    with pytest.raises(ValueError):
        t.configure_stage("adaptation", 8)
    with pytest.raises(ValueError):
        t.collect(1)


def test_training_namespaces_are_disjoint_and_target_streams_match():
    fit = training_seed(42, "racing", 0, "adaptation", False)
    assert fit == training_seed(42, "racing", 0, "from_scratch", False)
    assert fit != training_seed(42, "racing", 0, "adaptation", True)
    assert fit != training_seed(42, "snake", 0, "pretrain", False)
    for episode in range(30):
        assert training_seed(42, "racing", episode, "adaptation", False) % 2 == 0
        assert training_seed(42, "racing", episode, "adaptation", True) % 2 == 1


def good_gate(trainer):
    from arena.continual.calibration import GateCalibration

    action = torch.arange(10).repeat_interleave(8)
    target = torch.ones(80, trainer.config.hidden_size) * 0.2
    gate = GateCalibration()
    gate.refresh(
        target,
        target,
        torch.zeros_like(target),
        torch.zeros(80),
        torch.zeros(80),
        torch.full((80,), -20.0),
        torch.zeros(80, dtype=torch.bool),
        action,
        torch.ones(80, dtype=torch.bool),
        revision=trainer.policy_revision,
    )
    assert (
        gate.weights(
            torch.ones(1, 10, dtype=torch.bool), revision=trainer.policy_revision
        ).item()
        > 0
    )
    return gate


def test_world_mutation_invalidates_gate_even_without_calibration_warmup(fake_games):
    t = ready(fake_games)
    t.config = replace(t.config, warmup_steps=128)
    t.calibration = good_gate(t)
    t.learn_batch(world=True)
    assert t.world_updates == 1 and t.policy_revision == 0
    assert (
        t.calibration.weights(torch.ones(1, 10, dtype=torch.bool), revision=0).item()
        == 0
    )


def test_new_target_cannot_reuse_source_only_calibration(fake_games):
    t = ready(fake_games)
    t.configure_stage("adaptation", 64)
    assert len(t.calibration_replay) == 8 and "racing" not in t.calibration_replay.games
    t.calibration = good_gate(t)
    assert not t.refresh_calibration()
    assert (
        t.calibration.weights(torch.ones(1, 10, dtype=torch.bool), revision=0).item()
        == 0
    )


def test_calibration_support_counts_distinct_transitions_only(fake_games, monkeypatch):
    t = ready(fake_games)
    pixels, b = t._batch(t.calibration_replay, calibration=True)
    # Repeat one authenticated sampled sequence eight times: this may not turn
    # one transition into eight observations of independent model support.
    pixels = pixels[:1].repeat(8, 1, 1, 1, 1)
    b = {k: v[:1].repeat((8,) + (1,) * (v.ndim - 1)) for k, v in b.items()}
    monkeypatch.setattr(t, "_batch", lambda *args, **kwargs: (pixels, b))
    expected = int(
        (b["valid"][0, t.burn_in :] & ~b["resets"][0, t.burn_in + 1 :]).sum()
    )
    assert t.refresh_calibration()
    assert sum(s["support"] for s in t.calibration.stats) == expected


@pytest.mark.parametrize(
    "field,value",
    [
        ("global_steps", 201),
        ("stage_steps", 25),
        ("fit_steps", 200),
        ("policy_revision", 99),
        ("stage", "arbitrary"),
        ("next_actor", 5),
    ],
)
def test_snapshot_cannot_bypass_stage_or_total_limits(fake_games, field, value):
    t = ready(fake_games)
    metadata, tensors, observations = t.snapshot_payload()
    metadata[field] = value
    with pytest.raises(ValueError):
        ContinualTrainer.from_payload(
            metadata,
            tensors,
            observations,
            {"fit": t.replay, "calibration": t.calibration_replay, "source": None},
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("context_length", 1),
        ("n_step", 9),
        ("planning_fraction", 0.5),
        ("calibration_every_episodes", 1),
        ("adaptation_target_fraction", 1.0),
        ("learning_rate", float("nan")),
    ],
)
def test_invalid_correction_config_rejected(field, value):
    with pytest.raises(ValueError):
        compact(**{field: value})
