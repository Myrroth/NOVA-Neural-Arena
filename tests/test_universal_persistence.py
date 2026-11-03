"""Exact continuation after a verified full snapshot; no scientific training."""
import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from arena.universal.persistence import load_checkpoint, save_checkpoint
from arena.universal.trainer import LearnerConfig, UniversalTrainer


SOURCE = "development-synthetic-unit-fixture"
PROTOCOL = "a"*64


def compact_trainer():
    cfg = LearnerConfig(batch_size=2, burn_in=1, unroll=2, warmup_steps=1000,
                        capacity_per_game=256, replay_capacity=1024, threads=1)
    trainer = UniversalTrainer(1103, cfg)
    trainer.configure_stage("pretrain", 160)
    # Random mechanical collection only. Gradients below fit independent
    # synthetic pixel transitions, not these games or an official checkpoint.
    trainer.collect(24, learn=False)
    return trainer


def add_synthetic_memory(trainer):
    rng = np.random.default_rng(51)
    mask = np.ones(10, dtype=bool)
    obs = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    for i in range(8):
        nxt = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        trainer.replay.append("synthetic", obs, i % 10, .1, nxt, i == 7,
                              False, mask, mask, prev_action=(i-1) % 10 if i else 0,
                              prev_reward=.1 if i else 0.)
        obs = nxt


def weight_hash(trainer):
    digest = hashlib.sha256()
    for name, value in trainer.online.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().numpy().tobytes())
    return digest.hexdigest()


def test_exact_25_future_transitions_and_updates_after_reload(tmp_path):
    trainer = compact_trainer()
    add_synthetic_memory(trainer)
    receipt = save_checkpoint(trainer, tmp_path/"snapshot", source_revision=SOURCE,
                              protocol_sha256=PROTOCOL)
    initial_hash = weight_hash(trainer)
    expected = []
    # Use only synthetic entries for gradient fixtures, preserving real actor
    # state/RNG as part of the checkpoint continuation test.
    original_batch = trainer._batch
    def synthetic_batch():
        saved = trainer.replay.sample
        def sample(batch_size, burn_in, unroll, **kwargs):
            return saved(batch_size, burn_in, unroll, game_weights={"synthetic": 1.})
        trainer.replay.sample = sample
        try:
            return original_batch()
        finally:
            trainer.replay.sample = saved
    trainer._batch = synthetic_batch
    for i in range(25):
        trainer.collect(1, learn=False)
        loss = trainer.learn_batch(world=bool(i % 2))
        expected.append((loss, weight_hash(trainer),
                         {g: a["env"].snapshot() for g, a in trainer.actors.items()}))
    restored = load_checkpoint(tmp_path/"snapshot", expected_manifest_sha256=receipt["manifest_sha256"],
                               expected_source_revision=SOURCE, expected_protocol_sha256=PROTOCOL)
    assert weight_hash(restored) == initial_hash
    original_batch_r = restored._batch
    def restored_synthetic_batch():
        saved = restored.replay.sample
        def sample(batch_size, burn_in, unroll, **kwargs):
            return saved(batch_size, burn_in, unroll, game_weights={"synthetic": 1.})
        restored.replay.sample = sample
        try:
            return original_batch_r()
        finally:
            restored.replay.sample = saved
    restored._batch = restored_synthetic_batch
    for i, (loss, fingerprint, physical_states) in enumerate(expected):
        restored.collect(1, learn=False)
        assert restored.learn_batch(world=bool(i % 2)) == pytest.approx(loss, abs=1e-6)
        assert weight_hash(restored) == fingerprint
        assert {g: a["env"].snapshot() for g, a in restored.actors.items()} == physical_states


def test_corrupt_or_incompatible_snapshot_rejected_before_restore(tmp_path):
    trainer = compact_trainer()
    receipt = save_checkpoint(trainer, tmp_path/"snapshot", source_revision=SOURCE,
                              protocol_sha256=PROTOCOL)
    with pytest.raises(ValueError, match="mismatch"):
        load_checkpoint(tmp_path/"snapshot", expected_manifest_sha256=receipt["manifest_sha256"],
                        expected_source_revision="another-source", expected_protocol_sha256=PROTOCOL)
    with pytest.raises(FileExistsError):
        save_checkpoint(trainer, tmp_path/"snapshot", source_revision=SOURCE, protocol_sha256=PROTOCOL)
    target = tmp_path/"snapshot"/"state.pt"
    target.write_bytes(target.read_bytes()+b"corruption")
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path/"snapshot", expected_manifest_sha256=receipt["manifest_sha256"],
                        expected_source_revision=SOURCE, expected_protocol_sha256=PROTOCOL)


def test_finite_stage_noop_and_controlled_handoff():
    trainer = compact_trainer()
    with pytest.raises(ValueError, match="partially"):
        trainer.configure_stage("adaptation", 20)
    with pytest.raises(ValueError, match="remaining"):
        trainer.collect(200, learn=False)
    trainer.collect(136, learn=False)
    count = len(trainer.training_curve)
    trainer.collect(0, learn=False)
    assert len(trainer.training_curve) == count
    trainer.configure_stage("adaptation", 20)
    assert trainer.games == ("racing",)
    trainer.collect(20, learn=False)
    assert trainer.stage_steps == 20 and "racing" in trainer.training_baselines()


def test_inference_loader_uses_checkpoint_without_restoring_experiences(tmp_path, monkeypatch):
    from arena.universal.campaign import _RealBackend
    from arena.universal.replay import SequenceReplay
    trainer = compact_trainer()
    source = "c"*40
    receipt = save_checkpoint(trainer, tmp_path/"snapshot", source_revision=source,
                              protocol_sha256=PROTOCOL)
    def forbidden(*args, **kwargs):
        raise AssertionError("inference must not deserialize replay")
    monkeypatch.setattr(SequenceReplay, "load", forbidden)
    rng = torch.get_rng_state().clone()
    network = _RealBackend().load_policy(tmp_path,
        {"path": "snapshot", "manifest_sha256": receipt["manifest_sha256"],
         "model_seed": 1103, "stage": "before_adaptation"},
        {"source_revision": source, "protocol_sha256": PROTOCOL})
    assert torch.equal(torch.get_rng_state(), rng)
    assert not network.training and all(not p.requires_grad for p in network.parameters())
    for key, value in trainer.online.state_dict().items():
        assert torch.equal(value, network.state_dict()[key])


def test_candidate_protocol_matches_implemented_global_parameters_and_games():
    from dataclasses import asdict
    from arena.universal.protocol import protocol_config
    from arena.universal.envs import GAMES, make_game
    from arena.universal.network import VisualQNetwork
    protocol = protocol_config()
    assert asdict(LearnerConfig()) == protocol["learner_config"]
    assert VisualQNetwork.architecture_version == protocol["network_version"]
    assert {game: make_game(game).identity for game in GAMES} == protocol["environment_revisions"]
