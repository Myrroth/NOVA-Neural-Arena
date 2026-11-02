"""Synthetic frame chains only: persistence, capacity and temporal contracts."""
from collections import Counter
import hashlib
import json

import numpy as np
import pytest

from arena.universal.replay import SequenceReplay


def frame(value):
    return np.full((32, 32, 3), value, dtype=np.uint8)


def mask():
    value = np.zeros(10, dtype=bool)
    value[[0, 2]] = True
    return value


def episode(replay, game, base, length, *, closed=True, truncated=False):
    for step in range(length):
        replay.append(game, frame(base + step), 2 if step % 2 else 0, step + .25,
                      frame(base + step + 1), closed and step == length - 1 and not truncated,
                      closed and step == length - 1 and truncated, mask(), mask())


def assert_equal_batches(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], np.ndarray):
            np.testing.assert_array_equal(left[key], right[key], err_msg=key)
        else:
            assert left[key] == right[key]


def test_balanced_games_do_not_weight_long_game_more_heavily():
    replay = SequenceReplay(seed=17)
    episode(replay, "short", 10, 1)
    episode(replay, "medium", 40, 7)
    episode(replay, "long", 80, 23)
    batch = replay.sample(90, 4, 8)
    assert Counter(batch["games"]) == {"short": 30, "medium": 30, "long": 30}
    assert batch["frames"].shape == (90, 13, 32, 32, 3)
    assert batch["masks"].shape == (90, 13, 10)
    assert batch["actions"].shape == batch["valid"].shape == (90, 12)
    assert replay.game_counts() == {"short": 1, "medium": 7, "long": 23}
    snapshot = replay.game_counts()
    snapshot["short"] = 999
    assert replay.game_counts()["short"] == 1


def test_sequences_never_cross_episode_boundaries_and_keep_previous_inputs():
    replay = SequenceReplay(seed=21)
    episode(replay, "game", 10, 3)
    episode(replay, "game", 90, 4, truncated=True)
    batch = replay.sample(80, 4, 8)
    for row in range(80):
        valid = np.flatnonzero(batch["valid"][row])
        assert valid.size and batch["valid"][row, 4]
        assert np.array_equal(valid, np.arange(valid[0], valid[-1] + 1))
        for t in valid:
            current = int(batch["frames"][row, t, 0, 0, 0])
            following = int(batch["frames"][row, t + 1, 0, 0, 0])
            assert following == current + 1
            assert (10 <= current <= 12) or (90 <= current <= 93)
            assert batch["resets"][row, t] == (current in (10, 90))
            assert batch["prev_actions"][row, t + 1] == batch["actions"][row, t]
            assert batch["prev_rewards"][row, t + 1] == batch["rewards"][row, t]
            if batch["terminated"][row, t] or batch["truncated"][row, t]:
                assert t == valid[-1]
        assert not batch["resets"][row, valid[-1] + 1]


def test_explicit_adaptation_weights_are_balanced_without_game_specific_logic():
    replay = SequenceReplay(seed=71)
    for i, game in enumerate(("source-a", "source-b", "source-c", "source-d", "target")):
        episode(replay, game, i * 20, 5)
    weights = {game: .05 for game in replay.games if game != "target"}
    weights["target"] = .8
    counts = Counter(replay.sample(100, 0, 2, game_weights=weights)["games"])
    assert counts == {"target": 80, "source-a": 5, "source-b": 5, "source-c": 5, "source-d": 5}
    with pytest.raises(ValueError, match="available games"):
        replay.sample(2, 0, 1, game_weights={"unknown": 1})


def test_capacity_limits_and_evicted_prefix_is_not_a_fake_reset():
    replay = SequenceReplay(capacity_per_game=3, capacity_total=5, seed=4)
    episode(replay, "a", 10, 7)
    assert replay.game_counts() == {"a": 3}
    batch = replay.sample(40, 3, 2)
    for row in range(40):
        valid = np.flatnonzero(batch["valid"][row])
        assert all(not batch["resets"][row, t] for t in valid)
        assert int(batch["frames"][row, valid[0], 0, 0, 0]) >= 14
    episode(replay, "b", 80, 7)
    assert len(replay) == 5
    assert all(n <= 3 for n in replay.game_counts().values())


def test_append_owns_pixels_and_rejects_discontinuous_or_illegal_chains():
    replay = SequenceReplay(seed=9)
    observation, successor = frame(10), frame(11)
    replay.append("g", observation, 0, 1, successor, False, False, mask(), mask())
    observation[:] = 200
    successor[:] = 201
    sample = replay.sample(1, 0, 1)
    assert int(sample["frames"][0, 0, 0, 0, 0]) == 10
    assert int(sample["frames"][0, 1, 0, 0, 0]) == 11
    with pytest.raises(ValueError, match="discontinuous"):
        replay.append("g", frame(50), 0, 1, frame(51), False, False, mask(), mask())
    with pytest.raises(ValueError, match="legal controller"):
        replay.append("g", frame(11), 9, 1, frame(12), False, False, mask(), mask())
    assert len(replay) == 1


def test_save_restore_reproduces_next_samples_and_active_episode_extension(tmp_path):
    replay = SequenceReplay(capacity_per_game=20, capacity_total=30, seed=947)
    episode(replay, "a", 10, 6)
    episode(replay, "a", 40, 4, closed=False)
    episode(replay, "b", 90, 9)
    for _ in range(3):
        replay.sample(7, 4, 8)
    directory = tmp_path / "replay"
    replay.save(directory)
    restored = SequenceReplay.load(directory)
    assert restored.game_counts() == replay.game_counts()
    assert restored.capacity_total == 30 and restored.capacity_per_game == 20
    for _ in range(12):
        assert_equal_batches(replay.sample(7, 4, 8), restored.sample(7, 4, 8))
    for current in (replay, restored):
        current.append("a", frame(44), 2, 8.5, frame(45), False, True, mask(), mask())
    for _ in range(5):
        assert_equal_batches(replay.sample(5, 4, 8), restored.sample(5, 4, 8))
    with pytest.raises(FileExistsError):
        replay.save(directory)


def test_snapshot_hash_and_internal_terminal_corruption_are_rejected(tmp_path):
    replay = SequenceReplay(seed=88)
    episode(replay, "a", 10, 4)
    root = tmp_path / "source"
    replay.save(root)
    metadata = json.loads((root / "metadata.json").read_text())
    record = metadata["episodes"][0]
    archive = root / record["file"]
    with np.load(archive, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    arrays["terminated"][0] = True
    with archive.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    with pytest.raises(ValueError, match="SHA-256"):
        SequenceReplay.load(root)
    record["sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="internal terminal"):
        SequenceReplay.load(root)


def test_object_arrays_never_load_with_pickle_and_empty_replay_roundtrips(tmp_path):
    empty = SequenceReplay(seed=2)
    empty.save(tmp_path / "empty")
    restored = SequenceReplay.load(tmp_path / "empty")
    with pytest.raises(ValueError, match="empty replay"):
        restored.sample(1, 0, 1)
    replay = SequenceReplay(seed=8)
    episode(replay, "a", 10, 1)
    root = tmp_path / "object-fixture"
    replay.save(root)
    metadata = json.loads((root / "metadata.json").read_text())
    record = metadata["episodes"][0]
    archive = root / record["file"]
    with archive.open("wb") as handle:
        np.savez_compressed(handle, frames=np.array([{"synthetic": "object"}], dtype=object))
    record["sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="Object arrays"):
        SequenceReplay.load(root)


def test_stage_boundary_truncation_preserves_successor_and_sampler_rng(tmp_path):
    replay = SequenceReplay(seed=1234)
    episode(replay, "source", 10, 4, closed=False)
    before = replay._rng.getstate()
    assert replay.truncate_active("source") is True
    assert replay._rng.getstate() == before
    assert replay.truncate_active("source") is False
    assert len(replay) == 4
    batch = replay.sample(80, 0, 4)
    assert batch["truncated"].any()
    assert not batch["terminated"].any()
    for row in range(80):
        for t in np.flatnonzero(batch["truncated"][row]):
            assert int(batch["frames"][row, t + 1, 0, 0, 0]) == 14
            assert t == np.flatnonzero(batch["valid"][row])[-1]
    # The following reset is a new episode, not a continuation of pixel 14.
    episode(replay, "source", 90, 2, closed=False)
    replay.save(tmp_path / "stage-boundary")
    restored = SequenceReplay.load(tmp_path / "stage-boundary")
    for _ in range(3):
        assert_equal_batches(replay.sample(8, 4, 8), restored.sample(8, 4, 8))
