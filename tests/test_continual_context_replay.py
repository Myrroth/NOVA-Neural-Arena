"""Synthetic recorded-frame fixtures only; never a prepared-game interaction."""

import copy

import numpy as np
import pytest

from arena.continual.replay import InsufficientContext, sample_context_sequences
from arena.universal.replay import SequenceReplay


def append_episode(replay, game, length, *, base=20, truncated=False):
    frames = [np.full((32, 32, 3), base + i, dtype=np.uint8) for i in range(length + 1)]
    mask = np.ones(10, dtype=np.bool_)
    for i in range(length):
        final = i == length - 1
        replay.append(
            game,
            frames[i],
            i % 10,
            (i % 3 - 1) * 0.1,
            frames[i + 1],
            final and not truncated,
            final and truncated,
            mask,
            mask,
            prev_action=(i - 1) % 10 if i else 0,
            prev_reward=((i - 1) % 3 - 1) * 0.1 if i else 0.0,
        )
    return frames


def force_first_anchor(monkeypatch, replay):
    monkeypatch.setattr(replay._rng, "randrange", lambda total: 0)


def test_offset5_context16_excludes_every_anchor_before_first_retained_reset(
    monkeypatch,
):
    replay = SequenceReplay(capacity_per_game=40, capacity_total=40, seed=831151)
    append_episode(replay, "synthetic", 45)
    episode = replay._episodes[0]
    assert episode.offset == 5 and len(episode) == 40
    available = []
    limits = []
    for rank in range(29):

        def choose(total, rank=rank):
            limits.append(total)
            return rank

        monkeypatch.setattr(replay._rng, "randrange", choose)
        data = sample_context_sequences(replay, 1, 16, 4)
        available.append(int(data["start_indices"][0]))
    assert available == list(range(16, 45))
    assert limits == [29] * 29  # uniform rank over transitions, not episodes
    assert episode.offset == 5 and len(episode) == 40


def test_an_aligned_evicted_prefix_keeps_its_first_transition_eligible(monkeypatch):
    replay = SequenceReplay(capacity_per_game=32, capacity_total=32, seed=831152)
    frames = append_episode(replay, "synthetic", 48)
    assert replay._episodes[0].offset == 16
    force_first_anchor(monkeypatch, replay)
    data = sample_context_sequences(replay, 1, 16, 3)
    assert data["start_indices"].tolist() == [16]
    assert not data["valid"][0, :15].any()
    assert data["valid"][0, 15:].all()
    assert data["resets"][0, 15]
    np.testing.assert_array_equal(data["frames"][0, 15], frames[16])


def test_uniform_transition_ranks_across_unequal_episodes(monkeypatch):
    replay = SequenceReplay(capacity_per_game=64, capacity_total=64, seed=831153)
    append_episode(replay, "synthetic", 2, base=20)
    append_episode(replay, "synthetic", 5, base=100)
    selected = []
    for rank in range(7):
        monkeypatch.setattr(replay._rng, "randrange", lambda total, rank=rank: rank)
        data = sample_context_sequences(replay, 1, 4, 2)
        selected.append((int(data["episode_ids"][0]), int(data["start_indices"][0])))
    assert selected == [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]


def test_missing_selected_context_raises_without_dropping_game_or_consuming_rng():
    replay = SequenceReplay(capacity_per_game=3, capacity_total=20, seed=831154)
    append_episode(
        replay, "a", 8, base=20
    )  # retained absolute5..7; next reset16 is missing
    append_episode(replay, "b", 3, base=100)
    original_rng = replay._rng.getstate()
    original_counts = replay.game_counts()
    with pytest.raises(InsufficientContext, match="'a'"):
        sample_context_sequences(replay, 2, 16, 2)
    assert replay._rng.getstate() == original_rng
    assert replay.game_counts() == original_counts
    # Explicit caller weights may select only b; the adapter never does so itself.
    data = sample_context_sequences(replay, 4, 16, 2, game_weights={"a": 0.0, "b": 1.0})
    assert data["games"] == ["b"] * 4


def test_game_balance_and_rng_are_the_parent_balancer_then_eligible_rank_draws():
    replay = SequenceReplay(capacity_per_game=64, capacity_total=128, seed=831155)
    append_episode(replay, "a", 5, base=20)
    append_episode(replay, "b", 7, base=100)
    reference = copy.deepcopy(replay)
    expected_games = reference._game_batch(8, {"a": 1.0, "b": 3.0})
    expected_anchors = [
        reference._rng.randrange(5 if game == "a" else 7) for game in expected_games
    ]
    data = sample_context_sequences(replay, 8, 4, 3, game_weights={"a": 1.0, "b": 3.0})
    assert data["games"] == expected_games
    assert data["games"].count("a") == 2 and data["games"].count("b") == 6
    assert data["start_indices"].tolist() == expected_anchors
    assert replay._rng.getstate() == reference._rng.getstate()


def test_parent_replay_save_reload_reproduces_next_context_batch(tmp_path):
    replay = SequenceReplay(capacity_per_game=40, capacity_total=80, seed=831156)
    append_episode(replay, "a", 45, base=20)
    append_episode(replay, "b", 6, base=100, truncated=True)
    sample_context_sequences(replay, 8, 16, 5)
    replay.save(tmp_path / "synthetic-replay")
    restored = SequenceReplay.load(tmp_path / "synthetic-replay")
    expected = sample_context_sequences(replay, 8, 16, 5)
    actual = sample_context_sequences(restored, 8, 16, 5)
    assert expected.keys() == actual.keys()
    for name in expected:
        if isinstance(expected[name], np.ndarray):
            np.testing.assert_array_equal(expected[name], actual[name])
        else:
            assert expected[name] == actual[name]
    assert replay._rng.getstate() == restored._rng.getstate()


@pytest.mark.parametrize("truncated", [False, True])
def test_episode_boundaries_real_successors_flags_and_padding_never_borrow(truncated):
    replay = SequenceReplay(capacity_per_game=64, capacity_total=64, seed=831157)
    append_episode(replay, "synthetic", 3, base=20, truncated=truncated)
    append_episode(replay, "synthetic", 5, base=100, truncated=not truncated)
    data = sample_context_sequences(replay, 32, 4, 8)
    burn = 3
    for row, identifier in enumerate(data["episode_ids"]):
        episode = next(e for e in replay._episodes if e.identifier == identifier)
        anchor = int(data["start_indices"][row])
        real = len(episode) - anchor
        assert data["valid"][row, burn : burn + real].all()
        assert not data["valid"][row, burn + real :].any()
        np.testing.assert_array_equal(
            data["frames"][row, burn : burn + real + 1],
            np.asarray(episode.frames[anchor:]),
        )
        np.testing.assert_array_equal(
            data["terminated"][row, burn : burn + real], episode.terminated[anchor:]
        )
        np.testing.assert_array_equal(
            data["truncated"][row, burn : burn + real], episode.truncated[anchor:]
        )
        assert not data["frames"][row, burn + real + 1 :].any()
        assert data["masks"][row, burn + real + 1 :, 0].all()
        assert not data["masks"][row, burn + real + 1 :, 1:].any()


def test_every_learning_state_matches_a_full_periodic_controller_prefix():
    # Deterministic recurrent accumulator makes missing prefix state visible
    # without constructing a neural network or executing any simulator.
    replay = SequenceReplay(capacity_per_game=40, capacity_total=40, seed=831158)
    originals = append_episode(replay, "synthetic", 45, base=20)
    data = sample_context_sequences(replay, 32, 16, 8)
    burn = 15
    for row, anchor in enumerate(data["start_indices"]):
        state = 0.0
        states = []
        for frame, reset in zip(data["frames"][row], data["resets"][row]):
            if reset:
                state = 0.0
            state = 0.5 * state + float(frame[0, 0, 0])
            states.append(state)
        real = int(data["valid"][row, burn:].sum())
        for distance in range(real + 1):  # includes actual last successor
            absolute = int(anchor) + distance
            start = absolute - absolute % 16
            expected = 0.0
            for frame in originals[start : absolute + 1]:
                expected = 0.5 * expected + float(frame[0, 0, 0])
            assert states[burn + distance] == expected


def test_context1_is_zero_history_and_resets_every_real_observation():
    replay = SequenceReplay(capacity_per_game=3, capacity_total=3, seed=831159)
    append_episode(replay, "synthetic", 8)
    data = sample_context_sequences(replay, 8, 1, 2)
    assert data["start_indices"].min() >= 5
    assert data["resets"].all()
    assert data["valid"][:, 0].all()


@pytest.mark.parametrize(
    "batch,context,unroll",
    [(0, 4, 2), (True, 4, 2), (1, 0, 2), (1, 4, 0), (4097, 4, 2), (1, 4096, 2)],
)
def test_invalid_sizes_do_not_consume_parent_rng(batch, context, unroll):
    replay = SequenceReplay(seed=831160)
    append_episode(replay, "synthetic", 2)
    original = replay._rng.getstate()
    with pytest.raises(ValueError):
        sample_context_sequences(replay, batch, context, unroll)
    assert replay._rng.getstate() == original


def test_invalid_game_weights_and_empty_replay_fail_without_rng_change():
    replay = SequenceReplay(seed=831161)
    original = replay._rng.getstate()
    with pytest.raises(ValueError, match="empty"):
        sample_context_sequences(replay, 2, 4, 2)
    assert replay._rng.getstate() == original
    append_episode(replay, "synthetic", 2)
    original = replay._rng.getstate()
    with pytest.raises(ValueError, match="weights"):
        sample_context_sequences(replay, 2, 4, 2, game_weights={"unknown": 1.0})
    assert replay._rng.getstate() == original
