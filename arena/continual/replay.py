"""Exact finite-controller-context sampling over the immutable v1 replay format.

This adapter intentionally uses SequenceReplay schema 1's ``_episodes``,
``_game_batch`` and persistent ``_rng``. It does not change the historical
storage/eviction implementation. A retained episode fragment can lack the
controller history before its first periodic reset; those anchors are excluded.
No simulator, model or game-specific policy is constructed here.
"""

from __future__ import annotations

import numpy as np

from arena.universal.replay import SequenceReplay


class InsufficientContext(ValueError):
    """A selected game has no transition with reconstructible controller state."""


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _eligible(episode, context_length):
    # ``offset`` counts transitions evicted from the original episode. If the
    # fragment begins mid-context, its next real controller reset is the first
    # usable anchor; this is not a fabricated physical episode reset.
    first = (-episode.offset) % context_length
    return first, max(0, len(episode) - first)


def sample_context_sequences(
    replay: SequenceReplay,
    batch_size: int,
    context_length: int,
    unroll: int,
    *,
    game_weights=None,
):
    """Return the v1 sample layout with exactly reconstructible periodic memory.

    Burn-in is ``context_length - 1``. Games use the parent replay's balanced
    systematic sampler without dropping or silently reweighting any game.
    Within each selected game, sampling is uniform over ELIGIBLE transitions,
    rather than uniform over episodes. Every real observation receives the
    actor's periodic reset overlay at its original absolute episode index.

    Missing eligible context raises ``InsufficientContext``. Any failed sampling
    attempt restores the original parent sampler RNG, so it cannot leave a
    partly consumed, unreproducible selection behind. Successful sampling uses
    only that same persisted RNG; data and replay counters remain unchanged.
    """
    if not isinstance(replay, SequenceReplay) or replay.schema_version != 1:
        raise ValueError("context sampling requires SequenceReplay schema 1")
    for value, name in (
        (batch_size, "batch_size"),
        (context_length, "context_length"),
        (unroll, "unroll"),
    ):
        _positive(value, name)
    burn_in = context_length - 1
    time = burn_in + unroll
    if batch_size > 4096 or time > 4096:
        raise ValueError("requested replay batch exceeds sequence limits")
    rng_before = replay._rng.getstate()
    try:
        selected = replay._game_batch(batch_size, game_weights)
        catalogs = {}
        for game in selected:
            if game in catalogs:
                continue
            catalog = []
            for episode in replay._episodes:
                if episode.game != game:
                    continue
                first, count = _eligible(episode, context_length)
                if count:
                    catalog.append((episode, first, count))
            total = sum(count for _, _, count in catalog)
            if not total:
                raise InsufficientContext(
                    f"selected game {game!r} has no reconstructible context anchor"
                )
            catalogs[game] = catalog, total
        batch = {
            "frames": np.zeros((batch_size, time + 1, 32, 32, 3), dtype=np.uint8),
            "prev_actions": np.zeros((batch_size, time + 1), dtype=np.int64),
            "prev_rewards": np.zeros((batch_size, time + 1), dtype=np.float32),
            "resets": np.ones((batch_size, time + 1), dtype=np.bool_),
            "actions": np.zeros((batch_size, time), dtype=np.int64),
            "rewards": np.zeros((batch_size, time), dtype=np.float32),
            "terminated": np.zeros((batch_size, time), dtype=np.bool_),
            "truncated": np.zeros((batch_size, time), dtype=np.bool_),
            "valid": np.zeros((batch_size, time), dtype=np.bool_),
            "masks": np.zeros((batch_size, time + 1, 10), dtype=np.bool_),
            "games": selected,
            "episode_ids": np.zeros(batch_size, dtype=np.int64),
            "start_indices": np.zeros(batch_size, dtype=np.int64),
        }
        batch["masks"][:, :, 0] = True  # legal neutral input on padding only
        for row, game in enumerate(selected):
            catalog, total = catalogs[game]
            pick = replay._rng.randrange(total)
            for episode, first, count in catalog:
                if pick < count:
                    anchor = first + pick
                    break
                pick -= count
            start = max(0, anchor - burn_in)
            stop = min(len(episode), anchor + unroll)
            left = burn_in - (anchor - start)
            count = stop - start
            batch["episode_ids"][row] = episode.identifier
            batch["start_indices"][row] = episode.offset + anchor
            for name in ("frames", "prev_actions", "prev_rewards", "masks"):
                batch[name][row, left : left + count + 1] = np.asarray(
                    getattr(episode, name)[start : stop + 1]
                )
            original_resets = np.asarray(
                episode.resets[start : stop + 1], dtype=np.bool_
            )
            absolute = episode.offset + np.arange(start, stop + 1)
            batch["resets"][row, left : left + count + 1] = original_resets | (
                absolute % context_length == 0
            )
            for name in ("actions", "rewards", "terminated", "truncated"):
                batch[name][row, left : left + count] = np.asarray(
                    getattr(episode, name)[start:stop]
                )
            batch["valid"][row, left : left + count] = True
        return batch
    except Exception:
        replay._rng.setstate(rng_before)
        raise
