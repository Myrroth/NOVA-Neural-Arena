"""Bounded, game-balanced sequence replay with exact sampler persistence.

No model, simulator, pickle, expert policy or winning action is used. Frames
are kept once per episode boundary chain. Capacity eviction can remove an old
prefix; that retained prefix is NOT falsely marked as a true environment reset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
import re
import zipfile

import numpy as np


def _positive_int(value, name, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if zero else 'positive'} integer")
    return value


def _action(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= value < 10:
        raise ValueError("action must be a joypad index in 0..9")
    return int(value)


def _number(value):
    if (isinstance(value, (bool, np.bool_, complex, np.complexfloating))
            or not isinstance(value, (int, float, np.number)) or not math.isfinite(float(value))):
        raise ValueError("rewards must be finite numbers")
    result = np.float32(value)
    if not np.isfinite(result):
        raise ValueError("reward exceeds float32 replay range")
    return float(result)


def _frame(value):
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.shape != (32, 32, 3):
        raise ValueError("observations must be uint8 RGB frames with shape (32, 32, 3)")
    return array.copy()


def _mask(value, *, allow_empty=False):
    array = np.asarray(value)
    if array.dtype != np.bool_ or array.shape != (10,) or (not allow_empty and not array.any()):
        raise ValueError("mask must be bool with shape (10,) and contain a legal input")
    return array.copy()


def _flag(value):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError("terminal flags must be booleans")
    return bool(value)


def _tuples(value):
    return tuple(_tuples(v) for v in value) if isinstance(value, list) else value


@dataclass
class _Episode:
    game: str
    identifier: int
    offset: int = 0
    closed: bool = False
    frames: list = field(default_factory=list)
    masks: list = field(default_factory=list)
    prev_actions: list = field(default_factory=list)
    prev_rewards: list = field(default_factory=list)
    resets: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    terminated: list = field(default_factory=list)
    truncated: list = field(default_factory=list)

    def __len__(self):
        return len(self.actions)

    def trim(self, count):
        self.offset += count
        for name in ("frames", "masks", "prev_actions", "prev_rewards", "resets",
                     "actions", "rewards", "terminated", "truncated"):
            del getattr(self, name)[:count]


class SequenceReplay:
    """Uniform games by default; optional declared game weights for adaptation.

    ``sample`` uses balanced systematic game sampling, then a uniformly chosen
    stored transition within that game as the first learning transition. Prefix
    and suffix padding never borrow another episode. ``valid`` includes real
    burn-in transitions; the caller excludes its first ``burn_in`` positions
    from the learning loss.
    """

    schema_version = 1

    def __init__(self, capacity_per_game: int = 10000, seed: int = 0, *, capacity_total: int = 40000):
        self.capacity_per_game = _positive_int(capacity_per_game, "capacity_per_game")
        self.capacity_total = _positive_int(capacity_total, "capacity_total")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self._rng = random.Random(seed)
        self._episodes: list[_Episode] = []
        self._active: dict[str, _Episode] = {}
        self._next_identifier = 0

    def __len__(self):
        return sum(len(episode) for episode in self._episodes)

    @property
    def size(self):
        return len(self)

    @property
    def counts(self):
        result = {}
        for episode in self._episodes:
            result[episode.game] = result.get(episode.game, 0) + len(episode)
        return result

    @property
    def games(self):
        return tuple(sorted(game for game, count in self.counts.items() if count))

    def game_counts(self):
        """A new mapping, so callers cannot mutate replay's bookkeeping."""
        return dict(self.counts)

    def append(self, game, obs, action, reward, next_obs, terminated, truncated,
               mask, next_mask, prev_action=0, prev_reward=0):
        if not isinstance(game, str) or not game or len(game) > 128:
            raise ValueError("game must be a nonempty identifier of at most 128 characters")
        if game not in self.counts and len(self.counts) >= 128:
            raise ValueError("replay supports at most 128 games")
        action, reward = _action(action), _number(reward)
        previous_action, previous_reward = _action(prev_action), _number(prev_reward)
        ended, limited = _flag(terminated), _flag(truncated)
        if ended and limited:
            raise ValueError("prepared games use mutually exclusive terminal flags")
        observation, successor = _frame(obs), _frame(next_obs)
        available = _mask(mask)
        following = _mask(next_mask, allow_empty=ended)
        if not available[action]:
            raise ValueError("recorded action is not a legal controller input")
        episode = self._active.get(game)
        if episode is not None:
            if (not np.array_equal(episode.frames[-1], observation)
                    or not np.array_equal(episode.masks[-1], available)):
                raise ValueError("active episode observation/mask chain is discontinuous")
            # Previous input after the first frame is already known from replay;
            # optional first-frame arguments cannot overwrite that actual history.
        else:
            episode = _Episode(game, self._next_identifier)
            self._next_identifier += 1
            episode.frames.append(observation)
            episode.masks.append(available)
            episode.prev_actions.append(previous_action)
            episode.prev_rewards.append(previous_reward)
            episode.resets.append(True)
            self._episodes.append(episode)
            self._active[game] = episode
        episode.actions.append(action)
        episode.rewards.append(reward)
        episode.terminated.append(ended)
        episode.truncated.append(limited)
        episode.frames.append(successor)
        episode.masks.append(following)
        episode.prev_actions.append(action)
        episode.prev_rewards.append(reward)
        episode.resets.append(False)
        if ended or limited:
            episode.closed = True
            del self._active[game]
        self._trim()

    def _trim_game(self, game, amount):
        for episode in list(self._episodes):
            if episode.game != game or not amount:
                continue
            count = min(amount, len(episode))
            episode.trim(count)
            amount -= count
            if not len(episode) and episode.closed:
                self._episodes.remove(episode)

    def truncate_active(self, game):
        """Seal a discarded actor episode as an external truncation, without RNG.

        No transition or observation is invented. Its real final successor is
        retained for bootstrap; ``terminated`` remains false. This is needed
        before resetting an actor at a declared training-stage boundary.
        """
        if not isinstance(game, str) or not game:
            raise ValueError("game must be a nonempty identifier")
        episode = self._active.pop(game, None)
        if episode is None:
            return False
        if len(episode):
            episode.terminated[-1] = False
            episode.truncated[-1] = True
            episode.closed = True
        else:
            # Every transition of this active prefix was already evicted.
            self._episodes.remove(episode)
        return True

    def _trim(self):
        for game, count in self.counts.items():
            if count > self.capacity_per_game:
                self._trim_game(game, count - self.capacity_per_game)
        # Evict from the largest game first so a new adaptation game cannot
        # silently remove every transition of a small source game.
        while len(self) > self.capacity_total:
            counts = self.counts
            game = max(sorted(counts), key=lambda name: counts[name])
            self._trim_game(game, min(len(self) - self.capacity_total, 1))

    def _game_batch(self, batch_size, game_weights):
        games = self.games
        if not games:
            raise ValueError("cannot sample empty replay")
        if game_weights is None:
            weights = [1.0] * len(games)
        else:
            if not isinstance(game_weights, dict) or not game_weights:
                raise ValueError("game_weights must be a nonempty mapping")
            for game, value in game_weights.items():
                if (game not in games or type(value) not in (int, float)
                        or not math.isfinite(value) or value < 0):
                    raise ValueError("game weights must name available games and be finite/nonnegative")
            weights = [float(game_weights.get(game, 0.0)) for game in games]
            if not any(weights):
                raise ValueError("at least one game weight must be positive")
        total = sum(weights)
        if not math.isfinite(total) or total <= 0:
            raise ValueError("game weight total must be finite and positive")
        cumulative, value = [], 0.0
        for weight in weights:
            value += weight / total
            cumulative.append(value)
        cumulative[-1] = 1.0
        offset = self._rng.random()
        selected = []
        for i in range(batch_size):
            point = (i + offset) / batch_size
            index = 0
            while point >= cumulative[index]:
                index += 1
            selected.append(games[index])
        self._rng.shuffle(selected)
        return selected

    def sample(self, batch_size: int, burn_in: int, unroll: int, *, game_weights=None):
        _positive_int(batch_size, "batch_size")
        _positive_int(burn_in, "burn_in", zero=True)
        _positive_int(unroll, "unroll")
        time = burn_in + unroll
        if batch_size > 4096 or time > 4096:
            raise ValueError("requested replay batch exceeds sequence limits")
        selected = self._game_batch(batch_size, game_weights)
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
        batch["masks"][:, :, 0] = True  # harmless legal input on neutral padding
        for row, game in enumerate(selected):
            candidates = [episode for episode in self._episodes if episode.game == game and len(episode)]
            pick = self._rng.randrange(sum(len(episode) for episode in candidates))
            for episode in candidates:
                if pick < len(episode):
                    break
                pick -= len(episode)
            anchor = pick
            start = max(0, anchor - burn_in)
            stop = min(len(episode), anchor + unroll)
            left = burn_in - (anchor - start)
            count = stop - start
            batch["episode_ids"][row] = episode.identifier
            batch["start_indices"][row] = episode.offset + anchor
            for name in ("frames", "prev_actions", "prev_rewards", "resets", "masks"):
                batch[name][row, left:left + count + 1] = np.asarray(getattr(episode, name)[start:stop + 1])
            for name in ("actions", "rewards", "terminated", "truncated"):
                batch[name][row, left:left + count] = np.asarray(getattr(episode, name)[start:stop])
            batch["valid"][row, left:left + count] = True
        return batch

    def save(self, directory):
        """Write a NEW directory of bounded per-episode NPZ files and JSON state."""
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=False)
        records = []
        dtypes = {"frames": np.uint8, "masks": np.bool_, "prev_actions": np.int64,
                  "prev_rewards": np.float32, "resets": np.bool_, "actions": np.int64,
                  "rewards": np.float32, "terminated": np.bool_, "truncated": np.bool_}
        for episode in self._episodes:
            filename = f"episode-{episode.identifier:012d}.npz"
            arrays = {name: np.asarray(getattr(episode, name), dtype=dtype)
                      for name, dtype in dtypes.items()}
            with (root / filename).open("xb") as handle:
                np.savez_compressed(handle, **arrays)
            records.append({"file": filename, "sha256": hashlib.sha256((root / filename).read_bytes()).hexdigest(),
                            "game": episode.game, "identifier": episode.identifier,
                            "offset": episode.offset, "closed": episode.closed,
                            "transitions": len(episode)})
        metadata = {"schema_version": self.schema_version,
                    "capacity_per_game": self.capacity_per_game, "capacity_total": self.capacity_total,
                    "next_identifier": self._next_identifier, "rng_state": self._rng.getstate(),
                    "episodes": records, "active_episode_ids": {game: ep.identifier for game, ep in self._active.items()}}
        with (root / "metadata.json").open("x", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")

    @classmethod
    def load(cls, directory):
        """Restore data and sampler RNG without unpickling any Python object."""
        root = Path(directory)
        metadata_path = root / "metadata.json"
        if metadata_path.is_symlink() or metadata_path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("invalid replay metadata file")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if type(metadata) is not dict or metadata["schema_version"] != cls.schema_version:
                raise ValueError("unsupported replay schema")
            replay = cls(metadata["capacity_per_game"], capacity_total=metadata["capacity_total"])
            replay._next_identifier = _positive_int(metadata["next_identifier"], "next_identifier", zero=True)
            replay._rng.setstate(_tuples(metadata["rng_state"]))
            records = metadata["episodes"]
            if type(records) is not list or len(records) > replay.capacity_total + 128:
                raise ValueError("replay episode catalog exceeds capacity")
            identifiers = set()
            total = 0
            for record in records:
                identifier = _positive_int(record["identifier"], "identifier", zero=True)
                count = _positive_int(record["transitions"], "transitions", zero=True)
                offset = _positive_int(record["offset"], "offset", zero=True)
                filename = record["file"]
                if (type(filename) is not str or not re.fullmatch(r"episode-[0-9]{12}\.npz", filename)
                        or filename != f"episode-{identifier:012d}.npz"
                        or identifier in identifiers or identifier >= replay._next_identifier
                        or type(record["closed"]) is not bool
                        or not isinstance(record["game"], str) or not record["game"]
                        or len(record["game"]) > 128):
                    raise ValueError("invalid replay episode identity")
                total += count
                if total > replay.capacity_total or count > replay.capacity_per_game:
                    raise ValueError("replay saved transition capacity exceeded")
                path = root / filename
                if path.is_symlink() or path.stat().st_size > (count + 1) * 4096 + 65536:
                    raise ValueError("invalid replay episode file")
                if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                    raise ValueError("replay episode SHA-256 mismatch")
                with zipfile.ZipFile(path) as archive:
                    if sum(info.file_size for info in archive.infolist()) > (count + 1) * 4096 + 65536:
                        raise ValueError("replay episode expands beyond its capacity")
                with np.load(path, allow_pickle=False) as loaded:
                    arrays = {name: loaded[name] for name in loaded.files}
                cls._validate_arrays(arrays, count, offset, record["closed"])
                episode = _Episode(record["game"], identifier, offset, record["closed"])
                for name, array in arrays.items():
                    setattr(episode, name, [item.copy() if isinstance(item, np.ndarray) else item.item()
                                          for item in array])
                replay._episodes.append(episode)
                identifiers.add(identifier)
            active = metadata["active_episode_ids"]
            if type(active) is not dict or len(active) > 128:
                raise ValueError("invalid active episode catalog")
            for game, identifier in active.items():
                matches = [ep for ep in replay._episodes if ep.identifier == identifier and ep.game == game and not ep.closed]
                if len(matches) != 1:
                    raise ValueError("active episode reference is invalid")
                replay._active[game] = matches[0]
            if (any(not ep.closed and replay._active.get(ep.game) is not ep for ep in replay._episodes)
                    or any(n > replay.capacity_per_game for n in replay.counts.values())):
                raise ValueError("active episodes or per-game capacity are inconsistent")
            return replay
        except (KeyError, TypeError, OSError, IndexError, OverflowError, zipfile.BadZipFile) as exc:
            raise ValueError("invalid replay snapshot") from exc

    @staticmethod
    def _validate_arrays(arrays, count, offset, closed):
        specifications = {"frames": (np.uint8, (count + 1, 32, 32, 3)),
                          "masks": (np.bool_, (count + 1, 10)),
                          "prev_actions": (np.int64, (count + 1,)),
                          "prev_rewards": (np.float32, (count + 1,)),
                          "resets": (np.bool_, (count + 1,)),
                          "actions": (np.int64, (count,)), "rewards": (np.float32, (count,)),
                          "terminated": (np.bool_, (count,)), "truncated": (np.bool_, (count,))}
        if set(arrays) != set(specifications):
            raise ValueError("replay NPZ schema differs")
        for name, (dtype, shape) in specifications.items():
            if arrays[name].dtype != dtype or arrays[name].shape != shape:
                raise ValueError(f"replay array type/shape differs: {name}")
        if (not np.isfinite(arrays["rewards"]).all() or not np.isfinite(arrays["prev_rewards"]).all()
                or ((arrays["actions"] < 0) | (arrays["actions"] >= 10)).any()
                or ((arrays["prev_actions"] < 0) | (arrays["prev_actions"] >= 10)).any()
                or arrays["resets"][1:].any() or bool(arrays["resets"][0]) != (offset == 0)):
            raise ValueError("invalid replay history values")
        ends = arrays["terminated"] | arrays["truncated"]
        if ((arrays["terminated"] & arrays["truncated"]).any() or ends[:-1].any()
                or bool(ends[-1] if count else False) != closed):
            raise ValueError("replay episode contains an internal terminal boundary")
        if count:
            if (not np.array_equal(arrays["prev_actions"][1:], arrays["actions"])
                    or not np.array_equal(arrays["prev_rewards"][1:], arrays["rewards"])
                    or not arrays["masks"][:-1].any(axis=1).all()
                    or not arrays["masks"][np.arange(count), arrays["actions"]].all()):
                raise ValueError("replay previous inputs/masks disagree with transitions")
        if not arrays["masks"][-1].any() and not (count and arrays["terminated"][-1]):
            raise ValueError("nonterminal successor has no legal action")
