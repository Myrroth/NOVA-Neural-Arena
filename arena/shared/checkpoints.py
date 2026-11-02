"""Read-only, bounded checkpoint loading for new evaluation infrastructure.

Never constructs an agent/optimizer. Official model identity is pinned in the
historical catalog; the supplied digest in the generic loader is a trust input.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
from io import BytesIO
import math
from pathlib import Path

import torch

from arena.agents.dqn import DQNConfig, QNetwork
from arena.evaluation.frozen_phase2 import ARTIFACTS, verify_artifact_directory
from arena.shared.integrity import read_verified_bytes, safe_relative_path


def _validated_config(value: object) -> DQNConfig:
    defaults = asdict(DQNConfig())
    if type(value) is not dict or set(value) != set(defaults):
        raise ValueError("checkpoint config must contain exactly the DQNConfig fields")
    for key, default in defaults.items():
        number = value[key]
        if type(default) is int:
            if type(number) is not int:
                raise ValueError(f"checkpoint config requires an integer: {key}")
        elif type(number) not in (int, float) or not math.isfinite(number):
            raise ValueError(f"checkpoint config requires a finite number: {key}")
    if not (1 <= value["observation_dim"] <= 4096 and 1 <= value["actions"] <= 4096
            and 1 <= value["hidden"] <= 1024):
        raise ValueError("checkpoint network dimensions exceed evaluation limits")
    parameters = (value["observation_dim"] * value["hidden"] + value["hidden"] ** 2
                  + value["hidden"] * value["actions"] + 2 * value["hidden"] + value["actions"])
    if parameters > 5_000_000:
        raise ValueError("checkpoint network exceeds parameter limit")
    if not (0 <= value["gamma"] <= 1 and 0 < value["learning_rate"] <= 1):
        raise ValueError("checkpoint gamma or learning rate is outside evaluation limits")
    try:
        return DQNConfig(**value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid checkpoint DQN configuration") from exc


def load_frozen_checkpoint(path: str | Path, *, expected_sha256: str,
                           expected_config: dict | None = None) -> QNetwork:
    """Verify bounded bytes, deserialize weights only, validate tensors and freeze.

    No inference is performed. SHA verification and deserialization use the same
    in-memory bytes, closing the historical verify-then-reopen race. This is not
    a sandbox for arbitrary untrusted PyTorch files: the digest must be trusted.
    """
    data_bytes = read_verified_bytes(path, expected_sha256=expected_sha256)
    try:
        payload = torch.load(BytesIO(data_bytes), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("checkpoint cannot be loaded in weights-only mode") from exc
    if type(payload) is not dict or "config" not in payload or "online" not in payload:
        raise ValueError("invalid checkpoint payload")
    config = _validated_config(payload["config"])
    if expected_config is not None and payload["config"] != expected_config:
        raise ValueError("checkpoint config differs from trusted expected config")
    state = payload["online"]
    if type(state) not in (dict, OrderedDict):
        raise ValueError("checkpoint online weights must be a state dictionary")
    # Construction initializes weights; fork_rng prevents loading from changing
    # the caller's initialization schedule for untrained controls.
    with torch.random.fork_rng(devices=[]):
        network = QNetwork(config)
    expected = network.state_dict()
    if set(state) != set(expected):
        raise ValueError("checkpoint tensor names differ from network schema")
    for name, reference in expected.items():
        value = state[name]
        if (type(value) is not torch.Tensor or value.layout != torch.strided
                or value.device.type != "cpu" or value.dtype != reference.dtype
                or tuple(value.shape) != tuple(reference.shape)):
            raise ValueError(f"invalid checkpoint tensor type, shape or dtype: {name}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"non-finite checkpoint tensor: {name}")
    network.load_state_dict(state, strict=True)
    network.eval()
    network.requires_grad_(False)
    return network


def load_official_checkpoint(directory: str | Path, *, seed: int) -> QNetwork:
    """Reverify official provenance and pinned identity on every load; no bypass."""
    if type(seed) is not int or seed not in ARTIFACTS:
        raise ValueError("seed is not an official training seed")
    verified = verify_artifact_directory(directory, seed=seed)
    model = load_frozen_checkpoint(
        safe_relative_path(directory, "model.pt"),
        expected_sha256=ARTIFACTS[seed][1], expected_config=verified["config"],
    )
    if verified["config"] != asdict(DQNConfig()):
        raise ValueError("official checkpoint does not use the frozen pilot configuration")
    return model
