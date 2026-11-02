"""Adversarial integrity checks; synthetic weights only, no games or training."""
from dataclasses import asdict
import hashlib
import json

import pytest
import torch

from arena.agents.dqn import DQNConfig, QNetwork
from arena.shared.checkpoints import load_frozen_checkpoint, load_official_checkpoint
from arena.shared.integrity import (
    read_verified_bytes, safe_relative_path, sha256_file,
    verify_checksum_manifest, write_new_json,
)


@pytest.fixture
def checkpoint(tmp_path):
    config = DQNConfig()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(943)
        net = QNetwork(config)
    payload = {"config": asdict(config), "online": net.state_dict()}
    path = tmp_path / "fixture.pt"
    torch.save(payload, path)
    return path, payload


def test_load_is_frozen_without_optimizer_and_preserves_rng(checkpoint, monkeypatch):
    path, payload = checkpoint
    original_load = torch.load
    calls = []
    def guarded_load(*args, **kwargs):
        calls.append(kwargs)
        return original_load(*args, **kwargs)
    monkeypatch.setattr(torch, "load", guarded_load)
    def forbidden(*args, **kwargs):
        pytest.fail("a read-only checkpoint load must not construct an optimizer")
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    before = torch.random.get_rng_state().clone()
    model = load_frozen_checkpoint(path, expected_sha256=sha256_file(path),
                                    expected_config=payload["config"])
    assert torch.equal(before, torch.random.get_rng_state())
    assert len(calls) == 1 and calls[0] == {"map_location": "cpu", "weights_only": True}
    assert not model.training
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    for key, tensor in payload["online"].items():
        assert torch.equal(tensor, model.state_dict()[key])


def test_bad_hash_is_rejected_before_deserialization(checkpoint, monkeypatch):
    path, _ = checkpoint
    def forbidden(*args, **kwargs):
        pytest.fail("deserialization preceded integrity check")
    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_frozen_checkpoint(path, expected_sha256="0" * 64)


def test_loader_consumes_the_verified_snapshot(checkpoint, monkeypatch):
    path, _ = checkpoint
    original = torch.load
    digest = sha256_file(path)
    def replace_disk_file_then_load(*args, **kwargs):
        path.write_bytes(b"changed after verified bytes were captured")
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, "load", replace_disk_file_then_load)
    model = load_frozen_checkpoint(path, expected_sha256=digest)
    assert not model.training
    assert sha256_file(path) != digest


@pytest.mark.parametrize("mutation,match", [
    ("nan", "non-finite"), ("dtype", "dtype"), ("shape", "shape"),
    ("names", "names"), ("config_nan", "finite"),
    ("config_float_dimension", "integer"), ("config_huge", "dimensions"),
    ("config_lr", "learning rate"), ("config_extra", "exactly"),
])
def test_rejects_invalid_checkpoint_schema(checkpoint, mutation, match):
    path, payload = checkpoint
    key = next(iter(payload["online"]))
    if mutation == "nan":
        payload["online"][key].flatten()[0] = float("nan")
    elif mutation == "dtype":
        payload["online"][key] = payload["online"][key].double()
    elif mutation == "shape":
        payload["online"][key] = torch.zeros(1)
    elif mutation == "names":
        del payload["online"][key]
    elif mutation == "config_nan":
        payload["config"]["gamma"] = float("nan")
    elif mutation == "config_float_dimension":
        payload["config"]["hidden"] = 64.0
    elif mutation == "config_huge":
        payload["config"]["hidden"] = 10**9
    elif mutation == "config_lr":
        payload["config"]["learning_rate"] = -1
    elif mutation == "config_extra":
        payload["config"]["unused"] = True
    torch.save(payload, path)
    with pytest.raises(ValueError, match=match):
        load_frozen_checkpoint(path, expected_sha256=sha256_file(path))


def test_trusted_config_is_required_to_match(checkpoint):
    path, payload = checkpoint
    expected = dict(payload["config"], gamma=0.9)
    with pytest.raises(ValueError, match="trusted expected config"):
        load_frozen_checkpoint(path, expected_sha256=sha256_file(path), expected_config=expected)


def test_official_loader_does_not_accept_preverified_bypass(checkpoint):
    path, _ = checkpoint
    with pytest.raises(TypeError):
        load_official_checkpoint(path.parent, seed=11, verified={"config": asdict(DQNConfig())})
    with pytest.raises(ValueError, match="unsafe artifact file"):
        load_official_checkpoint(path.parent, seed=11)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a\\b", "C:/a", "a//b",
                                  "a/./b", "CON.txt", "nested/NUL", "a.", "a ", ""])
def test_manifest_path_validation_is_portable(tmp_path, name):
    with pytest.raises(ValueError, match="artifact path"):
        safe_relative_path(tmp_path, name)


def test_checksums_verify_nested_paths_and_exact_set(tmp_path):
    (tmp_path / "sub").mkdir()
    file = tmp_path / "sub/metrics.json"
    file.write_bytes(b"{}\n")
    (tmp_path / "SHA256SUMS").write_text(f"{sha256_file(file)}  sub/metrics.json\n")
    result = verify_checksum_manifest(tmp_path, expected_files={"sub/metrics.json"})
    assert result == {"sub/metrics.json": sha256_file(file)}
    with pytest.raises(ValueError, match="member set"):
        verify_checksum_manifest(tmp_path, expected_files={"model.pt"})
    file.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_checksum_manifest(tmp_path)


def test_duplicate_checksum_and_traversal_rejected(tmp_path):
    digest = hashlib.sha256(b"x").hexdigest()
    (tmp_path / "a").write_bytes(b"x")
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text(f"{digest}  a\n{digest}  a\n")
    with pytest.raises(ValueError, match="duplicate"):
        verify_checksum_manifest(tmp_path)
    manifest.write_text(f"{digest}  ../a\n")
    with pytest.raises(ValueError, match="unsafe artifact path"):
        verify_checksum_manifest(tmp_path)


def test_bounded_read_and_exclusive_json_write(tmp_path):
    path = tmp_path / "a"
    path.write_bytes(b"123")
    with pytest.raises(ValueError, match="byte limit"):
        read_verified_bytes(path, expected_sha256=sha256_file(path), max_bytes=2)
    out = tmp_path / "result.json"
    write_new_json(out, {"status": "new"})
    before = out.read_bytes()
    with pytest.raises(FileExistsError):
        write_new_json(out, {"status": "replacement"})
    assert out.read_bytes() == before
    assert json.loads(out.read_text()) == {"status": "new"}
    with pytest.raises(ValueError):
        write_new_json(tmp_path / "invalid.json", {"x": float("nan")})
    assert not (tmp_path / "invalid.json").exists()
