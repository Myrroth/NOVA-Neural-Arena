"""Synthetic snapshot checks only: no game, model policy or campaign is run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from arena.continual import persistence as persist
from arena.universal.replay import SequenceReplay

SOURCE, PROTOCOL = "a" * 40, "b" * 64
ARCHITECTURE = "continual-persistence-synthetic-v2"


def replay(seed):
    value = SequenceReplay(capacity_per_game=16, capacity_total=32, seed=seed)
    mask = np.ones(10, dtype=bool)
    frame = np.full((32, 32, 3), seed, dtype=np.uint8)
    for i in range(3):
        after = np.full_like(frame, seed + i + 1)
        value.append("synthetic", frame, i, 0.1 * i, after, i == 2, False, mask, mask)
        frame = after
    return value


class Payload:
    def __init__(self, *, source=True, observations=True):
        self.online = SimpleNamespace(architecture_version=ARCHITECTURE)
        self.replay = replay(1)
        self.calibration_replay = replay(2)
        self.source_replay = replay(3) if source else None
        self.metadata = {
            "schema_version": 2,
            "stage": "adaptation",
            "model_seed": 123,
            "q_updates": 7,
            "world_updates": 3,
            "gate": {"revision": 7, "stats": {}},
            "actor_history": {"synthetic": [0, 1, 2]},
        }
        self.tensors = {
            "online": {"weight": torch.arange(6).reshape(2, 3).float()},
            "target": {"weight": torch.ones(2, 3)},
            "source_teacher": {"weight": torch.zeros(2, 3)} if source else None,
            "policy_optimizer": {
                "state": {0: {"step": torch.tensor(7.0), "exp_avg": torch.ones(2, 3)}},
                "param_groups": [{"lr": 0.003, "params": [0]}],
            },
            "world_optimizer": {
                "state": {0: {"step": torch.tensor(3.0)}},
                "param_groups": [{"lr": 0.001, "params": [0]}],
            },
            "torch_rng": torch.get_rng_state().clone(),
        }
        self.observations = (
            {"synthetic": np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)}
            if observations
            else {}
        )

    def snapshot_payload(self):
        return self.metadata, self.tensors, self.observations


@pytest.fixture(autouse=True)
def snapshot_factory(monkeypatch):
    monkeypatch.setattr(persist, "_architecture_version", lambda: ARCHITECTURE)
    monkeypatch.setattr(
        persist,
        "_restore_trainer",
        lambda metadata, tensors, observations, replays: SimpleNamespace(
            metadata=metadata,
            tensors=tensors,
            observations=observations,
            replays=replays,
        ),
    )


def saved(tmp_path, **kwargs):
    trainer = Payload(**kwargs)
    root = tmp_path / "new-v2-snapshot"
    receipt = persist.save_checkpoint(
        trainer, root, source_revision=SOURCE, protocol_sha256=PROTOCOL
    )
    return trainer, root, receipt


def load(root, receipt, **kwargs):
    return persist.load_checkpoint(
        root,
        expected_manifest_sha256=receipt["manifest_sha256"],
        expected_source_revision=kwargs.get("source", SOURCE),
        expected_protocol_sha256=kwargs.get("protocol", PROTOCOL),
    )


def rehash(root):
    names = sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.name != "SHA256SUMS"
    )
    raw = "".join(
        f"{hashlib.sha256((root/n).read_bytes()).hexdigest()}  {n}\n" for n in names
    ).encode()
    (root / "SHA256SUMS").write_bytes(raw)
    return {"manifest_sha256": hashlib.sha256(raw).hexdigest()}


def forbid_deserialization(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("rejected bytes must never reach torch.load")

    monkeypatch.setattr(torch, "load", forbidden)


def test_all_memories_and_two_optimizer_states_roundtrip_without_model_or_games(
    tmp_path, monkeypatch
):
    trainer, root, receipt = saved(tmp_path)
    original_metadata = dict(trainer.metadata)
    original_load = torch.load
    calls = []

    def checked_load(*args, **kwargs):
        calls.append(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    restored = load(root, receipt)
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    assert (
        trainer.metadata == original_metadata
    )  # saving does not mutate live metadata.
    assert (
        restored.metadata["q_updates"] == 7 and restored.metadata["world_updates"] == 3
    )
    assert (
        restored.metadata["schema_version"] == 2
        and restored.metadata["has_source_replay"] is True
    )
    assert torch.equal(
        restored.tensors["source_teacher"]["weight"],
        trainer.tensors["source_teacher"]["weight"],
    )
    assert restored.tensors["policy_optimizer"]["state"][0]["step"].item() == 7
    assert restored.tensors["world_optimizer"]["state"][0]["step"].item() == 3
    assert np.array_equal(
        restored.observations["synthetic"], trainer.observations["synthetic"]
    )
    for name, original in (
        ("fit", trainer.replay),
        ("calibration", trainer.calibration_replay),
        ("source", trainer.source_replay),
    ):
        actual, expected = restored.replays[name].sample(5, 2, 3), original.sample(
            5, 2, 3
        )
        for field in actual:
            if isinstance(actual[field], np.ndarray):
                assert np.array_equal(actual[field], expected[field]), (name, field)
            else:
                assert actual[field] == expected[field]


def test_unconfigured_snapshot_can_omit_teacher_and_actor_images(tmp_path):
    _, root, receipt = saved(tmp_path, source=False, observations=False)
    restored = load(root, receipt)
    assert restored.replays["source"] is None
    assert restored.observations == {}
    assert not (root / "replay/source").exists()


@pytest.mark.parametrize(
    "member", ["state.pt", "observations.npz", "replay/source/metadata.json"]
)
def test_modified_members_fail_before_any_deserialization(
    tmp_path, monkeypatch, member
):
    _, root, receipt = saved(tmp_path)
    with (root / member).open("ab") as handle:
        handle.write(b"untrusted")
    forbid_deserialization(monkeypatch)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load(root, receipt)


def test_wrong_manifest_or_source_or_protocol_fail_before_deserialization(
    tmp_path, monkeypatch
):
    _, root, receipt = saved(tmp_path)
    forbid_deserialization(monkeypatch)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load(root, {"manifest_sha256": "0" * 64})
    for override in ({"source": "c" * 40}, {"protocol": "d" * 64}):
        with pytest.raises(ValueError, match="mismatch"):
            load(root, receipt, **override)


@pytest.mark.parametrize(
    "change", ["runtime", "schema", "architecture", "actions", "source_replay_flag"]
)
def test_self_consistent_but_incompatible_metadata_rejected_before_weights(
    tmp_path, monkeypatch, change
):
    _, root, _ = saved(tmp_path)
    metadata = json.loads((root / "metadata.json").read_text())
    if change == "runtime":
        metadata["versions"]["python"] = "0.0.0"
    if change == "schema":
        metadata["schema_version"] = 1
    if change == "architecture":
        metadata["architecture"] = "historical-v1"
    if change == "actions":
        metadata["action_names"] = ["expert_action"]
    if change == "source_replay_flag":
        metadata["has_source_replay"] = 1
    (root / "metadata.json").write_text(json.dumps(metadata))
    receipt = rehash(root)
    forbid_deserialization(monkeypatch)
    with pytest.raises(ValueError, match="mismatch"):
        load(root, receipt)


def test_unlisted_and_unknown_listed_files_rejected(tmp_path, monkeypatch):
    _, root, receipt = saved(tmp_path)
    (root / "foreign-data.txt").write_text("not part of the checkpoint")
    forbid_deserialization(monkeypatch)
    with pytest.raises(ValueError, match="unlisted"):
        load(root, receipt)
    with pytest.raises(ValueError, match="unknown checkpoint member"):
        load(root, rehash(root))


def test_replay_catalog_cannot_hide_a_listed_episode(tmp_path, monkeypatch):
    _, root, _ = saved(tmp_path)
    p = root / "replay/source/metadata.json"
    meta = json.loads(p.read_text())
    meta["episodes"] = []
    p.write_text(json.dumps(meta))
    forbid_deserialization(monkeypatch)
    with pytest.raises(ValueError, match="exactly match"):
        load(root, rehash(root))


@pytest.mark.parametrize("kind", ["object", "expanded"])
def test_authenticated_observation_archive_still_obeys_type_and_expansion_limits(
    tmp_path, monkeypatch, kind
):
    _, root, _ = saved(tmp_path)
    array = (
        np.array([{"arbitrary": "object"}], dtype=object)
        if kind == "object"
        else np.zeros(persist.MAX_OBSERVATION_BYTES + 1, dtype=np.uint8)
    )
    np.savez_compressed(root / "observations.npz", hostile=array)
    forbid_deserialization(monkeypatch)
    with pytest.raises(ValueError, match="numeric observation|byte bound"):
        load(root, rehash(root))


def test_manifest_rejects_traversal_duplicate_case_and_self_reference(
    tmp_path, monkeypatch
):
    _, root, _ = saved(tmp_path)
    original = (root / "SHA256SUMS").read_text()
    example = original.splitlines()[0]
    for line in (
        "0" * 64 + "  ../outside.npz",
        example.upper(),
        "0" * 64 + "  SHA256SUMS",
    ):
        raw = (original + line + "\n").encode()
        (root / "SHA256SUMS").write_bytes(raw)
        receipt = {"manifest_sha256": hashlib.sha256(raw).hexdigest()}
        forbid_deserialization(monkeypatch)
        with pytest.raises(ValueError):
            load(root, receipt)


def test_existing_destination_is_never_replaced_and_identity_must_be_explicit(tmp_path):
    trainer, root, receipt = saved(tmp_path)
    before = (root / "SHA256SUMS").read_bytes()
    with pytest.raises(FileExistsError):
        persist.save_checkpoint(
            trainer, root, source_revision=SOURCE, protocol_sha256=PROTOCOL
        )
    assert (root / "SHA256SUMS").read_bytes() == before
    for source in ("", "development", "A" * 40):
        with pytest.raises(ValueError, match="Git SHA"):
            persist.save_checkpoint(
                trainer,
                tmp_path / "invalid",
                source_revision=source,
                protocol_sha256=PROTOCOL,
            )
    assert receipt["schema_version"] == 2


def test_save_rejects_pickle_objects_and_nonfinite_or_object_observations(tmp_path):
    trainer = Payload()
    trainer.tensors["unsupported"] = object()
    with pytest.raises(ValueError, match="unsupported object"):
        persist.save_checkpoint(
            trainer,
            tmp_path / "objects",
            source_revision=SOURCE,
            protocol_sha256=PROTOCOL,
        )
    assert not (tmp_path / "objects").exists()
    trainer = Payload()
    trainer.observations["bad"] = np.array([float("nan")])
    with pytest.raises(ValueError, match="finite numeric"):
        persist.save_checkpoint(
            trainer, tmp_path / "nan", source_revision=SOURCE, protocol_sha256=PROTOCOL
        )


def test_linked_parent_is_rejected_before_access(tmp_path, monkeypatch):
    original = Path.lstat

    def fake_lstat(path):
        info = original(path)
        if path == tmp_path:
            return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
        return info

    monkeypatch.setattr(Path, "lstat", fake_lstat)
    with pytest.raises(ValueError, match="links or junctions"):
        persist.save_checkpoint(
            Payload(),
            tmp_path / "child",
            source_revision=SOURCE,
            protocol_sha256=PROTOCOL,
        )
