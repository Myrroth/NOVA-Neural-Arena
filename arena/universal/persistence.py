"""Autocontained new-lineage snapshots with trusted manifests, never pickle fallback."""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform

import gymnasium
import numpy as np
import torch

from arena.shared.integrity import read_verified_bytes, verify_checksum_manifest
from arena.universal.envs import ACTION_NAMES
from arena.universal.network import VisualQNetwork
from arena.universal.replay import SequenceReplay
from arena.universal.trainer import UniversalTrainer


def save_checkpoint(trainer: UniversalTrainer, destination, *, source_revision: str,
                    protocol_sha256: str):
    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=False)
    metadata, tensors, observations = trainer.snapshot_payload()
    metadata.update(source_revision=source_revision, protocol_sha256=protocol_sha256,
                    action_names=list(ACTION_NAMES), observation_shape=[32, 32, 3],
                    architecture=trainer.online.architecture_version,
                    versions={"python": platform.python_version(), "torch": torch.__version__,
                              "numpy": np.__version__, "gymnasium": gymnasium.__version__,
                              "system": platform.system(), "machine": platform.machine()})
    (directory/"metadata.json").write_text(json.dumps(metadata, allow_nan=False,
                                                     sort_keys=True, indent=2)+"\n", encoding="utf-8")
    torch.save(tensors, directory/"state.pt")
    np.savez_compressed(directory/"observations.npz", **observations)
    trainer.replay.save(directory/"replay")
    entries = []
    for file in sorted(directory.rglob("*")):
        if file.is_file():
            entries.append(f"{hashlib.sha256(file.read_bytes()).hexdigest()}  {file.relative_to(directory).as_posix()}")
    manifest = ("\n".join(entries)+"\n").encode("utf-8")
    (directory/"SHA256SUMS").write_bytes(manifest)
    verify_checksum_manifest(directory)
    return {"path": str(directory), "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "files_verified": len(entries), "source_revision": source_revision,
            "protocol_sha256": protocol_sha256}


def load_checkpoint(directory, *, expected_manifest_sha256: str,
                    expected_source_revision: str, expected_protocol_sha256: str):
    root = Path(directory)
    read_verified_bytes(root/"SHA256SUMS", expected_sha256=expected_manifest_sha256,
                        max_bytes=1024*1024)
    hashes = verify_checksum_manifest(root)
    files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if files != set(hashes)|{"SHA256SUMS"}:
        raise ValueError("unlisted checkpoint files")
    metadata = json.loads(read_verified_bytes(root/"metadata.json",
                           expected_sha256=hashes["metadata.json"], max_bytes=32*1024*1024))
    expected_versions = {"python": platform.python_version(), "torch": torch.__version__,
                         "numpy": np.__version__, "gymnasium": gymnasium.__version__,
                         "system": platform.system(), "machine": platform.machine()}
    if (metadata["source_revision"] != expected_source_revision or
            metadata["protocol_sha256"] != expected_protocol_sha256 or
            metadata["action_names"] != list(ACTION_NAMES) or
            metadata["observation_shape"] != [32, 32, 3] or
            metadata["architecture"] != VisualQNetwork.architecture_version or
            metadata["versions"] != expected_versions):
        raise ValueError("checkpoint code/protocol/contract/runtime mismatch")
    raw = read_verified_bytes(root/"state.pt", expected_sha256=hashes["state.pt"],
                              max_bytes=64*1024*1024)
    tensors = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
    obs_raw = read_verified_bytes(root/"observations.npz",
                                  expected_sha256=hashes["observations.npz"], max_bytes=1024*1024)
    with np.load(BytesIO(obs_raw), allow_pickle=False) as archive:
        observations = {name: archive[name] for name in archive.files}
    replay = SequenceReplay.load(root/"replay")
    return UniversalTrainer.from_payload(metadata, tensors, observations, replay)
