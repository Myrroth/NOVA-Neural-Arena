"""Verified v2 snapshots, including retention/calibration memory and optimizers.

The trusted manifest digest must come from an independent catalog. Loading does
not collect transitions, update models or accept historical v1 checkpoints.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
from io import BytesIO
import json
from pathlib import Path
import platform
import re
import stat
import zipfile

import gymnasium
import numpy as np
import torch

from arena.shared.integrity import (
    read_verified_bytes,
    safe_relative_path,
    sha256_file,
    validate_sha256,
)
from arena.universal.envs import ACTION_NAMES
from arena.universal.replay import SequenceReplay


SCHEMA_VERSION = 2
MAX_FILES = 120_000
MAX_MANIFEST_BYTES = 16 * 1024**2
MAX_METADATA_BYTES = 32 * 1024**2
MAX_STATE_BYTES = 64 * 1024**2
MAX_OBSERVATION_BYTES = 4 * 1024**2
REPLAY_LIMITS = {"fit": 40_000, "calibration": 8_000, "source": 40_000}


def _source(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("source revision must be an explicit lowercase Git SHA")
    return value


def _runtime():
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "gymnasium": gymnasium.__version__,
        "system": platform.system(),
        "machine": platform.machine(),
    }


def _architecture_version():
    from arena.continual.network import ContinualQNetwork

    return ContinualQNetwork.architecture_version


def _restore_trainer(metadata, tensors, observations, replays):
    from arena.continual.trainer import ContinualTrainer

    return ContinualTrainer.from_payload(metadata, tensors, observations, replays)


def _plain_path(path, *, directory):
    """Check links before resolution; reject ancestor junctions as well."""
    path = Path(path).absolute()
    for component in (path, *path.parents):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("checkpoint paths must not contain links or junctions")
    if path.exists() and not (path.is_dir() if directory else path.is_file()):
        raise ValueError("unexpected checkpoint path type")
    return path


def _files(root):
    files = set()
    # pathlib 3.11 does not recurse into directory symlinks; inspect all yielded
    # directories too so their presence cannot be hidden by that behavior.
    for path in root.rglob("*"):
        _plain_path(path, directory=path.is_dir())
        if path.is_file():
            files.add(path.relative_to(root).as_posix())
            if len(files) > MAX_FILES + 1:
                raise ValueError("checkpoint file count exceeds its bound")
        elif not path.is_dir():
            raise ValueError("checkpoint contains a nonregular member")
    return files


def _unique_json(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate checkpoint JSON key")
        value[key] = item
    return value


def _json(raw):
    def invalid(_):
        raise ValueError("nonfinite checkpoint JSON constant")

    try:
        value = json.loads(raw, object_pairs_hook=_unique_json, parse_constant=invalid)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("invalid checkpoint JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("checkpoint JSON must be an object")
    return value


def _manifest(root, expected_digest):
    raw = read_verified_bytes(
        root / "SHA256SUMS",
        expected_sha256=expected_digest,
        max_bytes=MAX_MANIFEST_BYTES,
    )
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise ValueError("manifest must be UTF-8") from exc
    if not 1 <= len(lines) <= MAX_FILES:
        raise ValueError("checkpoint manifest entry count exceeds its bound")
    hashes, folded = {}, set()
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise ValueError("invalid checkpoint manifest entry")
        digest, name = match.groups()
        if name == "SHA256SUMS" or name.casefold() in folded:
            raise ValueError("duplicate or self-referential checkpoint member")
        path = safe_relative_path(root, name)
        if not path.is_file() or not hmac.compare_digest(sha256_file(path), digest):
            raise ValueError("checkpoint member SHA-256 mismatch: " + name)
        folded.add(name.casefold())
        hashes[name] = digest
    if _files(root) != set(hashes) | {"SHA256SUMS"}:
        raise ValueError("unlisted checkpoint files")
    return hashes


def _bounded_zip(raw, maximum, *, members, allow_empty=False):
    try:
        with zipfile.ZipFile(BytesIO(raw)) as archive:
            infos = archive.infolist()
            if (
                not (0 if allow_empty else 1) <= len(infos) <= members
                or sum(i.file_size for i in infos) > maximum
                or any(
                    i.flag_bits & 1 or stat.S_ISLNK(i.external_attr >> 16)
                    for i in infos
                )
                or len({i.filename.casefold() for i in infos}) != len(infos)
            ):
                raise ValueError("serialized tensor/array archive exceeds its bounds")
    except zipfile.BadZipFile as exc:
        raise ValueError("invalid serialized tensor/array archive") from exc


def _observations(raw):
    _bounded_zip(raw, MAX_OBSERVATION_BYTES + 128 * 1024, members=256, allow_empty=True)
    try:
        with np.load(BytesIO(raw), allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    except (ValueError, TypeError, OSError, EOFError) as exc:
        raise ValueError("invalid numeric observation archive") from exc
    _check_observations(arrays)
    return arrays


def _check_observations(arrays):
    if (
        not isinstance(arrays, dict)
        or len(arrays) > 256
        or any(not isinstance(k, str) or not k or len(k) > 128 for k in arrays)
    ):
        raise ValueError("invalid observation array catalog")
    total = 0
    for array in arrays.values():
        if (
            not isinstance(array, np.ndarray)
            or array.dtype.hasobject
            or array.dtype.kind not in "biufc"
            or not np.isfinite(array).all()
        ):
            raise ValueError("observations must contain finite numeric arrays only")
        total += array.nbytes
    if total > MAX_OBSERVATION_BYTES:
        raise ValueError("observation arrays exceed their byte bound")


def _check_tensors(value):
    """Reject arbitrary objects before writing an otherwise unloadable pickle."""
    total = 0

    def visit(item):
        nonlocal total
        if isinstance(item, torch.Tensor):
            if (
                item.layout != torch.strided
                or item.device.type != "cpu"
                or not bool(torch.isfinite(item).all())
            ):
                raise ValueError("snapshot tensors must be finite dense CPU tensors")
            total += item.numel() * item.element_size()
        elif isinstance(item, dict):
            if any(type(k) not in (str, int) for k in item):
                raise ValueError(
                    "tensor-state dictionary keys must be strings or integers"
                )
            for x in item.values():
                visit(x)
        elif isinstance(item, (tuple, list)):
            for x in item:
                visit(x)
        elif item is None or type(item) in (str, bool, int):
            return
        elif type(item) is float and np.isfinite(item):
            return
        else:
            raise ValueError("unsupported object in tensor snapshot")
        if total > MAX_STATE_BYTES:
            raise ValueError("snapshot tensors exceed their byte bound")

    visit(value)


def _replay_catalog(root, slot, hashes):
    prefix = f"replay/{slot}/"
    metadata_name = prefix + "metadata.json"
    raw = read_verified_bytes(
        root / metadata_name,
        expected_sha256=hashes[metadata_name],
        max_bytes=16 * 1024**2,
    )
    meta = _json(raw)
    limit = REPLAY_LIMITS[slot]
    for key in ("capacity_total", "capacity_per_game"):
        if type(meta.get(key)) is not int or not 1 <= meta[key] <= limit:
            raise ValueError("replay configured capacity exceeds snapshot bound")
    records = meta.get("episodes")
    if not isinstance(records, list) or len(records) > limit + 128:
        raise ValueError("replay episode catalog exceeds snapshot bound")
    names, count = {metadata_name}, 0
    for record in records:
        if (
            not isinstance(record, dict)
            or type(record.get("transitions")) is not int
            or record["transitions"] < 0
            or record["transitions"] > limit
            or not isinstance(record.get("file"), str)
            or not re.fullmatch(r"episode-[0-9]{12}\.npz", record["file"])
        ):
            raise ValueError("invalid replay snapshot catalog")
        count += record["transitions"]
        name = prefix + record["file"]
        if name in names:
            raise ValueError("duplicate replay snapshot catalog member")
        names.add(name)
    if count > meta["capacity_total"] or names != {
        n for n in hashes if n.startswith(prefix)
    }:
        raise ValueError("replay catalog does not exactly match authenticated files")


def save_checkpoint(
    trainer, destination, *, source_revision: str, protocol_sha256: str
):
    source_revision, protocol_sha256 = _source(source_revision), validate_sha256(
        protocol_sha256
    )
    root = _plain_path(destination, directory=True)
    if root.exists():
        raise FileExistsError(root)
    metadata, tensors, observations = trainer.snapshot_payload()
    metadata = copy.deepcopy(metadata)
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION
    ):
        raise ValueError("v2 trainer snapshot schema required")
    _check_observations(observations)
    _check_tensors(tensors)
    replays = {
        "fit": trainer.replay,
        "calibration": trainer.calibration_replay,
        "source": trainer.source_replay,
    }
    for slot, replay in replays.items():
        if replay is not None and (
            not isinstance(replay, SequenceReplay)
            or replay.capacity_total > REPLAY_LIMITS[slot]
            or replay.capacity_per_game > REPLAY_LIMITS[slot]
        ):
            raise ValueError("invalid or oversized replay for snapshot")
    if any(replays[k] is None for k in ("fit", "calibration")):
        raise ValueError("fit and calibration replay are required")
    architecture = _architecture_version()
    if trainer.online.architecture_version != architecture:
        raise ValueError("v2 network architecture required")
    metadata.update(
        schema_version=SCHEMA_VERSION,
        source_revision=source_revision,
        protocol_sha256=protocol_sha256,
        action_names=list(ACTION_NAMES),
        observation_shape=[32, 32, 3],
        architecture=architecture,
        versions=_runtime(),
        has_source_replay=replays["source"] is not None,
    )
    encoded = (
        json.dumps(metadata, allow_nan=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    if len(encoded) > MAX_METADATA_BYTES:
        raise ValueError("snapshot metadata exceeds its byte bound")
    root.mkdir(parents=True, exist_ok=False)
    (root / "metadata.json").write_bytes(encoded)
    torch.save(tensors, root / "state.pt")
    np.savez_compressed(root / "observations.npz", **observations)
    if (root / "state.pt").stat().st_size > MAX_STATE_BYTES or (
        root / "observations.npz"
    ).stat().st_size > MAX_OBSERVATION_BYTES:
        raise ValueError("serialized tensor/array archive exceeds its byte bound")
    _bounded_zip((root / "state.pt").read_bytes(), MAX_STATE_BYTES, members=100_000)
    _observations((root / "observations.npz").read_bytes())
    for slot, replay in replays.items():
        if replay is not None:
            replay.save(root / "replay" / slot)
    files = sorted(_files(root))
    manifest = "".join(f"{sha256_file(root/n)}  {n}\n" for n in files).encode("utf-8")
    if len(files) > MAX_FILES or len(manifest) > MAX_MANIFEST_BYTES:
        raise ValueError("snapshot manifest exceeds its bound")
    (root / "SHA256SUMS").write_bytes(manifest)
    digest = hashlib.sha256(manifest).hexdigest()
    _manifest(root, digest)
    return {
        "path": str(root),
        "manifest_sha256": digest,
        "files_verified": len(files),
        "source_revision": source_revision,
        "protocol_sha256": protocol_sha256,
        "schema_version": SCHEMA_VERSION,
        "has_source_replay": replays["source"] is not None,
    }


def load_checkpoint(
    directory,
    *,
    expected_manifest_sha256: str,
    expected_source_revision: str,
    expected_protocol_sha256: str,
):
    source, protocol = _source(expected_source_revision), validate_sha256(
        expected_protocol_sha256
    )
    root = _plain_path(directory, directory=True)
    hashes = _manifest(root, expected_manifest_sha256)
    required = {"metadata.json", "state.pt", "observations.npz"}
    if not required <= set(hashes):
        raise ValueError("required snapshot members missing")
    metadata = _json(
        read_verified_bytes(
            root / "metadata.json",
            expected_sha256=hashes["metadata.json"],
            max_bytes=MAX_METADATA_BYTES,
        )
    )
    if (
        type(metadata.get("schema_version")) is not int
        or metadata["schema_version"] != SCHEMA_VERSION
        or metadata.get("source_revision") != source
        or metadata.get("protocol_sha256") != protocol
        or metadata.get("action_names") != list(ACTION_NAMES)
        or metadata.get("observation_shape") != [32, 32, 3]
        or metadata.get("architecture") != _architecture_version()
        or metadata.get("versions") != _runtime()
        or type(metadata.get("has_source_replay")) is not bool
    ):
        raise ValueError("checkpoint code/protocol/contract/runtime mismatch")
    slots = (
        ("fit", "calibration", "source")
        if metadata["has_source_replay"]
        else ("fit", "calibration")
    )
    if any(
        n not in required and not any(n.startswith(f"replay/{s}/") for s in slots)
        for n in hashes
    ):
        raise ValueError("unknown checkpoint member")
    for slot in slots:
        if f"replay/{slot}/metadata.json" not in hashes:
            raise ValueError("required replay snapshot missing")
        _replay_catalog(root, slot, hashes)
    obs_raw = read_verified_bytes(
        root / "observations.npz",
        expected_sha256=hashes["observations.npz"],
        max_bytes=MAX_OBSERVATION_BYTES,
    )
    observations = _observations(obs_raw)
    state_raw = read_verified_bytes(
        root / "state.pt", expected_sha256=hashes["state.pt"], max_bytes=MAX_STATE_BYTES
    )
    _bounded_zip(state_raw, MAX_STATE_BYTES, members=100_000)
    tensors = torch.load(BytesIO(state_raw), map_location="cpu", weights_only=True)
    _check_tensors(tensors)
    replays = {slot: SequenceReplay.load(root / "replay" / slot) for slot in slots}
    replays.setdefault("source", None)
    restored = _restore_trainer(metadata, tensors, observations, replays)
    # Loading is read-only; also catch a changed replay file during consumption.
    _manifest(root, expected_manifest_sha256)
    return restored
