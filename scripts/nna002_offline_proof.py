"""Recover existing checkpoints and recorded decisions; no scientific data collection."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform
import stat
import subprocess
import sys
import time
import traceback
import zipfile

SOURCE = "dc71247b2269d41c8d2aeb8c7d489977fac3ad3a"
PROTOCOL = "93591ede5951438f92432fc6601773aaafe28b2ba5840309aa7bd3611170a324"
ARTIFACT_SHA = "f396acfcbc8de35e67cbe79be4b6b986d7b536019bca2a62a3effa60ccefa129"
ROOT_SHA = "2797b5280d367be9e067d8528f2b0c8973081342eb248aacca19054475b18840"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def equal(a, b, torch):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k], torch) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(equal(x, y, torch) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def run(args):
    started, cpu_started = time.monotonic(), time.process_time()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = {"status": "STOPPED", "source_revision": SOURCE, "protocol_sha256": PROTOCOL,
              "original_scientific_run": 41551291622, "original_artifact": 13195542205,
              "original_artifact_sha256": ARTIFACT_SHA, "new_environment_interactions": 0,
              "new_training_updates": 0, "new_evaluation_episodes": 0,
              "scientific_conclusion": "NOT_ASSESSED_OFFLINE_RECOVERY_ONLY",
              "script_sha256": sha(__file__), "restored_models": [], "failure": None}
    try:
        source_root = args.source_root.resolve()
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip()
        if head != SOURCE or platform.python_version() != "3.11.17" or platform.system() != "Linux":
            raise ValueError("recovery requires the original source A and Python3.11.17/Linux")
        sys.path.insert(0, str(source_root))
        from arena.shared.integrity import safe_relative_path, verify_checksum_manifest, read_verified_bytes
        from arena.universal.protocol import protocol_sha256
        if protocol_sha256() != PROTOCOL or sha(args.archive) != ARTIFACT_SHA:
            raise ValueError("source protocol or original artifact digest mismatch")
        extracted = output.parent/"recorded-original"
        extracted.mkdir(exist_ok=False)
        with zipfile.ZipFile(args.archive) as archive:
            infos = archive.infolist()
            if len(infos) > 100_000 or sum(x.file_size for x in infos) > 4*1024**3:
                raise ValueError("artifact exceeds finite extraction limits")
            seen = set()
            for info in infos:
                name = info.filename.rstrip("/")
                if name.casefold() in seen or stat.S_ISLNK(info.external_attr >> 16) or info.flag_bits & 1:
                    raise ValueError("duplicate, linked or encrypted artifact member")
                seen.add(name.casefold())
                path = safe_relative_path(extracted, name)
                if info.is_dir():
                    path.mkdir(parents=True, exist_ok=True)
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                length = 0
                with archive.open(info) as src, path.open("xb") as dst:
                    while block := src.read(1024**2):
                        length += len(block)
                        if length > info.file_size:
                            raise ValueError("artifact member exceeded declared size")
                        dst.write(block)
                if length != info.file_size:
                    raise ValueError("incomplete artifact member")
        roots = list(extracted.rglob("TRAIN_RESERVED.json"))
        if len(roots) != 1:
            raise ValueError("scientific root is not unique")
        root = roots[0].parent
        if sha(root/"SHA256SUMS") != ROOT_SHA:
            raise ValueError("scientific root manifest mismatch")
        files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
        hashes = verify_checksum_manifest(root, expected_files=files-{"SHA256SUMS"})
        import numpy as np
        import torch
        import gymnasium
        from arena.universal.envs import PixelGame
        from arena.universal.trainer import UniversalTrainer
        from arena.universal.persistence import load_checkpoint
        from arena.universal.causal_trace import export_causal_trace, network_sha256
        from arena.universal.campaign import _rss_bytes
        from arena.universal.envs import _json_value

        def forbidden(*unused, **kwargs):
            raise RuntimeError("offline proof forbids simulator steps, collection and gradient updates")
        PixelGame.step = forbidden
        UniversalTrainer.collect = forbidden
        UniversalTrainer.learn_batch = forbidden
        torch.optim.Adam.step = forbidden
        torch.Tensor.backward = forbidden
        torch.autograd.backward = forbidden
        status["runtime"] = {"python": platform.python_version(), "system": platform.system(),
                             "machine": platform.machine(), "torch": torch.__version__,
                             "numpy": np.__version__, "gymnasium": gymnasium.__version__}
        index = json.loads((root/"model_index.json").read_text())
        expected = {f"{seed}:{stage}" for seed in (1103,2207,3301)
                    for stage in ("before_adaptation", "after_adaptation", "from_scratch")}
        if set(index) != expected:
            raise ValueError("exactly the nine fixed checkpoint slots are required")
        with (root/"evaluation/episodes.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for key, entry in sorted(index.items()):
            checkpoint = safe_relative_path(root, entry["path"])
            trainer = load_checkpoint(checkpoint, expected_manifest_sha256=entry["manifest_sha256"],
                                      expected_source_revision=SOURCE, expected_protocol_sha256=PROTOCOL)
            original_metadata = json.loads((checkpoint/"metadata.json").read_text())
            metadata, tensors, observations = trainer.snapshot_payload()
            if any(metadata[k] != original_metadata[k] for k in metadata):
                raise ValueError("recovered counters, actors, environment or RNG metadata differ")
            checkpoint_hashes = verify_checksum_manifest(checkpoint)
            original_tensors = torch.load(BytesIO(read_verified_bytes(checkpoint/"state.pt",
                                          expected_sha256=checkpoint_hashes["state.pt"],
                                          max_bytes=64*1024**2)),
                                          map_location="cpu", weights_only=True)
            if not equal(original_tensors, tensors, torch):
                raise ValueError("recovered online/target/optimizer/hidden/Torch RNG tensors differ")
            with np.load(checkpoint/"observations.npz", allow_pickle=False) as originals:
                if set(originals.files) != set(observations) or any(not np.array_equal(originals[g], observations[g]) for g in originals.files):
                    raise ValueError("recovered observations differ")
            replay = json.loads((checkpoint/"replay/metadata.json").read_text())
            if (_json_value(trainer.replay._rng.getstate()) != replay["rng_state"] or
                    len(trainer.replay) != sum(e["transitions"] for e in replay["episodes"])):
                raise ValueError("recovered replay sampling state or counts differ")
            fingerprint = network_sha256(trainer.online)
            expected_fingerprints = {r["network_sha256"] for r in rows
                                     if int(r["model_seed"]) == entry["model_seed"] and r["stage"] == entry["stage"]}
            if expected_fingerprints != {fingerprint}:
                raise ValueError("recovered online fingerprint differs from recorded evaluation")
            status["restored_models"].append({"slot": key, "manifest_sha256": entry["manifest_sha256"],
                 "global_steps": trainer.global_steps, "q_updates": trainer.q_updates,
                 "world_updates": trainer.world_updates, "replay_transitions": len(trainer.replay),
                 "replay_games": trainer.replay.game_counts(), "network_sha256": fingerprint,
                 "all_metadata_and_tensors_exact": True})
            for actor in trainer.actors.values():
                actor["env"].close()
            del trainer, original_tensors, tensors, observations
            gc.collect()
            if _rss_bytes() > 3*1024**3 or time.process_time()-cpu_started > 600:
                raise RuntimeError("finite offline proof resource limit reached")
        write(output/"CHECKPOINT_RECOVERY.json", {"status": "RESTORED_EXACTLY_WITHOUT_ADVANCING",
              "models": status["restored_models"], "new_environment_interactions": 0,
              "new_training_updates": 0, "future_25_updates_reexecuted": False,
              "limitation": "future25 reproduction remains the prior synthetic CI proof, not a new scientific rollout"})
        selected = index["2207:after_adaptation"]
        status["causal_trace"] = export_causal_trace(root/"evaluation", root/selected["path"], output/"causal",
             expected_evaluation_manifest_sha256=hashes["evaluation/SHA256SUMS"],
             expected_checkpoint_manifest_sha256=selected["manifest_sha256"],
             source_revision=SOURCE, protocol_sha256=PROTOCOL)
        if status["causal_trace"]["status"] != "VERIFIED":
            raise ValueError("causal reconstruction did not meet frozen tolerance")
        verify_checksum_manifest(root, expected_files=files-{"SHA256SUMS"})
        if _rss_bytes() > 3*1024**3 or time.process_time()-cpu_started > 600:
            raise RuntimeError("finite offline proof resource limit reached")
        status["peak_rss_bytes"] = _rss_bytes()
        status["status"] = "VERIFIED"
    except BaseException as exc:
        status["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
    finally:
        status["elapsed_seconds"] = time.monotonic()-started
        status["cpu_seconds"] = time.process_time()-cpu_started
        write(output/"OFFLINE_PROOF_STATUS.json", status)
        (output/"SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(output).as_posix()}\n"
                                        for p in sorted(output.rglob("*")) if p.is_file()), encoding="utf-8")
    print(json.dumps({"status": status["status"], "restored_models": len(status["restored_models"]),
                      "failure": status["failure"], "new_environment_interactions": 0, "new_training_updates": 0}))
    return 0 if status["status"] == "VERIFIED" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
