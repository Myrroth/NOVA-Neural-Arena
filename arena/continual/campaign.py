"""Candidate finite continual-v2 orchestration; default is a no-ML plan.

Importing this module runs no game, learning, workflow or network operation.
Permit references are supplied by an authorized operator; this code validates
their schema/provenance binding, not the truth of an external approval message.
The workflow still needs its independently audited durable reservation gate.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import urlparse

from arena.shared.integrity import (
    read_verified_bytes,
    safe_relative_path,
    sha256_file,
    verify_checksum_manifest,
)
from . import protocol as p


class CampaignStopped(RuntimeError):
    """Finite execution stopped; its directory is intentionally not reusable."""


def _write_json(path: Path, value, *, exclusive=False):
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    else:
        temporary = path.with_name(path.name + ".partial")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)


def _manifest(root: Path):
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise CampaignStopped("artifact paths must not contain symlinks")
        if path.is_file() and path != root / "SHA256SUMS":
            entries.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")
    verify_checksum_manifest(root)


def _source_state():
    root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    )
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True
    )
    return {"root": str(root), "head": head, "clean": not dirty.strip()}


def validate_permit(permit: dict, *, source_state=None, _test_only=False):
    """Fail before directories, trainer construction or any environment action."""
    if not isinstance(permit, dict) or permit.get("approved") is not True:
        raise CampaignStopped("an explicit approved permit is required")
    if permit.get("authorization_type") != "new-explicit-continual-003":
        raise CampaignStopped(
            "consumed001/002 authorization cannot be reused for this new lineage"
        )
    if not _test_only:
        config = p.protocol_config()
        if (
            config.get("status") != "FROZEN_APPROVAL_REQUIRED"
            or config.get("unresolved_before_freeze") != []
            or config.get("execution_authorized") is not False
        ):
            raise CampaignStopped(
                "candidate/unresolved protocol is not frozen; permit JSON alone cannot authorize it"
            )
        raise CampaignStopped(
            "production once-only remote reservation transport is not yet certified"
        )
    revision, digest = permit.get("source_revision", ""), permit.get(
        "protocol_sha256", ""
    )
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise CampaignStopped(
            "permit source_revision must be a 40-character commit SHA"
        )
    if digest != p.protocol_sha256():
        raise CampaignStopped("permit does not match the current canonical protocol")
    for field_ in ("human_approval_reference", "reservation_id"):
        value = permit.get(field_)
        if not isinstance(value, str) or not value.strip():
            raise CampaignStopped(f"permit missing {field_}")
    parsed = urlparse(str(permit.get("planning_approval_reference", "")))
    if parsed.scheme != "https" or not parsed.netloc or not parsed.path:
        raise CampaignStopped("planning approval must be an explicit HTTPS source URL")
    state = _source_state() if source_state is None else source_state
    if state.get("head") != revision or state.get("clean") is not True:
        raise CampaignStopped(
            "real source HEAD must match the permit and working tree must be clean"
        )
    return state


def _rss_bytes():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                *[
                    (name, ctypes.c_size_t)
                    for name in (
                        "PeakWorkingSetSize",
                        "WorkingSetSize",
                        "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage",
                        "QuotaPeakNonPagedPoolUsage",
                        "QuotaNonPagedPoolUsage",
                        "PagefileUsage",
                        "PeakPagefileUsage",
                    )
                ],
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        ctypes.windll.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        ):
            raise CampaignStopped("cannot measure process memory")
        return int(counters.PeakWorkingSetSize)
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


@dataclass
class Budget:
    total_steps: int = 0
    training_steps: int = 0
    evaluation_steps: int = 0
    q_updates: int = 0
    world_updates: int = 0
    prior_elapsed_seconds: float = 0.0
    prior_cpu_seconds: float = 0.0
    max_steps: int = p.MAX_TOTAL_STEPS
    clock: object = field(default=time.monotonic, repr=False)
    cpu_clock: object = field(default=time.process_time, repr=False)
    rss_reader: object = field(default=_rss_bytes, repr=False)
    started: float = field(init=False)
    cpu_started: float = field(init=False)
    peak_rss_bytes: int = 0

    def __post_init__(self):
        self.started = self.clock()
        self.cpu_started = self.cpu_clock()

    def elapsed(self):
        return self.prior_elapsed_seconds + self.clock() - self.started

    def cpu_elapsed(self):
        return self.prior_cpu_seconds + self.cpu_clock() - self.cpu_started

    def check(self):
        self.peak_rss_bytes = max(self.peak_rss_bytes, int(self.rss_reader()))
        if self.elapsed() >= 285 * 60:
            raise CampaignStopped(
                "285-minute internal STOP; preserve finalization margin"
            )
        if self.cpu_elapsed() >= 285 * 60:
            raise CampaignStopped(
                "285-minute process CPU STOP; preserve finalization margin"
            )
        if self.peak_rss_bytes > 3 * 1024**3:
            raise CampaignStopped("3 GiB process memory limit exceeded")
        if (
            self.total_steps > self.max_steps
            or self.training_steps > p.MAX_TRAINING_STEPS
            or self.evaluation_steps > p.MAX_EVALUATION_STEPS
            or self.q_updates > p.MAX_Q_UPDATES
            or self.world_updates > p.MAX_WORLD_UPDATES
            or self.q_updates + self.world_updates > 101_250
        ):
            raise CampaignStopped("absolute campaign interaction/update limit exceeded")

    def account(
        self, *, training_steps=0, evaluation_steps=0, q_updates=0, world_updates=0
    ):
        counts = (training_steps, evaluation_steps, q_updates, world_updates)
        if any(type(value) is not int or value < 0 for value in counts):
            raise CampaignStopped("counter regression or noninteger budget accounting")
        self.training_steps += training_steps
        self.evaluation_steps += evaluation_steps
        self.total_steps += training_steps + evaluation_steps
        self.q_updates += q_updates
        self.world_updates += world_updates
        self.check()

    def report(self):
        return {
            key: getattr(self, key)
            for key in (
                "total_steps",
                "training_steps",
                "evaluation_steps",
                "q_updates",
                "world_updates",
                "peak_rss_bytes",
            )
        } | {
            "elapsed_seconds": self.elapsed(),
            "cpu_elapsed_seconds": self.cpu_elapsed(),
        }


@dataclass(frozen=True)
class _Plan:
    """Private test seam; real entrypoints always use the canonical dimensions."""

    model_seeds: tuple = p.MODEL_SEEDS
    source_steps_per_game: int = p.STEPS_PER_GAME
    target_steps: int = p.STEPS_PER_GAME
    max_episode_steps: int = p.MAX_EPISODE_STEPS
    max_total_steps: int = p.MAX_TOTAL_STEPS
    test_only: bool = False


class _Journal:
    def __init__(self, root: Path, name: str):
        self.root, self.name = root, name
        self.path = root / f"{name}.jsonl"
        self.handle = self.path.open("x", encoding="utf-8", newline="\n")
        self.fields = set()

    def append(self, rows):
        for row in rows:
            self.fields.update(row)
            self.handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def flush_csv(self):
        self.handle.flush()
        temporary = self.root / f"{self.name}.csv.partial"
        with temporary.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=sorted(self.fields))
            writer.writeheader()
            with self.path.open(encoding="utf-8") as source:
                for line in source:
                    row = json.loads(line)
                    writer.writerow(
                        {
                            key: (
                                json.dumps(value, sort_keys=True, separators=(",", ":"))
                                if isinstance(value, (dict, list))
                                else value
                            )
                            for key, value in row.items()
                        }
                    )
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, self.root / f"{self.name}.csv")

    def close(self):
        try:
            self.flush_csv()
        finally:
            self.handle.close()


class _RealBackend:
    def __init__(self):
        import torch

        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)

    def trainer(self, seed):
        from .trainer import ContinualConfig, ContinualTrainer

        return ContinualTrainer(
            seed, ContinualConfig(**p.protocol_config()["learner_config"])
        )

    def save(self, trainer, path, permit):
        from .persistence import save_checkpoint

        return save_checkpoint(
            trainer,
            path,
            source_revision=permit["source_revision"],
            protocol_sha256=permit["protocol_sha256"],
        )

    def load_policy(self, root, entry, permit):
        from .persistence import load_frozen_controller

        expected_stage = {
            "before_adaptation": "pretrain",
            "after_adaptation": "adaptation",
            "from_scratch": "from_scratch",
        }[entry["stage"]]
        return load_frozen_controller(
            safe_relative_path(root, entry["path"]),
            expected_manifest_sha256=entry["manifest_sha256"],
            expected_source_revision=permit["source_revision"],
            expected_protocol_sha256=permit["protocol_sha256"],
            expected_model_seed=entry["model_seed"],
            expected_stage=expected_stage,
        )

    def untrained(self, seed):
        import torch
        from .calibration import GateCalibration
        from .network import ContinualQNetwork
        from .policy import FrozenPolicyController
        from .trainer import ContinualConfig

        config = ContinualConfig(**p.protocol_config()["learner_config"])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            network = (
                ContinualQNetwork(
                    hidden_size=config.hidden_size, latent_size=config.latent_size
                )
                .eval()
                .requires_grad_(False)
            )
        return FrozenPolicyController(
            network,
            calibration=GateCalibration(),
            policy_revision=0,
            context_length=config.context_length,
            gamma=config.gamma,
        )

    def evaluate(self, network, **kwargs):
        from .evaluation import evaluate

        return evaluate(network, **kwargs)


def _setup(permit, backend, plan, source_state):
    test_only = backend is not None
    if test_only:
        if plan is None or not plan.test_only or source_state is None:
            raise CampaignStopped(
                "fake backends require an explicitly test-only plan/source state"
            )
    elif plan is not None or source_state is not None:
        raise CampaignStopped(
            "real execution cannot override canonical plan or Git state"
        )
    source = validate_permit(permit, source_state=source_state, _test_only=test_only)
    return backend or _RealBackend(), plan or _Plan(), source, test_only


def _close_trainer(trainer):
    for actor in getattr(trainer, "actors", {}).values():
        actor["env"].close()


def _claim(root, permit, source, test_only):
    if root.exists():
        raise CampaignStopped(
            "output already exists; overwrite/resume/rerun is prohibited"
        )
    reservations = (
        root.parent / ".test-nna003-reservations"
        if test_only
        else Path(source["root"]) / "experiments" / "continual-003-reservations"
    )
    reservations.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(permit["reservation_id"].encode()).hexdigest()
    try:
        _write_json(
            reservations / f"{identity}.json",
            {"permit": permit, "output": str(root.resolve()), "test_only": test_only},
            exclusive=True,
        )
    except FileExistsError as exc:
        raise CampaignStopped(
            "reservation already consumed; no retry or alternate output"
        ) from exc
    root.mkdir(parents=True, exist_ok=False)
    _write_json(
        root / "TRAIN_RESERVED.json",
        {"permit": permit, "test_only": test_only},
        exclusive=True,
    )


def _run_stage(trainer, stage, steps, budget, curves, losses):
    trainer.configure_stage(stage, steps)
    if stage == "pretrain" and tuple(trainer.games) != p.SOURCE_GAMES:
        raise CampaignStopped(
            "source pretraining must contain four games and exclude Racing"
        )
    if stage != "pretrain" and tuple(trainer.games) != (p.TARGET_GAME,):
        raise CampaignStopped("target collection must contain Racing only")
    curve_cursor, loss_cursor = len(trainer.training_curve), len(trainer.loss_history)
    initial = (trainer.global_steps, trainer.q_updates, trainer.world_updates)
    while trainer.stage_steps < steps:
        budget.check()
        count = min(
            128, steps - trainer.stage_steps, budget.max_steps - budget.total_steps
        )
        if count <= 0:
            raise CampaignStopped("no remaining interaction budget")
        before = (trainer.global_steps, trainer.q_updates, trainer.world_updates)
        collection_error = None
        try:
            trainer.collect(count, learn=True)
        except BaseException as exc:
            collection_error = exc
        try:
            # Account first: journal IO can fail after physical interactions.
            budget.account(
                training_steps=int(trainer.global_steps - before[0]),
                q_updates=int(trainer.q_updates - before[1]),
                world_updates=int(trainer.world_updates - before[2]),
            )
        except BaseException as exc:
            if collection_error is None:
                collection_error = exc
            else:
                collection_error.add_note(f"accounting failure: {exc}")
        try:
            curves.append(trainer.training_curve[curve_cursor:])
            losses.append(
                [
                    {"model_seed": trainer.model_seed, "stage": stage, **row}
                    for row in trainer.loss_history[loss_cursor:]
                ]
            )
            curve_cursor, loss_cursor = len(trainer.training_curve), len(
                trainer.loss_history
            )
        except BaseException as exc:
            if collection_error is None:
                collection_error = exc
            else:
                collection_error.add_note(f"journal failure: {exc}")
        if collection_error is not None:
            raise collection_error
        if trainer.global_steps - before[0] != count:
            raise CampaignStopped("trainer did not consume the exact bounded chunk")
    if stage == "pretrain":
        stats = trainer.training_baselines()
        if "racing" in stats or set(stats) != set(p.SOURCE_GAMES):
            raise CampaignStopped("Racing leaked into source training statistics")
        if any(stats[game]["support"] != steps // 4 for game in p.SOURCE_GAMES):
            raise CampaignStopped("source game exposure is not exactly balanced")
    return (
        trainer.global_steps - initial[0],
        trainer.q_updates - initial[1],
        trainer.world_updates - initial[2],
    )


def _update_audit(trainer, steps, deltas):
    config = p.protocol_config()["learner_config"]
    scheduled = {
        kind: max(
            0,
            steps // config[interval]
            - (config["warmup_steps"] - 1) // config[interval],
        )
        for kind, interval in (("q", "q_interval"), ("world", "world_interval"))
    }
    return {
        "scheduled_opportunities": scheduled,
        "attempted": dict(trainer.update_attempts),
        "warmup_or_replay_skipped_opportunities": {
            kind: scheduled[kind] - trainer.update_attempts[kind]
            for kind in ("q", "world")
        },
        "skipped_insufficient_context": dict(trainer.skipped_updates),
        "successful": {"q": deltas[1], "world": deltas[2]},
        "compute_matched": False,
    }


def _finalize_artifacts(root, status_name, status, operations, *, extra_roots=()):
    """Try independent cleanup actions; journal failures cannot imply success."""

    def record(phase, exc):
        error = {"phase": phase, "type": type(exc).__name__, "message": str(exc)}
        if status["failure"] is None:
            status["failure"] = error
        else:
            status["failure"].setdefault("finalization_errors", []).append(error)
        status["status"] = "STOPPED"

    for name, operation in operations:
        try:
            operation()
        except BaseException as exc:
            record(name, exc)
    _write_json(root / status_name, status)
    for target in (root, *extra_roots):
        try:
            _manifest(target)
        except BaseException as exc:
            record("manifest", exc)
            # Preserve a truthful receipt even if sealing the artifact fails.
            _write_json(root / status_name, status)


def train_once(
    output,
    permit,
    *,
    _backend=None,
    _plan=None,
    _source=None,
    _clock=time.monotonic,
    _rss_reader=_rss_bytes,
):
    backend, plan, source, test_only = _setup(permit, _backend, _plan, _source)
    root = Path(output)
    _claim(root, permit, source, test_only)
    budget = Budget(
        max_steps=plan.max_total_steps, clock=_clock, rss_reader=_rss_reader
    )
    curves, losses = _Journal(root, "training_curve"), _Journal(root, "training_losses")
    index, trainer, failure = {}, None, None
    _write_json(root / "protocol.json", p.protocol_config(), exclusive=True)
    try:
        for seed in plan.model_seeds:
            trainer = backend.trainer(seed)
            for stage, label, steps in (
                ("pretrain", "before_adaptation", 4 * plan.source_steps_per_game),
                ("adaptation", "after_adaptation", plan.target_steps),
            ):
                deltas = _run_stage(trainer, stage, steps, budget, curves, losses)
                destination = root / "models" / str(seed) / label
                receipt = backend.save(trainer, destination, permit)
                index[f"{seed}:{label}"] = {
                    "path": destination.relative_to(root).as_posix(),
                    "manifest_sha256": receipt["manifest_sha256"],
                    "model_seed": seed,
                    "stage": label,
                    "training_baselines": trainer.training_baselines(),
                    "stage_counters": list(deltas),
                    "update_audit": _update_audit(trainer, steps, deltas),
                }
                budget.check()
            adaptation_counts = index[f"{seed}:after_adaptation"]["stage_counters"]
            _close_trainer(trainer)
            trainer = backend.trainer(seed)
            scratch_counts = _run_stage(
                trainer, "from_scratch", plan.target_steps, budget, curves, losses
            )
            # Both arms have identical physical interactions and scheduled
            # opportunities. Reserved calibration and insufficient context can
            # change realized updates; expose them, never silently match/redo.
            if scratch_counts[0] != adaptation_counts[0]:
                raise CampaignStopped(
                    "adaptation/scratch target interaction budgets differ"
                )
            destination = root / "models" / str(seed) / "from_scratch"
            receipt = backend.save(trainer, destination, permit)
            index[f"{seed}:from_scratch"] = {
                "path": destination.relative_to(root).as_posix(),
                "manifest_sha256": receipt["manifest_sha256"],
                "model_seed": seed,
                "stage": "from_scratch",
                "training_baselines": trainer.training_baselines(),
                "stage_counters": list(scratch_counts),
                "update_audit": _update_audit(
                    trainer, plan.target_steps, scratch_counts
                ),
                "paired_adaptation_counters": adaptation_counts,
                "comparison_scope": "matched physical steps and scheduled opportunities; realized updates disclosed",
            }
            _close_trainer(trainer)
            trainer = None
        expected = len(plan.model_seeds) * (
            4 * plan.source_steps_per_game + 2 * plan.target_steps
        )
        if budget.training_steps != expected or len(index) != 3 * len(plan.model_seeds):
            raise CampaignStopped("training dimensions incomplete")
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        if trainer is not None:
            try:
                backend.save(trainer, root / "partial_checkpoint", permit)
            except Exception as snapshot_error:
                failure["partial_checkpoint_error"] = str(snapshot_error)
    finally:
        status = {
            "status": "STOPPED" if failure else "SUCCESS",
            "failure": failure,
            "test_only": test_only,
            "permit": permit,
            "budget": budget.report(),
            "model_count": len(index),
            "scientific_conclusion": "NOT_EVALUATED",
        }
        operations = [
            ("close_curves", curves.close),
            ("close_losses", losses.close),
            (
                "write_model_index",
                lambda: _write_json(root / "model_index.json", index, exclusive=True),
            ),
        ]
        if trainer is not None:
            operations.insert(0, ("close_trainer", lambda: _close_trainer(trainer)))
        _finalize_artifacts(root, "TRAIN_STATUS.json", status, operations)
        failure = status["failure"]
    if failure:
        raise CampaignStopped(
            f"training stopped; partial artifacts preserved: {failure['message']}"
        )
    return status


def _write_demonstration(root, cell, trajectory, episode_row):
    import numpy as np

    destination = root / "demonstrations" / cell.game
    destination.mkdir(parents=True, exist_ok=False)
    frames, next_frames, records = [], [], []
    for step in trajectory:
        row = dict(step)
        frames.append(row.pop("rgb"))
        next_frames.append(row.pop("next_rgb"))
        records.append(row)
    np.savez_compressed(
        destination / "frames.npz", rgb=np.stack(frames), next_rgb=np.stack(next_frames)
    )
    _write_json(
        destination / "trajectory.json",
        {
            "cell": asdict(cell),
            "steps": records,
            "network_sha256": episode_row["network_sha256"],
            "environment_id": episode_row["environment_id"],
            "note": "existing final evaluation episode; no extra rollout",
        },
        exclusive=True,
    )


def evaluate_final_once(
    directory,
    permit,
    *,
    _backend=None,
    _plan=None,
    _source=None,
    _clock=time.monotonic,
    _rss_reader=_rss_bytes,
    _cells=None,
):
    backend, plan, source, test_only = _setup(permit, _backend, _plan, _source)
    if _cells is not None and not test_only:
        raise CampaignStopped(
            "real final evaluation cannot override its canonical matrix"
        )
    root = Path(directory)
    train = json.loads((root / "TRAIN_STATUS.json").read_text(encoding="utf-8"))
    reservation = json.loads((root / "TRAIN_RESERVED.json").read_text(encoding="utf-8"))
    if (
        train.get("status") != "SUCCESS"
        or train.get("test_only") != test_only
        or train.get("permit") != permit
        or reservation != {"permit": permit, "test_only": test_only}
    ):
        raise CampaignStopped(
            "all training must finish under the identical permit before final evaluation"
        )
    if (root / "evaluation").exists():
        raise CampaignStopped("final evaluation already reserved; no resume/rerun")
    verify_checksum_manifest(root)
    output = root / "evaluation"
    output.mkdir(exist_ok=False)
    _write_json(
        output / "EVALUATION_RESERVED.json",
        {"permit": permit, "test_only": test_only},
        exclusive=True,
    )
    prior = train["budget"]
    budget = Budget(
        total_steps=prior["total_steps"],
        training_steps=prior["training_steps"],
        q_updates=prior["q_updates"],
        world_updates=prior["world_updates"],
        prior_elapsed_seconds=prior["elapsed_seconds"],
        max_steps=plan.max_total_steps,
        prior_cpu_seconds=prior["cpu_elapsed_seconds"],
        clock=_clock,
        rss_reader=_rss_reader,
    )
    journals = {
        name: _Journal(output, name)
        for name in ("episodes", "action_pairs", "model_predictions")
    }
    failure, completed = None, 0
    try:
        budget.check()
        index = json.loads((root / "model_index.json").read_text(encoding="utf-8"))
        expected_keys = {
            f"{seed}:{stage}"
            for seed in plan.model_seeds
            for stage in ("before_adaptation", "after_adaptation", "from_scratch")
        }
        if set(index) != expected_keys:
            raise CampaignStopped("exactly nine trained checkpoints are required")
        cache = {
            key: backend.load_policy(root, entry, permit)
            for key, entry in index.items()
        }
        for seed in plan.model_seeds:
            cache[f"{seed}:untrained"] = backend.untrained(seed)
        cells = tuple(p.evaluation_matrix() if _cells is None else _cells)
        for cell in cells:
            budget.check()
            if budget.total_steps + plan.max_episode_steps > plan.max_total_steps:
                raise CampaignStopped(
                    "insufficient remaining budget for the next bounded final episode"
                )
            if cell.policy == "random":
                network, baselines = None, None
            elif cell.policy == "untrained":
                network, baselines = cache[f"{cell.model_seed}:untrained"], None
            else:
                label = "after_adaptation" if cell.stage == "ablation" else cell.stage
                network = cache[f"{cell.model_seed}:{label}"]
                baselines = index[f"{cell.model_seed}:{label}"]["training_baselines"]
            demonstration = (
                not test_only
                and cell.stage == "after_adaptation"
                and cell.policy == "shared"
                and cell.model_seed == p.MODEL_SEEDS[1]
                and cell.environment_seed == p.TEST_SEEDS[0]
            )
            before_evaluation = budget.evaluation_steps
            row, pairs, predictions, trajectory = backend.evaluate(
                network,
                **asdict(cell),
                max_steps=plan.max_episode_steps,
                training_baselines=baselines,
                record_world_predictions=cell.stage == "after_adaptation"
                and cell.policy == "shared",
                record_trajectory=demonstration,
                include_frames=demonstration,
                before_step=budget.check,
                after_step=lambda: budget.account(evaluation_steps=1),
            )
            if any(row.get(name) != value for name, value in asdict(cell).items()):
                raise CampaignStopped(
                    "evaluator returned a different final evaluation cell"
                )
            if (
                type(row.get("steps")) is not int
                or not 1 <= row["steps"] <= plan.max_episode_steps
            ):
                raise CampaignStopped("invalid final evaluation step counter")
            shared_after = cell.stage == "after_adaptation" and cell.policy == "shared"
            if shared_after and (
                len(pairs) != row["steps"] or len(predictions) != row["steps"]
            ):
                raise CampaignStopped(
                    "every shared-after step needs its world prediction and same-state action pair"
                )
            if not shared_after and (pairs or predictions):
                raise CampaignStopped(
                    "unexpected attribution/prediction rows outside shared-after"
                )
            journals["episodes"].append([row])
            journals["action_pairs"].append(pairs)
            journals["model_predictions"].append(predictions)
            if budget.evaluation_steps - before_evaluation != row["steps"]:
                raise CampaignStopped(
                    "per-interaction evaluation accounting differs from episode row"
                )
            completed += 1
            if demonstration:
                _write_demonstration(output, cell, trajectory, row)
        if completed != len(cells):
            raise CampaignStopped("incomplete final evaluation matrix")
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        status = {
            "status": "STOPPED" if failure else "SUCCESS",
            "failure": failure,
            "test_only": test_only,
            "permit": permit,
            "budget": budget.report(),
            "completed_episodes": completed,
            "scientific_conclusion": "REQUIRES_INDEPENDENT_ANALYSIS",
            "inference_runtime_note": "inference compatibility is not an exact cross-platform resume claim",
        }
        _finalize_artifacts(
            output,
            "EVALUATION_STATUS.json",
            status,
            [("close_" + name, journal.close) for name, journal in journals.items()],
            extra_roots=(root,),
        )
        failure = status["failure"]
    if failure:
        raise CampaignStopped(
            f"final evaluation stopped; partial artifacts preserved: {failure['message']}"
        )
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--plan", action="store_true")
    choice.add_argument("--train-once", action="store_true")
    choice.add_argument("--evaluate-final-once", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--permit", type=Path)
    args = parser.parse_args(argv)
    if not args.train_once and not args.evaluate_final_once:
        print(
            p.canonical_json(
                {
                    "protocol_sha256": p.protocol_sha256(),
                    "protocol": p.protocol_config(),
                    "execution_authorized": False,
                }
            )
        )
        return 0
    if args.output is None or args.permit is None:
        parser.error("run modes require --output and --permit")
    permit = json.loads(args.permit.read_text(encoding="utf-8"))
    operation = train_once if args.train_once else evaluate_final_once
    try:
        result = operation(args.output, permit)
    except CampaignStopped as exc:
        parser.exit(2, f"STOP: {exc}\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
