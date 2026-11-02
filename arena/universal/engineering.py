"""Bounded CPU preflight planning and measurement, never a scientific campaign.

The CLI defaults to a plan. ``--run`` measures a random mechanical fixture and
updates a different, noise-only replay. Environment observations NEVER enter
that replay. No checkpoint or experimental game-learning result is produced.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time

from arena.universal.protocol import (
    GAMES, MAX_EVALUATION_STEPS, MAX_MINUTES, MAX_Q_UPDATES,
    MAX_TRAINING_STEPS, MAX_WORLD_UPDATES, protocol_config, protocol_sha256,
)

ENGINEERING_SEED = 810200
MECHANICAL_TOTAL_LIMIT = 1000
PREVIOUS_RANDOM_SMOKE_STEPS = 500
PREVIOUS_EVALUATOR_UNIT_STEPS = 25
REMAINING_MECHANICAL_STEPS = (MECHANICAL_TOTAL_LIMIT - PREVIOUS_RANDOM_SMOKE_STEPS
                              - PREVIOUS_EVALUATOR_UNIT_STEPS)
SYNTHETIC_UPDATE_LIMIT = 100
MARGIN = 1.20
RSS_LIMIT_BYTES = 3 * 1024**3
# A conservative storage allowance includes two owned RGB frames for a
# one-transition episode, masks, Python containers/scalars and episode metadata.
# It is an engineering estimate, not an allocator-level mathematical guarantee.
REPLAY_BYTES_PER_TRANSITION_ALLOWANCE = 16 * 1024

# This is a narrowly authenticated correction of ONE completed technical run,
# not a generic means to promote partial/failed benchmarks into measurements.
RECOVERABLE_ORIGINAL_RECEIPT_SHA256 = "aa7ee8964afa6f412eec308c6dba6a8763b82cdcfe72cc3fb74828c956ecc97d"
RECOVERABLE_SOURCE_COMMIT = "bd447baa601dd4ef5dfe18e9932ac75e3dc313da"
RECOVERABLE_PROTOCOL_SHA256 = "96927386422c7cdb3949ec2aa908deefd5230c2394e570bafda81a3627da3053"
RECOVERABLE_RSS_FAILURE = {"type": "ValueError", "message": "actual current/peak RSS counters are required"}
RECOVERABLE_SOURCE_FILES_SHA256 = {
    "arena/universal/__init__.py": "6a3aace66604527ffc7667a8ec34cab8852981f16acfaf05db1c60f00fe7958d",
    "arena/universal/analysis.py": "ee42992ac9523d044631d26e627cab4c8adb75131e61c697cdd613d8e7d79993",
    "arena/universal/campaign.py": "5ed8a243b0295d9484a222b13fe24ff2f9b64c111d8a35e9f37d4dc89b2c4ac9",
    "arena/universal/engineering.py": "c5041586d0a1f536847234409f9e17494c9410794a9124f41633d922389347f7",
    "arena/universal/envs.py": "e83f727f24b9814fe4fcb3578a82fc923b52fb24dfcaa4818535de8ed09064a9",
    "arena/universal/evaluation.py": "55c71e7527af23b60cd27bad156cd89ef10bed9bfd28b7d90d0329ff6721d466",
    "arena/universal/network.py": "52cd61f21c6fec0c20e81bff9ab7a15c123ee82889c80d2fc404496a8b734a45",
    "arena/universal/persistence.py": "c1df0f665bad6af667f94a8164e4dac5a650ecbd35b511c35845a4b43887c75d",
    "arena/universal/protocol.py": "6a012d2f09e57b4b1d512cd9518babd554ed59563b005bb10421382f85fd17ab",
    "arena/universal/replay.py": "afe5b4e00cacd15d5084069273e6cda604c86c72b5a3165eced78c50af990151",
    "arena/universal/trainer.py": "0d312531ac051027970d2bf4d8dd3c001c53f0f9c38dcd92d1c811c38ad44d89",
    "pyproject.toml": "eab5ebfffeb43ba91584d2bebdf89d737c427030bea58603df120734a87991b4",
    "requirements-ci.txt": "5ba4047bbe77232fbe0c6d59eef6ffc4b50d6aae1a0c71a1be6ce95096f51d8a",
}
RECOVERABLE_LEARNER_CONFIG = {
    "batch_size": 16, "burn_in": 4, "capacity_per_game": 10000, "episode_steps": 400,
    "epsilon_end": .05, "gamma": .98, "hidden_size": 64, "latent_size": 64,
    "learning_rate": .0003, "max_training_steps": 1080000, "planning_fraction": .2,
    "q_interval": 16, "replay_capacity": 40000, "target_tau": .01, "threads": 2,
    "unroll": 8, "warmup_steps": 128, "world_interval": 32,
}
RECOVERABLE_PERFORMANCE_PATHS = tuple(
    f"arena/universal/{name}.py" for name in ("network", "replay", "trainer", "envs", "evaluation"))


@dataclass(frozen=True)
class EngineeringConfig:
    mechanical_steps: int = REMAINING_MECHANICAL_STEPS
    synthetic_updates: int = SYNTHETIC_UPDATE_LIMIT
    seed: int = ENGINEERING_SEED

    def __post_init__(self):
        if (type(self.mechanical_steps) is not int or
                not len(GAMES) <= self.mechanical_steps <= REMAINING_MECHANICAL_STEPS or
                self.mechanical_steps % len(GAMES)):
            raise ValueError("mechanical steps must balance five games within the remaining 475-step budget")
        if (type(self.synthetic_updates) is not int or
                not 2 <= self.synthetic_updates <= SYNTHETIC_UPDATE_LIMIT or
                self.synthetic_updates % 2):
            raise ValueError("synthetic updates must be even, with Q/world pairs, in 2..100")
        if type(self.seed) is not int or self.seed != ENGINEERING_SEED:
            raise ValueError("this technical fixture uses seed 810200, never the reserved test seeds")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process_memory() -> dict:
    """Actual current RSS and process high-water RSS, without a third-party app."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("page_faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "peak_working_set", "working_set", "peak_paged_pool", "paged_pool",
                    "peak_nonpaged_pool", "nonpaged_pool", "pagefile", "peak_pagefile", "private_usage")]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        return {"rss_bytes": int(counters.working_set),
                "peak_rss_bytes": int(counters.peak_working_set),
                "source": "Windows GetProcessMemoryInfo working set / PeakWorkingSetSize"}
    if sys.platform.startswith("linux"):
        import resource
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
        return {"rss_bytes": resident_pages * int(os.sysconf("SC_PAGE_SIZE")),
                "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
                "source": "Linux /proc/self/statm and getrusage(RUSAGE_SELF), KiB converted to bytes"}
    raise RuntimeError("actual RSS measurement is implemented for Windows and Linux only")


def timing_summary(values) -> dict:
    values = list(values)
    if not values or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("timings require finite nonnegative observations")
    ordered = sorted(float(v) for v in values)

    def percentile(q):
        position = (len(ordered) - 1) * q
        lower, upper = math.floor(position), math.ceil(position)
        return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])

    return {"samples": len(values), "p50_seconds": statistics.median(ordered),
            "p95_seconds": percentile(.95), "maximum_seconds": ordered[-1],
            "percentile_interpolation": "linear"}


def summarize_memory(records) -> dict:
    """Retain raw RSS counters and use their conservative global maximum.

    Linux statm resident is an approximate counter; it and getrusage are also
    sampled separately. Therefore RSS <= reported high-water is not a valid
    cross-provider per-record invariant. Positive actual values are required,
    and neither a lower raw counter nor an inconvenient observation is dropped.
    """
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("actual positive current/peak RSS counters are required")
    for record in records:
        if (type(record) is not dict or type(record.get("rss_bytes")) is not int
                or record["rss_bytes"] <= 0 or type(record.get("peak_rss_bytes")) is not int
                or record["peak_rss_bytes"] <= 0
                or not isinstance(record.get("source"), str) or not record["source"]):
            raise ValueError("actual positive current/peak RSS counters are required")
    current = max(record["rss_bytes"] for record in records)
    reported = max(record["peak_rss_bytes"] for record in records)
    return {"baseline": copy.deepcopy(records[0]), "observations": copy.deepcopy(list(records)),
            "observed_current_rss_maximum": current,
            "actual_process_peak_rss_bytes": reported,
            "reported_process_peak_rss_maximum": reported,
            "conservative_peak_rss_bytes": max(current, reported),
            "peak_used_for_projection": "max of ALL reported high-water and observed current RSS values",
            "raw_counters_modified": False,
            "cross_provider_inversions": [
                {"record_index": index, "rss_above_reported_peak_bytes": record["rss_bytes"] - record["peak_rss_bytes"]}
                for index, record in enumerate(records) if record["rss_bytes"] > record["peak_rss_bytes"]],
            "counter_caveat": "Linux statm resident is approximate and providers are not sampled atomically",
            "counter_documentation": "https://man7.org/linux/man-pages/man5/proc_pid_statm.5.html"}


def project_campaign(timings: dict, *, observed_peak_rss_bytes: int,
                     synthetic_replay_transitions: int, replay_capacity: int = 40000) -> dict:
    """Project the fixed maximum workload from measured p95 operation timings.

    Each interaction includes CPU perception, recurrent policy, full one-step
    world planning and random mechanics/rendering. Both process CPU and wall
    projections are reported; the stricter value governs this conservative gate.
    """
    if (type(observed_peak_rss_bytes) is not int or observed_peak_rss_bytes <= 0
            or type(replay_capacity) is not int or not 1 <= replay_capacity <= 40000
            or type(synthetic_replay_transitions) is not int
            or not 0 <= synthetic_replay_transitions <= replay_capacity):
        raise ValueError("actual positive peak RSS and valid replay counts are required")
    counts = {"step": MAX_TRAINING_STEPS + MAX_EVALUATION_STEPS,
              "q": MAX_Q_UPDATES, "world": MAX_WORLD_UPDATES}
    projections = {}
    for clock in ("wall", "process_cpu"):
        total = 0.0
        for operation, count in counts.items():
            summary = timings[operation][clock]
            p95 = summary["p95_seconds"]
            if (type(summary["samples"]) is not int or summary["samples"] < 1
                    or type(p95) not in (int, float) or not math.isfinite(p95) or p95 <= 0):
                raise ValueError("each operation needs an actual positive p95 timing")
            total += count * p95
        projections[clock + "_seconds_with_margin"] = MARGIN * total
    missing = replay_capacity - synthetic_replay_transitions
    additional_replay = missing * REPLAY_BYTES_PER_TRANSITION_ALLOWANCE
    estimated_rss = math.ceil(MARGIN * (observed_peak_rss_bytes + additional_replay))
    governing_seconds = max(projections.values())
    fits = governing_seconds <= MAX_MINUTES * 60 and estimated_rss <= RSS_LIMIT_BYTES
    return {**projections, "governing_seconds_with_margin": governing_seconds,
            "governing_minutes_with_margin": governing_seconds / 60,
            "margin_multiplier": MARGIN, "maximum_minutes": MAX_MINUTES,
            "training_steps": MAX_TRAINING_STEPS, "evaluation_steps_including_ablations": MAX_EVALUATION_STEPS,
            "q_updates": MAX_Q_UPDATES, "world_updates": MAX_WORLD_UPDATES,
            "projected_step_calls": counts["step"], "replay_capacity": replay_capacity,
            "observed_peak_rss_bytes": observed_peak_rss_bytes,
            "additional_replay_bytes_allowance": additional_replay,
            "replay_bytes_per_transition_allowance": REPLAY_BYTES_PER_TRANSITION_ALLOWANCE,
            "estimated_rss_bytes_with_margin": estimated_rss,
            "maximum_rss_bytes": RSS_LIMIT_BYTES, "within_projected_limits": fits,
            "memory_estimate_is_allocator_guarantee": False,
            "excludes_unmeasured_checkpoint_and_external_io_time": True}


def source_provenance() -> dict:
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "arena/universal", "requirements-ci.txt", "pyproject.toml"],
        cwd=root, text=True).splitlines()
    paths = sorted((root / "arena/universal").glob("*.py"))
    paths += [root / "requirements-ci.txt", root / "pyproject.toml"]
    candidate = protocol_config()
    return {"source_commit": commit, "relevant_sources_clean": not dirty,
            "dirty_source_records": dirty,
            "source_files_sha256": {path.relative_to(root).as_posix(): _sha(path) for path in paths},
            "protocol_sha256": protocol_sha256(), "protocol_version": candidate["version"],
            "protocol_status": candidate["status"], "architecture_config": candidate["architecture"],
            "scientific_execution_authorized_by_this_module": False}


def _base_receipt(config, source):
    return {"schema_version": 1, "kind": "ENGINEERING_PREFLIGHT_NOT_SCIENTIFIC_RESULT",
            "config": asdict(config), "source": source,
            "technical_seed": config.seed,
            "mechanical_budget": {"global_limit": MECHANICAL_TOTAL_LIMIT,
                                  "previous_random_smoke": PREVIOUS_RANDOM_SMOKE_STEPS,
                                  "previous_evaluator_unit": PREVIOUS_EVALUATOR_UNIT_STEPS,
                                  "requested_here": config.mechanical_steps,
                                  "maximum_new_steps_here": REMAINING_MECHANICAL_STEPS},
            "synthetic_update_limit": SYNTHETIC_UPDATE_LIMIT,
            "scientific_environment_training_steps": 0,
            "local_timing_is_github_runner_timing": False,
            "matching_hardware_ci_measurement_required_if_different": True,
            "projection_is_full_campaign_runtime_guarantee": False,
            "resume_25_receipt": "PENDING_SEPARATE_TECHNICAL_TEST_NOT_PERFORMED_HERE",
            "no_scientific_checkpoints": True}


def _write_receipt(root, receipt):
    path = root / "receipt.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with (root / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        handle.write(f"{_sha(path)}  receipt.json\n")


def plan_preflight(output, config=None, *, source_provider=source_provenance):
    """Create a new plan receipt without importing Torch or constructing games."""
    config = config or EngineeringConfig()
    receipt = _base_receipt(config, source_provider())
    receipt.update(status="NOT_EXECUTED", actual_measurements=False, timings=None, projection=None)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    _write_receipt(root, receipt)
    return receipt


def _validate_recoverable_receipt(receipt):
    """Fail closed on any incomplete fixture, different failure or source identity."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(type(receipt) is dict and receipt.get("schema_version") == 1,
            "invalid original engineering receipt schema")
    require(receipt.get("kind") == "ENGINEERING_PREFLIGHT_NOT_SCIENTIFIC_RESULT"
            and receipt.get("status") == "FAILED" and receipt.get("failure") == RECOVERABLE_RSS_FAILURE,
            "only the specific completed RSS cross-counter validation failure is recoverable")
    require(receipt.get("actual_measurements") is True
            and receipt.get("no_scientific_checkpoints") is True
            and type(receipt.get("scientific_environment_training_steps")) is int
            and receipt["scientific_environment_training_steps"] == 0,
            "original must contain actual technical measurements and zero scientific training")
    require(receipt.get("config") == {"mechanical_steps": 475, "synthetic_updates": 100, "seed": 810200}
            and receipt.get("technical_seed") == 810200 and receipt.get("synthetic_update_limit") == 100,
            "original bounded fixture configuration differs")
    require(receipt.get("mechanical_budget") == {
        "global_limit": 1000, "previous_random_smoke": 500, "previous_evaluator_unit": 25,
        "requested_here": 475, "maximum_new_steps_here": 475}, "original mechanical budget differs")
    require(receipt.get("mechanical_attempts") == 475 and receipt.get("synthetic_update_attempts") == 100
            and receipt.get("mechanical_steps_by_game") == {game: 95 for game in GAMES},
            "original fixture is not complete across all five games")
    source = receipt.get("source", {})
    require(source.get("source_commit") == RECOVERABLE_SOURCE_COMMIT
            and source.get("protocol_sha256") == RECOVERABLE_PROTOCOL_SHA256
            and source.get("protocol_version") == "mg002-plano-v0.1-implementation-2"
            and source.get("protocol_status") == "CANDIDATE_NOT_FROZEN"
            and source.get("relevant_sources_clean") is True
            and source.get("dirty_source_records") == []
            and source.get("scientific_execution_authorized_by_this_module") is False,
            "original source commit/protocol/cleanliness identity differs")
    require(source.get("source_files_sha256") == RECOVERABLE_SOURCE_FILES_SHA256,
            "original neural/environment/trainer/source file hashes differ")
    architecture = source.get("architecture_config", {})
    expected_architecture = {"observation": [32, 32, 3], "actions": 10, "gru_units": 64,
                             "batch_size": 16, "burn_in": 4, "unroll": 8, "gamma": .98,
                             "per_game_heads": False, "world_weight": .2, "replay_capacity": 40000}
    require(all(architecture.get(key) == value for key, value in expected_architecture.items()),
            "original neural architecture contract differs")
    state = receipt.get("fixture_state", {})
    require(state.get("learner_config") == RECOVERABLE_LEARNER_CONFIG
            and state.get("cpu_threads") == 2 and state.get("torch_version") == "2.10.0+cpu"
            and state.get("numpy_version") == "2.3.5", "original learner/runtime configuration differs")
    require(state.get("global_environment_training_steps") == 0
            and state.get("training_curve_records") == 0 and state.get("trainer_stage") == "unconfigured"
            and state.get("synthetic_replay_only") is True and state.get("mechanical_steps") == 475
            and state.get("q_updates") == state.get("world_updates") == 50
            and state.get("replay_transitions") == 512
            and state.get("replay_game_counts") == {f"synthetic-noise-{i}": 128 for i in range(4)},
            "original synthetic-only fixture counters are incomplete or inconsistent")
    require(state.get("mechanical_policy") == "uniform random legal inputs; computed network outputs are not selected"
            and state.get("mechanical_first_environment_seeds") == {
                game: 810200 + 1000 * index for index, game in enumerate(GAMES)},
            "original random mechanical policy/seed namespace differs")
    before = receipt.get("untrained_fingerprint_before_mechanics")
    after_mechanics = receipt.get("untrained_fingerprint_after_mechanics")
    after_updates = receipt.get("fixture_fingerprint_after_synthetic_updates")
    require(all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                for value in (before, after_mechanics, after_updates))
            and before == after_mechanics and after_updates != before,
            "original unchanged-mechanics/changed-synthetic neural fingerprint contract failed")
    samples = receipt.get("partial_timing_samples", {})
    require(set(samples) == {"step", "q", "world"}, "original raw timing operations are incomplete")
    summaries = {}
    for operation, count in (("step", 475), ("q", 50), ("world", 50)):
        require(type(samples[operation]) is dict and set(samples[operation]) == {"wall", "process_cpu"},
                "original timing clocks are incomplete")
        summaries[operation] = {}
        for clock in ("wall", "process_cpu"):
            values = samples[operation][clock]
            require(type(values) is list and len(values) == count
                    and all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in values),
                    "original actual timing sample counts/values are incomplete")
            summaries[operation][clock] = timing_summary(values)
    require(receipt.get("timings") == summaries,
            "recorded timing summaries differ from complete original raw samples")
    records = receipt.get("partial_memory_records", [])
    require(type(records) is list and len(records) == 4
            and all(record.get("source") ==
                    "Linux /proc/self/statm and getrusage(RUSAGE_SELF), KiB converted to bytes"
                    for record in records if type(record) is dict),
            "original Linux memory provenance is incomplete")
    memory = summarize_memory(records)
    require(bool(memory["cross_provider_inversions"]),
            "original data do not exhibit the specifically recoverable counter inversion")
    require((MAX_TRAINING_STEPS, MAX_EVALUATION_STEPS, MAX_Q_UPDATES, MAX_WORLD_UPDATES,
             MAX_MINUTES, MARGIN, RSS_LIMIT_BYTES, REPLAY_BYTES_PER_TRANSITION_ALLOWANCE) ==
            (1080000, 694000, 67500, 33750, 300, 1.2, 3 * 1024**3, 16 * 1024),
            "offline correction cannot silently change the measured projection contract")
    return summaries, memory, state


def recover_completed_preflight(original_receipt, output, *, expected_original_sha256,
                                source_provider=source_provenance):
    """Derive a correction from ONE authenticated completed fixture, with no rerun.

    This never creates a PASS measurement. The original remains FAILED and is
    not edited; Chat 1 must explicitly decide whether the derived calculation
    satisfies its engineering gate before scientific execution can be frozen.
    """
    if (expected_original_sha256 != RECOVERABLE_ORIGINAL_RECEIPT_SHA256
            or re.fullmatch(r"[0-9a-f]{64}", expected_original_sha256 or "") is None):
        raise ValueError("only the pinned original receipt SHA-256 may be recovered")
    original = Path(original_receipt)
    if original.is_symlink() or not original.is_file() or original.stat().st_size > 1024 * 1024:
        raise ValueError("original receipt must be a bounded regular file")
    data = original.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_original_sha256:
        raise ValueError("original receipt SHA-256 does not match the authenticated measurement")
    receipt = json.loads(data)
    summaries, memory, state = _validate_recoverable_receipt(receipt)
    derivation_source = source_provider()
    current_hashes = derivation_source.get("source_files_sha256", {})
    if any(current_hashes.get(path) != RECOVERABLE_SOURCE_FILES_SHA256[path]
           for path in RECOVERABLE_PERFORMANCE_PATHS):
        raise ValueError("current performance-critical neural/environment sources differ from the measured source")
    projection = project_campaign(
        summaries, observed_peak_rss_bytes=memory["conservative_peak_rss_bytes"],
        synthetic_replay_transitions=state["replay_transitions"],
        replay_capacity=state["learner_config"]["replay_capacity"])
    correction = {
        "schema_version": 1,
        "kind": "ENGINEERING_OFFLINE_CORRECTION_NOT_NEW_MEASUREMENT",
        "status": "DERIVED_FROM_COMPLETE_MEASUREMENTS",
        "actual_measurements": False,
        "uses_authenticated_original_actual_measurements": True,
        "new_measurements": False, "new_mechanical_steps": 0, "new_synthetic_updates": 0,
        "scientific_environment_training_steps": 0, "no_scientific_checkpoints": True,
        "original_receipt": {"path": str(original.resolve()), "sha256": expected_original_sha256,
                             "status": receipt["status"], "failure": copy.deepcopy(receipt["failure"]),
                             "modified": False},
        "original_artifact_origin": {
            "run_id": 41500156977, "job_id": 125465721718,
            "coordinator_verified_archive_sha256": "72dd6cddf504d2192532dfea50138e22f7779032d23ca456179f1c5e093809a0",
            "coordinator_verified_archive_bytes": 12550,
            "archive_revalidated_by_this_offline_function": False,
            "receipt_bytes_independently_revalidated": True},
        "source": copy.deepcopy(receipt["source"]),
        "derivation_source": derivation_source,
        "performance_critical_sources_unchanged": {
            path: current_hashes[path] for path in RECOVERABLE_PERFORMANCE_PATHS},
        "config": copy.deepcopy(receipt["config"]),
        "mechanical_budget": copy.deepcopy(receipt["mechanical_budget"]),
        "mechanical_steps_by_game": copy.deepcopy(receipt["mechanical_steps_by_game"]),
        "mechanical_attempts_already_spent": receipt["mechanical_attempts"],
        "synthetic_update_attempts_already_spent": receipt["synthetic_update_attempts"],
        "fixture_state": copy.deepcopy(state),
        "untrained_fingerprint_before_mechanics": receipt["untrained_fingerprint_before_mechanics"],
        "untrained_fingerprint_after_mechanics": receipt["untrained_fingerprint_after_mechanics"],
        "fixture_fingerprint_after_synthetic_updates": receipt["fixture_fingerprint_after_synthetic_updates"],
        "original_raw_timing_samples": copy.deepcopy(receipt["partial_timing_samples"]),
        "timings": summaries, "memory": memory, "projection": projection,
        "projected_limits_decision": "WITHIN_LIMITS" if projection["within_projected_limits"] else "EXCEEDS_LIMITS",
        "original_relabelled_as_pass": False,
        "gate_acceptance": "PENDING_EXPLICIT_CHAT1_DECISION_BEFORE_FREEZE",
        "scientific_execution_authorized_by_this_correction": False,
        "resume_25_receipt": receipt["resume_25_receipt"],
        "hardware_metadata_not_present_in_original": "hardware" not in receipt,
        "timing_scope": "recorded Linux GitHub job only; not proof of equivalence to another machine/job",
        "correction_method": "validate positive raw counters, preserve every record, project using max(all observed RSS, all reported peaks)",
        "failure_cause": "per-record RSS <= high-water was incorrectly assumed across Linux statm/getrusage providers",
        "method_documentation": ["https://man7.org/linux/man-pages/man5/proc_pid_statm.5.html",
                                 "https://man7.org/linux/man-pages/man2/getrusage.2.html"],
    }
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    _write_receipt(root, correction)
    return correction


class _RealBackend:
    """Mechanical actors and a wholly separate synthetic replay in one process."""
    def __init__(self, config):
        import numpy as np
        import torch
        from arena.universal.envs import make_game
        from arena.universal.trainer import LearnerConfig, UniversalTrainer
        self.np, self.torch, self.make_game = np, torch, make_game
        self.config = config
        self.trainer = UniversalTrainer(config.seed, LearnerConfig())
        self.trainer.online.eval()
        if next(self.trainer.online.parameters()).device.type != "cpu":
            raise RuntimeError("engineering measurements require CPU")
        self.actors = {}
        self.rng = np.random.default_rng(config.seed + 17)
        self.noise_rng = np.random.default_rng(config.seed + 31)
        self.mechanical_steps = 0

    def fingerprint(self):
        digest = hashlib.sha256()
        for name, tensor in sorted(self.trainer.online.state_dict().items()):
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def mechanical_step(self, game):
        np, torch, trainer = self.np, self.torch, self.trainer
        if game not in self.actors:
            env = self.make_game(game, max_steps=trainer.config.episode_steps)
            seed = self.config.seed + 1000 * GAMES.index(game)
            observation, _ = env.reset(seed=seed)
            self.actors[game] = {"env": env, "observation": observation, "hidden": trainer.online.initial(1),
                                 "action": 0, "reward": 0., "episode": 0, "first_seed": seed}
        actor = self.actors[game]
        available = actor["env"].action_mask()
        pixels = torch.as_tensor(actor["observation"][None], dtype=torch.uint8).permute(0, 3, 1, 2).contiguous()
        with torch.inference_mode():
            q, state, _ = trainer.online.step(pixels, torch.tensor([actor["action"]]),
                                             torch.tensor([actor["reward"]], dtype=torch.float32), actor["hidden"])
            plan = trainer.online.plan_values(state, torch.as_tensor(available[None]), gamma=trainer.config.gamma)
            values = (1 - trainer.config.planning_fraction) * q + trainer.config.planning_fraction * plan
            values.masked_fill_(~torch.as_tensor(available[None]), -torch.inf)
            if not bool(torch.isfinite(values[0, torch.as_tensor(available)]).all()):
                raise FloatingPointError("nonfinite full-policy inference")
            values.argmax(-1).item()  # Materialize the full policy, but NEVER act on it.
        action = int(self.rng.choice(np.flatnonzero(available)))
        observation, reward, terminated, truncated, _ = actor["env"].step(action)
        self.mechanical_steps += 1
        actor.update(observation=observation, hidden=state, action=action, reward=float(reward))
        if terminated or truncated:
            actor["episode"] += 1
            observation, _ = actor["env"].reset(seed=actor["first_seed"] + actor["episode"])
            actor.update(observation=observation, hidden=trainer.online.initial(1), action=0, reward=0.)

    def prepare_synthetic(self):
        np, replay, rng = self.np, self.trainer.replay, self.noise_rng
        if len(replay):
            raise RuntimeError("environment transitions must never enter engineering replay")
        # Eight 64-transition chains exercise the actual sequence sampler and
        # learner shapes; no simulator was used to generate a single pixel here.
        for chain in range(8):
            game = f"synthetic-noise-{chain % 4}"
            observation = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
            available = np.ones(10, dtype=bool)
            for step in range(64):
                following = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
                replay.append(game, observation, int(rng.integers(10)), float(rng.uniform(-1, 1)),
                              following, step == 63, False, available, available)
                observation = following

    def update(self, *, world):
        return self.trainer.learn_batch(world=world)

    def state_receipt(self):
        trainer = self.trainer
        return {"global_environment_training_steps": trainer.global_steps,
                "trainer_stage": trainer.stage, "training_curve_records": len(trainer.training_curve),
                "q_updates": trainer.q_updates, "world_updates": trainer.world_updates,
                "mechanical_steps": self.mechanical_steps,
                "replay_transitions": len(trainer.replay), "replay_game_counts": trainer.replay.game_counts(),
                "synthetic_replay_only": bool(trainer.replay.game_counts()) and
                    all(name.startswith("synthetic-noise-") for name in trainer.replay.game_counts()),
                "learner_config": asdict(trainer.config), "torch_version": self.torch.__version__,
                "numpy_version": self.np.__version__, "cpu_threads": self.torch.get_num_threads(),
                "mechanical_policy": "uniform random legal inputs; computed network outputs are not selected",
                "mechanical_first_environment_seeds": {game: actor["first_seed"] for game, actor in self.actors.items()}}

    def close(self):
        for actor in self.actors.values():
            actor["env"].close()


def run_preflight(output, config=None, *, backend_factory=None,
                  wall_clock=time.perf_counter, cpu_clock=time.process_time,
                  memory_reader=process_memory, source_provider=source_provenance):
    """Explicit technical measurements, never automatically called by tests/import.

    Dependency injection supports bounded unit fixtures. Injected measurements
    are always labelled TEST_ONLY and can never issue an engineering PASS.
    """
    config = config or EngineeringConfig()
    source = source_provider()
    actual = (backend_factory is None and wall_clock is time.perf_counter
              and cpu_clock is time.process_time and memory_reader is process_memory)
    if actual and not source["relevant_sources_clean"]:
        raise ValueError("commit the bound implementation before collecting an actual timing receipt")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    receipt = _base_receipt(config, source)
    receipt.update(status="FAILED", actual_measurements=actual,
                   mechanical_attempts=0, synthetic_update_attempts=0)
    samples = {name: {"wall": [], "process_cpu": []} for name in ("step", "q", "world")}
    memory_records = []
    backend = None
    try:
        memory_records.append(memory_reader())
        backend = _RealBackend(config) if backend_factory is None else backend_factory(config)
        memory_records.append(memory_reader())
        receipt["untrained_fingerprint_before_mechanics"] = backend.fingerprint()
        per_game = {game: 0 for game in GAMES}
        for index in range(config.mechanical_steps):
            game = GAMES[index % len(GAMES)]
            receipt["mechanical_attempts"] += 1
            wall, cpu = wall_clock(), cpu_clock()
            backend.mechanical_step(game)
            samples["step"]["process_cpu"].append(cpu_clock() - cpu)
            samples["step"]["wall"].append(wall_clock() - wall)
            per_game[game] += 1
        receipt["mechanical_steps_by_game"] = per_game
        receipt["untrained_fingerprint_after_mechanics"] = backend.fingerprint()
        if receipt["untrained_fingerprint_before_mechanics"] != receipt["untrained_fingerprint_after_mechanics"]:
            raise RuntimeError("random mechanical timing unexpectedly changed neural weights")
        memory_records.append(memory_reader())
        backend.prepare_synthetic()
        for index in range(config.synthetic_updates):
            world = bool(index % 2)
            name = "world" if world else "q"
            receipt["synthetic_update_attempts"] += 1
            wall, cpu = wall_clock(), cpu_clock()
            loss = backend.update(world=world)
            samples[name]["process_cpu"].append(cpu_clock() - cpu)
            samples[name]["wall"].append(wall_clock() - wall)
            if not math.isfinite(float(loss)):
                raise FloatingPointError("nonfinite synthetic fixture loss")
        receipt["fixture_state"] = state = backend.state_receipt()
        receipt["fixture_fingerprint_after_synthetic_updates"] = backend.fingerprint()
        if (state["global_environment_training_steps"] != 0 or state["training_curve_records"] != 0
                or not state["synthetic_replay_only"] or state["mechanical_steps"] != config.mechanical_steps
                or state["q_updates"] != config.synthetic_updates // 2
                or state["world_updates"] != config.synthetic_updates // 2):
            raise RuntimeError("technical fixture counters or synthetic-only replay contract differ")
        if receipt["fixture_fingerprint_after_synthetic_updates"] == receipt["untrained_fingerprint_after_mechanics"]:
            raise RuntimeError("synthetic updates did not change fixture weights")
        memory_records.append(memory_reader())
        receipt["timings"] = summaries = {
            name: {clock: timing_summary(values) for clock, values in clocks.items()}
            for name, clocks in samples.items()}
        receipt["memory"] = summarize_memory(memory_records)
        receipt["projection"] = projection = project_campaign(
            summaries, observed_peak_rss_bytes=receipt["memory"]["conservative_peak_rss_bytes"],
            synthetic_replay_transitions=state["replay_transitions"],
            replay_capacity=state["learner_config"]["replay_capacity"])
        if source_provider() != source:
            raise RuntimeError("bound source/protocol changed during preflight")
        receipt["hardware"] = {"platform": platform.platform(), "machine": platform.machine(),
                               "processor": platform.processor(), "logical_cpus": os.cpu_count(),
                               "python": platform.python_version(), "cpu_threads": state.get("cpu_threads")}
        receipt["status"] = ("TEST_ONLY_NOT_EVIDENCE" if not actual else
                             "PASS" if projection["within_projected_limits"] else "FAIL_PROJECTED_LIMITS")
    except Exception as exc:
        receipt["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        receipt["partial_timing_samples"] = samples
        receipt["partial_memory_records"] = memory_records
    finally:
        if backend is not None:
            backend.close()
    _write_receipt(root, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="NEW technical receipt directory")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="explicitly measure the bounded technical fixture")
    mode.add_argument("--recover-from", type=Path,
                      help="OFFLINE correction of the single authenticated completed RSS-validation failure")
    parser.add_argument("--expected-original-sha256", help="required independently authenticated original receipt digest")
    parser.add_argument("--steps", type=int, default=REMAINING_MECHANICAL_STEPS)
    parser.add_argument("--synthetic-updates", type=int, default=SYNTHETIC_UPDATE_LIMIT)
    args = parser.parse_args()
    if args.recover_from:
        if not args.expected_original_sha256:
            parser.error("offline recovery requires --expected-original-sha256")
        if args.steps != REMAINING_MECHANICAL_STEPS or args.synthetic_updates != SYNTHETIC_UPDATE_LIMIT:
            parser.error("offline recovery cannot change the original measured fixture workload")
        receipt = recover_completed_preflight(args.recover_from, args.output,
                                              expected_original_sha256=args.expected_original_sha256)
        print(json.dumps({"status": receipt["status"], "output": str(args.output),
                          "new_measurements": False, "new_mechanical_steps": 0,
                          "new_synthetic_updates": 0, "gate_acceptance": receipt["gate_acceptance"],
                          "projected_limits_decision": receipt["projected_limits_decision"]}, sort_keys=True))
        return
    if args.expected_original_sha256:
        parser.error("--expected-original-sha256 is only valid with --recover-from")
    config = EngineeringConfig(args.steps, args.synthetic_updates)
    receipt = run_preflight(args.output, config) if args.run else plan_preflight(args.output, config)
    print(json.dumps({"status": receipt["status"], "output": str(args.output),
                      "actual_measurements": receipt["actual_measurements"],
                      "scientific_environment_training_steps": 0}, sort_keys=True))
    if args.run and receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
