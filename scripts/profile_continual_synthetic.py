"""Opt-in, bounded synthetic cost profile; never a scientific experiment.

The default is a plan and imports no ML runtime. --run constructs NEW random
weights and random pixel replay only. It never configures/steps an environment,
loads a checkpoint, uses evaluation seeds or reuses historical G1 timings.

Resource checks are cooperative after setup, around operations and throughout
finalization. They cannot interrupt a blocked native import/kernel or IO; an
outer CI/process timeout must provide that hard watchdog.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SEED = 830003
MAX_UPDATES = MAX_FORWARDS = 100
MAX_WALL_SECONDS, MAX_CPU_SECONDS, MAX_RSS_BYTES = 120.0, 240.0, 3 * 1024**3
SOURCE_FILES = (
    "scripts/profile_continual_synthetic.py",
    "arena/continual/__init__.py",
    "arena/continual/trainer.py",
    "arena/continual/network.py",
    "arena/continual/replay.py",
    "arena/continual/objectives.py",
    "arena/continual/calibration.py",
    "arena/continual/policy.py",
    "arena/continual/evaluation.py",
    "arena/universal/network.py",
    "arena/universal/replay.py",
    "arena/universal/envs.py",
    "arena/universal/evaluation.py",
    "arena/universal/campaign.py",
    "pyproject.toml",
    "requirements-ci.txt",
)
FORWARD_OPERATIONS = (
    "actor_uncached_prefix8",
    "actor_uncached_prefix16",
    "plan_values_all10",
    "frozen_control_step",
    "shared_after_audit_step",
)


@dataclass(frozen=True)
class ProfileConfig:
    updates: int = 100
    forwards: int = 100
    context_length: int = 16
    threads: int = 2

    def __post_init__(self):
        if (
            type(self.updates) is not int
            or not 0 <= self.updates <= MAX_UPDATES
            or self.updates % 2
        ):
            raise ValueError(
                "updates must be even and in 0..100 (at most 50 Q and 50 world)"
            )
        if type(self.forwards) is not int or not 0 <= self.forwards <= MAX_FORWARDS:
            raise ValueError("forwards must be in 0..100 total standalone operations")
        if type(self.context_length) is not int or self.context_length not in (8, 16):
            raise ValueError("context_length must be 8 or 16")
        if type(self.threads) is not int or self.threads not in (1, 2):
            raise ValueError("threads must be 1 or 2")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def provenance():
    def git(*args):
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None

    status = git("status", "--porcelain")
    return {
        "git_sha": git("rev-parse", "HEAD"),
        "tracked_or_untracked_dirty": bool(status) if status is not None else None,
        "git_status": status.splitlines() if status else [],
        "source_files_sha256": {
            name: sha(ROOT / name) if (ROOT / name).is_file() else None
            for name in SOURCE_FILES
        },
        "hash_convention": "raw local file bytes; source may be dirty and must be fixed separately for science",
    }


def metadata_runtime():
    versions = {}
    for name in ("torch", "numpy", "gymnasium"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "installed_distribution_versions": versions,
        "ml_runtime_loaded_by_profile": False,
    }


def new_output(value):
    path = Path(value)
    if ".." in path.parts or str(path).startswith(("\\\\", "//")):
        raise ValueError("output must be an ordinary new local path")
    path = path.absolute()
    for component in (path, *path.parents):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("output path must not contain symlinks or reparse points")
    if path.exists():
        raise FileExistsError("output already exists; choose a new directory")
    path.mkdir(parents=True, exist_ok=False)
    return path


def summary(values):
    if not values:
        return {
            "count": 0,
            "total_seconds": 0.0,
            "p50_seconds": None,
            "p95_seconds": None,
            "maximum_seconds": None,
        }
    ordered = sorted(values)
    if any(not math.isfinite(v) or v < 0 for v in ordered):
        raise ValueError("timing samples must be finite and nonnegative")

    def percentile(fraction):
        index = fraction * (len(ordered) - 1)
        lower, upper = math.floor(index), math.ceil(index)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return {
        "count": len(values),
        "total_seconds": sum(values),
        "p50_seconds": percentile(0.5),
        "p95_seconds": percentile(0.95),
        "maximum_seconds": ordered[-1],
    }


class _MaskOnly:
    """A controller-input mask, deliberately not a simulator or Gym Env."""

    def action_mask(self):
        import numpy as np

        return np.ones(10, dtype=np.bool_)


class SyntheticBackend:
    def __init__(self, config):
        # Import only after explicit --run. Small CLI tests use a fresh process,
        # so inter-op thread limits are established before any tensor execution.
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = str(config.threads)
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        import copy
        import numpy as np
        import torch

        torch.set_num_interop_threads(config.threads)
        torch.set_default_device("cpu")
        import arena.continual.trainer as module
        from arena.continual.calibration import GateCalibration
        from arena.continual.evaluation import _raw_model_values
        from arena.continual.network import ContinualQNetwork
        from arena.continual.policy import FrozenPolicyController
        from arena.universal.campaign import _rss_bytes

        self.torch, self.np, self.read_peak_rss = torch, np, _rss_bytes
        self.hooks = []
        self.kernel_counts = defaultdict(lambda: {"calls": 0, "items": 0})
        self.original_make_game = module.make_game
        self.module = module
        self.forbidden_environment_calls = 0

        def forbidden(*args, **kwargs):
            self.forbidden_environment_calls += 1
            raise RuntimeError(
                "synthetic profiler forbids environment construction or interaction"
            )

        module.make_game = forbidden
        try:
            learner_config = module.ContinualConfig(
                context_length=config.context_length, threads=config.threads
            )
            self.trainer = module.ContinualTrainer(SEED, learner_config)
            self.trainer.configure_stage = forbidden
            self.trainer.collect = forbidden
            rng = np.random.default_rng(SEED)
            self.episode_length = 33  # crosses both 8- and 16-observation boundaries

            def append(replay, label, episodes):
                for episode in range(episodes):
                    frame = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
                    previous_action, previous_reward = 0, 0.0
                    for step in range(self.episode_length):
                        nxt = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
                        action = int(rng.integers(0, 10))
                        reward = float(rng.uniform(-1, 1))
                        last = step == self.episode_length - 1
                        mask = np.ones(10, dtype=np.bool_)
                        replay.append(
                            label,
                            frame,
                            action,
                            reward,
                            nxt,
                            last and episode % 2 == 0,
                            last and episode % 2 == 1,
                            mask,
                            mask,
                            prev_action=previous_action,
                            prev_reward=previous_reward,
                        )
                        frame, previous_action, previous_reward = nxt, action, reward

            for label in module.SOURCE_GAMES:
                append(self.trainer.replay, label, 16)
            self.trainer.source_replay = copy.deepcopy(self.trainer.replay)
            self.trainer.source_teacher = (
                copy.deepcopy(self.trainer.online).requires_grad_(False).eval()
            )
            append(self.trainer.replay, module.TARGET_GAME, 16)
            append(self.trainer.calibration_replay, module.TARGET_GAME, 32)
            self.trainer.stage = "adaptation"
            self.trainer.games = (module.TARGET_GAME,)
            # stage_budget/global/fit/calibration interaction counters remain zero;
            # these are artificial arrays, not collected gameplay transitions.
            self.teacher_before = self.fingerprint(self.trainer.source_teacher)
            self.forward_frames = {
                n: rng.integers(0, 256, (n, 32, 32, 3), dtype=np.uint8) for n in (8, 16)
            }
            # This separate NEW random network is never optimized or loaded.
            # Cold calibration does not claim that any learned plan is valid.
            self.inference_network = (
                ContinualQNetwork(
                    latent_size=learner_config.latent_size,
                    hidden_size=learner_config.hidden_size,
                )
                .cpu()
                .eval()
                .requires_grad_(False)
            )
            self.inference_controller = FrozenPolicyController(
                self.inference_network,
                calibration=GateCalibration(),
                policy_revision=0,
                context_length=config.context_length,
                gamma=learner_config.gamma,
            )
            self.raw_model_values = _raw_model_values
            self.inference_phase = config.context_length // 2
            self.inference_hidden = self.inference_controller._hidden.detach().clone()
            self.inference_frame = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
            self.inference_next_frame = rng.integers(
                0, 256, (32, 32, 3), dtype=np.uint8
            )
            self.inference_mask = np.ones(10, dtype=np.bool_)
            self.inference_batch_mask = torch.from_numpy(self.inference_mask).unsqueeze(
                0
            )
            self.inference_before = self.fingerprint(self.inference_network)
            self.inference_operations = {
                "frozen_control_step": 0,
                "shared_after_audit_step": 0,
            }
            networks = {
                role: getattr(self.trainer, role)
                for role in ("online", "target", "source_teacher")
            }
            networks["frozen_inference"] = self.inference_network
            for role, network in networks.items():
                for component in ("encoder", "memory", "transition", "q_head"):
                    key = role + "." + component
                    self.kernel_counts[key]  # include explicit zero setup counts

                    def observe(_module, args, _output, key=key):
                        self.kernel_counts[key]["calls"] += 1
                        self.kernel_counts[key]["items"] += int(args[0].shape[0])

                    self.hooks.append(
                        getattr(network, component).register_forward_hook(observe)
                    )
            self.setup_kernel_counts = {
                k: dict(v) for k, v in self.kernel_counts.items()
            }
            self.runtime = {
                "python": platform.python_version(),
                "system": platform.system(),
                "machine": platform.machine(),
                "torch": str(torch.__version__),
                "numpy": np.__version__,
                "gymnasium": importlib.metadata.version("gymnasium"),
                "device": "cpu",
                "intraop_threads": torch.get_num_threads(),
                "interop_threads": torch.get_num_interop_threads(),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cuda_initialized": torch.cuda.is_initialized(),
                "ml_runtime_loaded_by_profile": True,
            }
            if (
                self.runtime["cuda_initialized"]
                or max(self.runtime["intraop_threads"], self.runtime["interop_threads"])
                > 2
            ):
                raise RuntimeError("CPU-only two-thread profile contract violated")
        except BaseException:
            self.close()
            raise

    def fingerprint(self, network):
        digest = hashlib.sha256()
        for name, tensor in sorted(network.state_dict().items()):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def forward(self, operation):
        torch, np, trainer = self.torch, self.np, self.trainer
        with torch.inference_mode():
            if operation in ("frozen_control_step", "shared_after_audit_step"):
                return self._frozen_forward(
                    audit=operation == "shared_after_audit_step"
                )
            if operation == "plan_values_all10":
                hidden = torch.zeros(
                    (1, trainer.config.hidden_size), dtype=torch.float32
                )
                values = trainer.online.plan_values(
                    hidden, torch.ones((1, 10), dtype=torch.bool), trainer.config.gamma
                )
                return float(values.max())
            if operation not in ("actor_uncached_prefix8", "actor_uncached_prefix16"):
                raise ValueError("unknown standalone synthetic forward operation")
            length = 8 if operation == "actor_uncached_prefix8" else 16
            original = trainer.config
            trainer.config = replace(original, context_length=length)
            try:
                frames = self.forward_frames[length]
                actor = {
                    "env": _MaskOnly(),
                    "observation": frames[-1],
                    "previous_action": 0,
                    "previous_reward": 0.0,
                    "history": [(frame, 0, 0.0) for frame in frames[:-1]],
                    "_cached_revision": -1,
                }
                values, _, _ = trainer._observe_actor(actor)
                return float(values.max())
            finally:
                trainer.config = original

    def _frozen_forward(self, *, audit):
        """Real controller/fork APIs, artificial fixed-phase inputs only.

        Direct state restoration is a declared benchmarking fixture, not a
        causally collected history. Its clone/assignment cost is included.
        There are no unreported neural warm-up forwards or simulator calls.
        """
        controller = self.inference_controller
        controller._hidden = self.inference_hidden.detach().clone()
        controller._observations = self.inference_phase
        before = controller.fork() if audit else None
        decision = controller.act(self.inference_frame, 0, 0.0, self.inference_mask)
        operation = "shared_after_audit_step" if audit else "frozen_control_step"
        if not audit:
            self.inference_operations[operation] += 1
            return float(decision["q_mixed"].max())

        # These are the evaluator's same-history counterfactual branches.
        reset = before.fork().act(
            self.inference_frame, 0, 0.0, self.inference_mask, gru_reset=True
        )
        hidden_pixels = before.act(
            self.inference_frame, 0, 0.0, self.inference_mask, pixels_hidden=True
        )
        # Conservative cost only: all THREE actual hypothetical ten-action
        # plans are audited even though cold calibration keeps control mix=0.
        # These raw values never replace a controller decision or its gate.
        raw_plans = [
            self.raw_model_values(controller, item, self.inference_batch_mask)
            for item in (decision, reset, hidden_pixels)
        ]
        action = decision["action"]
        action_tensor = self.torch.tensor([action], dtype=self.torch.long)
        predicted, reward, done_logit = self.inference_network.model(
            decision["hidden"], action_tensor
        )
        predicted, reward, done_logit = (
            value.detach().clone() for value in (predicted, reward, done_logit)
        )
        next_pixels, _, next_reward, _ = controller._inputs(
            self.inference_next_frame, action, 0.25, self.inference_mask
        )
        _, target_hidden, _ = self.inference_network.step(
            next_pixels, action_tensor, next_reward, decision["hidden"]
        )
        # Audit arithmetic is included, using synthetic reward/terminal inputs.
        # Continuous memory is used even if control would reset next time.
        model_error = float((predicted - target_hidden).square().mean().item())
        copy_error = float((decision["hidden"] - target_hidden).square().mean().item())
        reward_error = abs(float(reward[0].item()) - 0.25)
        terminal_error = float(self.torch.sigmoid(done_logit)[0].item()) ** 2
        plan_checksums = sum(float(values.max()) for values in raw_plans)
        self.inference_operations[operation] += 1
        return model_error + copy_error + reward_error + terminal_error + plan_checksums

    def update(self, world):
        return self.trainer.learn_batch(world=world)

    def state(self):
        t = self.trainer
        if any(
            (
                t.global_steps,
                t.fit_steps,
                t.calibration_steps,
                self.forbidden_environment_calls,
            )
        ):
            raise RuntimeError("synthetic profile advanced an environment counter")
        return {
            "global_environment_training_steps": t.global_steps,
            "fit_environment_steps": t.fit_steps,
            "calibration_environment_steps": t.calibration_steps,
            "forbidden_environment_calls": self.forbidden_environment_calls,
            "q_updates": t.q_updates,
            "world_updates": t.world_updates,
            "learner_config": asdict(t.config),
            "teacher_is_frozen": all(
                not p.requires_grad for p in t.source_teacher.parameters()
            ),
            "teacher_parameters_unchanged": self.teacher_before
            == self.fingerprint(t.source_teacher),
            "teacher_source": "own newly initialized network; not an officially trained model",
            "inference_network_source": "separate newly initialized frozen network; never optimized or loaded",
            "inference_parameters_unchanged": self.inference_before
            == self.fingerprint(self.inference_network),
            "inference_is_frozen": not self.inference_network.training
            and all(not p.requires_grad for p in self.inference_network.parameters()),
            "inference_setup_kernel_counts": self.setup_kernel_counts,
            "inference_operations": dict(self.inference_operations),
            "inference_fixture": {
                "observation_phase": self.inference_phase,
                "hidden": "zero construction state restored inside each measured call",
                "neural_warmup_forwards": 0,
                "calibration": "cold copied state; no validation or gate PASS claimed",
                "control_step": "one real act with cold gate; standalone plan_values_all10 separately measures potential lookahead cost",
                "shared_audit_step": "two real forks, three real acts, three explicit ten-action raw plans, chosen-action forecast, continuous GRU next target and audit arithmetic",
                "shared_audit_cost_scope": "conservative hypothetical planning cost even with cold gate off; no game stepping, output row serialization or episode-boundary fingerprint IO",
            },
            "replay": {
                name: {
                    "transitions": len(replay),
                    "episodes": len(replay._episodes),
                    "games": replay.game_counts(),
                    "stored_frame_bytes": sum(
                        f.nbytes for e in replay._episodes for f in e.frames
                    ),
                }
                for name, replay in (
                    ("fit", t.replay),
                    ("protected_source", t.source_replay),
                    ("calibration", t.calibration_replay),
                )
            },
            "synthetic_episode_length": self.episode_length,
            "instrumented_kernel_counts": {
                k: dict(v) for k, v in self.kernel_counts.items()
            },
        }

    def close(self):
        try:
            for hook in self.hooks:
                hook.remove()
        finally:
            self.module.make_game = self.original_make_game


def profile(output, config, *, run=False, backend_factory=None):
    started, cpu_started = time.perf_counter(), time.process_time()
    out = new_output(output)
    source_before = None
    operations = {
        name: {
            "attempted": 0,
            "completed": 0,
            "wall_seconds": [],
            "process_cpu_seconds": [],
        }
        for name in ("q_update", "world_update", *FORWARD_OPERATIONS)
    }
    result = {
        "schema_version": 1,
        "kind": "SYNTHETIC_COST_PROFILE_ONLY",
        "status": "NOT_EXECUTED_SYNTHETIC_PLAN",
        "actual_measurements": False,
        "injected_backend_fixture": backend_factory is not None,
        "config": asdict(config),
        "synthetic_seed": SEED,
        "source": source_before,
        "runtime": {
            "ml_runtime_loaded_by_profile": False,
            "metadata_unavailable": True,
        },
        "operations": operations,
        "failure": None,
        "new_game_episodes": 0,
        "new_environment_interactions": 0,
        "official_models_loaded": False,
        "scientific_conclusion": "NOT_ASSESSED",
        "historical_G1_measurements_reused": False,
        "scientific_budget_projection": None,
        "limits": {
            "synthetic_updates": MAX_UPDATES,
            "q_updates": 50,
            "world_updates": 50,
            "standalone_forward_operations": MAX_FORWARDS,
            "cpu_threads": 2,
            "wall_seconds": MAX_WALL_SECONDS,
            "process_cpu_seconds": MAX_CPU_SECONDS,
            "process_peak_rss_bytes": MAX_RSS_BYTES,
        },
        "counting_scope": "At most 100 standalone operations TOTAL, round-robin over five kernels; these are separate from the NN forwards necessarily executed inside <=100 updates. Nested encoder/GRU/transition/Q-head calls and batch items are reported separately; instrumentation overhead is included.",
        "forward_scope": "Uncached actor prefixes8/16, ten-action planning, actual frozen controller step, and actual same-history shared audit with three conservatively forced raw plans. New frozen inference weights are separate from optimizer weights. No plan timing or cold gate is evidence of calibration PASS.",
        "forward_round_robin": list(FORWARD_OPERATIONS),
        "limitations": [
            "Random RGB/rewards are not gameplay or learned skills.",
            "Synthetic timings do not guarantee scientific runtime, RSS or competence.",
            "Full-capacity replay, environment physics, evaluation, checkpoint IO and remote hardware are not measured.",
            "Resource limits are cooperative checks after setup, around operations and during finalization; they cannot preempt blocked C imports/kernels or IO. The outer CI/process timeout is the hard watchdog.",
            "Plan imports no ML runtime and creates no neural network. --run is explicit opt-in.",
        ],
    }
    backend, exit_code, peak_rss = None, 0, None

    def failed(exc, *, field=None, kind=None):
        nonlocal exit_code
        detail = {"type": kind or type(exc).__name__, "message": str(exc)[:500]}
        if field is not None:
            result[field] = detail
        result["failure"] = result["failure"] or detail
        result["status"] = "STOPPED_SYNTHETIC_PROFILE_NOT_LEARNING_EVIDENCE"
        if isinstance(exc, KeyboardInterrupt):
            exit_code = 130
        elif exit_code != 130:
            exit_code = 2

    def bounds():
        nonlocal peak_rss
        if backend is not None:
            rss = backend.read_peak_rss()
            if type(rss) is not int or rss <= 0:
                raise RuntimeError(
                    "profile RSS measurement must be a positive byte count"
                )
            peak_rss = rss if peak_rss is None else max(peak_rss, rss)
        if (
            time.perf_counter() - started > MAX_WALL_SECONDS
            or time.process_time() - cpu_started > MAX_CPU_SECONDS
            or peak_rss is not None
            and peak_rss > MAX_RSS_BYTES
        ):
            raise RuntimeError("bounded synthetic profile resource limit reached")

    try:
        try:
            source_before = provenance()
            result["source"] = source_before
        except BaseException as exc:
            failed(exc, field="initial_provenance_failure")
            raise
        try:
            result["runtime"] = metadata_runtime()
        except BaseException as exc:
            failed(exc, field="runtime_metadata_failure")
            raise
        bounds()
        if run:
            result["status"] = (
                "TEST_ONLY_SYNTHETIC_PROFILE_NOT_LEARNING_EVIDENCE"
                if backend_factory is None
                else "TEST_ONLY_INJECTED_PROFILE_FIXTURE_NOT_MEASUREMENT"
            )
            backend = (backend_factory or SyntheticBackend)(config)
            result["runtime"] = backend.runtime
            # Even a zero-operation request imports and allocates this backend.
            bounds()
            result["synthetic_state_before"] = backend.state()
            bounds()

            def measure(name, callback):
                bounds()
                op = operations[name]
                op["attempted"] += 1
                wall, cpu = time.perf_counter(), time.process_time()
                before_kernels = {
                    k: dict(v) for k, v in getattr(backend, "kernel_counts", {}).items()
                }
                try:
                    value = callback()
                except BaseException:
                    op.setdefault("failed_attempt_timings", []).append(
                        {
                            "wall_seconds": time.perf_counter() - wall,
                            "process_cpu_seconds": time.process_time() - cpu,
                        }
                    )
                    raise
                finally:
                    for key, after in getattr(backend, "kernel_counts", {}).items():
                        entry = op.setdefault(
                            "instrumented_kernel_counts", {}
                        ).setdefault(key, {"calls": 0, "items": 0})
                        for metric in ("calls", "items"):
                            entry[metric] += after[metric] - before_kernels.get(
                                key, {}
                            ).get(metric, 0)
                elapsed, consumed = (
                    time.perf_counter() - wall,
                    time.process_time() - cpu,
                )
                if not math.isfinite(float(value)):
                    raise ValueError(
                        "profile operation returned nonfinite synthetic data"
                    )
                op["wall_seconds"].append(elapsed)
                op["process_cpu_seconds"].append(consumed)
                op["completed"] += 1
                bounds()

            for i in range(config.forwards):
                name = FORWARD_OPERATIONS[i % len(FORWARD_OPERATIONS)]
                measure(name, lambda name=name: backend.forward(name))
            for i in range(config.updates):
                world = bool(i % 2)
                measure(
                    "world_update" if world else "q_update",
                    lambda world=world: backend.update(world),
                )
            result["synthetic_state_after"] = backend.state()
            counts = result["synthetic_state_after"]
            if (
                counts["q_updates"] != config.updates // 2
                or counts["world_updates"] != config.updates // 2
            ):
                raise RuntimeError(
                    "synthetic optimizer update counts differ from the explicit request"
                )
            if not counts["teacher_parameters_unchanged"]:
                raise RuntimeError("protected synthetic teacher changed")
            if counts.get("inference_parameters_unchanged") is False:
                raise RuntimeError("separate frozen inference network changed")
            result["actual_measurements"] = backend_factory is None
    except BaseException as exc:
        failed(exc)
    finally:
        if backend is not None:
            try:
                result["synthetic_state_after"] = backend.state()
            except BaseException as exc:
                failed(exc, field="finalization_failure")
            finally:
                try:
                    backend.close()
                except BaseException as exc:
                    failed(exc, field="cleanup_failure")
        result["source_after"] = None
        try:
            result["source_after"] = provenance()
            if (
                source_before is not None
                and result["source_after"]["source_files_sha256"]
                != source_before["source_files_sha256"]
            ):
                failed(
                    RuntimeError("profile source files changed during measurement"),
                    field="source_change_failure",
                    kind="SourceChanged",
                )
        except BaseException as exc:
            failed(exc, field="final_provenance_failure")
        for op in operations.values():
            op["wall_summary"] = summary(op["wall_seconds"])
            op["process_cpu_summary"] = summary(op["process_cpu_seconds"])
        try:
            bounds()
        except BaseException as exc:
            failed(exc, field="final_resource_failure")
        result["elapsed_wall_seconds"] = time.perf_counter() - started
        result["elapsed_process_cpu_seconds"] = time.process_time() - cpu_started
        result["process_peak_rss_bytes"] = peak_rss
        result["memory_scope"] = (
            "OS process peak RSS helper, not current RSS; setup/import/data allocation and cleanup/provenance finalization are included in timing and peak. Receipt/manifest sealing is bounded only by the outer process watchdog."
        )
        path = out / "profile.json"
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        # If serialization/initial write crossed a cooperative limit, correct
        # only this own new file before publishing its checksum manifest.
        # No measured operation is retried or any earlier output overwritten.
        try:
            bounds()
        except BaseException as exc:
            failed(exc, field="receipt_sealing_resource_failure")
            result["elapsed_wall_seconds"] = time.perf_counter() - started
            result["elapsed_process_cpu_seconds"] = time.process_time() - cpu_started
            result["process_peak_rss_bytes"] = peak_rss
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
        (out / "SHA256SUMS").write_text(
            f"{sha(path)}  profile.json\n", encoding="utf-8"
        )
    return result, exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--plan", action="store_true", help="default: write a plan without loading ML"
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="explicit opt-in to bounded synthetic-only profiling",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--forwards", type=int, default=100)
    parser.add_argument("--context-length", type=int, choices=(8, 16), default=16)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    args = parser.parse_args(argv)
    try:
        config = ProfileConfig(
            args.updates, args.forwards, args.context_length, args.threads
        )
        result, code = profile(args.output, config, run=args.run)
    except (ValueError, FileExistsError, OSError) as exc:
        parser.exit(2, f"synthetic profile rejected: {exc}\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "actual_measurements": result["actual_measurements"],
                "profile_path": str(args.output / "profile.json"),
                "completed_operations": {
                    k: v["completed"] for k, v in result["operations"].items()
                },
                "failure": result["failure"],
            }
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
