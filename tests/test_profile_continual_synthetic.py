"""CLI/provenance checks and ONE four-update artificial-pixel smoke fixture."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts.profile_continual_synthetic import ProfileConfig, profile, summary
import scripts.profile_continual_synthetic as profile_module


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/profile_continual_synthetic.py"


def cli(tmp_path, *args, env=None):
    output = tmp_path / "new-profile"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=45,
        env=env,
    )
    return result, output


def test_default_plan_never_imports_ml_or_constructs_a_backend(tmp_path):
    def forbidden(_):
        raise AssertionError("plan constructed the synthetic learner")

    result, code = profile(
        tmp_path / "plan", ProfileConfig(), backend_factory=forbidden
    )
    assert code == 0 and result["status"] == "NOT_EXECUTED_SYNTHETIC_PLAN"
    assert result["actual_measurements"] is False
    assert result["runtime"]["ml_runtime_loaded_by_profile"] is False
    assert "synthetic_state_before" not in result
    assert all(
        op["completed"] == op["attempted"] == 0 for op in result["operations"].values()
    )
    digest, member = (tmp_path / "plan/SHA256SUMS").read_text().split()
    assert member == "profile.json"
    assert (
        hashlib.sha256((tmp_path / "plan/profile.json").read_bytes()).hexdigest()
        == digest
    )
    assert result["scientific_budget_projection"] is None


def test_plan_cli_cannot_import_a_shadow_ml_runtime(tmp_path):
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    for name in ("torch", "numpy", "gymnasium"):
        (shadow / (name + ".py")).write_text(
            'raise AssertionError("ML import forbidden in plan")\n'
        )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(shadow)
    result, out = cli(tmp_path, "--plan", env=env)
    assert result.returncode == 0, result.stderr
    report = json.loads((out / "profile.json").read_text(encoding="utf-8"))
    assert report["config"] == {
        "updates": 100,
        "forwards": 100,
        "context_length": 16,
        "threads": 2,
    }
    assert report["synthetic_seed"] == 830003
    assert report["source"]["git_sha"]
    assert all(value for value in report["source"]["source_files_sha256"].values())


@pytest.mark.parametrize(
    "args",
    [
        ("--updates", "101"),
        ("--updates", "1"),
        ("--updates", "-2"),
        ("--forwards", "101"),
        ("--forwards", "-1"),
        ("--threads", "3"),
        ("--context-length", "32"),
        ("--plan", "--run"),
        ("--seed", "830004"),
    ],
)
def test_cli_bounds_reject_before_any_output_or_model(tmp_path, args):
    result, out = cli(tmp_path, "--run", *args)
    assert result.returncode == 2
    assert not out.exists()


def test_existing_output_is_preserved_and_never_reused(tmp_path):
    out = tmp_path / "old"
    out.mkdir()
    (out / "original.txt").write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        profile(out, ProfileConfig())
    assert (out / "original.txt").read_bytes() == b"keep"
    assert not (out / "profile.json").exists()


def test_timing_summary_uses_actual_samples_and_explicit_missing_values():
    assert summary([])["p95_seconds"] is None
    assert summary([1.0, 2.0, 3.0, 4.0]) == {
        "count": 4,
        "total_seconds": 10.0,
        "p50_seconds": 2.5,
        "p95_seconds": pytest.approx(3.85),
        "maximum_seconds": 4.0,
    }
    with pytest.raises(ValueError):
        summary([float("nan")])


def test_runtime_failure_preserves_new_stopped_receipt_without_retry(tmp_path):
    calls = []

    def broken(_):
        calls.append(1)
        raise RuntimeError("synthetic constructor fixture failure")

    report, code = profile(
        tmp_path / "failed",
        ProfileConfig(updates=4, forwards=4),
        run=True,
        backend_factory=broken,
    )
    assert code == 2 and calls == [1]
    assert report["status"] == "STOPPED_SYNTHETIC_PROFILE_NOT_LEARNING_EVIDENCE"
    assert report["failure"]["message"] == "synthetic constructor fixture failure"
    assert (tmp_path / "failed/profile.json").is_file()
    assert (tmp_path / "failed/SHA256SUMS").is_file()


def test_state_and_cleanup_failures_still_preserve_stopped_receipt(tmp_path):
    class BrokenState:
        runtime = {"fixture": True}

        def __init__(self, config):
            self.closed = False

        def state(self):
            raise RuntimeError("fixture state unavailable")

        def read_peak_rss(self):
            return 1

        def close(self):
            self.closed = True
            raise RuntimeError("fixture cleanup failed")

    instance = BrokenState(None)
    report, code = profile(
        tmp_path / "state-failure",
        ProfileConfig(updates=0, forwards=0),
        run=True,
        backend_factory=lambda config: instance,
    )
    assert code == 2 and instance.closed
    assert report["failure"]["message"] == "fixture state unavailable"
    assert report["cleanup_failure"]["message"] == "fixture cleanup failed"
    assert report["status"] == "STOPPED_SYNTHETIC_PROFILE_NOT_LEARNING_EVIDENCE"
    assert (tmp_path / "state-failure/profile.json").is_file()
    assert (tmp_path / "state-failure/SHA256SUMS").is_file()


def test_injected_backend_never_claims_actual_profile_measurements(tmp_path):
    class Fixture:
        runtime = {"fixture": True}

        def __init__(self, config):
            pass

        def read_peak_rss(self):
            return 1

        def state(self):
            return {
                "q_updates": 0,
                "world_updates": 0,
                "teacher_parameters_unchanged": True,
            }

        def close(self):
            pass

    report, code = profile(
        tmp_path / "injected",
        ProfileConfig(updates=0, forwards=0),
        run=True,
        backend_factory=Fixture,
    )
    assert code == 0 and report["actual_measurements"] is False
    assert report["status"] == "TEST_ONLY_INJECTED_PROFILE_FIXTURE_NOT_MEASUREMENT"
    assert report["injected_backend_fixture"] is True


@pytest.fixture
def fake_profile_runtime(monkeypatch):
    """No ML, environment, clock wait or source hashing is involved."""
    clock = SimpleNamespace(wall=0.0, cpu=0.0, rss=1, events=[])
    monkeypatch.setattr(
        profile_module,
        "time",
        SimpleNamespace(
            perf_counter=lambda: clock.wall, process_time=lambda: clock.cpu
        ),
    )
    monkeypatch.setattr(
        profile_module,
        "provenance",
        lambda: {"source_files_sha256": {"fixture.py": "unchanged"}},
    )
    monkeypatch.setattr(profile_module, "metadata_runtime", lambda: {"fixture": True})

    class FakeBackend:
        runtime = {"fixture": True}

        def __init__(self, config):
            clock.events.append("setup")

        def read_peak_rss(self):
            return clock.rss

        def state(self):
            clock.events.append("state")
            return {
                "q_updates": 0,
                "world_updates": 0,
                "teacher_parameters_unchanged": True,
            }

        def forward(self, name):
            raise AssertionError("zero-operation fixture attempted a forward")

        def update(self, world):
            raise AssertionError("zero-operation fixture attempted an update")

        def close(self):
            clock.events.append("close")

    return clock, FakeBackend


def assert_stopped_receipt(output, report):
    saved = json.loads((output / "profile.json").read_text(encoding="utf-8"))
    assert saved == report
    assert saved["status"] == "STOPPED_SYNTHETIC_PROFILE_NOT_LEARNING_EVIDENCE"
    assert saved["actual_measurements"] is False
    digest, member = (output / "SHA256SUMS").read_text().split()
    assert member == "profile.json"
    assert hashlib.sha256((output / member).read_bytes()).hexdigest() == digest
    return saved


@pytest.mark.parametrize(
    ("resource", "value"),
    [
        ("wall", profile_module.MAX_WALL_SECONDS + 1),
        ("cpu", profile_module.MAX_CPU_SECONDS + 1),
        ("rss", profile_module.MAX_RSS_BYTES + 1),
    ],
)
def test_zero_operations_still_enforce_backend_setup_limits(
    tmp_path, fake_profile_runtime, resource, value
):
    clock, FakeBackend = fake_profile_runtime

    class ExpensiveSetup(FakeBackend):
        def __init__(self, config):
            super().__init__(config)
            setattr(clock, resource, value)

    output = tmp_path / "setup-limit"
    report, code = profile(
        output,
        ProfileConfig(updates=0, forwards=0),
        run=True,
        backend_factory=ExpensiveSetup,
    )
    assert code == 2 and clock.events == ["setup", "state", "close"]
    saved = assert_stopped_receipt(output, report)
    assert "resource limit reached" in saved["failure"]["message"]
    assert all(op["attempted"] == 0 for op in saved["operations"].values())
    assert saved["elapsed_wall_seconds"] == clock.wall
    assert saved["elapsed_process_cpu_seconds"] == clock.cpu
    assert saved["process_peak_rss_bytes"] == clock.rss


@pytest.mark.parametrize("phase", ["cleanup", "final_provenance"])
def test_finalization_time_is_included_and_can_stop_zero_operation_profile(
    tmp_path, monkeypatch, fake_profile_runtime, phase
):
    clock, FakeBackend = fake_profile_runtime

    class ExpensiveCleanup(FakeBackend):
        def close(self):
            super().close()
            if phase == "cleanup":
                clock.wall = profile_module.MAX_WALL_SECONDS + 1

    calls = []

    def provenance_fixture():
        calls.append(1)
        if len(calls) == 2 and phase == "final_provenance":
            clock.wall = profile_module.MAX_WALL_SECONDS + 1
        return {"source_files_sha256": {"fixture.py": "unchanged"}}

    monkeypatch.setattr(profile_module, "provenance", provenance_fixture)
    output = tmp_path / "finalization-limit"
    report, code = profile(
        output,
        ProfileConfig(updates=0, forwards=0),
        run=True,
        backend_factory=ExpensiveCleanup,
    )
    assert code == 2 and calls == [1, 1]
    saved = assert_stopped_receipt(output, report)
    assert saved["final_resource_failure"]["type"] == "RuntimeError"
    assert saved["elapsed_wall_seconds"] == profile_module.MAX_WALL_SECONDS + 1
    assert clock.events[-1] == "close"


@pytest.mark.parametrize("phase", ["initial_provenance", "runtime_metadata"])
@pytest.mark.parametrize("final_unavailable", [False, True])
def test_setup_metadata_failures_preserve_stop_receipt_without_constructing_backend(
    tmp_path, monkeypatch, fake_profile_runtime, phase, final_unavailable
):
    clock, FakeBackend = fake_profile_runtime
    calls = []

    def provenance_fixture():
        calls.append(1)
        if phase == "initial_provenance" and len(calls) == 1:
            raise OSError("initial provenance unavailable")
        if final_unavailable and len(calls) == 2:
            raise OSError("final provenance unavailable")
        return {"source_files_sha256": {"fixture.py": "unchanged"}}

    def metadata_fixture():
        if phase == "runtime_metadata":
            raise RuntimeError("runtime metadata unavailable")
        return {"fixture": True}

    monkeypatch.setattr(profile_module, "provenance", provenance_fixture)
    monkeypatch.setattr(profile_module, "metadata_runtime", metadata_fixture)
    output = tmp_path / "metadata-failure"
    report, code = profile(
        output,
        ProfileConfig(updates=0, forwards=0),
        run=True,
        backend_factory=FakeBackend,
    )
    assert code == 2 and clock.events == []
    saved = assert_stopped_receipt(output, report)
    assert saved[f"{phase}_failure"] == saved["failure"]
    if final_unavailable:
        assert saved["source_after"] is None
        assert (
            saved["final_provenance_failure"]["message"]
            == "final provenance unavailable"
        )
    else:
        assert saved["source_after"]["source_files_sha256"]
    if phase == "initial_provenance":
        assert saved["source"] is None


@pytest.mark.parametrize("original_failure", [False, True])
def test_final_provenance_failure_has_stop_receipt_and_preserves_first_failure(
    tmp_path, monkeypatch, fake_profile_runtime, original_failure
):
    clock, FakeBackend = fake_profile_runtime
    calls = []

    def provenance_fixture():
        calls.append(1)
        if len(calls) == 2:
            raise OSError("final provenance unavailable")
        return {"source_files_sha256": {"fixture.py": "unchanged"}}

    def backend_factory(config):
        if original_failure:
            raise RuntimeError("original setup failure")
        return FakeBackend(config)

    monkeypatch.setattr(profile_module, "provenance", provenance_fixture)
    output = tmp_path / "final-provenance-failure"
    report, code = profile(
        output,
        ProfileConfig(updates=0, forwards=0),
        run=True,
        backend_factory=backend_factory,
    )
    assert code == 2
    saved = assert_stopped_receipt(output, report)
    assert (
        saved["final_provenance_failure"]["message"] == "final provenance unavailable"
    )
    assert saved["failure"]["message"] == (
        "original setup failure" if original_failure else "final provenance unavailable"
    )
    assert saved["source_after"] is None
    assert calls == [1, 1]


def test_source_change_does_not_overwrite_original_failure(
    tmp_path, monkeypatch, fake_profile_runtime
):
    calls = []

    def provenance_fixture():
        calls.append(1)
        return {"source_files_sha256": {"fixture.py": str(len(calls))}}

    def broken_backend(config):
        raise RuntimeError("original setup failure")

    monkeypatch.setattr(profile_module, "provenance", provenance_fixture)
    output = tmp_path / "changed-source"
    report, code = profile(
        output,
        ProfileConfig(updates=0, forwards=0),
        run=True,
        backend_factory=broken_backend,
    )
    assert code == 2
    saved = assert_stopped_receipt(output, report)
    assert saved["failure"]["message"] == "original setup failure"
    assert saved["source_change_failure"]["type"] == "SourceChanged"


def test_receipt_write_crossing_limit_is_corrected_before_manifest(
    tmp_path, monkeypatch, fake_profile_runtime
):
    clock, FakeBackend = fake_profile_runtime
    dump = profile_module.json.dump
    writes = []

    def slow_dump(*args, **kwargs):
        writes.append(1)
        dump(*args, **kwargs)
        clock.wall = profile_module.MAX_WALL_SECONDS + 1

    monkeypatch.setattr(profile_module.json, "dump", slow_dump)
    output = tmp_path / "receipt-limit"
    report, code = profile(
        output,
        ProfileConfig(updates=0, forwards=0),
        run=True,
        backend_factory=FakeBackend,
    )
    assert code == 2 and writes == [1, 1]
    saved = assert_stopped_receipt(output, report)
    assert saved["receipt_sealing_resource_failure"]["type"] == "RuntimeError"
    assert saved["elapsed_wall_seconds"] == profile_module.MAX_WALL_SECONDS + 1


def test_four_update_opt_in_profiles_only_artificial_replay_and_fresh_weights(tmp_path):
    result, out = cli(
        tmp_path, "--run", "--updates", "4", "--forwards", "5", "--threads", "2"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((out / "profile.json").read_text(encoding="utf-8"))
    assert report["status"] == "TEST_ONLY_SYNTHETIC_PROFILE_NOT_LEARNING_EVIDENCE"
    assert report["actual_measurements"] is True and report["failure"] is None
    counts = {name: op["completed"] for name, op in report["operations"].items()}
    assert counts == {
        "q_update": 2,
        "world_update": 2,
        "actor_uncached_prefix8": 1,
        "actor_uncached_prefix16": 1,
        "plan_values_all10": 1,
        "frozen_control_step": 1,
        "shared_after_audit_step": 1,
    }
    state = report["synthetic_state_after"]
    assert state["q_updates"] == state["world_updates"] == 2
    assert (
        state["global_environment_training_steps"]
        == state["forbidden_environment_calls"]
        == 0
    )
    assert state["fit_environment_steps"] == state["calibration_environment_steps"] == 0
    assert state["teacher_is_frozen"] and state["teacher_parameters_unchanged"]
    assert state["inference_is_frozen"] and state["inference_parameters_unchanged"]
    assert state["inference_operations"] == {
        "frozen_control_step": 1,
        "shared_after_audit_step": 1,
    }
    assert all(
        item["calls"] == item["items"] == 0
        for item in state["inference_setup_kernel_counts"].values()
    )
    assert (
        report["operations"]["shared_after_audit_step"]["instrumented_kernel_counts"][
            "frozen_inference.encoder"
        ]["calls"]
        == 4
    )
    assert state["replay"]["fit"]["transitions"] == 2640
    assert state["replay"]["protected_source"]["transitions"] == 2112
    assert state["replay"]["calibration"]["transitions"] == 1056
    assert state["synthetic_episode_length"] == 33
    assert state["instrumented_kernel_counts"]["source_teacher.encoder"]["calls"] > 0
    assert state["instrumented_kernel_counts"]["target.encoder"]["calls"] > 0
    assert state["instrumented_kernel_counts"]["online.transition"]["items"] > 10
    assert (
        report["runtime"]["intraop_threads"]
        == report["runtime"]["interop_threads"]
        == 2
    )
    assert (
        report["runtime"]["device"] == "cpu"
        and not report["runtime"]["cuda_initialized"]
    )
    assert report["process_peak_rss_bytes"] > 0
    assert (
        report["source"]["source_files_sha256"]
        == report["source_after"]["source_files_sha256"]
    )
    digest, member = (out / "SHA256SUMS").read_text().split()
    assert (
        member == "profile.json"
        and hashlib.sha256((out / member).read_bytes()).hexdigest() == digest
    )
    assert report["official_models_loaded"] is False
    assert report["new_environment_interactions"] == report["new_game_episodes"] == 0
