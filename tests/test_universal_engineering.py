"""Fake-clock/data tests: zero games, real-environment learning or benchmarks."""
import hashlib
import json
import copy

import pytest

from arena.universal import engineering as eng


def source():
    return {"source_commit": "a" * 40, "protocol_sha256": "b" * 64,
            "relevant_sources_clean": True, "protocol_version": "unit-fixture-only",
            "architecture_config": {"batch_size": 16, "burn_in": 4, "unroll": 8},
            "source_files_sha256": {**eng.RECOVERABLE_SOURCE_FILES_SHA256,
                                    "unit-fixture.py": "c" * 64}}


class Clock:
    def __init__(self, increment):
        self.increment, self.value = increment, 0.

    def __call__(self):
        self.value += self.increment
        return self.value


def memory():
    return {"rss_bytes": 120 * 1024**2, "peak_rss_bytes": 160 * 1024**2,
            "source": "FAKE_UNIT_TEST_NOT_EVIDENCE"}


class Backend:
    def __init__(self, config):
        self.config = config
        self.mechanics = []
        self.q = self.world = 0
        self.synthetic = False
        self.closed = False

    def fingerprint(self):
        return "d" * 64 if not self.q + self.world else "e" * 64

    def mechanical_step(self, game):
        assert game in eng.GAMES
        assert not self.synthetic
        self.mechanics.append(game)

    def prepare_synthetic(self):
        self.synthetic = True

    def update(self, *, world):
        assert self.synthetic
        if world:
            self.world += 1
        else:
            self.q += 1
        return .1

    def state_receipt(self):
        return {"global_environment_training_steps": 0, "training_curve_records": 0,
                "synthetic_replay_only": True, "replay_transitions": 512,
                "q_updates": self.q, "world_updates": self.world,
                "mechanical_steps": len(self.mechanics),
                "learner_config": {"replay_capacity": 40000}, "cpu_threads": 2}

    def close(self):
        self.closed = True


def timing(wall=.001, cpu=.002):
    return {operation: {"wall": eng.timing_summary([wall] * 10),
                        "process_cpu": eng.timing_summary([cpu] * 10)}
            for operation in ("step", "q", "world")}


def test_ledger_and_seed_bounds_cannot_spend_the_earlier_525_steps_again():
    config = eng.EngineeringConfig()
    assert config.mechanical_steps == 475
    assert config.synthetic_updates == 100 and config.seed == 810200
    assert eng.PREVIOUS_RANDOM_SMOKE_STEPS + eng.PREVIOUS_EVALUATOR_UNIT_STEPS + config.mechanical_steps == 1000
    for arguments in ({"mechanical_steps": 500}, {"mechanical_steps": 476},
                      {"mechanical_steps": 0}, {"synthetic_updates": 102},
                      {"synthetic_updates": 3}, {"seed": 920001}):
        with pytest.raises(ValueError):
            eng.EngineeringConfig(**arguments)


def test_percentiles_and_fixed_projection_use_20_percent_margin_and_all_work():
    assert eng.timing_summary([0., 10.])["p50_seconds"] == 5
    assert eng.timing_summary([0., 10.])["p95_seconds"] == 9.5
    values = timing()
    values["q"]["wall"] = eng.timing_summary([.010] * 10)
    values["world"]["wall"] = eng.timing_summary([.020] * 10)
    result = eng.project_campaign(values, observed_peak_rss_bytes=160 * 1024**2,
                                  synthetic_replay_transitions=512)
    expected = 1.2 * ((1_080_000 + 694_000) * .001 + 67_500 * .010 + 33_750 * .020)
    assert result["wall_seconds_with_margin"] == pytest.approx(expected)
    assert result["projected_step_calls"] == 1_774_000
    assert result["additional_replay_bytes_allowance"] == (40000 - 512) * 16 * 1024
    assert result["estimated_rss_bytes_with_margin"] >= 1.2 * (160 * 1024**2 + (40000 - 512) * 16 * 1024)
    assert result["within_projected_limits"] is True
    assert eng.project_campaign(timing(wall=1, cpu=1), observed_peak_rss_bytes=100,
                                synthetic_replay_transitions=0)["within_projected_limits"] is False
    assert eng.project_campaign(timing(), observed_peak_rss_bytes=3 * 1024**3,
                                synthetic_replay_transitions=0)["within_projected_limits"] is False


def test_projection_never_accepts_missing_zero_or_fabricated_measurement_values():
    for number in (0, -1, float("nan"), float("inf")):
        values = timing()
        values["q"]["wall"]["p95_seconds"] = number
        with pytest.raises(ValueError, match="positive p95"):
            eng.project_campaign(values, observed_peak_rss_bytes=1000,
                                 synthetic_replay_transitions=1)
    with pytest.raises(ValueError):
        eng.project_campaign(timing(), observed_peak_rss_bytes=0, synthetic_replay_transitions=1)
    with pytest.raises(ValueError):
        eng.timing_summary([])


def test_plan_does_not_construct_backend_and_cannot_claim_a_pass(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("plan must not construct games or a learner")
    monkeypatch.setattr(eng, "_RealBackend", forbidden)
    output = tmp_path / "plan"
    result = eng.plan_preflight(output, source_provider=source)
    assert result["status"] == "NOT_EXECUTED"
    assert result["actual_measurements"] is False
    assert result["projection"] is None
    assert result["resume_25_receipt"].startswith("PENDING")
    assert result["local_timing_is_github_runner_timing"] is False
    expected, name = (output / "SHA256SUMS").read_text().split()
    assert name == "receipt.json"
    assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    with pytest.raises(FileExistsError):
        eng.plan_preflight(output, source_provider=source)


def test_fake_clock_backend_respects_counts_but_never_issues_actual_evidence(tmp_path):
    instances = []
    def factory(config):
        backend = Backend(config)
        instances.append(backend)
        return backend
    output = tmp_path / "injected-fixture"
    result = eng.run_preflight(output, eng.EngineeringConfig(10, 4),
                               backend_factory=factory, wall_clock=Clock(.001),
                               cpu_clock=Clock(.002), memory_reader=memory, source_provider=source)
    assert result["status"] == "TEST_ONLY_NOT_EVIDENCE"
    assert result["actual_measurements"] is False
    assert result["mechanical_attempts"] == 10
    assert result["synthetic_update_attempts"] == 4
    assert set(result["mechanical_steps_by_game"].values()) == {2}
    assert result["fixture_state"]["q_updates"] == result["fixture_state"]["world_updates"] == 2
    assert result["untrained_fingerprint_before_mechanics"] == result["untrained_fingerprint_after_mechanics"]
    assert result["fixture_fingerprint_after_synthetic_updates"] != result["untrained_fingerprint_before_mechanics"]
    assert result["scientific_environment_training_steps"] == 0
    assert instances[0].closed
    assert json.loads((output / "receipt.json").read_text())["status"] == "TEST_ONLY_NOT_EVIDENCE"


def test_failed_fixture_is_preserved_without_retry_or_extra_steps(tmp_path):
    class FailingBackend(Backend):
        def mechanical_step(self, game):
            super().mechanical_step(game)
            if len(self.mechanics) == 3:
                raise RuntimeError("unit-fixture stop")
    instances = []
    def factory(config):
        value = FailingBackend(config)
        instances.append(value)
        return value
    result = eng.run_preflight(tmp_path / "partial", eng.EngineeringConfig(10, 4),
                               backend_factory=factory, wall_clock=Clock(.001),
                               cpu_clock=Clock(.002), memory_reader=memory, source_provider=source)
    assert result["status"] == "FAILED"
    assert result["mechanical_attempts"] == 3 and result["synthetic_update_attempts"] == 0
    assert result["failure"]["message"] == "unit-fixture stop"
    assert len(instances) == 1 and instances[0].closed


def linux_counter_records():
    provider = "Linux /proc/self/statm and getrusage(RUSAGE_SELF), KiB converted to bytes"
    return [{"peak_rss_bytes": peak, "rss_bytes": rss, "source": provider}
            for peak, rss in ((22269952, 21999616), (376401920, 376606720),
                              (404586496, 404738048), (449413120, 445739008))]


def test_linux_approximate_counter_inversions_retain_raw_values_and_global_peak():
    records = linux_counter_records()
    original = copy.deepcopy(records)
    summary = eng.summarize_memory(records)
    assert records == original == summary["observations"]
    assert summary["conservative_peak_rss_bytes"] == 449413120
    assert summary["actual_process_peak_rss_bytes"] == 449413120
    assert summary["observed_current_rss_maximum"] == 445739008
    assert summary["cross_provider_inversions"] == [
        {"record_index": 1, "rss_above_reported_peak_bytes": 204800},
        {"record_index": 2, "rss_above_reported_peak_bytes": 151552}]
    summary["observations"][0]["rss_bytes"] = 1
    assert records == original
    # Also cover a current RSS larger than EVERY reported peak, not just an
    # intermediate reading. It must conservatively govern the projection.
    values = [{"rss_bytes": 200, "peak_rss_bytes": 100, "source": "unit-counters"},
              {"rss_bytes": 150, "peak_rss_bytes": 150, "source": "unit-counters"}]
    result = eng.summarize_memory(values)
    assert result["actual_process_peak_rss_bytes"] == 150
    assert result["conservative_peak_rss_bytes"] == 200


@pytest.mark.parametrize("field,value", [("rss_bytes", 0), ("peak_rss_bytes", 0),
                                        ("rss_bytes", -1), ("peak_rss_bytes", True),
                                        ("rss_bytes", "100")])
def test_memory_correction_never_accepts_nonpositive_or_noninteger_counters(field, value):
    records = linux_counter_records()
    records[1][field] = value
    with pytest.raises(ValueError, match="positive current/peak"):
        eng.summarize_memory(records)


def test_injected_preflight_no_longer_fails_on_valid_linux_cross_provider_inversions(tmp_path):
    records = iter(linux_counter_records())
    result = eng.run_preflight(tmp_path / "linux-inversion-unit-fixture", eng.EngineeringConfig(10, 4),
                               backend_factory=Backend, wall_clock=Clock(.001), cpu_clock=Clock(.002),
                               memory_reader=lambda: next(records), source_provider=source)
    assert result["status"] == "TEST_ONLY_NOT_EVIDENCE"
    assert result["memory"]["conservative_peak_rss_bytes"] == 449413120
    assert result["projection"]["observed_peak_rss_bytes"] == 449413120
    assert "failure" not in result


def complete_failed_fixture():
    """Fabricated timing rows only; no games, neural code or benchmarks run."""
    samples = {operation: {"wall": [.001] * count, "process_cpu": [.002] * count}
               for operation, count in (("step", 475), ("q", 50), ("world", 50))}
    return {
        "unit_test_fake_original": True,
        "schema_version": 1, "kind": "ENGINEERING_PREFLIGHT_NOT_SCIENTIFIC_RESULT",
        "status": "FAILED", "failure": copy.deepcopy(eng.RECOVERABLE_RSS_FAILURE),
        "actual_measurements": True, "no_scientific_checkpoints": True,
        "scientific_environment_training_steps": 0,
        "config": {"mechanical_steps": 475, "synthetic_updates": 100, "seed": 810200},
        "technical_seed": 810200, "synthetic_update_limit": 100,
        "mechanical_attempts": 475, "synthetic_update_attempts": 100,
        "mechanical_steps_by_game": {game: 95 for game in eng.GAMES},
        "mechanical_budget": {"global_limit": 1000, "previous_random_smoke": 500,
                              "previous_evaluator_unit": 25, "requested_here": 475,
                              "maximum_new_steps_here": 475},
        "source": {"source_commit": eng.RECOVERABLE_SOURCE_COMMIT,
                   "protocol_sha256": eng.RECOVERABLE_PROTOCOL_SHA256,
                   "protocol_version": "mg002-plano-v0.1-implementation-2",
                   "protocol_status": "CANDIDATE_NOT_FROZEN", "relevant_sources_clean": True,
                   "dirty_source_records": [], "scientific_execution_authorized_by_this_module": False,
                   "source_files_sha256": copy.deepcopy(eng.RECOVERABLE_SOURCE_FILES_SHA256),
                   "architecture_config": {"observation": [32, 32, 3], "actions": 10,
                                            "gru_units": 64, "batch_size": 16, "burn_in": 4,
                                            "unroll": 8, "gamma": .98, "per_game_heads": False,
                                            "world_weight": .2, "replay_capacity": 40000}},
        "fixture_state": {"learner_config": copy.deepcopy(eng.RECOVERABLE_LEARNER_CONFIG),
                          "cpu_threads": 2, "torch_version": "2.10.0+cpu", "numpy_version": "2.3.5",
                          "global_environment_training_steps": 0, "training_curve_records": 0,
                          "trainer_stage": "unconfigured", "synthetic_replay_only": True,
                          "mechanical_steps": 475, "q_updates": 50, "world_updates": 50,
                          "replay_transitions": 512,
                          "replay_game_counts": {f"synthetic-noise-{i}": 128 for i in range(4)},
                          "mechanical_policy": "uniform random legal inputs; computed network outputs are not selected",
                          "mechanical_first_environment_seeds": {
                              game: 810200 + 1000 * index for index, game in enumerate(eng.GAMES)}},
        "untrained_fingerprint_before_mechanics": "d" * 64,
        "untrained_fingerprint_after_mechanics": "d" * 64,
        "fixture_fingerprint_after_synthetic_updates": "e" * 64,
        "partial_timing_samples": samples,
        "timings": {operation: {clock: eng.timing_summary(values) for clock, values in clocks.items()}
                    for operation, clocks in samples.items()},
        "partial_memory_records": linux_counter_records(),
        "resume_25_receipt": "PENDING_SEPARATE_UNIT_FIXTURE_ONLY",
    }


def authenticated_unit_fixture(tmp_path, monkeypatch, mutate=None):
    value = complete_failed_fixture()
    if mutate is not None:
        mutate(value)
    original = tmp_path / "fake-original-failed-receipt.json"
    original.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    # Production has a literal, independently verified original digest. Only
    # this unit test substitutes its fabricated fixture; no exported PASS exists.
    monkeypatch.setattr(eng, "RECOVERABLE_ORIGINAL_RECEIPT_SHA256", digest)
    return original, digest


def test_offline_recovery_never_remeasures_or_changes_original_status(tmp_path, monkeypatch):
    original, digest = authenticated_unit_fixture(tmp_path, monkeypatch)
    original_bytes = original.read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError("offline correction must not rerun games, updates, timers or RSS measurements")
    monkeypatch.setattr(eng, "_RealBackend", forbidden)
    monkeypatch.setattr(eng, "run_preflight", forbidden)
    monkeypatch.setattr(eng, "process_memory", forbidden)
    output = tmp_path / "derived"
    result = eng.recover_completed_preflight(original, output,
                                             expected_original_sha256=digest, source_provider=source)
    assert original.read_bytes() == original_bytes
    assert json.loads(original.read_text())["status"] == "FAILED"
    assert result["status"] == "DERIVED_FROM_COMPLETE_MEASUREMENTS"
    assert result["actual_measurements"] is False and result["new_measurements"] is False
    assert result["new_mechanical_steps"] == result["new_synthetic_updates"] == 0
    assert result["original_relabelled_as_pass"] is False
    assert result["gate_acceptance"] == "PENDING_EXPLICIT_CHAT1_DECISION_BEFORE_FREEZE"
    assert result["scientific_execution_authorized_by_this_correction"] is False
    assert result["memory"]["observations"] == linux_counter_records()
    assert result["projection"]["observed_peak_rss_bytes"] == 449413120
    assert result["projected_limits_decision"] == "WITHIN_LIMITS"
    expected, name = (output / "SHA256SUMS").read_text().split()
    assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    with pytest.raises(FileExistsError):
        eng.recover_completed_preflight(original, output, expected_original_sha256=digest, source_provider=source)


def test_recovery_requires_exact_authenticated_receipt_bytes(tmp_path, monkeypatch):
    original, digest = authenticated_unit_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="pinned original"):
        eng.recover_completed_preflight(original, tmp_path / "wrong-anchor",
                                        expected_original_sha256="0" * 64, source_provider=source)
    original.write_bytes(original.read_bytes() + b" ")
    with pytest.raises(ValueError, match="does not match"):
        eng.recover_completed_preflight(original, tmp_path / "tampered",
                                        expected_original_sha256=digest, source_provider=source)
    assert not (tmp_path / "tampered").exists()


def test_recovery_does_not_apply_old_timings_to_changed_current_neural_code(tmp_path, monkeypatch):
    original, digest = authenticated_unit_fixture(tmp_path, monkeypatch)
    changed = source()
    changed["source_files_sha256"]["arena/universal/network.py"] = "0" * 64
    with pytest.raises(ValueError, match="current performance-critical"):
        eng.recover_completed_preflight(original, tmp_path / "changed-neural-implementation",
                                        expected_original_sha256=digest, source_provider=lambda: changed)
    assert not (tmp_path / "changed-neural-implementation").exists()


@pytest.mark.parametrize("case", ["wrong_failure", "not_actual", "incomplete", "bad_neural_hash",
                                 "changed_mechanical_weights", "environment_learning", "wrong_source",
                                 "wrong_protocol", "wrong_summary", "incomplete_updates"])
def test_recovery_rejects_incomplete_or_different_contracts_even_behind_a_unit_anchor(tmp_path, monkeypatch, case):
    def mutate(value):
        if case == "wrong_failure":
            value["failure"]["message"] = "unrelated failure"
        elif case == "not_actual":
            value["actual_measurements"] = False
        elif case == "incomplete":
            value["partial_timing_samples"]["world"]["wall"].pop()
        elif case == "bad_neural_hash":
            value["source"]["source_files_sha256"]["arena/universal/network.py"] = "0" * 64
        elif case == "changed_mechanical_weights":
            value["untrained_fingerprint_after_mechanics"] = "f" * 64
        elif case == "environment_learning":
            value["fixture_state"]["global_environment_training_steps"] = 1
        elif case == "wrong_source":
            value["source"]["source_commit"] = "0" * 40
        elif case == "wrong_protocol":
            value["source"]["protocol_sha256"] = "0" * 64
        elif case == "wrong_summary":
            value["timings"]["q"]["wall"]["p95_seconds"] += .001
        else:
            value["fixture_state"]["q_updates"] = 49
    original, digest = authenticated_unit_fixture(tmp_path, monkeypatch, mutate)
    with pytest.raises(ValueError):
        eng.recover_completed_preflight(original, tmp_path / "invalid-derived",
                                        expected_original_sha256=digest, source_provider=source)
    assert not (tmp_path / "invalid-derived").exists()
