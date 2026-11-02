"""Fake-clock/data tests: zero games, real-environment learning or benchmarks."""
import hashlib
import json

import pytest

from arena.universal import engineering as eng


def source():
    return {"source_commit": "a" * 40, "protocol_sha256": "b" * 64,
            "relevant_sources_clean": True, "protocol_version": "unit-fixture-only",
            "architecture_config": {"batch_size": 16, "burn_in": 4, "unroll": 8},
            "source_files_sha256": {"unit-fixture.py": "c" * 64}}


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
