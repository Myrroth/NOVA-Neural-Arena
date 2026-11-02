"""Fake orchestration only: SIX fake train counters + ONE fake evaluation step.

No environment, neural network, optimizer, real checkpoint or reserved seed runs.
Permits in these fixtures are explicitly synthetic and are not authorizations.
"""
import hashlib
import json
from pathlib import Path

import pytest

from arena.shared.integrity import verify_checksum_manifest
from arena.universal import campaign as c
from arena.universal import protocol as p


class FakeTrainer:
    def __init__(self, seed, backend):
        self.model_seed = seed
        self.backend = backend
        self.global_steps = self.q_updates = self.world_updates = 0
        self.training_curve, self.loss_history, self.actors = [], [], {}
        self.stats = {}

    def configure_stage(self, stage, budget):
        self.stage, self.stage_budget, self.stage_steps = stage, budget, 0
        self.games = p.SOURCE_GAMES if stage == "pretrain" else ("racing",)
        self.backend.stages.append((self.model_seed, stage, self.games, budget))

    def collect(self, count, *, learn):
        assert learn is True
        actual = 1 if self.backend.fail else count
        self.global_steps += actual
        self.stage_steps += actual
        for game in self.games:
            self.stats[game] = self.stats.get(game, 0)+actual//len(self.games)
        self.training_curve.append({"model_seed": self.model_seed, "stage": self.stage,
                                    "game": self.games[0], "training_step": self.stage_steps,
                                    "global_step": self.global_steps, "progress_normalized": .1})
        if self.backend.fail:
            raise RuntimeError("synthetic interruption, not a failed game")

    def training_baselines(self):
        return {game: {"source": "training", "support": support,
                        "reward_mean": 0., "terminal_rate": 0.} for game, support in self.stats.items()}


class FakeBackend:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.stages, self.trainers, self.saved, self.loaded, self.evaluated = [], [], [], [], []
        self.initialized = []

    def trainer(self, seed):
        value = FakeTrainer(seed, self)
        self.trainers.append(value)
        return value

    def save(self, trainer, path, permit):
        path.mkdir(parents=True, exist_ok=False)
        payload = json.dumps({"fixture": True, "stage": trainer.stage}).encode()
        (path/"fake.json").write_bytes(payload)
        manifest = f"{hashlib.sha256(payload).hexdigest()}  fake.json\n".encode()
        (path/"SHA256SUMS").write_bytes(manifest)
        self.saved.append(path)
        return {"manifest_sha256": hashlib.sha256(manifest).hexdigest()}

    def load_policy(self, root, entry, permit):
        self.loaded.append(entry["path"])
        return "FAKE-NETWORK-NO-INFERENCE:"+entry["path"]

    def untrained(self, seed):
        self.initialized.append(seed)
        return "FAKE-UNTRAINED-NO-INFERENCE"

    def evaluate(self, network, **kwargs):
        self.evaluated.append((network, kwargs))
        assert kwargs["max_steps"] == 1
        assert kwargs["environment_seed"] == 810201
        return ({key: kwargs[key] for key in ("stage", "policy", "game", "model_seed", "environment_seed")}
                | {"episode": 1, "steps": 1, "return": 0., "score": 0.},
                [{"synthetic_fixture": True, "step": 1}], [{"synthetic_fixture": True, "step": 1}], [])


@pytest.fixture
def setup(tmp_path):
    source = {"root": str(tmp_path), "head": "a"*40, "clean": True}
    permit = {"source_revision": "a"*40, "protocol_sha256": p.protocol_sha256(),
              "human_approval_reference": "SYNTHETIC TEST FIXTURE - NOT HUMAN APPROVAL",
              "planning_approval_reference": "https://example.com/synthetic-test-only",
              "reservation_id": "SYNTHETIC-TEST-RESERVATION", "approved": True,
              "authorization_type": "new-explicit-multigame-002"}
    plan = c._Plan(model_seeds=(1103,), source_steps_per_game=1, target_steps=1,
                   max_episode_steps=1, max_total_steps=7, test_only=True)
    return permit, source, plan


def kwargs(backend, source, plan):
    return {"_backend": backend, "_source": source, "_plan": plan,
            "_rss_reader": lambda: 1, "_clock": lambda: 0.}


def test_fake_train_then_one_fake_eval_account_cumulative_steps_without_double_count(tmp_path, setup):
    permit, source, plan = setup
    backend = FakeBackend()
    root = tmp_path/"test-only-run"
    trained = c.train_once(root, permit, **kwargs(backend, source, plan))
    assert trained["status"] == "SUCCESS" and trained["test_only"] is True
    assert trained["budget"]["total_steps"] == trained["budget"]["training_steps"] == 6
    assert len(backend.trainers) == 2 and len(backend.saved) == 3
    assert backend.stages == [(1103, "pretrain", p.SOURCE_GAMES, 4),
                               (1103, "adaptation", ("racing",), 1),
                               (1103, "from_scratch", ("racing",), 1)]
    assert (root/"training_curve.csv").is_file()
    assert verify_checksum_manifest(root)
    cell = p.EvaluationCell("after_adaptation", "shared", "snake", 1103, 810201)
    result = c.evaluate_final_once(root, permit, _cells=(cell,), **kwargs(backend, source, plan))
    assert result["status"] == "SUCCESS" and result["budget"]["total_steps"] == 7
    assert result["budget"]["training_steps"] == 6 and result["budget"]["evaluation_steps"] == 1
    assert len(backend.loaded) == 3 and backend.initialized == [1103]
    assert len(backend.evaluated) == 1
    assert backend.evaluated[0][1]["record_world_predictions"] is True
    assert result["scientific_conclusion"] == "REQUIRES_INDEPENDENT_ANALYSIS"
    assert verify_checksum_manifest(root/"evaluation") and verify_checksum_manifest(root)


def test_directory_and_consumed_reservation_block_rerun_and_alternate_output(tmp_path, setup):
    permit, source, plan = setup
    backend = FakeBackend()
    root = tmp_path/"test-only-run"
    c.train_once(root, permit, **kwargs(backend, source, plan))
    with pytest.raises(c.CampaignStopped, match="already exists"):
        c.train_once(root, permit, **kwargs(backend, source, plan))
    with pytest.raises(c.CampaignStopped, match="reservation already consumed"):
        c.train_once(tmp_path/"alternate", permit, **kwargs(backend, source, plan))
    cell = p.EvaluationCell("after_adaptation", "shared", "snake", 1103, 810201)
    c.evaluate_final_once(root, permit, _cells=(cell,), **kwargs(backend, source, plan))
    with pytest.raises(c.CampaignStopped, match="already reserved"):
        c.evaluate_final_once(root, permit, _cells=(cell,), **kwargs(backend, source, plan))


@pytest.mark.parametrize("field,value", [("approved", False), ("approved", 1),
    ("authorization_type", "old-phase5"), ("protocol_sha256", "0"*64),
    ("source_revision", "b"*40), ("human_approval_reference", ""),
    ("planning_approval_reference", "http://example.com/fixture"), ("reservation_id", "")])
def test_invalid_permit_never_constructs_trainer_or_output(tmp_path, setup, field, value):
    permit, source, plan = setup
    invalid = {**permit, field: value}
    backend = FakeBackend()
    output = tmp_path/"never-created"
    with pytest.raises(c.CampaignStopped):
        c.train_once(output, invalid, **kwargs(backend, source, plan))
    assert not output.exists() and not backend.trainers


def test_unclean_source_refused_before_fake_run(tmp_path, setup):
    permit, source, plan = setup
    with pytest.raises(c.CampaignStopped, match="working tree"):
        c.train_once(tmp_path/"never-created", permit, **kwargs(FakeBackend(), {**source, "clean": False}, plan))


def test_partial_failure_burns_attempt_and_preserves_logs_manifest_and_snapshot(tmp_path, setup):
    permit, source, plan = setup
    backend = FakeBackend(fail=True)
    root = tmp_path/"interrupted-fixture"
    with pytest.raises(c.CampaignStopped, match="partial artifacts preserved"):
        c.train_once(root, permit, **kwargs(backend, source, plan))
    status = json.loads((root/"TRAIN_STATUS.json").read_text())
    assert status["status"] == "STOPPED" and status["budget"]["training_steps"] == 1
    assert (root/"partial_checkpoint"/"SHA256SUMS").is_file()
    assert (root/"training_curve.csv").is_file() and verify_checksum_manifest(root)
    with pytest.raises(c.CampaignStopped, match="all training must finish"):
        c.evaluate_final_once(root, permit, _cells=(), **kwargs(backend, source, plan))


def test_budget_stops_on_wall_clock_memory_and_update_limits():
    now = [0.]
    budget = c.Budget(clock=lambda: now[0], rss_reader=lambda: 1)
    budget.account(training_steps=7, q_updates=2, world_updates=1)
    assert budget.report()["total_steps"] == 7
    now[0] = 285*60
    with pytest.raises(c.CampaignStopped, match="285-minute"):
        budget.check()
    with pytest.raises(c.CampaignStopped, match="memory"):
        c.Budget(clock=lambda: 0., rss_reader=lambda: 3*1024**3+1).check()
    with pytest.raises(c.CampaignStopped, match="update limit"):
        c.Budget(clock=lambda: 0., rss_reader=lambda: 1).account(q_updates=p.MAX_Q_UPDATES+1)


def test_process_cpu_budget_accumulates_independently_from_wall_time():
    wall, cpu = [100.], [200.]
    budget = c.Budget(prior_elapsed_seconds=12., prior_cpu_seconds=34.,
                      clock=lambda: wall[0], cpu_clock=lambda: cpu[0], rss_reader=lambda: 1)
    wall[0] += 5.
    cpu[0] += 10.
    report = budget.report()
    assert report["elapsed_seconds"] == 17.
    assert report["cpu_elapsed_seconds"] == 44.
    next_process = c.Budget(prior_elapsed_seconds=report["elapsed_seconds"],
                            prior_cpu_seconds=report["cpu_elapsed_seconds"],
                            clock=lambda: wall[0], cpu_clock=lambda: cpu[0], rss_reader=lambda: 1)
    cpu[0] += 285*60-44
    with pytest.raises(c.CampaignStopped, match="process CPU"):
        next_process.check()
    assert next_process.elapsed() == 17.


def test_real_path_cannot_accept_test_plan_or_override_git_state(tmp_path, setup):
    permit, source, plan = setup
    with pytest.raises(c.CampaignStopped, match="cannot override"):
        c.train_once(tmp_path/"never-real", permit, _plan=plan, _source=source)
    assert not (tmp_path/"never-real").exists()


def test_candidate_protocol_blocks_real_path_even_with_approved_permit_before_actions(setup, monkeypatch):
    permit, source, _ = setup
    config = p.protocol_config()
    config.update(status="CANDIDATE_NOT_FROZEN", unresolved_before_freeze=["synthetic unresolved gate"])
    monkeypatch.setattr(p, "protocol_config", lambda: config)
    permit["protocol_sha256"] = p.protocol_sha256()
    monkeypatch.setattr(c, "_source_state", lambda: pytest.fail("candidate must fail before Git/backend/environment actions"))
    with pytest.raises(c.CampaignStopped, match="not frozen"):
        c.validate_permit(permit)


def test_inference_tensor_gate_rejects_missing_shape_dtype_and_nonfinite():
    torch = pytest.importorskip("torch")
    expected = {"synthetic_weight": torch.zeros(1, 2)}
    c._verify_tensor_state({"synthetic_weight": torch.ones(1, 2)}, expected)
    for invalid in ({}, {"synthetic_weight": torch.zeros(2, 1)},
                    {"synthetic_weight": torch.zeros(1, 2, dtype=torch.float64)},
                    {"synthetic_weight": torch.tensor([[float("nan"), 0.]])}):
        with pytest.raises(c.CampaignStopped, match="architecture|tensor"):
            c._verify_tensor_state(invalid, expected)
    assert not c._finite_payload({"optimizer_fixture": [torch.tensor(float("inf"))]})
