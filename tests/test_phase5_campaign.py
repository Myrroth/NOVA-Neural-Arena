"""No official episodes, checkpoint reads or training in these tests."""
import json
import sys
import pytest

from arena.evaluation import phase5_campaign as p5


def test_approved_manifest_derived_from_immutable_candidate_without_episodes(monkeypatch):
    monkeypatch.setattr(p5, "_evaluate_episode",
                        lambda *args, **kwargs: pytest.fail("no campaign permitted in validation"))
    manifest = p5.approved_manifest()
    assert manifest["candidate_protocol_sha256"] == p5.SOURCE_CANDIDATE_SHA
    assert manifest["total_episodes"] == 32200
    assert manifest["max_total_steps"] == 6_440_000
    assert len(manifest["scenarios"]) == 92
    assert len(manifest["primary_contrasts"]) == 6
    assert len(manifest["policies"]) == 7
    assert manifest["primary_endpoint"] == "score"
    assert manifest["hypothesis_tests"] is False
    assert manifest["noninferiority_or_equivalence"] is False
    assert len(manifest["approved_protocol_sha256"]) == 64
    assert manifest["approved_protocol_sha256"] == p5.approved_manifest()["approved_protocol_sha256"]


def test_campaign_refuses_missing_auth_before_any_model_or_game(monkeypatch, tmp_path):
    for name in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_REPOSITORY",
                 "NNA_PHASE5_EVALUATOR_SHA", "NNA_PHASE5_APPROVED_HASH",
                 "NNA_PHASE5_RESERVATION_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(p5, "verify_official_policies",
                        lambda *_: pytest.fail("Must not load real checkpoint"))
    with pytest.raises(RuntimeError, match="authorization"):
        p5.campaign(tmp_path / "never-read", tmp_path / "not-created")
    assert not (tmp_path / "not-created").exists()


def test_validate_cli_prints_manifest_without_checkpoint_access(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["phase5", "--validate-only"])
    monkeypatch.setattr(p5, "verify_official_policies",
                        lambda *_: pytest.fail("No model access in validation"))
    p5.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["total_episodes"] == 32200
    assert payload["status"] == "PRE_REGISTERED_FOR_CHAT1_FINAL_APPROVAL"


def test_campaign_cli_rejects_no_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["phase5", "--campaign"])
    with pytest.raises(SystemExit) as exc:
        p5.main()
    assert exc.value.code == 2
    assert not (tmp_path / "out").exists()
