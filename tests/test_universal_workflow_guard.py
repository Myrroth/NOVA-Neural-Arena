"""Offline fixtures for the reservation gate; no API, rollout or authorization."""
import copy
import json

import pytest

from arena.universal import workflow_guard as g


@pytest.fixture
def event_and_approval():
    repo = {"full_name": g.REPOSITORY, "private": False}
    event = {"action": "labeled", "label": {"name": g.APPROVAL_LABEL},
             "repository": repo, "number": 13, "sender": {"login": "Myrroth"},
             "pull_request": {"number": 13,
                 "head": {"ref": g.HEAD_BRANCH, "sha": "b"*40, "repo": repo},
                 "base": {"ref": g.BASE_BRANCH, "repo": repo}}}
    approval = {"approval_type": g.APPROVAL_TYPE, "approved": True,
                "workflow_head": "b"*40, "source_commit": "a"*40,
                "protocol_sha256": "c"*64,
                "human_approval_reference": "SYNTHETIC FIXTURE, NOT REAL APPROVAL",
                "planning_approval_url": "https://github.com/Myrroth/NOVA-Neural-Arena/issues/12#issuecomment-123",
                "budgets": copy.deepcopy(g.APPROVAL_BUDGET)}
    return event, approval


def valid(event, approval, comments=(), attempt=1):
    return g.validate_event(event, approval, "12345", attempt, comments)


def test_valid_gate_is_not_a_usable_permit_or_a_reservation(event_and_approval):
    event, approval = event_and_approval
    before = copy.deepcopy((event, approval))
    result = valid(event, approval, [{"body": "Discussion mentions "+g.RESERVATION_PREFIX+" but consumes nothing."}])
    assert (event, approval) == before
    assert result["reservation_text"].startswith(g.RESERVATION_PREFIX+"12345 attempt 1")
    assert result["permit"]["reservation_id"] is None
    assert result["permit"]["source_revision"] == "a"*40
    assert result["permit"]["workflow_head"] == "b"*40
    result["permit"]["budgets"]["automatic_retry"] = True
    assert approval["budgets"]["automatic_retry"] is False


@pytest.mark.parametrize("path,value", [
    (("action",), "synchronize"), (("label", "name"), "nna-mg002-preflight"),
    (("repository", "private"), True), (("number",), 12),
    (("pull_request", "head", "sha"), "d"*40),
    (("pull_request", "head", "ref"), "main"),
    (("pull_request", "head", "repo", "full_name"), "other/fork"),
    (("pull_request", "base", "ref"), "main"), (("sender", "login"), "other"),
])
def test_event_mismatch_never_returns_a_reservation(event_and_approval, path, value):
    event, approval = event_and_approval
    target = event
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(g.GateRejected):
        valid(event, approval)


@pytest.mark.parametrize("attempt", [0, 2, "2", True, 1.0, None])
def test_failed_or_cancelled_attempt_cannot_be_retried(event_and_approval, attempt):
    with pytest.raises(g.GateRejected):
        valid(*event_and_approval, attempt=attempt)


def test_remote_reservation_burns_current_and_future_runs(event_and_approval):
    with pytest.raises(g.GateRejected, match="already reserved"):
        valid(*event_and_approval, comments=[{"body": g.RESERVATION_PREFIX+"12345 attempt 1 | failed"}])


@pytest.mark.parametrize("field,value", [
    ("approval_type", "old-phase5"), ("approved", 1),
    ("source_commit", "main"), ("protocol_sha256", "c"*63),
    ("human_approval_reference", "Autorizacao Fase 5"),
    ("human_approval_reference", "https://github.com/Myrroth/NOVA-Neural-Arena/issues/6#issuecomment-123"),
    ("human_approval_reference", "PENDING"),
    ("planning_approval_url", "https://github.com/Myrroth/NOVA-Neural-Arena/issues/12"),
])
def test_missing_or_old_approval_is_not_reinterpreted(event_and_approval, field, value):
    event, approval = event_and_approval
    approval[field] = value
    with pytest.raises(g.GateRejected):
        valid(event, approval)


def test_exact_finite_budget_is_mandatory(event_and_approval):
    event, approval = event_and_approval
    del approval["budgets"]
    with pytest.raises(g.GateRejected):
        valid(event, approval)
    approval["budgets"] = {**g.APPROVAL_BUDGET, "training_steps": 1_079_999}
    with pytest.raises(g.GateRejected, match="limits"):
        valid(event, approval)
    approval["budgets"] = {**g.APPROVAL_BUDGET, "automatic_retry": 0}
    with pytest.raises(g.GateRejected, match="limits"):
        valid(event, approval)


def owner_comment(approval):
    return {"user": {"login": "Myrroth"}, "author_association": "OWNER",
            "issue_url": "https://api.github.com/repos/Myrroth/NOVA-Neural-Arena/issues/12",
            "body": g.APPROVAL_MARKER+json.dumps(approval)}


def test_owner_attestation_parser_requires_trusted_api_metadata(event_and_approval):
    comment = owner_comment(event_and_approval[1])
    assert g.parse_approval_comment(comment) == event_and_approval[1]
    for field in ("user", "author_association", "issue_url"):
        incomplete = {k: v for k, v in comment.items() if k != field}
        with pytest.raises(g.GateRejected):
            g.parse_approval_comment(incomplete)
    with pytest.raises(g.GateRejected):
        g.parse_approval_comment({**comment, "user": {"login": "someone"}})
    for raw in ('{"approved":true,"approved":false}', '{"approved":NaN}', '[]', '{} {}'):
        with pytest.raises(g.GateRejected):
            g.parse_approval_comment({**comment, "body": g.APPROVAL_MARKER+raw})
