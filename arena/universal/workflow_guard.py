"""Side-effect-free pre-reservation checks for the future multigame workflow.

This module does not run a campaign, read credentials, access a network, import
the learner, or supply a default approval. An approval is an operator's audit
attestation, not proof that a human authorized anything. Work must obtain trusted
NEW user approval before applying the campaign label.

The caller must fetch ALL Issue 12 comments under a global, non-cancelling
workflow concurrency lock. After validation it must POST ``reservation_text`` to
Issue 12 and confirm the returned positive comment ID BEFORE inserting that ID
in ``permit['reservation_id']`` or constructing a trainer. Any reservation burns
the campaign, including a failed run and the current run. Validation alone never
creates a usable campaign permit: the returned reservation_id is None.

``workflow_head`` binds the PR event; ``source_commit`` binds the later immutable
checkout. The workflow must separately verify that checkout and the canonical
frozen protocol hash/limits before executing. Neither source is frozen here.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import json
import re


REPOSITORY = "Myrroth/NOVA-Neural-Arena"
OWNER = "Myrroth"
PR_NUMBER = 13
HEAD_BRANCH = "feat/nna-multigame-002"
BASE_BRANCH = "chore/nna-closure-final"
APPROVAL_LABEL = "nna-mg002-campaign-approved"
APPROVAL_TYPE = "new-explicit-multigame-002"
APPROVAL_MARKER = "NNA-MULTIJOGOS-002:APPROVAL\n"
RESERVATION_PREFIX = "NNA-MULTIJOGOS-002:RESERVED | immutable run "

# Explicit budget attestation. Exact values prevent approving a smaller
# budget which the fixed runner would silently exceed. Keep aligned at freeze.
APPROVAL_BUDGET = {
    "training_steps": 1_080_000,
    "evaluation_steps": 694_000,
    "total_steps": 1_800_000,
    "q_updates": 67_500,
    "world_updates": 33_750,
    "combined_updates": 101_250,
    "evaluation_episode_steps": 400,
    "cpu_minutes": 300,
    "rss_bytes": 3*1024**3,
    "campaign_attempts": 1,
    "automatic_retry": False,
    "paid_resources": False,
}
_APPROVAL_FIELDS = frozenset({"approval_type", "approved", "workflow_head", "source_commit",
                              "protocol_sha256", "human_approval_reference", "planning_approval_url", "budgets"})
_SHA40 = re.compile(r"[0-9a-f]{40}\Z")
_SHA64 = re.compile(r"[0-9a-f]{64}\Z")
_PLANNING_URL = re.compile(
    r"https://github\.com/Myrroth/NOVA-Neural-Arena/issues/12#issuecomment-[1-9][0-9]*\Z")
_OLD_MISSION = re.compile(r"NNA[-_ ]FECHAMENTO[-_ ]001|(?:PHASE|FASE)[ _-]*5", re.IGNORECASE)
_OLD_GITHUB_REFERENCE = re.compile(
    r"https://github\.com/Myrroth/NOVA-Neural-Arena/(?:issues/6|pull/(?:[1-9]|1[0-2]))(?:[/#?]|$)",
    re.IGNORECASE)


class GateRejected(ValueError):
    """The workflow must stop before reservation or scientific execution."""


def _mapping(value, name):
    if not isinstance(value, Mapping):
        raise GateRejected(f"{name} must be an object")
    return value


def _positive_id(value, name):
    if type(value) is int and value > 0:
        return str(value)
    if isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value):
        return value
    raise GateRejected(f"{name} must be a positive integer ID")


def _owner_if_present(container, field):
    if field in container:
        actor = _mapping(container[field], field)
        if actor.get("login") != OWNER:
            raise GateRejected(f"{field} actor must be the exact repository owner")


def parse_approval_comment(comment):
    """Parse one explicit owner attestation; this does not validate or grant it.

    GitHub's user/association/issue metadata is required. The caller
    remains responsible for obtaining that metadata from the trusted API, rather
    than treating a locally supplied comment object as authentication.
    """
    comment = _mapping(comment, "approval comment")
    if _mapping(comment.get("user"), "approval author").get("login") != OWNER:
        raise GateRejected("approval comment author must be the repository owner")
    if comment.get("author_association") != "OWNER":
        raise GateRejected("approval comment author association must be OWNER")
    if comment.get("issue_url") != (
            f"https://api.github.com/repos/{REPOSITORY}/issues/12"):
        raise GateRejected("approval comment must belong to Issue 12")
    body = comment.get("body")
    if not isinstance(body, str) or not body.startswith(APPROVAL_MARKER):
        raise GateRejected("approval comment must begin with the exact approval marker")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise GateRejected("approval JSON contains duplicate keys")
            result[key] = value
        return result

    def invalid_constant(value):
        raise GateRejected(f"approval JSON contains nonfinite constant {value}")

    try:
        value = json.loads(body[len(APPROVAL_MARKER):], object_pairs_hook=unique_object,
                           parse_constant=invalid_constant)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise GateRejected("approval marker must be followed by one JSON object") from exc
    return dict(_mapping(value, "approval JSON"))


def _validate_approval(approval):
    approval = _mapping(approval, "approval")
    if set(approval) != _APPROVAL_FIELDS:
        raise GateRejected("approval has missing or unexpected schema fields")
    if approval["approval_type"] != APPROVAL_TYPE or approval["approved"] is not True:
        raise GateRejected("a NEW explicit multigame-002 approval is required; old approval cannot be reused")
    for name, pattern in (("workflow_head", _SHA40), ("source_commit", _SHA40),
                          ("protocol_sha256", _SHA64)):
        if not isinstance(approval[name], str) or not pattern.fullmatch(approval[name]):
            raise GateRejected(f"approval {name} must be an explicit lowercase full SHA")
    reference = approval["human_approval_reference"]
    if (not isinstance(reference, str) or not reference.strip() or reference != reference.strip()
            or any(ord(char) < 32 for char in reference)
            or re.search(r"\b(?:PENDING|TODO|TBD)\b", reference, re.IGNORECASE)):
        raise GateRejected("human_approval_reference must be explicit, nonempty and not PENDING")
    if _OLD_MISSION.search(reference) or _OLD_GITHUB_REFERENCE.search(reference):
        raise GateRejected("old mission approval references cannot authorize multigame-002")
    if (not isinstance(approval["planning_approval_url"], str) or
            not _PLANNING_URL.fullmatch(approval["planning_approval_url"])):
        raise GateRejected("planning_approval_url must be an exact positive Issue 12 comment URL")
    if "budgets" in approval:
        budgets = _mapping(approval["budgets"], "budgets")
        if set(budgets) != set(APPROVAL_BUDGET):
            raise GateRejected("explicit budgets must contain the complete fixed budget schema")
        if any(type(budgets[name]) is not type(expected) or budgets[name] != expected
               for name, expected in APPROVAL_BUDGET.items()):
            raise GateRejected("explicit budgets must match the finite fixed campaign limits")
    return approval


def validate_event(payload, approval, run_id, run_attempt, comments):
    """Return reservation text and an UNRESERVED permit, without side effects.

    ``comments`` is the complete, paginated Issue 12 comment list. Only a body
    beginning at byte/character zero with RESERVATION_PREFIX consumes the single
    campaign. A quotation or narrative mention elsewhere cannot consume it.
    """
    payload = _mapping(payload, "event")
    approval = _validate_approval(approval)
    run_id = _positive_id(run_id, "run_id")
    if (type(run_attempt) is not int or run_attempt != 1) and run_attempt != "1":
        raise GateRejected("only run_attempt 1 is permitted, including after failure")
    if isinstance(run_attempt, bool):
        raise GateRejected("only run_attempt 1 is permitted, including after failure")
    repository = _mapping(payload.get("repository"), "repository")
    if repository.get("full_name") != REPOSITORY or repository.get("private") is not False:
        raise GateRejected("event repository mismatch")
    if payload.get("action") != "labeled":
        raise GateRejected("only the labeled pull-request event is permitted")
    label = _mapping(payload.get("label"), "label")
    if label.get("name") != APPROVAL_LABEL:
        raise GateRejected("campaign approval label mismatch")
    pr = _mapping(payload.get("pull_request"), "pull_request")
    if (type(pr.get("number")) is not int or pr["number"] != PR_NUMBER or
            type(payload.get("number")) is not int or payload["number"] != PR_NUMBER):
        raise GateRejected("only PR 13 can reserve multigame-002")
    head, base = _mapping(pr.get("head"), "head"), _mapping(pr.get("base"), "base")
    if (head.get("ref") != HEAD_BRANCH or base.get("ref") != BASE_BRANCH or
            _mapping(head.get("repo"), "head repo").get("full_name") != REPOSITORY or
            _mapping(base.get("repo"), "base repo").get("full_name") != REPOSITORY):
        raise GateRejected("PR branch/repository mismatch; forks cannot reserve the campaign")
    if head.get("sha") != approval["workflow_head"]:
        raise GateRejected("PR head must match the explicitly approved workflow_head")
    _owner_if_present(payload, "sender")
    if not isinstance(comments, Sequence) or isinstance(comments, (str, bytes, bytearray)):
        raise GateRejected("comments must be the complete Issue 12 comment list")
    for comment in comments:
        comment = _mapping(comment, "Issue 12 comment")
        body = comment.get("body")
        if not isinstance(body, str):
            raise GateRejected("Issue 12 comment body must be text")
        if body.startswith(RESERVATION_PREFIX):
            raise GateRejected("campaign already reserved; no rerun, even after failure or in the current run")
    reservation = (
        f"{RESERVATION_PREFIX}{run_id} attempt 1 | source {approval['source_commit']}"
        f" | protocol {approval['protocol_sha256']} | workflow {approval['workflow_head']}"
        " | no reruns, even on failure"
    )
    permit = {
        "authorization_type": approval["approval_type"],
        "approved": approval["approved"],
        "source_revision": approval["source_commit"],
        "protocol_sha256": approval["protocol_sha256"],
        "human_approval_reference": approval["human_approval_reference"],
        "planning_approval_reference": approval["planning_approval_url"],
        "reservation_id": None,
        "workflow_head": approval["workflow_head"],
        "workflow_run_id": run_id,
        "workflow_run_attempt": 1,
    }
    if "budgets" in approval:
        permit["budgets"] = copy.deepcopy(dict(approval["budgets"]))
    return {"reservation_text": reservation, "permit": permit}
