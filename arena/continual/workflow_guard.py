"""Pure validation for a distinct, once-only NNA-CONTINUAL-003 campaign.

No credential, network, learner, environment or approval is created on import.
Owner records attest an already obtained human consent; they cannot authenticate
that conversation. All API metadata must come from a trusted reservation job.
The global non-cancelling workflow lock must cover read + POST. Any prior003
reservation burns the attempt, regardless of source or outcome. Validation only
returns an unreserved permit; production needs a confirmed remote receipt.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import json
import re


REPOSITORY = "Myrroth/NOVA-Neural-Arena"
OWNER = "Myrroth"
PR_NUMBER = 14
HEAD_BRANCH = "fix/nna-learning-v2"
BASE_BRANCH = "feat/nna-multigame-002"
APPROVAL_LABEL = "nna-continual003-campaign-approved"
APPROVAL_TYPE = "new-explicit-continual-003"
APPROVAL_MARKER = "NNA-CONTINUAL-003:APPROVAL\n"
RESERVATION_PREFIX = "NNA-CONTINUAL-003:RESERVED | immutable run "

OWNER_ID = 288351784
MISSION = "NNA-CONTINUAL-003"
WORKFLOW_FILE = ".github/workflows/nna-continual003-once.yml"


def approval_budget():
    """Exact pure-protocol budget snapshot; no learner/runtime is imported."""
    from .protocol import protocol_config

    limits = protocol_config()["limits"]
    result = {
        name: limits[name]
        for name in (
            "training_steps",
            "evaluation_steps",
            "total_steps",
            "q_updates",
            "world_updates",
            "combined_updates",
            "evaluation_episode_steps",
            "cpu_minutes",
            "rss_bytes",
            "campaign_attempts",
            "automatic_retry",
            "paid_resources",
        )
    }
    result["wall_minutes"] = limits.get("wall_minutes", limits["cpu_minutes"])
    fallback = limits.get("internal_stop_minutes")
    result["internal_stop_cpu_minutes"] = limits.get(
        "internal_stop_cpu_minutes", fallback
    )
    result["internal_stop_wall_minutes"] = limits.get(
        "internal_stop_wall_minutes", fallback
    )
    return result


APPROVAL_BUDGET = approval_budget()
_APPROVAL_FIELDS = frozenset(
    {
        "approval_type",
        "approved",
        "workflow_head",
        "source_commit",
        "protocol_sha256",
        "human_approval_reference",
        "planning_approval_url",
        "budgets",
    }
)
_SHA40 = re.compile(r"[0-9a-f]{40}\Z")
_SHA64 = re.compile(r"[0-9a-f]{64}\Z")
_PLANNING_URL = re.compile(
    r"https://github\.com/Myrroth/NOVA-Neural-Arena/pull/14#issuecomment-[1-9][0-9]*\Z"
)
_OLD_MISSION = re.compile(
    r"NNA[-_ ](?:FECHAMENTO[-_ ]001|MULTIJOGOS[-_ ]002)|multigame[-_ ]002|(?:PHASE|FASE)[ _-]*5",
    re.IGNORECASE,
)
_OLD_GITHUB_REFERENCE = re.compile(
    r"https://github\.com/Myrroth/NOVA-Neural-Arena/(?:issues/(?:6|12)|pull/(?:[1-9]|1[0-3]))(?:[/#?]|$)",
    re.IGNORECASE,
)


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


def require_owner(value, name):
    user = _mapping(value, name)
    if (
        user.get("login") != OWNER
        or type(user.get("id")) is not int
        or user["id"] != OWNER_ID
    ):
        raise GateRejected(f"{name} must be the exact repository owner login and ID")


def parse_approval_comment(comment):
    """Parse one explicit owner attestation; this does not validate or grant it.

    GitHub's user/association/issue metadata is required. The caller
    remains responsible for obtaining that metadata from the trusted API, rather
    than treating a locally supplied comment object as authentication.
    """
    comment = _mapping(comment, "approval comment")
    require_owner(comment.get("user"), "approval comment author")
    if comment.get("author_association") != "OWNER":
        raise GateRejected("approval comment author association must be OWNER")
    if comment.get("issue_url") != (
        f"https://api.github.com/repos/{REPOSITORY}/issues/14"
    ):
        raise GateRejected("approval comment must belong to PR 14")
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
        value = json.loads(
            body[len(APPROVAL_MARKER) :],
            object_pairs_hook=unique_object,
            parse_constant=invalid_constant,
        )
    except (json.JSONDecodeError, RecursionError) as exc:
        raise GateRejected(
            "approval marker must be followed by one JSON object"
        ) from exc
    return dict(_mapping(value, "approval JSON"))


def _validate_approval(approval):
    approval = _mapping(approval, "approval")
    if set(approval) != _APPROVAL_FIELDS:
        raise GateRejected("approval has missing or unexpected schema fields")
    if approval["approval_type"] != APPROVAL_TYPE or approval["approved"] is not True:
        raise GateRejected(
            "a NEW explicit continual-003 approval is required; old approval cannot be reused"
        )
    for name, pattern in (
        ("workflow_head", _SHA40),
        ("source_commit", _SHA40),
        ("protocol_sha256", _SHA64),
    ):
        if not isinstance(approval[name], str) or not pattern.fullmatch(approval[name]):
            raise GateRejected(
                f"approval {name} must be an explicit lowercase full SHA"
            )
    reference = approval["human_approval_reference"]
    if (
        not isinstance(reference, str)
        or not reference.strip()
        or reference != reference.strip()
        or any(ord(char) < 32 for char in reference)
        or re.search(r"\b(?:PENDING|TODO|TBD)\b", reference, re.IGNORECASE)
    ):
        raise GateRejected(
            "human_approval_reference must be explicit, nonempty and not PENDING"
        )
    if _OLD_MISSION.search(reference) or _OLD_GITHUB_REFERENCE.search(reference):
        raise GateRejected(
            "old mission approval references cannot authorize continual-003"
        )
    if not isinstance(
        approval["planning_approval_url"], str
    ) or not _PLANNING_URL.fullmatch(approval["planning_approval_url"]):
        raise GateRejected(
            "planning_approval_url must be an exact positive PR 14 comment URL"
        )
    if "budgets" in approval:
        budgets = _mapping(approval["budgets"], "budgets")
        if set(budgets) != set(approval_budget()):
            raise GateRejected(
                "explicit budgets must contain the complete fixed budget schema"
            )
        if any(
            type(budgets[name]) is not type(expected) or budgets[name] != expected
            for name, expected in approval_budget().items()
        ):
            raise GateRejected(
                "explicit budgets must match the finite fixed campaign limits"
            )
    return approval


def validate_event(payload, approval, run_id, run_attempt, comments):
    """Return reservation text and an UNRESERVED permit, without side effects.

    ``comments`` is the complete, paginated PR 14 comment list. Only a body
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
    if (
        repository.get("full_name") != REPOSITORY
        or repository.get("private") is not False
    ):
        raise GateRejected("event repository mismatch")
    if payload.get("action") != "labeled":
        raise GateRejected("only the labeled pull-request event is permitted")
    label = _mapping(payload.get("label"), "label")
    if label.get("name") != APPROVAL_LABEL:
        raise GateRejected("campaign approval label mismatch")
    pr = _mapping(payload.get("pull_request"), "pull_request")
    if (
        type(pr.get("number")) is not int
        or pr["number"] != PR_NUMBER
        or type(payload.get("number")) is not int
        or payload["number"] != PR_NUMBER
    ):
        raise GateRejected("only PR 14 can reserve continual-003")
    head, base = _mapping(pr.get("head"), "head"), _mapping(pr.get("base"), "base")
    if (
        head.get("ref") != HEAD_BRANCH
        or base.get("ref") != BASE_BRANCH
        or _mapping(head.get("repo"), "head repo").get("full_name") != REPOSITORY
        or _mapping(base.get("repo"), "base repo").get("full_name") != REPOSITORY
    ):
        raise GateRejected(
            "PR branch/repository mismatch; forks cannot reserve the campaign"
        )
    if head.get("sha") != approval["workflow_head"]:
        raise GateRejected("PR head must match the explicitly approved workflow_head")
    require_owner(payload.get("sender"), "label sender")
    if (
        pr.get("state") != "open"
        or pr.get("merged") is not False
        or pr.get("merged_at") is not None
    ):
        raise GateRejected("PR 14 must remain open and unmerged")
    if _mapping(head.get("repo"), "head repo").get("fork") is not False:
        raise GateRejected("fork repositories cannot reserve the campaign")
    if not isinstance(comments, Sequence) or isinstance(
        comments, (str, bytes, bytearray)
    ):
        raise GateRejected("comments must be the complete PR 14 comment list")
    for comment in comments:
        comment = _mapping(comment, "PR 14 comment")
        body = comment.get("body")
        if not isinstance(body, str):
            raise GateRejected("PR 14 comment body must be text")
        if body.startswith(RESERVATION_PREFIX):
            raise GateRejected(
                "campaign already reserved; no rerun, even after failure or in the current run"
            )
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
        "workflow_revision": approval["workflow_head"],
        "workflow_run_id": run_id,
        "workflow_run_attempt": 1,
    }
    if "budgets" in approval:
        permit["budgets"] = copy.deepcopy(dict(approval["budgets"]))
    return {"reservation_text": reservation, "permit": permit}
