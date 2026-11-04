"""Trusted reservation adapter and offline receipt validation for campaign003.

Imports are standard-library only and perform no requests. reserve() uses only
its supplied API callable; the CLI's fixed-host transport does not retry GET or
POST. A lost POST can already consume the campaign: never repeat it. The caller
must hold the global non-cancelling workflow lock and must first obtain actual
new human consent and actual Planning review. Owner JSON is an attestation of
those events, not their authentication. The learner must receive only the hashed
receipt/permit in a separate read-only job, never this job's write credential.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from . import workflow_guard as guard


REPO_PATH = f"/repos/{guard.REPOSITORY}"
ISSUE_URL = f"https://api.github.com/repos/{guard.REPOSITORY}/issues/14"
COMMENT_PATH = REPO_PATH + "/issues/14/comments"
COMMENT_API_PATH = REPO_PATH + "/issues/comments/"
COMMENT_WEB_URL = f"https://github.com/{guard.REPOSITORY}/pull/14#issuecomment-"
PR_PATH = REPO_PATH + "/pulls/14"
RECERTIFICATION_PREFIX = "NNA-CONTINUAL-003:RECERTIFIED | source "
MAX_COMMENT_PAGES = 100
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


def receipt_bytes(value):
    """The one canonical UTF8 serialization, including its final newline."""
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


class ReservationRejected(guard.GateRejected):
    """No confirmed reservation/permit was obtained; do not start science."""

    def __init__(self, message, *, diagnostic=None):
        self.diagnostic = (
            _safe_diagnostic(diagnostic) if diagnostic is not None else None
        )
        if self.diagnostic is not None:
            status = self.diagnostic["http_status"]
            message += (
                f" [category={self.diagnostic['category']}; "
                f"http_status={status if status is not None else 'unavailable'}]"
            )
        super().__init__(message)


class ReservationUncertain(ReservationRejected):
    """The single POST was attempted; its outcome must not be retried."""


class ReservationTransportError(ReservationRejected):
    """Only a fixed category and numeric status may cross the HTTP boundary."""

    def __init__(self, category, http_status=None):
        super().__init__(
            "reservation API request failed; no automatic retry",
            diagnostic={"category": category, "http_status": http_status},
        )


_ERROR_CATEGORIES = frozenset(
    {
        "http_error",
        "timeout",
        "redirect",
        "unexpected_status",
        "invalid_json",
        "response_too_large",
        "network_error",
        "transport_error",
        "validation_rejected",
        "invalid_post_response",
        "output_io_error",
        "internal_error",
    }
)


def _safe_diagnostic(value):
    """Never preserve exception messages, URLs, bodies, headers or credentials."""
    value = value if isinstance(value, Mapping) else {}
    category = value.get("category")
    if not isinstance(category, str) or category not in _ERROR_CATEGORIES:
        category = "transport_error"
    status = value.get("http_status")
    return {
        "category": category,
        "http_status": status if type(status) is int and 100 <= status <= 599 else None,
    }


def _error_diagnostic(error, *, request=False):
    if isinstance(error, ReservationRejected) and error.diagnostic is not None:
        return _safe_diagnostic(error.diagnostic)
    if isinstance(error, HTTPError):
        status = error.code
        redirect = type(status) is int and 300 <= status <= 399
        return _safe_diagnostic(
            {
                "category": "redirect" if redirect else "http_error",
                "http_status": status,
            }
        )
    if isinstance(error, TimeoutError) or (
        isinstance(error, URLError) and isinstance(error.reason, TimeoutError)
    ):
        return _safe_diagnostic({"category": "timeout"})
    if isinstance(error, URLError):
        return _safe_diagnostic({"category": "network_error"})
    if isinstance(error, ReservationRejected):
        return _safe_diagnostic({"category": "validation_rejected"})
    if isinstance(error, OSError):
        return _safe_diagnostic(
            {"category": "network_error" if request else "output_io_error"}
        )
    return _safe_diagnostic(
        {"category": "transport_error" if request else "internal_error"}
    )


def _object(value, name):
    if not isinstance(value, Mapping):
        raise ReservationRejected(f"{name} must be a JSON object")
    return value


def _id(value, name):
    if type(value) is int and value > 0:
        return str(value)
    if isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value):
        return value
    raise ReservationRejected(f"{name} must be a positive ID")


def _sha(value, length, name):
    if not isinstance(value, str) or not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
        raise ReservationRejected(f"{name} must be an explicit lowercase full SHA")


def _location(comment, name, *, expected_id=None):
    comment = _object(comment, name)
    comment_id = _id(comment.get("id"), f"{name} id")
    if (
        expected_id is not None
        and comment_id != expected_id
        or comment.get("issue_url") != ISSUE_URL
        or comment.get("html_url") != COMMENT_WEB_URL + comment_id
    ):
        raise ReservationRejected(f"{name} must be the exact PR 14 comment")
    return comment_id


def _owner(comment, name):
    try:
        guard.require_owner(comment.get("user"), f"{name} author")
    except guard.GateRejected:
        raise ReservationRejected(
            f"{name} must be authored by the exact repository OWNER"
        ) from None
    if comment.get("author_association") != "OWNER":
        raise ReservationRejected(f"{name} must have OWNER author association")


def _get(api, path):
    try:
        return api("GET", path)
    except Exception as exc:
        # A transport exception may contain credentials or response details.
        raise ReservationRejected(
            "GitHub GET failed; no reservation was posted",
            diagnostic=_error_diagnostic(exc, request=True),
        ) from None


def _all_comments(api):
    comments, seen_ids = [], set()
    for page in range(1, MAX_COMMENT_PAGES + 1):
        response = _get(api, f"{COMMENT_PATH}?per_page=100&page={page}")
        if not isinstance(response, list) or len(response) > 100:
            raise ReservationRejected(
                "PR 14 pagination must return JSON arrays of at most 100 comments"
            )
        for comment in response:
            comment = _object(comment, "PR 14 comment")
            comment_id = _location(comment, "PR 14 comment")
            if comment_id in seen_ids:
                raise ReservationRejected(
                    "duplicate paginated comment ID; completeness cannot be established"
                )
            if not isinstance(comment.get("body"), str):
                raise ReservationRejected("PR 14 comment body must be text")
            seen_ids.add(comment_id)
            comments.append(comment)
        if len(response) < 100:
            return comments, page
    raise ReservationRejected(
        "PR 14 exceeded the 100-page bound; completeness cannot be established"
    )


def _verify_source_and_live_pr(api, event, source, workflow):
    live = _object(_get(api, PR_PATH), "live PR")
    expected = event["pull_request"]
    for field in ("number", "state", "merged", "merged_at"):
        if live.get(field) != expected.get(field):
            raise ReservationRejected(
                "live PR identity/state changed from the approved event"
            )
    if (
        live.get("state") != "open"
        or live.get("merged") is not False
        or live.get("merged_at") is not None
    ):
        raise ReservationRejected("live PR must be open and unmerged")
    for side in ("head", "base"):
        actual = _object(live.get(side), "live PR " + side)
        snapshot = _object(expected.get(side), "event PR " + side)
        if (
            actual.get("ref") != snapshot.get("ref")
            or _object(actual.get("repo"), "live repo").get("full_name")
            != guard.REPOSITORY
        ):
            raise ReservationRejected("live PR branch/repository mismatch")
    if (
        live["head"].get("sha") != workflow
        or live["head"]["repo"].get("fork") is not False
    ):
        raise ReservationRejected(
            "live PR head must remain approved workflow B without a fork"
        )
    comparison = _object(
        _get(api, REPO_PATH + f"/compare/{source}...{workflow}"), "source comparison"
    )
    files = comparison.get("files")
    if (
        comparison.get("status") != "ahead"
        or comparison.get("behind_by") != 0
        or _object(comparison.get("base_commit"), "comparison base").get("sha")
        != source
        or _object(comparison.get("merge_base_commit"), "comparison ancestor").get(
            "sha"
        )
        != source
        or not isinstance(files, list)
        or len(files) != 1
        or _object(files[0], "changed file").get("filename") != guard.WORKFLOW_FILE
        or files[0].get("status") not in ("added", "modified")
        or files[0].get("previous_filename") is not None
    ):
        raise ReservationRejected(
            "source A must be ancestor of B with only the new campaign workflow changed"
        )


def reserve(
    event,
    expected_source40,
    expected_protocol64,
    run_id,
    run_attempt,
    api,
    *,
    expected_workflow40,
    before_post=None,
):
    """Verify all gates, POST once, then return ``{'receipt', 'permit'}``.

    API responses must be decoded trusted GitHub JSON, not user-supplied approval
    objects. All comment pages are read before any POST. Owner attestations for
    other immutable source/head/protocol combinations do not match this attempt.
    Duplicate matching approvals, malformed marked approvals, or any durable
    reservation consume/deny the attempt rather than selecting a convenient one.
    ``before_post`` is an optional persistence callback after validation and
    before the sole POST. Its failure prevents the API call. No filesystem is
    used here unless the caller explicitly supplies that callback.
    """
    _sha(expected_source40, 40, "expected_source40")
    _sha(expected_protocol64, 64, "expected_protocol64")
    _sha(expected_workflow40, 40, "expected_workflow40")
    run_id = _id(run_id, "run_id")
    if not (
        (type(run_attempt) is int and run_attempt == 1)
        or (type(run_attempt) is str and run_attempt == "1")
    ):
        raise ReservationRejected(
            "only run_attempt 1 is permitted; no retry after failure"
        )
    if not callable(api):
        raise ReservationRejected("an explicit API callable is required")
    if before_post is not None and not callable(before_post):
        raise ReservationRejected("before_post must be callable when provided")
    event = _object(event, "event")
    pr = _object(event.get("pull_request"), "pull_request")
    head = _object(pr.get("head"), "head")
    workflow_head = head.get("sha")
    _sha(workflow_head, 40, "event head")
    if workflow_head != expected_workflow40:
        raise ReservationRejected(
            "event head differs from expected workflow revision B"
        )
    repository = _object(_get(api, REPO_PATH), "repository response")
    if (
        repository.get("full_name") != guard.REPOSITORY
        or repository.get("private") is not False
    ):
        raise ReservationRejected("the exact public repository is required")
    _verify_source_and_live_pr(api, event, expected_source40, expected_workflow40)
    comments, pages = _all_comments(api)
    matches = []
    for comment in comments:
        if comment["body"].startswith(guard.APPROVAL_MARKER):
            try:
                approval = guard.parse_approval_comment(comment)
            except guard.GateRejected:
                raise ReservationRejected(
                    "a marked approval has invalid owner metadata or JSON"
                ) from None
            if (
                approval.get("workflow_head") == workflow_head
                and approval.get("source_commit") == expected_source40
                and approval.get("protocol_sha256") == expected_protocol64
            ):
                matches.append((comment, approval))
    if len(matches) != 1:
        raise ReservationRejected("exactly one matching owner approval is required")
    approval_comment, approval = matches[0]
    try:
        gated = guard.validate_event(event, approval, run_id, run_attempt, comments)
    except guard.GateRejected:
        raise ReservationRejected(
            "workflow event, finite approval or existing reservation gate rejected"
        ) from None
    approval_id = _location(approval_comment, "approval")
    # validate_event has already constrained this URL to the exact PR 14 URL.
    planning_url = approval["planning_approval_url"]
    planning_id = planning_url.removeprefix(COMMENT_WEB_URL)
    _id(planning_id, "planning comment id")
    recertification = _get(api, COMMENT_API_PATH + planning_id)
    _location(recertification, "recertification", expected_id=planning_id)
    _owner(recertification, "recertification")
    required_line = f"{RECERTIFICATION_PREFIX}{expected_source40} | protocol {expected_protocol64} | workflow {expected_workflow40}"
    if (
        not isinstance(recertification.get("body"), str)
        or recertification["body"].split("\n", 1)[0] != required_line
    ):
        raise ReservationRejected(
            "recertification first line must literally bind this source and protocol"
        )
    reservation_text = gated["reservation_text"]
    if before_post is not None:
        attempt = {
            "schema_version": 1,
            "mission": guard.MISSION,
            "repository": guard.REPOSITORY,
            "pr_number": guard.PR_NUMBER,
            "source_revision": expected_source40,
            "workflow_revision": expected_workflow40,
            "protocol_sha256": expected_protocol64,
            "workflow_run_id": run_id,
            "workflow_run_attempt": 1,
            "phase": "reservation_post",
            "status": "ATTEMPTED",
            "post_outcome": "UNCONFIRMED",
            "automatic_retry": False,
        }
        try:
            # A durable checkpoint is required BEFORE crossing the POST
            # boundary. It records intent, never proves remote delivery.
            before_post(attempt)
        except Exception:
            raise ReservationRejected(
                "attempt checkpoint failed; no reservation POST was called",
                diagnostic={"category": "output_io_error"},
            ) from None
    try:
        posted = api("POST", COMMENT_PATH, {"body": reservation_text})
    except Exception as exc:
        raise ReservationUncertain(
            "reservation POST unconfirmed; the campaign may be consumed; NEVER retry",
            diagnostic=_error_diagnostic(exc, request=True),
        ) from None
    try:
        reservation_id = _location(posted, "posted reservation")
        if posted.get("body") != reservation_text:
            raise ReservationRejected(
                "posted reservation body differs from the requested immutable marker"
            )
    except (ReservationRejected, AttributeError, TypeError):
        raise ReservationUncertain(
            "reservation POST response is ambiguous; the campaign may be consumed; NEVER retry",
            diagnostic={"category": "invalid_post_response"},
        ) from None
    reservation_url = COMMENT_WEB_URL + reservation_id
    permit = copy.deepcopy(gated["permit"])
    permit["reservation_id"] = reservation_url
    receipt = {
        "schema_version": 2,
        "mission": guard.MISSION,
        "repository": guard.REPOSITORY,
        "pr_number": guard.PR_NUMBER,
        "workflow_run_id": run_id,
        "workflow_run_attempt": 1,
        "workflow_revision": expected_workflow40,
        "source_revision": expected_source40,
        "protocol_sha256": expected_protocol64,
        "approval_comment_id": approval_id,
        "approval_comment_url": COMMENT_WEB_URL + approval_id,
        "human_approval_reference": approval["human_approval_reference"],
        "planning_approval_reference": planning_url,
        "reservation_comment_id": reservation_id,
        "reservation_id": reservation_url,
        "reservation_text": reservation_text,
        "comment_pages_checked": pages,
        "comments_checked": len(comments),
        "budgets": copy.deepcopy(approval["budgets"]),
        "source_relation": "sourceA ancestor of workflowB; final tree difference is only the approved workflow file",
        "truth_scope": "trusted workflow API audit of owner attestations, not authentication of a human chat or Chat1 identity",
    }
    digest = hashlib.sha256(receipt_bytes(receipt)).hexdigest()
    permit["reservation_receipt_sha256"] = digest
    return {"receipt": receipt, "permit": permit, "receipt_sha256": digest}


def validate_receipt(
    receipt,
    permit,
    *,
    expected_receipt_sha256,
    expected_source_revision,
    expected_protocol_sha256,
    expected_workflow_revision,
    expected_run_id,
    expected_run_attempt,
):
    """Bind an authenticated job artifact to this run, without network or ML.

    The expected digest must come from the trusted reservation job output, not
    from an editable local permit. This proves transport/binding, not the truth
    of a chat or absence of arbitrary modifications by a privileged operator.
    """
    receipt, permit = _object(receipt, "receipt"), _object(permit, "permit")
    for value, size, name in (
        (expected_receipt_sha256, 64, "receipt SHA256"),
        (expected_source_revision, 40, "source revision"),
        (expected_protocol_sha256, 64, "protocol SHA256"),
        (expected_workflow_revision, 40, "workflow revision"),
    ):
        _sha(value, size, name)
    run_id = _id(expected_run_id, "run_id")
    if type(expected_run_attempt) not in (str, int) or str(expected_run_attempt) != "1":
        raise ReservationRejected("only run attempt 1 can consume a receipt")
    if hashlib.sha256(receipt_bytes(receipt)).hexdigest() != expected_receipt_sha256:
        raise ReservationRejected("remote receipt checksum mismatch")
    bindings = {
        "source_revision": expected_source_revision,
        "protocol_sha256": expected_protocol_sha256,
        "workflow_revision": expected_workflow_revision,
        "workflow_run_id": run_id,
        "workflow_run_attempt": 1,
    }
    if (
        receipt.get("schema_version") != 2
        or receipt.get("mission") != guard.MISSION
        or receipt.get("repository") != guard.REPOSITORY
        or receipt.get("pr_number") != guard.PR_NUMBER
        or any(
            type(receipt.get(k)) is not type(v) or receipt[k] != v
            for k, v in bindings.items()
        )
        or any(
            type(permit.get(k)) is not type(v) or permit[k] != v
            for k, v in bindings.items()
        )
        or permit.get("authorization_type") != guard.APPROVAL_TYPE
        or permit.get("approved") is not True
        or permit.get("reservation_receipt_sha256") != expected_receipt_sha256
    ):
        raise ReservationRejected(
            "receipt/permit does not bind this exact new003 execution"
        )
    for name in (
        "human_approval_reference",
        "planning_approval_reference",
        "reservation_id",
        "budgets",
    ):
        if receipt.get(name) != permit.get(name):
            raise ReservationRejected(
                "receipt and permit differ on authorization or budget"
            )
    approval_id = _id(receipt.get("approval_comment_id"), "approval comment ID")
    reservation_id = _id(
        receipt.get("reservation_comment_id"), "reservation comment ID"
    )
    if (
        receipt.get("approval_comment_url") != COMMENT_WEB_URL + approval_id
        or receipt.get("reservation_id") != COMMENT_WEB_URL + reservation_id
    ):
        raise ReservationRejected("receipt comment identities differ from PR14")
    # Reuse the exact approval schema checks, including distinct human reference,
    # literal PR14 Planning URL and all current finite budget values/types.
    guard._validate_approval(
        {
            "approval_type": permit["authorization_type"],
            "approved": permit["approved"],
            "source_commit": expected_source_revision,
            "workflow_head": expected_workflow_revision,
            "protocol_sha256": expected_protocol_sha256,
            "human_approval_reference": permit.get("human_approval_reference"),
            "planning_approval_url": permit.get("planning_approval_reference"),
            "budgets": permit.get("budgets"),
        }
    )
    expected_marker = (
        f"{guard.RESERVATION_PREFIX}{run_id} attempt 1 | source {expected_source_revision}"
        f" | protocol {expected_protocol_sha256} | workflow {expected_workflow_revision}"
        " | no reruns, even on failure"
    )
    if receipt.get("reservation_text") != expected_marker:
        raise ReservationRejected("receipt reservation marker binding mismatch")
    return copy.deepcopy(dict(receipt))


def read_and_validate_receipt(
    directory,
    *,
    expected_receipt_sha256,
    expected_source_revision,
    expected_protocol_sha256,
    expected_workflow_revision,
    expected_run_id,
    expected_run_attempt,
    permit=None,
):
    """Read two bounded ordinary JSON files from the gate artifact directory."""
    root = Path(directory).absolute()
    for path in (root, *root.parents):
        info = path.lstat()
        if path.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ReservationRejected(
                "receipt directory cannot contain symlinks/reparse points"
            )

    def read(name):
        path = root / name
        info = path.lstat()
        if (
            path.is_symlink()
            or getattr(info, "st_file_attributes", 0) & 0x400
            or not path.is_file()
            or info.st_size > 64 * 1024
        ):
            raise ReservationRejected(
                "receipt members must be bounded ordinary JSON files"
            )
        raw = path.read_bytes()
        if len(raw) > 64 * 1024:
            raise ReservationRejected("receipt member exceeds size bound")
        try:
            value = json.loads(raw)
            canonical = receipt_bytes(value)
        except (ValueError, RecursionError):
            raise ReservationRejected("receipt member is not finite JSON") from None
        if canonical != raw:
            raise ReservationRejected("receipt member is not canonical JSON")
        return value

    receipt = read("receipt.json")
    artifact_permit = read("permit.json")
    if permit is not None and permit != artifact_permit:
        raise ReservationRejected("caller permit differs from gate artifact")
    return validate_receipt(
        receipt,
        artifact_permit,
        expected_receipt_sha256=expected_receipt_sha256,
        expected_source_revision=expected_source_revision,
        expected_protocol_sha256=expected_protocol_sha256,
        expected_workflow_revision=expected_workflow_revision,
        expected_run_id=expected_run_id,
        expected_run_attempt=expected_run_attempt,
    )


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _github_api(token):
    """Construct a fixed-host HTTPS transport; no request occurs here."""
    if not isinstance(token, str) or not token or any(ord(c) < 32 for c in token):
        raise ReservationRejected("GH_TOKEN must be supplied through the environment")
    opener = build_opener(_NoRedirect())
    allowed_get = re.compile(
        re.escape(REPO_PATH)
        + r"(?:/issues/14/comments\?per_page=100&page=[1-9][0-9]*|/issues/comments/[1-9][0-9]*|/pulls/14|/compare/[0-9a-f]{40}\.\.\.[0-9a-f]{40})?\Z"
    )

    def api(method, path, payload=None):
        if not (
            (
                method == "GET"
                and isinstance(path, str)
                and allowed_get.fullmatch(path)
                and payload is None
            )
            or (
                method == "POST"
                and path == COMMENT_PATH
                and isinstance(payload, dict)
                and set(payload) == {"body"}
                and isinstance(payload["body"], str)
            )
        ):
            raise ReservationRejected(
                "transport only permits the fixed reservation API endpoints"
            )
        data = (
            None
            if payload is None
            else json.dumps(payload, allow_nan=False).encode("utf-8")
        )
        request = Request(
            "https://api.github.com" + path,
            data=data,
            method=method,
            headers={
                "Authorization": "Bearer " + token,
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "NNA-CONTINUAL-003-reservation",
            },
        )
        observed_status = None
        try:
            with opener.open(request, timeout=30) as response:
                observed_status = response.status
                if response.geturl() != request.full_url:
                    raise ReservationTransportError("redirect", observed_status)
                if observed_status != (201 if method == "POST" else 200):
                    raise ReservationTransportError(
                        "unexpected_status", observed_status
                    )
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise ReservationTransportError(
                        "response_too_large", observed_status
                    )
                try:
                    return json.loads(raw)
                except (ValueError, UnicodeError, RecursionError):
                    raise ReservationTransportError(
                        "invalid_json", observed_status
                    ) from None
        except ReservationTransportError:
            raise
        except Exception as exc:
            diagnostic = _error_diagnostic(exc, request=True)
            if diagnostic["http_status"] is None:
                diagnostic["http_status"] = _safe_diagnostic(
                    {"http_status": observed_status}
                )["http_status"]
            # Do not expose the raw exception, its request URL, body or headers.
            raise ReservationTransportError(
                diagnostic["category"], diagnostic["http_status"]
            ) from None

    return api


def _exclusive_output(path):
    output = Path(path).expanduser().resolve()
    if any((ancestor / ".git").exists() for ancestor in (output, *output.parents)):
        raise ReservationRejected(
            "receipt output must be outside every Git working tree"
        )
    try:
        output.mkdir(parents=False, exist_ok=False)
    except OSError:
        raise ReservationRejected(
            "receipt output must be a new writable directory"
        ) from None
    return output


def _write_diagnostic(output, name, value):
    """Write new state via a flushed temporary file in the owned output dir."""
    target = output / name
    if target.exists():
        raise FileExistsError("diagnostic state already exists")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=name + ".", suffix=".partial", dir=output, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(receipt_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass  # A cleanup error must not erase the original IO failure.


def _public_context(args):
    def sha(value, length):
        return (
            value
            if isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value)
            else None
        )

    run = os.environ.get("GITHUB_RUN_ID")
    return {
        "schema_version": 1,
        "mission": guard.MISSION,
        "repository": guard.REPOSITORY,
        "pr_number": guard.PR_NUMBER,
        "source_revision": sha(args.expected_source, 40),
        "workflow_revision": sha(args.expected_workflow, 40),
        "protocol_sha256": sha(args.expected_protocol, 64),
        "workflow_run_id": (
            run if isinstance(run, str) and re.fullmatch(r"[1-9][0-9]*", run) else None
        ),
        "workflow_run_attempt": (
            1 if os.environ.get("GITHUB_RUN_ATTEMPT") == "1" else None
        ),
        "automatic_retry": False,
    }


def _write_stopped(output, context, state, error):
    if output is None:
        return
    summary = {
        **context,
        **state,
        "status": "STOPPED",
        "requires_manual_inspection": state["post_boundary_entered"],
        "error": _error_diagnostic(error),
    }
    try:
        _write_diagnostic(output, "AUTHORIZATION_STATUS.json", summary)
    except Exception:
        # Do not replace the first error with an output error or print either
        # exception's potentially sensitive message. ATTEMPTED remains intact
        # after any POST boundary that could have been crossed.
        print(
            "diagnostic summary persistence failed; original failure retained",
            file=sys.stderr,
        )


def main(argv=None):
    """Future workflow CLI. Never invoke as an engineering preflight."""
    parser = argparse.ArgumentParser(
        description="Reserve one newly approved continual campaign; never run training"
    )
    parser.add_argument("--expected-source", required=True)
    parser.add_argument("--expected-protocol", required=True)
    parser.add_argument("--expected-workflow", required=True)
    parser.add_argument(
        "--output", required=True, help="new directory outside any Git working tree"
    )
    args = parser.parse_args(argv)
    output = None
    context = _public_context(args)
    state = {
        "phase": "protocol_validation",
        "post_boundary_entered": False,
        "reservation_confirmed": False,
    }
    try:
        from .protocol import protocol_config, protocol_sha256

        config = protocol_config()
        if (
            config.get("status") != "FROZEN_APPROVAL_REQUIRED"
            or config.get("execution_authorized") is not False
            or config.get("unresolved_before_freeze") != []
            or protocol_sha256() != args.expected_protocol
        ):
            raise ReservationRejected(
                "candidate/unresolved or mismatched protocol cannot reserve production"
            )
        # Claim the output directory before any remote mutation; never overwrite.
        state["phase"] = "output_claim"
        output = _exclusive_output(args.output)
        state["phase"] = "event_input"
        event_path = os.environ.get("GITHUB_EVENT_PATH")
        if not event_path:
            raise ReservationRejected("GITHUB_EVENT_PATH is required")
        with Path(event_path).open(encoding="utf-8") as handle:
            event = json.load(handle)

        def checkpoint(attempt):
            state["phase"] = "attempt_checkpoint"
            _write_diagnostic(output, "ATTEMPTED.json", attempt)
            state["post_boundary_entered"] = True
            state["phase"] = "reservation_post"

        state["phase"] = "approval_validation"
        result = reserve(
            event,
            args.expected_source,
            args.expected_protocol,
            os.environ.get("GITHUB_RUN_ID"),
            os.environ.get("GITHUB_RUN_ATTEMPT"),
            _github_api(os.environ.get("GH_TOKEN")),
            expected_workflow40=args.expected_workflow,
            before_post=checkpoint,
        )
        state["reservation_confirmed"] = True
        state["phase"] = "receipt_persistence"
        for name in ("receipt", "permit"):
            with (output / f"{name}.json").open(
                "x", encoding="utf-8", newline="\n"
            ) as handle:
                handle.write(receipt_bytes(result[name]).decode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
        with (output / "SHA256SUMS").open(
            "x", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(result["receipt_sha256"] + "  receipt.json\n")
        output_file = os.environ.get("GITHUB_OUTPUT")
        if output_file:
            with Path(output_file).open("a", encoding="utf-8", newline="\n") as handle:
                handle.write("receipt_sha256=" + result["receipt_sha256"] + "\n")
                handle.write(
                    "reservation_id=" + result["permit"]["reservation_id"] + "\n"
                )
        state["phase"] = "status_persistence"
        _write_diagnostic(
            output,
            "AUTHORIZATION_STATUS.json",
            {
                **context,
                **state,
                "phase": "complete",
                "status": "CONFIRMED",
                "requires_manual_inspection": False,
                "error": None,
            },
        )
        state["phase"] = "complete"
    except ReservationRejected as exc:
        print(str(exc), file=sys.stderr)
        _write_stopped(output, context, state, exc)
        return 1
    except Exception as exc:
        print(
            "reservation stopped; inspect the issue before any action; NEVER retry a POST",
            file=sys.stderr,
        )
        _write_stopped(output, context, state, exc)
        return 1
    print(
        "Reservation confirmed; receipt and permit saved. Scientific execution remains a separate workflow step."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
