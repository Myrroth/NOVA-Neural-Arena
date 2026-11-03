"""Prepare the future single campaign reservation; importing does nothing.

``reserve`` uses only the supplied ``api(method, path, payload=None)`` callable.
The caller must hold the workflow's global, non-cancelling concurrency lock.
An owner approval comment is an audit attestation: Work must first obtain NEW
trusted human consent and literal source/protocol recertification. This adapter
does not itself authenticate a human conversation or authorize science.

No request is retried. In particular an unconfirmed POST may already have burned
the campaign; it must never be repeated after timeout, failure or bad response.
The optional CLI is for a future explicitly approved workflow, not preflight.
It only reserves and writes the receipt/permit; it never imports a trainer.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
import json
import os
from pathlib import Path
import re
import sys
from urllib.request import HTTPRedirectHandler, Request, build_opener

from . import workflow_guard as guard


REPO_PATH = f"/repos/{guard.REPOSITORY}"
ISSUE_URL = f"https://api.github.com/repos/{guard.REPOSITORY}/issues/12"
COMMENT_PATH = REPO_PATH+"/issues/12/comments"
COMMENT_API_PATH = REPO_PATH+"/issues/comments/"
COMMENT_WEB_URL = f"https://github.com/{guard.REPOSITORY}/issues/12#issuecomment-"
RECERTIFICATION_PREFIX = "NNA-MULTIJOGOS-002:RECERTIFIED | source "
MAX_COMMENT_PAGES = 100
MAX_RESPONSE_BYTES = 8*1024*1024


class ReservationRejected(guard.GateRejected):
    """No confirmed reservation/permit was obtained; do not start science."""


class ReservationUncertain(ReservationRejected):
    """The single POST was attempted; its outcome must not be retried."""


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
    if (expected_id is not None and comment_id != expected_id or
            comment.get("issue_url") != ISSUE_URL or
            comment.get("html_url") != COMMENT_WEB_URL+comment_id):
        raise ReservationRejected(f"{name} must be the exact Issue 12 comment")
    return comment_id


def _owner(comment, name):
    if (_object(comment.get("user"), f"{name} author").get("login") != guard.OWNER or
            comment.get("author_association") != "OWNER"):
        raise ReservationRejected(f"{name} must be authored by the repository OWNER")


def _get(api, path):
    try:
        return api("GET", path)
    except Exception:
        # A transport exception may contain credentials or response details.
        raise ReservationRejected("GitHub GET failed; no reservation was posted") from None


def _all_comments(api):
    comments, seen_ids = [], set()
    for page in range(1, MAX_COMMENT_PAGES+1):
        response = _get(api, f"{COMMENT_PATH}?per_page=100&page={page}")
        if not isinstance(response, list) or len(response) > 100:
            raise ReservationRejected("Issue 12 pagination must return JSON arrays of at most 100 comments")
        for comment in response:
            comment = _object(comment, "Issue 12 comment")
            comment_id = _location(comment, "Issue 12 comment")
            if comment_id in seen_ids:
                raise ReservationRejected("duplicate paginated comment ID; completeness cannot be established")
            if not isinstance(comment.get("body"), str):
                raise ReservationRejected("Issue 12 comment body must be text")
            seen_ids.add(comment_id)
            comments.append(comment)
        if len(response) < 100:
            return comments, page
    raise ReservationRejected("Issue 12 exceeded the 100-page bound; completeness cannot be established")


def reserve(event, expected_source40, expected_protocol64, run_id, run_attempt, api):
    """Verify all gates, POST once, then return ``{'receipt', 'permit'}``.

    API responses must be decoded trusted GitHub JSON, not user-supplied approval
    objects. All comment pages are read before any POST. Owner attestations for
    other immutable source/head/protocol combinations do not match this attempt.
    Duplicate matching approvals, malformed marked approvals, or any durable
    reservation consume/deny the attempt rather than selecting a convenient one.
    """
    _sha(expected_source40, 40, "expected_source40")
    _sha(expected_protocol64, 64, "expected_protocol64")
    run_id = _id(run_id, "run_id")
    if not ((type(run_attempt) is int and run_attempt == 1) or
            (type(run_attempt) is str and run_attempt == "1")):
        raise ReservationRejected("only run_attempt 1 is permitted; no retry after failure")
    if not callable(api):
        raise ReservationRejected("an explicit API callable is required")
    event = _object(event, "event")
    pr = _object(event.get("pull_request"), "pull_request")
    head = _object(pr.get("head"), "head")
    workflow_head = head.get("sha")
    _sha(workflow_head, 40, "event head")
    repository = _object(_get(api, REPO_PATH), "repository response")
    if repository.get("full_name") != guard.REPOSITORY or repository.get("private") is not False:
        raise ReservationRejected("the exact public repository is required")
    comments, pages = _all_comments(api)
    matches = []
    for comment in comments:
        if comment["body"].startswith(guard.APPROVAL_MARKER):
            try:
                approval = guard.parse_approval_comment(comment)
            except guard.GateRejected:
                raise ReservationRejected("a marked approval has invalid owner metadata or JSON") from None
            if (approval.get("workflow_head") == workflow_head and
                    approval.get("source_commit") == expected_source40 and
                    approval.get("protocol_sha256") == expected_protocol64):
                matches.append((comment, approval))
    if len(matches) != 1:
        raise ReservationRejected("exactly one matching owner approval is required")
    approval_comment, approval = matches[0]
    try:
        gated = guard.validate_event(event, approval, run_id, run_attempt, comments)
    except guard.GateRejected:
        raise ReservationRejected("workflow event, finite approval or existing reservation gate rejected") from None
    approval_id = _location(approval_comment, "approval")
    # validate_event has already constrained this URL to the exact Issue 12 URL.
    planning_url = approval["planning_approval_url"]
    planning_id = planning_url.removeprefix(COMMENT_WEB_URL)
    _id(planning_id, "planning comment id")
    recertification = _get(api, COMMENT_API_PATH+planning_id)
    _location(recertification, "recertification", expected_id=planning_id)
    _owner(recertification, "recertification")
    required_line = f"{RECERTIFICATION_PREFIX}{expected_source40} | protocol {expected_protocol64}"
    if (not isinstance(recertification.get("body"), str) or
            recertification["body"].split("\n", 1)[0] != required_line):
        raise ReservationRejected("recertification first line must literally bind this source and protocol")
    reservation_text = gated["reservation_text"]
    try:
        posted = api("POST", COMMENT_PATH, {"body": reservation_text})
    except Exception:
        raise ReservationUncertain("reservation POST unconfirmed; the campaign may be consumed; NEVER retry") from None
    try:
        reservation_id = _location(posted, "posted reservation")
        if posted.get("body") != reservation_text:
            raise ReservationRejected("posted reservation body differs from the requested immutable marker")
    except (ReservationRejected, AttributeError, TypeError):
        raise ReservationUncertain("reservation POST response is ambiguous; the campaign may be consumed; NEVER retry") from None
    reservation_url = COMMENT_WEB_URL+reservation_id
    permit = copy.deepcopy(gated["permit"])
    permit["reservation_id"] = reservation_url
    receipt = {
        "schema_version": 1, "mission": "NNA-MULTIJOGOS-002", "repository": guard.REPOSITORY,
        "workflow_run_id": run_id, "workflow_run_attempt": 1, "workflow_head": workflow_head,
        "source_commit": expected_source40, "protocol_sha256": expected_protocol64,
        "approval_comment_id": approval_id, "approval_comment_url": COMMENT_WEB_URL+approval_id,
        "planning_approval_url": planning_url, "reservation_comment_id": reservation_id,
        "reservation_url": reservation_url, "reservation_text": reservation_text,
        "comment_pages_checked": pages, "comments_checked": len(comments),
        "budgets": copy.deepcopy(approval["budgets"]),
    }
    return {"receipt": receipt, "permit": permit}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _github_api(token):
    """Construct a fixed-host HTTPS transport; no request occurs here."""
    if not isinstance(token, str) or not token or any(ord(c) < 32 for c in token):
        raise ReservationRejected("GH_TOKEN must be supplied through the environment")
    opener = build_opener(_NoRedirect())
    allowed_get = re.compile(
        re.escape(REPO_PATH)+r"(?:/issues/12/comments\?per_page=100&page=[1-9][0-9]*|/issues/comments/[1-9][0-9]*)?\Z")

    def api(method, path, payload=None):
        if not ((method == "GET" and isinstance(path, str) and allowed_get.fullmatch(path) and payload is None)
                or (method == "POST" and path == COMMENT_PATH and isinstance(payload, dict)
                    and set(payload) == {"body"} and isinstance(payload["body"], str))):
            raise ReservationRejected("transport only permits the fixed reservation API endpoints")
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
        request = Request("https://api.github.com"+path, data=data, method=method,
                          headers={"Authorization": "Bearer "+token, "Accept": "application/vnd.github+json",
                                   "Content-Type": "application/json", "X-GitHub-Api-Version": "2022-11-28",
                                   "User-Agent": "NNA-MULTIJOGOS-002-reservation"})
        try:
            with opener.open(request, timeout=30) as response:
                if response.geturl() != request.full_url or response.status != (201 if method == "POST" else 200):
                    raise ReservationRejected("unexpected reservation API status or endpoint")
                raw = response.read(MAX_RESPONSE_BYTES+1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise ReservationRejected("reservation API response exceeds the finite size bound")
                return json.loads(raw)
        except Exception:
            # Suppress raw HTTP errors/URLs/headers, which can contain secrets.
            raise ReservationRejected("reservation API request failed; no automatic retry") from None
    return api


def _exclusive_output(path):
    output = Path(path).expanduser().resolve()
    if any((ancestor/".git").exists() for ancestor in (output, *output.parents)):
        raise ReservationRejected("receipt output must be outside every Git working tree")
    try:
        output.mkdir(parents=False, exist_ok=False)
    except OSError:
        raise ReservationRejected("receipt output must be a new writable directory") from None
    return output


def main(argv=None):
    """Future workflow CLI. Never invoke as an engineering preflight."""
    parser = argparse.ArgumentParser(description="Reserve one newly approved multigame campaign; never run training")
    parser.add_argument("--expected-source", required=True)
    parser.add_argument("--expected-protocol", required=True)
    parser.add_argument("--output", required=True, help="new directory outside any Git working tree")
    args = parser.parse_args(argv)
    try:
        # Claim the output directory before any remote mutation; never overwrite.
        output = _exclusive_output(args.output)
        event_path = os.environ.get("GITHUB_EVENT_PATH")
        if not event_path:
            raise ReservationRejected("GITHUB_EVENT_PATH is required")
        with Path(event_path).open(encoding="utf-8") as handle:
            event = json.load(handle)
        result = reserve(event, args.expected_source, args.expected_protocol,
                         os.environ.get("GITHUB_RUN_ID"), os.environ.get("GITHUB_RUN_ATTEMPT"),
                         _github_api(os.environ.get("GH_TOKEN")))
        for name in ("receipt", "permit"):
            with (output/f"{name}.json").open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(result[name], indent=2, sort_keys=True, allow_nan=False)+"\n")
                handle.flush()
                os.fsync(handle.fileno())
    except ReservationRejected as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception:
        print("reservation stopped; inspect the issue before any action; NEVER retry a POST", file=sys.stderr)
        return 1
    print("Reservation confirmed; receipt and permit saved. Scientific execution remains a separate workflow step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
