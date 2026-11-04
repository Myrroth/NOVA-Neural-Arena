"""Invented GitHub API fixtures only; no consent, request, model or gameplay."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from arena.continual import workflow_guard as g
from arena.continual import workflow_reservation as r


A, B, P, RUN = "a" * 40, "b" * 40, "c" * 64, "12345"


def comment(number, body, *, owner=True):
    return {
        "id": number,
        "body": body,
        "issue_url": r.ISSUE_URL,
        "html_url": r.COMMENT_WEB_URL + str(number),
        "user": {
            "login": g.OWNER if owner else "github-actions[bot]",
            "id": g.OWNER_ID if owner else 41898282,
        },
        "author_association": "OWNER" if owner else "NONE",
    }


def event():
    repo = {"full_name": g.REPOSITORY, "private": False, "fork": False}
    return {
        "action": "labeled",
        "number": 14,
        "label": {"name": g.APPROVAL_LABEL},
        "repository": copy.deepcopy(repo),
        "sender": {"login": g.OWNER, "id": g.OWNER_ID},
        "pull_request": {
            "number": 14,
            "state": "open",
            "merged": False,
            "merged_at": None,
            "head": {"sha": B, "ref": g.HEAD_BRANCH, "repo": copy.deepcopy(repo)},
            "base": {
                "sha": "d" * 40,
                "ref": g.BASE_BRANCH,
                "repo": copy.deepcopy(repo),
            },
        },
    }


def approval():
    return {
        "approval_type": g.APPROVAL_TYPE,
        "approved": True,
        "source_commit": A,
        "workflow_head": B,
        "protocol_sha256": P,
        "human_approval_reference": "SYNTHETIC TEST ONLY - NOT HUMAN CONSENT",
        "planning_approval_url": r.COMMENT_WEB_URL + "20",
        "budgets": g.approval_budget(),
    }


class FakeAPI:
    def __init__(self):
        self.event, self.approval = event(), approval()
        self.repo = self.event["repository"]
        self.live = copy.deepcopy(self.event["pull_request"])
        self.comparison = {
            "status": "ahead",
            "behind_by": 0,
            "base_commit": {"sha": A},
            "merge_base_commit": {"sha": A},
            "files": [{"filename": g.WORKFLOW_FILE, "status": "added"}],
        }
        self.recert = comment(
            20,
            f"{r.RECERTIFICATION_PREFIX}{A} | protocol {P} | workflow {B}\nFixture only",
        )
        self.pages = [
            [comment(10, g.APPROVAL_MARKER + json.dumps(self.approval)), self.recert]
        ]
        self.calls, self.posts = [], 0
        self.post_mode, self.fail_get = None, None

    def __call__(self, method, path, payload=None):
        self.calls.append((method, path))
        if method == "GET":
            if path == self.fail_get:
                raise RuntimeError("private-test-token must not be logged")
            if path == r.REPO_PATH:
                return copy.deepcopy(self.repo)
            if path == r.PR_PATH:
                return copy.deepcopy(self.live)
            if path == r.REPO_PATH + f"/compare/{A}...{B}":
                return copy.deepcopy(self.comparison)
            if path == r.COMMENT_API_PATH + "20":
                return copy.deepcopy(self.recert)
            if path.startswith(r.COMMENT_PATH + "?per_page=100&page="):
                index = int(path.rsplit("=", 1)[1]) - 1
                return copy.deepcopy(
                    self.pages[index] if index < len(self.pages) else []
                )
        if method == "POST" and path == r.COMMENT_PATH:
            self.posts += 1
            assert self.posts == 1, "A POST must never be automatically retried"
            posted = comment(9000, payload["body"], owner=False)
            self.pages[0].append(copy.deepcopy(posted))
            if self.post_mode == "lost":
                raise TimeoutError("private-test-token lost response")
            if self.post_mode == "bad":
                posted["body"] = "different"
            if self.post_mode == "none":
                return None
            return posted
        raise AssertionError("Unexpected fake API endpoint: " + path)


def perform(api, **changes):
    return r.reserve(
        changes.get("event", api.event),
        changes.get("source", A),
        changes.get("protocol", P),
        changes.get("run", RUN),
        changes.get("attempt", "1"),
        api,
        expected_workflow40=changes.get("workflow", B),
    )


def expected(result):
    return {
        "expected_receipt_sha256": result["receipt_sha256"],
        "expected_source_revision": A,
        "expected_protocol_sha256": P,
        "expected_workflow_revision": B,
        "expected_run_id": RUN,
        "expected_run_attempt": "1",
    }


def test_full_binding_and_exact_receipt_digest_can_be_consumed_offline(tmp_path):
    api = FakeAPI()
    result = perform(api)
    assert api.posts == 1
    assert result["permit"]["reservation_id"] == r.COMMENT_WEB_URL + "9000"
    assert result["permit"]["workflow_revision"] == B
    assert (
        hashlib.sha256(r.receipt_bytes(result["receipt"])).hexdigest()
        == result["receipt_sha256"]
    )
    for name in ("receipt", "permit"):
        (tmp_path / (name + ".json")).write_bytes(r.receipt_bytes(result[name]))
    actual = r.read_and_validate_receipt(
        tmp_path, permit=result["permit"], **expected(result)
    )
    assert (
        actual["human_approval_reference"] == api.approval["human_approval_reference"]
    )
    assert actual["comment_pages_checked"] == 1
    assert api.calls[-1] == ("POST", r.COMMENT_PATH)


@pytest.mark.parametrize(
    "changes",
    [
        {"attempt": 2},
        {"attempt": True},
        {"attempt": 1.0},
        {"run": 0},
        {"source": "main"},
        {"protocol": "c" * 63},
        {"workflow": "b" * 39},
    ],
)
def test_invalid_execution_binding_fails_before_any_api(changes):
    api = FakeAPI()
    with pytest.raises(r.ReservationRejected):
        perform(api, **changes)
    assert api.calls == []


@pytest.mark.parametrize(
    "target,key,value",
    [
        ("live", "state", "closed"),
        ("live", "merged", True),
        ("comparison", "status", "diverged"),
        ("comparison", "behind_by", 1),
        ("comparison", "merge_base_commit", {"sha": "d" * 40}),
        (
            "comparison",
            "files",
            [{"filename": "arena/continual/trainer.py", "status": "modified"}],
        ),
        (
            "comparison",
            "files",
            [
                {
                    "filename": g.WORKFLOW_FILE,
                    "status": "renamed",
                    "previous_filename": "another.yml",
                }
            ],
        ),
    ],
)
def test_live_pr_and_source_ancestry_are_checked_before_reservation(target, key, value):
    api = FakeAPI()
    getattr(api, target)[key] = value
    with pytest.raises(r.ReservationRejected):
        perform(api)
    assert api.posts == 0


def test_a_push_after_label_or_a_fork_cannot_use_stale_event():
    for change in (
        lambda api: api.live["head"].update(sha="d" * 40),
        lambda api: api.live["head"]["repo"].update(fork=True),
    ):
        api = FakeAPI()
        change(api)
        with pytest.raises(r.ReservationRejected):
            perform(api)
        assert api.posts == 0


def test_every_comment_page_is_read_and_any_prior003_reservation_burns_attempt():
    api = FakeAPI()
    api.pages[0] += [comment(100 + i, "discussion", owner=False) for i in range(98)]
    api.pages.append(
        [
            comment(
                9999,
                g.RESERVATION_PREFIX + "1 attempt 1 | different source failed",
                owner=False,
            )
        ]
    )
    with pytest.raises(r.ReservationRejected, match="existing reservation"):
        perform(api)
    assert api.posts == 0 and api.calls[-1][1].endswith("page=2")


def test_old002_markers_cannot_become_new003_approval_or_recertification():
    api = FakeAPI()
    api.pages[0][0]["body"] = api.pages[0][0]["body"].replace(
        "NNA-CONTINUAL-003", "NNA-MULTIJOGOS-002"
    )
    with pytest.raises(r.ReservationRejected, match="exactly one matching"):
        perform(api)
    assert api.posts == 0
    api = FakeAPI()
    api.recert["body"] = f"NNA-MULTIJOGOS-002:RECERTIFIED | source {A} | protocol {P}"
    with pytest.raises(r.ReservationRejected):
        perform(api)
    assert api.posts == 0


def test_duplicate_matching_owner_records_or_wrong_owner_id_cannot_reserve():
    api = FakeAPI()
    api.pages[0].append(comment(11, g.APPROVAL_MARKER + json.dumps(api.approval)))
    with pytest.raises(r.ReservationRejected, match="exactly one matching"):
        perform(api)
    assert api.posts == 0
    api = FakeAPI()
    api.pages[0][0]["user"]["id"] = 123
    with pytest.raises(r.ReservationRejected, match="owner metadata"):
        perform(api)
    assert api.posts == 0


def test_get_transport_failure_is_redacted_without_a_post():
    api = FakeAPI()
    api.fail_get = r.PR_PATH
    with pytest.raises(r.ReservationRejected) as error:
        perform(api)
    assert "private-test-token" not in str(error.value)
    assert api.posts == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("body", "Narrative approval, not the literal003 first line"),
        ("body", f"{r.RECERTIFICATION_PREFIX}{A} | protocol {P}"),
        ("body", f"{r.RECERTIFICATION_PREFIX}{A} | protocol {P} | workflow {'d'*40}"),
        ("user", {"login": g.OWNER, "id": 1}),
        ("author_association", "CONTRIBUTOR"),
        (
            "issue_url",
            "https://api.github.com/repos/Myrroth/NOVA-Neural-Arena/issues/12",
        ),
    ],
)
def test_planning_literal_and_owner_metadata_must_match_new_source_workflow(
    field, value
):
    api = FakeAPI()
    api.recert[field] = value
    with pytest.raises(r.ReservationRejected):
        perform(api)
    assert api.posts == 0


@pytest.mark.parametrize("outcome", ["lost", "bad", "none"])
def test_unconfirmed_post_never_retries_and_next_call_sees_consumed_attempt(outcome):
    api = FakeAPI()
    api.post_mode = outcome
    with pytest.raises(r.ReservationUncertain, match="NEVER retry") as error:
        perform(api)
    assert "private-test-token" not in str(error.value) and api.posts == 1
    with pytest.raises(r.ReservationRejected):
        perform(api)
    assert api.posts == 1


@pytest.mark.parametrize(
    "field",
    [
        "human_approval_reference",
        "reservation_id",
        "budgets",
        "source_revision",
        "workflow_revision",
        "workflow_run_id",
    ],
)
def test_receipt_and_permit_cannot_drift_or_be_reused_by_another_run(field):
    result = perform(FakeAPI())
    permit = copy.deepcopy(result["permit"])
    permit[field] = {} if field == "budgets" else "different"
    with pytest.raises(g.GateRejected):
        r.validate_receipt(result["receipt"], permit, **expected(result))


def test_file_tampering_and_old_receipt_are_not_accepted(tmp_path):
    result = perform(FakeAPI())
    for name in ("receipt", "permit"):
        (tmp_path / (name + ".json")).write_bytes(r.receipt_bytes(result[name]))
    original = (tmp_path / "receipt.json").read_bytes()
    (tmp_path / "receipt.json").write_bytes(original + b" ")
    with pytest.raises(r.ReservationRejected):
        r.read_and_validate_receipt(tmp_path, **expected(result))
    receipt = copy.deepcopy(result["receipt"])
    receipt["mission"] = "NNA-MULTIJOGOS-002"
    binding = expected(result)
    binding["expected_receipt_sha256"] = hashlib.sha256(
        r.receipt_bytes(receipt)
    ).hexdigest()
    permit = copy.deepcopy(result["permit"])
    permit["reservation_receipt_sha256"] = binding["expected_receipt_sha256"]
    with pytest.raises(r.ReservationRejected):
        r.validate_receipt(receipt, permit, **binding)


def test_import_needs_no_scientific_runtime():
    root = Path(__file__).resolve().parents[1]
    code = "import sys;import arena.continual.workflow_reservation;assert not {'torch','numpy','gymnasium'} & set(sys.modules)"
    result = subprocess.run(
        [sys.executable, "-S", "-c", code], cwd=root, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def configure_cli(monkeypatch, tmp_path, api):
    from arena.continual import protocol

    config = protocol.protocol_config()
    config.update(
        status="FROZEN_APPROVAL_REQUIRED",
        execution_authorized=False,
        unresolved_before_freeze=[],
    )
    monkeypatch.setattr(protocol, "protocol_config", lambda: config)
    monkeypatch.setattr(protocol, "protocol_sha256", lambda: P)
    event_file = tmp_path / "event.json"
    event_file.write_text(json.dumps(api.event), encoding="utf-8")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_file))
    monkeypatch.setenv("GITHUB_RUN_ID", RUN)
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("GH_TOKEN", "private-test-token")
    monkeypatch.setattr(r, "_github_api", lambda token: api)

    # The fake CLI exchange can live in the workspace's pytest folder. Hide
    # only .git detection here, leaving real exclusive mkdir/file operations.
    class ExternalPath(type(Path())):
        def exists(self):
            return False if self.name == ".git" else super().exists()

    monkeypatch.setattr(r, "Path", ExternalPath)
    output = tmp_path / "new-external-gate"
    return output, [
        "--expected-source",
        A,
        "--expected-workflow",
        B,
        "--expected-protocol",
        P,
        "--output",
        str(output),
    ]


def test_cli_fake_exchange_emits_exact_hash_and_consumed_output(
    monkeypatch, tmp_path, capsys
):
    api = FakeAPI()
    output, args = configure_cli(monkeypatch, tmp_path, api)
    outputs = tmp_path / "job-output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(outputs))
    assert r.main(args) == 0 and api.posts == 1
    assert sorted(p.name for p in output.iterdir()) == [
        "SHA256SUMS",
        "permit.json",
        "receipt.json",
    ]
    digest, member = (output / "SHA256SUMS").read_text().split()
    assert (
        member == "receipt.json"
        and hashlib.sha256((output / member).read_bytes()).hexdigest() == digest
    )
    assert outputs.read_text().splitlines() == [
        "receipt_sha256=" + digest,
        "reservation_id=" + r.COMMENT_WEB_URL + "9000",
    ]
    assert r.main(args) == 1 and api.posts == 1
    assert "private-test-token" not in str(capsys.readouterr())


def test_candidate_cli_fails_before_output_or_api(monkeypatch, tmp_path):
    from arena.continual import protocol

    config = protocol.protocol_config()
    config.update(status="CANDIDATE_NOT_AUTHORIZED")
    monkeypatch.setattr(protocol, "protocol_config", lambda: config)
    monkeypatch.setattr(
        r,
        "_github_api",
        lambda token: pytest.fail("candidate accessed credentials/transport"),
    )
    output = tmp_path / "not-created"
    assert (
        r.main(
            [
                "--expected-source",
                A,
                "--expected-workflow",
                B,
                "--expected-protocol",
                P,
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert not output.exists()


def test_cli_lost_post_never_writes_usable_permit(monkeypatch, tmp_path, capsys):
    api = FakeAPI()
    api.post_mode = "lost"
    output, args = configure_cli(monkeypatch, tmp_path, api)
    assert r.main(args) == 1 and api.posts == 1
    assert not (output / "permit.json").exists()
    assert "NEVER retry" in str(capsys.readouterr())


def test_fixed_https_transport_refuses_redirects_unlisted_paths_and_redacts_failure(
    monkeypatch,
):
    requests = []

    class Response:
        def __init__(self, request):
            self.request = request
            self.status = 201 if request.method == "POST" else 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def geturl(self):
            return self.request.full_url

        def read(self, limit):
            assert limit == r.MAX_RESPONSE_BYTES + 1
            return b'{"fixture":true}'

    class Opener:
        fail = False

        def open(self, request, timeout):
            assert (
                timeout == 30
                and request.get_header("Authorization") == "Bearer private-test-token"
            )
            requests.append(request)
            if self.fail:
                raise RuntimeError("private-test-token details")
            return Response(request)

    opener = Opener()
    handlers = []
    monkeypatch.setattr(
        r, "build_opener", lambda handler: (handlers.append(handler) or opener)
    )
    transport = r._github_api("private-test-token")
    assert (
        requests == []
        and handlers[0].redirect_request(
            None, None, 302, "redirect", {}, "https://example.com"
        )
        is None
    )
    for path in (r.REPO_PATH, r.PR_PATH, r.REPO_PATH + f"/compare/{A}...{B}"):
        assert transport("GET", path) == {"fixture": True}
    for path in (
        "https://example.com",
        r.REPO_PATH + "/../other",
        r.COMMENT_PATH,
        "//evil.example",
    ):
        with pytest.raises(r.ReservationRejected):
            transport("GET", path)
    assert len(requests) == 3
    opener.fail = True
    with pytest.raises(r.ReservationRejected) as error:
        transport("GET", r.REPO_PATH)
    assert "private-test-token" not in str(error.value) and len(requests) == 4
