"""Only fake API/HTTP responses: no GitHub request, consent or science occurs."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from arena.universal import workflow_guard as g
from arena.universal import workflow_reservation as r


SOURCE, WORKFLOW, PROTOCOL = "a"*40, "b"*40, "c"*64
RUN_ID = "12345"


def comment(comment_id, body, *, owner=True):
    return {"id": comment_id, "body": body, "issue_url": r.ISSUE_URL,
            "html_url": r.COMMENT_WEB_URL+str(comment_id),
            "user": {"login": g.OWNER if owner else "contributor"},
            "author_association": "OWNER" if owner else "CONTRIBUTOR"}


@pytest.fixture
def fixture():
    repo = {"full_name": g.REPOSITORY, "private": False}
    event = {"action": "labeled", "number": 13, "label": {"name": g.APPROVAL_LABEL},
             "sender": {"login": g.OWNER}, "repository": copy.deepcopy(repo),
             "pull_request": {"number": 13,
                 "head": {"sha": WORKFLOW, "ref": g.HEAD_BRANCH, "repo": copy.deepcopy(repo)},
                 "base": {"ref": g.BASE_BRANCH, "repo": copy.deepcopy(repo)}}}
    approval = {"approval_type": g.APPROVAL_TYPE, "approved": True,
                "workflow_head": WORKFLOW, "source_commit": SOURCE, "protocol_sha256": PROTOCOL,
                "human_approval_reference": "SYNTHETIC TEST ONLY; NOT REAL HUMAN CONSENT",
                "planning_approval_url": r.COMMENT_WEB_URL+"20", "budgets": copy.deepcopy(g.APPROVAL_BUDGET)}
    return event, approval


class FakeAPI:
    def __init__(self, approval):
        self.repo = {"full_name": g.REPOSITORY, "private": False}
        self.recert = comment(20, f"{r.RECERTIFICATION_PREFIX}{SOURCE} | protocol {PROTOCOL}\nSynthetic fixture only")
        self.pages = [[comment(10, g.APPROVAL_MARKER+json.dumps(approval)), copy.deepcopy(self.recert),
                       comment(30, "Narrative mentions "+g.RESERVATION_PREFIX+" without reserving.", owner=False)]]
        self.calls, self.posts = [], 0
        self.get_failure = None
        self.post_failure = False
        self.post_response = "default"

    def __call__(self, method, path, payload=None):
        self.calls.append((method, path, copy.deepcopy(payload)))
        if method == "GET":
            if path == self.get_failure:
                raise RuntimeError("unit-test-secret must never reach a log")
            if path == r.REPO_PATH:
                return copy.deepcopy(self.repo)
            if path.startswith(r.COMMENT_PATH+"?per_page=100&page="):
                page = int(path.rsplit("=", 1)[1])
                return copy.deepcopy(self.pages[page-1] if page <= len(self.pages) else [])
            if path == r.COMMENT_API_PATH+"20":
                return copy.deepcopy(self.recert)
        if method == "POST" and path == r.COMMENT_PATH:
            self.posts += 1
            assert self.posts == 1, "the adapter must never retry POST"
            actual = comment(9000, payload["body"], owner=False)
            self.pages[0].append(actual)  # Model persistence even if the response is lost.
            if self.post_failure:
                raise TimeoutError("unit-test-secret lost POST response")
            return copy.deepcopy(actual if self.post_response == "default" else self.post_response)
        raise AssertionError("unexpected fake API request")


def perform(event, api, **changes):
    return r.reserve(event, changes.get("source", SOURCE), changes.get("protocol", PROTOCOL),
                     changes.get("run_id", RUN_ID), changes.get("attempt", "1"), api)


def replace_approval(api, approval):
    api.pages[0][0]["body"] = g.APPROVAL_MARKER+json.dumps(approval)


def test_reservation_binds_owner_recertification_source_protocol_and_post_response(fixture):
    event, approval = fixture
    before = copy.deepcopy((event, approval))
    api = FakeAPI(approval)
    result = perform(event, api)
    assert (event, approval) == before
    assert [(method, path) for method, path, _ in api.calls] == [
        ("GET", r.REPO_PATH), ("GET", r.COMMENT_PATH+"?per_page=100&page=1"),
        ("GET", r.COMMENT_API_PATH+"20"), ("POST", r.COMMENT_PATH)]
    receipt, permit = result["receipt"], result["permit"]
    assert permit["reservation_id"] == receipt["reservation_url"] == r.COMMENT_WEB_URL+"9000"
    assert permit["source_revision"] == SOURCE and permit["workflow_head"] == WORKFLOW
    assert receipt["approval_comment_url"] == r.COMMENT_WEB_URL+"10"
    assert receipt["planning_approval_url"] == r.COMMENT_WEB_URL+"20"
    assert receipt["comment_pages_checked"] == 1 and receipt["comments_checked"] == 3
    assert receipt["reservation_text"] == api.calls[-1][2]["body"]
    assert receipt["reservation_text"].startswith(g.RESERVATION_PREFIX+RUN_ID+" attempt 1")
    assert f"source {SOURCE} | protocol {PROTOCOL} | workflow {WORKFLOW}" in receipt["reservation_text"]
    permit["budgets"]["automatic_retry"] = True
    assert approval["budgets"]["automatic_retry"] is False


def full_first_page(api):
    return [*api.pages[0][:2], *[comment(100+i, "Unrelated discussion", owner=False) for i in range(98)]]


def test_all_pages_checked_and_real_reservation_on_last_page_blocks_even_current_failed_run(fixture):
    event, approval = fixture
    api = FakeAPI(approval)
    api.pages = [full_first_page(api), [comment(9000, g.RESERVATION_PREFIX+RUN_ID+" attempt 1 | failed")]]
    with pytest.raises(r.ReservationRejected, match="existing reservation"):
        perform(event, api)
    assert api.posts == 0
    assert api.calls[-1][1].endswith("page=2")


def test_exactly_full_page_requires_an_additional_empty_page(fixture):
    event, approval = fixture
    api = FakeAPI(approval)
    api.pages = [full_first_page(api)]
    result = perform(event, api)
    assert result["receipt"]["comment_pages_checked"] == 2
    assert result["receipt"]["comments_checked"] == 100


def test_pagination_bound_never_posts_or_requests_page_101(fixture):
    event, _ = fixture
    calls = []

    def endless(method, path, payload=None):
        calls.append((method, path))
        assert method == "GET"
        if path == r.REPO_PATH:
            return {"full_name": g.REPOSITORY, "private": False}
        page = int(path.rsplit("=", 1)[1])
        return [comment(page*100+i, "No approval") for i in range(100)]

    with pytest.raises(r.ReservationRejected, match="100-page bound"):
        perform(event, endless)
    assert len(calls) == 101 and calls[-1][1].endswith("page=100")


@pytest.mark.parametrize("failure", ("not_array", "too_long", "duplicate", "missing_body", "wrong_issue", "wrong_url"))
def test_incomplete_or_ambiguous_comment_pages_are_rejected_before_post(fixture, failure):
    event, approval = fixture
    api = FakeAPI(approval)
    if failure == "not_array":
        api.pages[0] = {"error": "not a complete comment page"}
    elif failure == "too_long":
        api.pages[0] = [comment(100+i, "x") for i in range(101)]
    elif failure == "duplicate":
        api.pages[0].append(copy.deepcopy(api.pages[0][0]))
    elif failure == "missing_body":
        del api.pages[0][0]["body"]
    elif failure == "wrong_issue":
        api.pages[0][0]["issue_url"] = r.ISSUE_URL.replace("12", "6")
    else:
        api.pages[0][0]["html_url"] += "?misdirected=1"
    with pytest.raises(r.ReservationRejected):
        perform(event, api)
    assert api.posts == 0


@pytest.mark.parametrize("field,value", (("workflow_head", "d"*40), ("source_commit", "d"*40),
                                        ("protocol_sha256", "d"*64)))
def test_approval_must_match_each_literal_binding(fixture, field, value):
    event, approval = fixture
    approval[field] = value
    api = FakeAPI(approval)
    with pytest.raises(r.ReservationRejected, match="exactly one matching"):
        perform(event, api)
    assert api.posts == 0


def test_duplicate_matching_approvals_refused_and_stale_other_source_not_selected(fixture):
    event, approval = fixture
    api = FakeAPI(approval)
    api.pages[0].append(comment(11, g.APPROVAL_MARKER+json.dumps(approval)))
    with pytest.raises(r.ReservationRejected, match="exactly one matching"):
        perform(event, api)
    assert api.posts == 0
    stale = {**approval, "source_commit": "d"*40}
    api.pages[0][-1]["body"] = g.APPROVAL_MARKER+json.dumps(stale)
    assert perform(event, api)["receipt"]["approval_comment_id"] == "10"


@pytest.mark.parametrize("field,value", (("approved", 1), ("approval_type", "old-phase5"),
    ("human_approval_reference", "PENDING"), ("budgets", {}),
    ("planning_approval_url", "https://example.com/approval")))
def test_invalid_attestation_cannot_reach_recertification_or_post(fixture, field, value):
    event, approval = fixture
    approval[field] = value
    api = FakeAPI(approval)
    with pytest.raises(r.ReservationRejected, match="finite approval"):
        perform(event, api)
    assert api.posts == 0
    assert all(path != r.COMMENT_API_PATH+"20" for _, path, _ in api.calls)


def test_budget_and_owner_metadata_mandatory_in_approval(fixture):
    event, approval = fixture
    del approval["budgets"]
    api = FakeAPI(approval)
    with pytest.raises(r.ReservationRejected):
        perform(event, api)
    assert api.posts == 0
    api.pages[0][0]["user"]["login"] = "someone"
    with pytest.raises(r.ReservationRejected, match="owner metadata"):
        perform(event, api)
    assert api.posts == 0


@pytest.mark.parametrize("path,value", (("number", 8), ("action", "synchronize"),
                                       ("label", {"name": "neural-phase5-closure-approved"})))
def test_old_pr_or_wrong_event_cannot_post(fixture, path, value):
    event, approval = fixture
    event[path] = value
    api = FakeAPI(approval)
    with pytest.raises(r.ReservationRejected):
        perform(event, api)
    assert api.posts == 0


def test_fork_cannot_post_even_with_copied_approval(fixture):
    event, approval = fixture
    event["pull_request"]["head"]["repo"]["full_name"] = "someone/fork"
    api = FakeAPI(approval)
    with pytest.raises(r.ReservationRejected):
        perform(event, api)
    assert api.posts == 0


@pytest.mark.parametrize("change", ({"attempt": 2}, {"attempt": True}, {"attempt": 1.0},
    {"run_id": "0"}, {"source": "main"}, {"protocol": "c"*63}))
def test_invalid_execution_binding_fails_before_any_api(fixture, change):
    event, approval = fixture
    api = FakeAPI(approval)
    with pytest.raises(r.ReservationRejected):
        perform(event, api, **change)
    assert api.calls == []


@pytest.mark.parametrize("repo", ({"full_name": g.REPOSITORY, "private": True},
    {"full_name": "someone/NOVA-Neural-Arena", "private": False}, {"full_name": g.REPOSITORY}))
def test_repository_public_state_independently_confirmed(fixture, repo):
    event, approval = fixture
    api = FakeAPI(approval)
    api.repo = repo
    with pytest.raises(r.ReservationRejected, match="exact public repository"):
        perform(event, api)
    assert len(api.calls) == 1 and api.posts == 0


@pytest.mark.parametrize("field,value", (("id", 21), ("user", {"login": "someone"}),
    ("author_association", "CONTRIBUTOR"), ("issue_url", r.ISSUE_URL.replace("12", "6")),
    ("html_url", r.COMMENT_WEB_URL+"21"), ("body", "Narrative recertification"),
    ("body", f"{r.RECERTIFICATION_PREFIX}{'d'*40} | protocol {PROTOCOL}"),
    ("body", f"{r.RECERTIFICATION_PREFIX}{SOURCE} | protocol {'d'*64}"),
    ("body", f"{r.RECERTIFICATION_PREFIX}{SOURCE} | protocol {PROTOCOL}\r\nextra")))
def test_recertification_requires_exact_first_line_and_owner_issue_binding(fixture, field, value):
    event, approval = fixture
    api = FakeAPI(approval)
    api.recert[field] = value
    # Listing uses an independent snapshot, so this tests the direct GET too.
    with pytest.raises(r.ReservationRejected):
        perform(event, api)
    assert api.posts == 0


@pytest.mark.parametrize("path", (r.REPO_PATH, r.COMMENT_PATH+"?per_page=100&page=1", r.COMMENT_API_PATH+"20"))
def test_get_failure_is_redacted_and_never_posts(fixture, path):
    event, approval = fixture
    api = FakeAPI(approval)
    api.get_failure = path
    with pytest.raises(r.ReservationRejected) as error:
        perform(event, api)
    assert "unit-test-secret" not in str(error.value) and api.posts == 0


@pytest.mark.parametrize("outcome", ("lost", "none", "bad_id", "bool_id", "bad_body", "bad_location"))
def test_post_uncertainty_never_retries_or_returns_a_permit(fixture, outcome):
    event, approval = fixture
    api = FakeAPI(approval)
    if outcome == "lost":
        api.post_failure = True
    elif outcome == "none":
        api.post_response = None
    else:
        # A wrong body is enough for these malformed response fixtures.
        api.post_response = comment(9000, "different body")
        if outcome == "bad_id":
            api.post_response["id"] = 0
        elif outcome == "bool_id":
            api.post_response["id"] = True
        elif outcome == "bad_location":
            api.post_response["issue_url"] = r.ISSUE_URL.replace("12", "6")
    with pytest.raises(r.ReservationUncertain, match="NEVER retry") as error:
        perform(event, api)
    assert "unit-test-secret" not in str(error.value) and api.posts == 1
    # The same failed/current run is consumed once the server stored the marker.
    with pytest.raises(r.ReservationRejected):
        perform(event, api)
    assert api.posts == 1


def configure_cli(monkeypatch, tmp_path, event, api, *, isolate_git_detection=True):
    event_file = tmp_path/"event.json"
    event_file.write_text(json.dumps(event), encoding="utf-8")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_file))
    monkeypatch.setenv("GITHUB_RUN_ID", RUN_ID)
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("GH_TOKEN", "unit-test-secret")

    def fake_transport(token):
        assert token == "unit-test-secret"
        return api

    monkeypatch.setattr(r, "_github_api", fake_transport)
    if isolate_git_detection:
        # These tests exercise the CLI/API exchange even when pytest's temporary
        # directory lives inside the repository. Hide only the .git sentinel;
        # the real exclusive mkdir/file checks remain active. The dedicated Git
        # protection test below deliberately opts out of this fixture isolation.
        class ExternalFixturePath(type(Path())):
            def exists(self):
                return False if self.name == ".git" else super().exists()

        monkeypatch.setattr(r, "Path", ExternalFixturePath)
    output = tmp_path/"exclusive-output"
    return output, ["--expected-source", SOURCE, "--expected-protocol", PROTOCOL, "--output", str(output)]


def test_cli_with_fake_api_writes_exclusive_receipt_and_permit(fixture, monkeypatch, tmp_path, capsys):
    event, approval = fixture
    api = FakeAPI(approval)
    output, args = configure_cli(monkeypatch, tmp_path, event, api)
    assert r.main(args) == 0 and api.posts == 1
    receipt = json.loads((output/"receipt.json").read_text(encoding="utf-8"))
    permit = json.loads((output/"permit.json").read_text(encoding="utf-8"))
    assert permit["reservation_id"] == receipt["reservation_url"] == r.COMMENT_WEB_URL+"9000"
    assert sorted(p.name for p in output.iterdir()) == ["permit.json", "receipt.json"]
    assert r.main(args) == 1 and api.posts == 1
    assert "unit-test-secret" not in str(capsys.readouterr())


def test_cli_refuses_output_under_git_before_api(fixture, monkeypatch, tmp_path):
    event, approval = fixture
    api = FakeAPI(approval)
    output, args = configure_cli(monkeypatch, tmp_path, event, api, isolate_git_detection=False)
    (tmp_path/".git").mkdir()
    assert r.main(args) == 1 and not output.exists() and api.calls == []


def test_cli_post_loss_writes_no_permit_and_never_logs_token(fixture, monkeypatch, tmp_path, capsys):
    event, approval = fixture
    api = FakeAPI(approval)
    api.post_failure = True
    output, args = configure_cli(monkeypatch, tmp_path, event, api)
    assert r.main(args) == 1 and api.posts == 1
    assert list(output.iterdir()) == []
    assert "unit-test-secret" not in str(capsys.readouterr())


def test_transport_is_fixed_https_no_redirect_no_retry_and_redacts_failures(monkeypatch):
    opened, handlers = [], []

    class Response:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def geturl(self):
            return "https://api.github.com"+r.REPO_PATH
        def read(self, limit):
            assert limit == r.MAX_RESPONSE_BYTES+1
            return b'{"private":false}'

    class Opener:
        fail = False
        def open(self, request, *, timeout):
            opened.append(request)
            assert timeout == 30
            assert request.get_header("Authorization") == "Bearer unit-test-secret"
            if self.fail:
                raise RuntimeError("unit-test-secret private HTTP detail")
            return Response()

    opener = Opener()
    monkeypatch.setattr(r, "build_opener", lambda handler: (handlers.append(handler) or opener))
    api = r._github_api("unit-test-secret")
    assert opened == [] and isinstance(handlers[0], r._NoRedirect)
    assert handlers[0].redirect_request(None, None, 302, "redirect", {}, "https://example.com") is None
    assert api("GET", r.REPO_PATH) == {"private": False}
    for path in ("https://example.com", "//example.com", r.REPO_PATH+"/../other", r.COMMENT_PATH):
        with pytest.raises(r.ReservationRejected):
            api("GET", path)
    assert len(opened) == 1
    opener.fail = True
    with pytest.raises(r.ReservationRejected) as error:
        api("GET", r.REPO_PATH)
    assert len(opened) == 2 and "unit-test-secret" not in str(error.value)


def test_adapter_imports_with_stdlib_only_without_scientific_modules():
    root = Path(__file__).resolve().parents[1]
    code = ("import sys; import arena.universal.workflow_reservation; "
            "assert not {'torch', 'numpy', 'gymnasium'} & set(sys.modules); "
            "assert 'arena.universal.trainer' not in sys.modules")
    result = subprocess.run([sys.executable, "-S", "-c", code], cwd=root,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
