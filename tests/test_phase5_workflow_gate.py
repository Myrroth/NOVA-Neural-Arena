"""Static CI tests: no official Phase 5 evaluation or training is executed."""
from pathlib import Path
import re

WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/nna-phase5-once.yml"
EVALUATOR = "39461168db30edb90a25385fb3a1eaca8dfdde33"
PROTOCOL = "ceb65ad7a9529d091611a7e0f60cc3b937abfc76d00cc8fd766b07557ed5160a"


def test_phase5_pins_immutable_tested_code_and_approved_manifest():
    wf = WORKFLOW.read_text(encoding="utf-8")
    assert f"ref: {EVALUATOR}" in wf
    assert f"NNA_PHASE5_EVALUATOR_SHA: {EVALUATOR}" in wf
    assert f"NNA_PHASE5_APPROVED_HASH: {PROTOCOL}" in wf
    assert 'test "$(git rev-parse HEAD)" = "$NNA_PHASE5_EVALUATOR_SHA"' in wf
    assert "github.event.pull_request.head.sha" not in wf
    assert "persist-credentials: false" in wf


def test_phase5_strict_label_and_single_reservation():
    wf = WORKFLOW.read_text(encoding="utf-8")
    assert re.search(r"(?m)^on:\s*$", wf)
    assert "types: [labeled]" in wf
    assert "github.event.label.name == 'neural-phase5-closure-approved'" in wf
    assert "github.event.pull_request.number == 8" in wf
    assert "github.event.pull_request.head.ref == 'feat/nna-fechamento-001-phase5'" in wf
    assert "github.event.pull_request.base.ref == 'feat/nna-fechamento-001-pong'" in wf
    assert "concurrency:" in wf and "cancel-in-progress: false" in wf
    assert "GITHUB_RUN_ATTEMPT" in wf
    assert 'NNA-FECHAMENTO-001:PHASE5-RESERVED' in wf
    assert "issues: write" in wf
    assert "gh api --method POST" in wf
    assert "NNA_PHASE5_RESERVATION_ID" in wf
    assert not re.search(r"(?m)^\s*(push|workflow_dispatch|pull_request_target):", wf)


def test_phase5_full_budget_no_training_minimal_permissions_and_pinned_actions():
    wf = WORKFLOW.read_text(encoding="utf-8")
    assert "timeout-minutes: 60" in wf
    assert "contents: read" in wf and "actions: read" in wf
    assert "len(rows) == 32200" in wf and "steps_used'] <= 6440000" in wf
    assert "python -m arena.evaluation.phase5_campaign --campaign" in wf
    assert "sha256sum -c SHA256SUMS" in wf
    assert "retention-days: 30" in wf
    assert "if: always()" in wf
    assert "python -m arena.training.train" not in wf
    for line in wf.splitlines():
        if "uses: actions/" in line:
            assert re.search(r"uses: actions/[\w-]+@[a-f0-9]{40}", line)
