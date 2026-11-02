"""Prevent accidental unpinning of Phase 3 evaluator or weakening label gate.

Static checks only; this module never runs a game or downloads checkpoints.
"""
from pathlib import Path
import re

EVALUATOR_COMMIT = "8892980eccfb623de30f3803a7550aa1fa31a9c6"
WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/phase3-on-label.yml"


def test_workflow_uses_only_literal_approved_evaluator_sha():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert re.search(r"(?m)^\s+ref: " + EVALUATOR_COMMIT + r"$", text)
    assert re.search(r"(?m)^\s+APPROVED_EVALUATOR_SHA: " + EVALUATOR_COMMIT + r"$", text)
    assert 'test "$(git rev-parse HEAD)" = "$APPROVED_EVALUATOR_SHA"' in text
    assert "github.event.pull_request.head.sha" not in text
    assert "fetch-depth: 0" in text
    assert "persist-credentials: false" in text


def test_workflow_requires_new_phase3_label_and_exact_pr():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert re.search(r"(?m)^on:\s*$", text)
    assert re.search(r"(?m)^\s+types: \[labeled\]\s*$", text)
    assert "github.event.label.name == 'neural-phase3-generalization-approved'" in text
    assert "github.event.pull_request.number == 3" in text
    assert "github.event.pull_request.head.repo.full_name == github.repository" in text
    assert "github.event.pull_request.base.repo.full_name == github.repository" in text
    assert "github.event.pull_request.head.ref == 'feat/phase3-ood-board-sizes'" in text
    assert "github.event.pull_request.base.ref == 'feat/phase2-frozen-evaluation'" in text
    assert not re.search(r"(?m)^\s*(?:push|pull_request_target|workflow_dispatch):", text)
    assert "neural-phase2-evaluation-approved" not in text


def test_workflow_uses_only_frozen_inference_with_audit_and_limits():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "python -m arena.evaluation.phase3_generalization" in text
    assert "python -m arena.training.train" not in text
    assert "python -m arena.training.train" not in text
    assert "timeout-minutes: 35" in text
    assert "contents: read" in text and "actions: read" in text
    assert "len(rows) == 1050" in text
    assert "--authorized" in text
    assert "sha256sum -c SHA256SUMS" in text
    assert "run-id: 41350959549" in text
