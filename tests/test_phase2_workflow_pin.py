"""Static workflow regression checks — no training, no model evaluation."""
from pathlib import Path
import re

EVALUATOR_SHA = "f9e1b171a8c045f8bd7c874432114e4bc654665e"
WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/phase2-on-label.yml"


def test_frozen_evaluator_checkout_uses_reviewed_literal_commit():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert re.search(r"(?m)^\s+ref: " + EVALUATOR_SHA + r"$", workflow)
    assert re.search(r"(?m)^\s+EXPECTED_HEAD: " + EVALUATOR_SHA + r"$", workflow)
    assert 'test "$(git rev-parse HEAD)" = "$EXPECTED_HEAD"' in workflow
    assert "github.event.pull_request.head.sha" not in workflow


def test_only_new_approval_label_can_dispatch_phase2_evaluation():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert re.search(r"(?m)^on:\s*$", workflow)
    assert re.search(r"(?m)^\s+types: \[labeled\]\s*$", workflow)
    assert "github.event.label.name == 'neural-phase2-evaluation-approved'" in workflow
    assert "github.event.pull_request.number == 2" in workflow
    assert "github.event.pull_request.head.repo.full_name == github.repository" in workflow
    assert "github.event.pull_request.base.repo.full_name == github.repository" in workflow
    assert not re.search(r"(?m)^\s*(?:push|pull_request_target|workflow_dispatch):", workflow)
    assert "--authorized" in workflow
    assert "python -m arena.training.train" not in workflow
