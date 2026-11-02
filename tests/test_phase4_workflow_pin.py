"""Static regression checks for immutable Phase 4 checkout and approval gate.

Never execute real or synthetic evaluation; strictly inspect the workflow file.
"""
from pathlib import Path
import re

EVALUATOR_SHA = "8b4b6696e980191349e803218af6e6d55d8ff7e9"
SOURCE = Path(__file__).resolve().parents[1] / ".github/workflows/phase4-on-label.yml"


def test_phase4_uses_literal_validated_evaluator_and_sha_guard():
    wf = SOURCE.read_text(encoding="utf-8")
    assert re.search(r"(?m)^\s+ref: " + EVALUATOR_SHA + r"$", wf)
    assert re.search(r"(?m)^\s+APPROVED_EVALUATOR_SHA: " + EVALUATOR_SHA + r"$", wf)
    assert 'test "$(git rev-parse HEAD)" = "$APPROVED_EVALUATOR_SHA"' in wf
    assert "github.event.pull_request.head.sha" not in wf
    assert "fetch-depth: 0" in wf
    assert "persist-credentials: false" in wf


def test_phase4_requires_exclusive_label_in_exact_pr_and_branches():
    wf = SOURCE.read_text(encoding="utf-8")
    assert re.search(r"(?m)^on:\s*$", wf)
    assert re.search(r"(?m)^\s+types: \[labeled\]\s*$", wf)
    assert "github.event.label.name == 'neural-phase4-startstate-approved'" in wf
    assert "github.event.pull_request.number == 4" in wf
    assert "github.event.pull_request.head.repo.full_name == github.repository" in wf
    assert "github.event.pull_request.base.repo.full_name == github.repository" in wf
    assert "github.event.pull_request.head.ref == 'feat/phase4-startstate-robustness'" in wf
    assert "github.event.pull_request.base.ref == 'feat/phase3-ood-board-sizes'" in wf
    assert not re.search(r"(?m)^\s*(?:push|pull_request_target|workflow_dispatch):", wf)
    assert "neural-phase3-generalization-approved" not in wf


def test_phase4_never_trains_and_enforces_metrics_and_budgets():
    wf = SOURCE.read_text(encoding="utf-8")
    assert "python -m arena.evaluation.phase4_startstates" in wf
    assert "python -m arena.training.train" not in wf
    assert "optimizer.step" not in wf
    assert "timeout-minutes: 50" in wf
    assert "contents: read" in wf and "actions: read" in wf
    assert "--authorized" in wf
    assert "len(rows) == 4200" in wf
    assert "set(actual) == expected_scenarios()" in wf
    assert "initial_food_x" in wf and "initial_food_y" in wf
    assert "sha256sum -c SHA256SUMS" in wf
    assert "run-id: 41350959549" in wf
    assert "retention-days: 30" in wf
