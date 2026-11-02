"""Guard the CI boundary: synthetic tests and enumeration, never a campaign."""
from pathlib import Path
import re


SOURCE = Path(__file__).resolve().parents[1] / ".github/workflows/lab-infrastructure.yml"


def test_lab_ci_is_read_only_pr_validation_without_campaign_triggers():
    workflow = SOURCE.read_text(encoding="utf-8")
    assert re.search(r"(?m)^on:\s*$", workflow)
    assert re.search(r"(?m)^  pull_request:\s*$", workflow)
    assert "types: [opened, synchronize, reopened]" in workflow
    assert "branches: [feat/phase4-startstate-robustness]" in workflow
    assert not re.search(r"(?m)^\s*(push|pull_request_target|workflow_dispatch|schedule):", workflow)
    assert "labeled" not in workflow and "github.event.label" not in workflow
    assert "contents: read" in workflow
    assert not re.search(r"(?m)^\s+\w+: write\s*$", workflow)
    assert "persist-credentials: false" in workflow
    assert "timeout-minutes: 25" in workflow


def test_lab_ci_checks_frozen_files_using_full_history():
    workflow = SOURCE.read_text(encoding="utf-8")
    assert "fetch-depth: 0" in workflow
    frozen = {
        "24bd38e9e78a14e894f32d7d1c5b5b6d6d724463": (
            "arena/envs/snake.py arena/agents/dqn.py arena/training/train.py "
            "arena/evaluation/benchmark.py"),
        "f9e1b171a8c045f8bd7c874432114e4bc654665e": "arena/evaluation/frozen_phase2.py",
        "8892980eccfb623de30f3803a7550aa1fa31a9c6": "arena/evaluation/phase3_generalization.py",
        "8b4b6696e980191349e803218af6e6d55d8ff7e9": "arena/evaluation/phase4_startstates.py",
    }
    for commit, paths in frozen.items():
        assert f"git diff --exit-code {commit} HEAD -- {paths}" in workflow


def test_lab_ci_only_runs_synthetic_tests_and_phase5_enumeration():
    workflow = SOURCE.read_text(encoding="utf-8")
    assert "python-version: '3.11'" in workflow
    assert "'torch==2.10.0' --index-url https://download.pytorch.org/whl/cpu" in workflow
    assert "python -m pip install -c requirements-ci.txt -e '.[dev]'" in workflow
    assert "python -m pip check" in workflow
    assert re.search(r"(?m)^\s+python -m pytest -q\s*$", workflow)
    assert "python -m arena.evaluation.phase5_hardstates --validate-only" in workflow
    assert "manifest['status'] == 'UNAPPROVED'" in workflow
    assert "--campaign" not in workflow and "--authorized" not in workflow
    assert "python -m arena.training.train" not in workflow
    assert "actions/download-artifact" not in workflow
    assert "load_official_checkpoint" not in workflow
