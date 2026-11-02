"""The opt-in technical audit can deserialize trusted weights but cannot play."""
from pathlib import Path
import re

SOURCE = Path(__file__).resolve().parents[1] / ".github/workflows/lab-checkpoint-audit.yml"


def test_load_audit_is_separate_opt_in_and_pinned():
    workflow = SOURCE.read_text(encoding="utf-8")
    assert "types: [labeled]" in workflow
    assert "github.event.label.name == 'lab-checkpoint-load-audit'" in workflow
    assert "github.event.pull_request.number == 5" in workflow
    assert "github.event.pull_request.head.repo.full_name == github.repository" in workflow
    assert "github.event.pull_request.base.repo.full_name == github.repository" in workflow
    assert "ref: 6faec4b5a2524a801e71c7f72f8e4859a2eb5d2b" in workflow
    assert "git rev-parse HEAD" in workflow
    assert "contents: read" in workflow and "actions: read" in workflow
    assert "persist-credentials: false" in workflow
    assert not re.search(r"(?m)^\s*(push|pull_request_target|workflow_dispatch|schedule):", workflow)
    assert not re.search(r"(?m)^\s+\w+: write\s*$", workflow)
    assert all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in
               re.findall(r"uses: actions/[\w-]+@([^\s]+)", workflow))


def test_load_audit_has_no_inference_or_campaign_path():
    workflow = SOURCE.read_text(encoding="utf-8")
    assert "load_official_checkpoint(folder, seed=seed)" in workflow
    assert "official_episodes_executed': 0" in workflow
    assert "training_executed': False" in workflow
    assert "artifact-ids: '13101471957,13101072100,13100988601'" in workflow
    assert "run-id: 41350959549" in workflow
    for forbidden in ("--authorized", "--campaign", "greedy_action", "env.step(",
                      "evaluate_episode", "network(", "optimizer", "arena.training.train"):
        # The natural-language step name may state the absence of an optimizer.
        script = workflow.split("python - <<'PY'", 1)[1]
        assert forbidden not in script
