"""Reports reuse existing CSVs only; no episodes, model files or training."""
import csv
import json
import math
from pathlib import Path

import pytest

from arena.evaluation.reporting import create_report, load_rows
from arena.shared.integrity import sha256_file, verify_checksum_manifest


def fixture_source(path: Path, *, phase: int = 4, mutate=None):
    path.mkdir()
    rows = []
    for policy, seed, score, steps, reward, first in (
        ("trained-11", 500001, 2, 10, 1, 3), ("trained-11", 500002, 0, 20, -1, ""),
        ("random", 500001, 0, 10, -.1, ""), ("random", 500002, 1, 10, .9, 2),
    ):
        row = {"policy_id": policy, "environment_seed": seed, "score": score,
               "steps": steps, "reward": reward, "terminated": True,
               "truncated": False, "terminal_reason": "collision"}
        if phase >= 3:
            row.update(board_width=10, board_height=10, max_steps=200,
                       first_food_step=first, food_per_100_steps=100 * score / steps)
        if phase == 4:
            row.update(head_x=5, head_y=5, initial_direction="east")
        if phase == 5:
            row.update(scenario_id="synthetic-example", head_x=5, head_y=5, direction="east",
                       first_food_censored=first == "", repeated_state_visits=int(score == 0))
        rows.append(row)
    if mutate:
        mutate(rows)
    with (path / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (path / "config.json").write_text("{}\n", encoding="utf-8")
    (path / "SHA256SUMS").write_text("".join(
        f"{sha256_file(path / name)}  {name}\n" for name in ("episodes.csv", "config.json")), encoding="utf-8")
    return path


@pytest.mark.parametrize("phase", (2, 3, 4, 5))
def test_historical_and_synthetic_schemas_metrics_and_source_unchanged(tmp_path, phase):
    source = fixture_source(tmp_path / "source", phase=phase)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    output = tmp_path / "report"
    result = create_report(source, output, expected_csv_sha256=sha256_file(source / "episodes.csv"))
    assert result["episodes"] == 4 and result["groups"] == 2
    groups = json.loads((output / "summary.json").read_text())["groups"]
    trained = next(g for g in groups if g["policy_id"] == "trained-11")
    assert trained["score"]["mean"] == trained["score"]["median"] == 1
    assert trained["score"]["std"] == pytest.approx(math.sqrt(2))
    assert trained["reward"]["mean"] == 0 and trained["steps"]["mean"] == 15
    assert trained["food_per_100_steps"]["mean"] == 10
    assert trained["at_least_one_food_rate"] == .5
    assert trained["first_food_censoring_rate"] == (.5 if phase >= 3 else None)
    assert trained["repeated_state_episode_rate"] == (.5 if phase == 5 else None)
    assert trained["first_food_step_mean_if_observed"] == (3 if phase >= 3 else None)
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    verify_checksum_manifest(output, expected_files={"summary.json", "report.md", "provenance.json"})
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["no_new_episodes"] and provenance["no_model_deserialization"]
    assert "n/a" in (output / "report.md").read_text() if phase < 5 else True
    with pytest.raises(FileExistsError):
        create_report(source, output)
    with pytest.raises(ValueError, match="outside"):
        create_report(source, source / "derived")


@pytest.mark.parametrize("mutate", (
    lambda rows: rows.append(rows[0]),
    lambda rows: rows[0].update(score=-1),
    lambda rows: rows[0].update(steps=0),
    lambda rows: rows[0].update(reward="NaN"),
    lambda rows: rows[0].update(environment_seed="1.5"),
    lambda rows: rows[0].update(food_per_100_steps=99),
    lambda rows: rows[0].update(first_food_step=""),
    lambda rows: rows[0].update(first_food_step=11),
    lambda rows: rows[0].update(terminated=False),
    lambda rows: rows[0].update(terminated=False, truncated=True, terminal_reason="step_limit"),
    lambda rows: rows[0].update(head_x=10),
    lambda rows: rows[0].update(initial_direction="northeast"),
    lambda rows: rows[0].update(policy_id=""),
))
def test_invalid_rows_rejected_before_writing(tmp_path, mutate):
    source = fixture_source(tmp_path / "source", mutate=mutate)
    output = tmp_path / "bad"
    with pytest.raises(ValueError):
        create_report(source, output)
    assert not output.exists()


def test_manifest_tampering_and_independent_digest_anchor(tmp_path):
    source = fixture_source(tmp_path / "source")
    with pytest.raises(ValueError, match="independently"):
        create_report(source, tmp_path / "bad", expected_csv_sha256="a" * 64)
    assert not (tmp_path / "bad").exists()
    (source / "config.json").write_text("tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        load_rows(source)


def test_missing_schema_and_duplicate_header_rejected(tmp_path):
    for name, header in (("missing", "policy_id,score\na,1\n"),
                         ("duplicate", "policy_id,policy_id\na,a\n")):
        source = fixture_source(tmp_path / name)
        (source / "episodes.csv").write_text(header)
        (source / "SHA256SUMS").write_text(f"{sha256_file(source / 'episodes.csv')}  episodes.csv\n")
        with pytest.raises(ValueError, match="schema"):
            load_rows(source)


def test_optional_plots_have_readable_policy_labels_png_and_svg(tmp_path):
    pytest.importorskip("matplotlib")
    source = fixture_source(tmp_path / "source")
    output = tmp_path / "plots"
    create_report(source, output, plots=True)
    assert (output / "score-means.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    svg = (output / "score-means.svg").read_text(encoding="utf-8")
    assert "trained-11" in svg and "random" in svg
    assert "observed conditions" in svg and "no inferential intervals" in svg
    metadata = json.loads((output / "provenance.json").read_text())
    assert metadata["matplotlib"]["version"]
    assert metadata["matplotlib"]["backend"].lower() == "agg"
    verify_checksum_manifest(output)
