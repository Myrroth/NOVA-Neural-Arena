"""Synthetic authenticated-file fixtures ONLY, never a game or final-seed run.

920001/2207 below are metadata values required by the viewer schema. All frames,
returns, Q vectors and CSV cells are invented bytes. No policy, simulator,
training, evaluation campaign or external service is started by these tests.
The synthetic SUCCESS documents exercise parsing; they are not mission evidence.
"""
import base64
import csv
import hashlib
from io import BytesIO
import json
import re
import sys
import zipfile

import numpy as np
import pytest

from arena.universal import protocol as p
from arena.universal import viewer as v


NETWORK_SHA = "1"*64


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False, sort_keys=True), encoding="utf-8")


def seal(root):
    entries = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}"
               for path in sorted(root.rglob("*")) if path.is_file() and path != root/"SHA256SUMS"]
    manifest = ("\n".join(entries)+"\n").encode()
    (root/"SHA256SUMS").write_bytes(manifest)
    return hashlib.sha256(manifest).hexdigest()


def csv_rows(root):
    with (root/"episodes.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(root, rows):
    with (root/"episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def recording(tmp_path):
    root = tmp_path/"SYNTHETIC_evaluation_files_not_science"
    root.mkdir()
    rows = []
    for index, cell in enumerate(p.evaluation_matrix()):
        demo = cell.stage == "after_adaptation" and cell.policy == "shared" and cell.model_seed == 2207 and cell.environment_seed == 920001
        rows.append({**vars(cell), "episode": index+1, "steps": 2 if demo else 1,
                     "return": -.9 if demo else 0., "return_clipped": -.9 if demo else 0.,
                     "score": 0., "terminated": 1, "truncated": 0,
                     "environment_id": f"visual-{cell.game}-v1", "network_sha256": NETWORK_SHA,
                     "synthetic_unit_test_fixture": True})
    write_csv(root, rows)
    write_json(root/"EVALUATION_STATUS.json", {"status": "SUCCESS", "test_only": False,
               "completed_episodes": 1735, "synthetic_unit_test_fixture": True})
    originals = {}
    for game_index, game in enumerate(p.GAMES):
        directory = root/"demonstrations"/game
        directory.mkdir(parents=True)
        observations = np.stack([np.full((32, 32, 3), game_index*10+i,
                                         dtype=np.uint8) for i in range(3)])
        frames, after = observations[:-1], observations[1:]
        np.savez_compressed(directory/"frames.npz", rgb=frames, next_rgb=after)
        cell = {"stage": "after_adaptation", "policy": "shared", "game": game,
                "model_seed": 2207, "environment_seed": 920001}
        vector = json.dumps([1.]+[None]*9)
        trace = [{**cell, "step": i+1, "action": 0, "reward": reward,
                  "reward_clipped": reward, "terminated": i == 1, "truncated": False,
                  "rgb_sha256": hashlib.sha256(frames[i].tobytes()).hexdigest(),
                  "next_rgb_sha256": hashlib.sha256(after[i].tobytes()).hexdigest(),
                  "action_mask": [True]+[False]*9, "q_direct": vector,
                  "model_values": vector, "q_mixed": vector,
                  "synthetic_unit_test_fixture": True} for i, reward in enumerate((.1, -1.))]
        write_json(directory/"trajectory.json", {"cell": cell, "network_sha256": NETWORK_SHA,
                   "environment_id": f"visual-{game}-v1", "steps": trace,
                   "synthetic_unit_test_fixture": True})
        originals[game] = observations
    return root, tmp_path/"SYNTHETIC_viewer.html", seal(root), originals


def mutate_trace(root, mutation, game="snake"):
    path = root/"demonstrations"/game/"trajectory.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    mutation(value)
    write_json(path, value)


def test_viewer_replays_exact_five_authenticated_byte_clips_without_importing_learning(recording, monkeypatch):
    root, output, digest, originals = recording
    before = (root/"SHA256SUMS").read_bytes()
    for module in ("arena.universal.envs", "arena.universal.network", "arena.universal.evaluation",
                   "arena.universal.trainer", "torch"):
        monkeypatch.setitem(sys.modules, module, None)
    receipt = v.build_viewer(root, output, expected_manifest_sha256=digest)
    assert receipt["recorded_games"] == 5 and receipt["new_episodes"] == 0
    assert receipt["training_started_by_viewer"] is False
    assert receipt["model_seed"] == 2207 and receipt["environment_seed"] == 920001
    assert receipt["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert (root/"SHA256SUMS").read_bytes() == before
    html = output.read_text(encoding="utf-8")
    payload = json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.S).group(1))
    assert [clip["game"] for clip in payload] == list(p.GAMES)
    for clip in payload:
        assert len(clip["frames"]) == 3 and len(clip["steps"]) == 2
        assert clip["network_sha256"] == NETWORK_SHA
        decoded = np.stack([np.frombuffer(base64.b64decode(frame), dtype=np.uint8).reshape(32, 32, 3)
                            for frame in clip["frames"]])
        assert np.array_equal(decoded, originals[clip["game"]])
    assert "fetch(" not in html and "<script src" not in html and "WebSocket" not in html


def test_output_never_overwrites_existing_file_or_invalidates_authenticated_input(recording):
    root, output, digest, _ = recording
    output.write_text("existing user file", encoding="utf-8")
    with pytest.raises(FileExistsError):
        v.build_viewer(root, output, expected_manifest_sha256=digest)
    assert output.read_text() == "existing user file"
    with pytest.raises(ValueError, match="outside the authenticated"):
        v.build_viewer(root, root/"viewer.html", expected_manifest_sha256=digest)
    assert not (root/"viewer.html").exists()


def test_wrong_trusted_manifest_and_modified_member_fail_before_output(recording):
    root, output, digest, _ = recording
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        v.build_viewer(root, output, expected_manifest_sha256="0"*64)
    (root/"demonstrations"/"snake"/"frames.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        v.build_viewer(root, output, expected_manifest_sha256=digest)
    assert not output.exists()


def test_unlisted_extra_file_is_not_authenticated_by_an_otherwise_valid_manifest(recording):
    root, output, digest, _ = recording
    (root/"unlisted.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="unlisted"):
        v.build_viewer(root, output, expected_manifest_sha256=digest)
    assert not output.exists()


def test_sixth_demo_and_incomplete_matrix_are_not_silently_accepted(recording):
    root, output, _, _ = recording
    extra = root/"demonstrations"/"sixth"
    extra.mkdir()
    (extra/"frames.npz").write_bytes(b"not a registered game")
    with pytest.raises(ValueError, match="exactly the five"):
        v.build_viewer(root, output, expected_manifest_sha256=seal(root))


def test_missing_final_matrix_cell_blocks_even_when_all_five_clips_exist(recording):
    root, output, _, _ = recording
    write_csv(root, csv_rows(root)[:-1])
    with pytest.raises(ValueError, match="complete unique"):
        v.build_viewer(root, output, expected_manifest_sha256=seal(root))


@pytest.mark.parametrize("field,value", [("status", "STOPPED"), ("test_only", True),
                                         ("completed_episodes", 5)])
def test_stopped_test_only_or_partial_run_never_becomes_final_viewer(recording, field, value):
    root, output, _, _ = recording
    status = json.loads((root/"EVALUATION_STATUS.json").read_text())
    status[field] = value
    write_json(root/"EVALUATION_STATUS.json", status)
    with pytest.raises(ValueError, match="completed evaluation"):
        v.build_viewer(root, output, expected_manifest_sha256=seal(root))


@pytest.mark.parametrize("mutation,reason", [
    (lambda m: m["cell"].update(model_seed=1103), "fixed episode"),
    (lambda m: m["cell"].update(environment_seed=920002), "fixed episode"),
    (lambda m: m["steps"][0].update(step=2), "contiguous"),
    (lambda m: m["steps"][0].update(action=9), "available controls"),
    (lambda m: m["steps"][0].update(rgb_sha256="0"*64), "per-step trace hash"),
    (lambda m: m["steps"][0].update(reward_clipped=.5), "clipped reward"),
    (lambda m: m["steps"][0].update(terminated=True), "before the final"),
    (lambda m: m["steps"][-1].update(terminated=False), "prefix"),
    (lambda m: m["steps"][0].update(q_mixed=json.dumps([2.]+[None]*9)), "mixture"),
    (lambda m: m.update(network_sha256="not-a-hash"), "network SHA"),
    (lambda m: m.update(environment_id="<script>"), "environment identity"),
])
def test_authenticated_but_malformed_trace_fails_closed(recording, mutation, reason):
    root, output, _, _ = recording
    mutate_trace(root, mutation)
    with pytest.raises(ValueError, match=reason):
        v.build_viewer(root, output, expected_manifest_sha256=seal(root))
    assert not output.exists()


def test_current_and_next_frames_must_form_a_real_recorded_chain(recording):
    root, output, _, _ = recording
    path = root/"demonstrations"/"snake"/"frames.npz"
    with np.load(path, allow_pickle=False) as archive:
        frames, after = archive["rgb"], archive["next_rgb"]
    after[0] = 251
    np.savez_compressed(path, rgb=frames, next_rgb=after)
    mutate_trace(root, lambda m: m["steps"][0].update(next_rgb_sha256=hashlib.sha256(after[0].tobytes()).hexdigest()))
    with pytest.raises(ValueError, match="discontinuous"):
        v.build_viewer(root, output, expected_manifest_sha256=seal(root))


def test_five_specialist_hashes_cannot_pose_as_one_shared_policy(recording):
    root, output, _, _ = recording
    mutate_trace(root, lambda m: m.update(network_sha256="2"*64), game="pong")
    rows = csv_rows(root)
    for row in rows:
        if row["game"] == "pong" and row["stage"] == "after_adaptation" and row["model_seed"] == "2207" and row["environment_seed"] == "920001":
            row["network_sha256"] = "2"*64
    write_csv(root, rows)
    with pytest.raises(ValueError, match="SAME shared network"):
        v.build_viewer(root, output, expected_manifest_sha256=seal(root))


def test_trace_rewards_must_match_the_authenticated_episode_csv(recording):
    root, output, _, _ = recording
    mutate_trace(root, lambda m: m["steps"][0].update(reward=.2, reward_clipped=.2))
    with pytest.raises(ValueError, match="matrix return"):
        v.build_viewer(root, output, expected_manifest_sha256=seal(root))


def test_archive_header_cannot_allocate_an_unbounded_or_object_frame_array(monkeypatch):
    huge = BytesIO()
    np.lib.format.write_array_header_1_0(huge, {"descr": "|u1", "fortran_order": False,
                                              "shape": (2**30, 32, 32, 3)})
    compressed = BytesIO()
    with zipfile.ZipFile(compressed, "w") as archive:
        archive.writestr("rgb.npy", huge.getvalue())
        archive.writestr("next_rgb.npy", huge.getvalue())
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("invalid declared shape must fail before allocation"))
    with pytest.raises(ValueError, match="RGB header"):
        v._recorded_arrays(compressed.getvalue())


def test_recorded_text_is_json_escaped_not_executed_as_html(recording):
    root, output, _, _ = recording
    attack = "</script><script>globalThis.syntheticInjection=true</script>"
    mutate_trace(root, lambda m: m["steps"][0].update(synthetic_note=attack))
    v.build_viewer(root, output, expected_manifest_sha256=seal(root))
    html = output.read_text(encoding="utf-8")
    assert attack not in html and "\\u003c/script>" in html
