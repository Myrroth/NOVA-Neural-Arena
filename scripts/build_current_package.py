"""Build or verify the single ATUAL preservation package without running models.

Only explicitly selected, bounded inputs are included. Existing packages and
original scientific archives are never overwritten, unpacked in place, or saved
again. Cloud upload, readback and retirement are separate audited operations.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path
import re
import stat
import subprocess
import sys
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from arena.shared.integrity import read_verified_bytes, safe_relative_path, validate_sha256
from arena.shared.preservation import OFFICIAL_ARCHIVES

PACKAGE_NAME = "NOVA-Neural-Arena-ATUAL.zip"
APPROVED_SOURCE = "8468e0dad3a75ecfe06187d8dbbf7a523f468df0"
APPROVED_EXECUTOR = "39461168db30edb90a25385fb3a1eaca8dfdde33"
APPROVED_PROTOCOL = "ceb65ad7a9529d091611a7e0f60cc3b937abfc76d00cc8fd766b07557ed5160a"
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_PACKAGE_BYTES = 512 * 1024 * 1024
_MANIFEST = re.compile(r"([0-9a-f]{64})  (.+)")
_SIGNED = re.compile(r"https?://[^\s\"'<>]+[?&](?:sig|signature|access_token|token|X-Amz-[^=&#]+)=", re.I)
_TOKEN = re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")
_ALLOWED = {".json", ".csv", ".tsv", ".md", ".txt", ".py", ".png", ".svg", ".pdf", ".zip", ".ipynb"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(data: object) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _read(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"missing, unsafe or oversized package input: {path}")
    data = path.read_bytes()
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("package input grew beyond byte limit")
    return data


def _check_text(data: bytes, name: str) -> None:
    if Path(name).suffix.lower() in {".json", ".csv", ".tsv", ".md", ".txt", ".py", ".svg", ".ipynb"}:
        text = data.decode("utf-8-sig")
        if _SIGNED.search(text) or _TOKEN.search(text):
            raise ValueError(f"ephemeral URL or credential-like value in selected input: {name}")


def _zip_files(data: bytes) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    with zipfile.ZipFile(BytesIO(data)) as archive:
        infos = archive.infolist()
        if len(infos) > 2000 or sum(i.file_size for i in infos) > MAX_PACKAGE_BYTES:
            raise ValueError("archive exceeds recovery limits")
        folded: set[str] = set()
        for item in infos:
            safe_relative_path(Path.cwd() / "__package_path_validation__", item.filename)
            kind = stat.S_IFMT(item.external_attr >> 16)
            if item.is_dir() or kind not in (0, stat.S_IFREG) or item.flag_bits & 1:
                raise ValueError("archive must contain only ordinary unencrypted files")
            if item.filename.casefold() in folded:
                raise ValueError("case-colliding archive members")
            folded.add(item.filename.casefold())
            result[item.filename] = archive.read(item)  # verifies member CRC
    return result


def _verify_inner_manifest(files: dict[str, bytes], *, exact: bool) -> dict[str, str]:
    if "SHA256SUMS" not in files:
        raise ValueError("missing SHA256SUMS")
    listed: dict[str, str] = {}
    folded: set[str] = set()
    for line in files["SHA256SUMS"].decode("utf-8-sig").splitlines():
        match = _MANIFEST.fullmatch(line)
        if not match:
            raise ValueError("invalid package checksum entry")
        digest, name = match.groups()
        safe_relative_path(Path.cwd() / "__package_path_validation__", name)
        if name.casefold() in folded or name not in files or _sha(files[name]) != digest:
            raise ValueError("duplicate, missing or corrupt checksum member")
        listed[name] = digest
        folded.add(name.casefold())
    if not listed or (exact and set(listed) != set(files) - {"SHA256SUMS"}):
        raise ValueError("checksum member set differs from package contents")
    return listed


def verify_package(path: Path, expected_sha256: str, *, extract_to: Path | None = None) -> dict:
    """Check trusted outer identity and complete inner coverage before extraction."""
    data = read_verified_bytes(path, expected_sha256=expected_sha256, max_bytes=MAX_PACKAGE_BYTES)
    files = _zip_files(data)
    hashes = _verify_inner_manifest(files, exact=True)
    inventory = json.loads(files["INVENTORY.json"])
    if inventory.get("mission") != "NNA-FECHAMENTO-001" or inventory.get("schema_version") != 1:
        raise ValueError("unexpected ATUAL inventory identity")
    catalog = inventory.get("payload", {})
    if set(catalog) != set(files) - {"SHA256SUMS", "INVENTORY.json"} or any(
        entry != {"bytes": len(files[name]), "sha256": _sha(files[name])}
        for name, entry in catalog.items()
    ):
        raise ValueError("inventory content catalog differs from actual payload")
    if extract_to is not None:
        extract_to.mkdir(parents=True, exist_ok=False)
        for name, content in files.items():
            target = safe_relative_path(extract_to, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as output:
                output.write(content)
        for name, digest in hashes.items():
            if _sha(safe_relative_path(extract_to, name).read_bytes()) != digest:
                raise ValueError("local recovery differs from package")
    return {"bytes": len(data), "sha256": _sha(data), "members_verified": len(hashes),
            "mission": inventory["mission"], "local_recovery_verified": extract_to is not None}


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True, stderr=subprocess.STDOUT).strip()


def _gather(directory: Path, prefix: str, payload: dict[str, bytes]) -> None:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"missing selected directory: {directory}")
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    if not files or len(files) > 500:
        raise ValueError("selected evidence/analysis directory must have 1..500 files")
    for path in files:
        relative = path.relative_to(directory).as_posix()
        safe_relative_path(directory, relative)
        if ((path.suffix.lower() not in _ALLOWED and path.name != "SHA256SUMS")
                or any(part.startswith(".") for part in path.relative_to(directory).parts)):
            raise ValueError(f"unapproved evidence file kind/path: {relative}")
        content = _read(path)
        name = prefix + "/" + relative
        _check_text(content, name)
        payload[name] = content


def build_package(inputs_path: Path, out: Path, *, repo: Path = REPO_ROOT) -> dict:
    """Build only after final scientific data/report and a committed source exist."""
    cfg = json.loads(_read(inputs_path))
    if cfg.get("schema_version") != 1 or cfg.get("mission") != "NNA-FECHAMENTO-001":
        raise ValueError("wrong package input schema/mission")
    if out.exists():
        raise FileExistsError("package destination already exists; never overwrite a generation")
    if _git(repo, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("commit final tracked source/documentation before packaging")
    head = _git(repo, "rev-parse", "HEAD")
    if cfg.get("final_source_commit") != head or cfg.get("approved_source_commit") != APPROVED_SOURCE:
        raise ValueError("source commit must match current committed HEAD and fixed approved source")
    if cfg.get("scientific_executor_commit") != APPROVED_EXECUTOR or cfg.get("protocol_sha256") != APPROVED_PROTOCOL:
        raise ValueError("scientific source/protocol differs from approved identity")

    payload: dict[str, bytes] = {}
    metadata = json.loads(_read(Path(cfg["historical_metadata_path"])))
    historical = [a for run in metadata.values() for a in run["artifacts"]]
    if len(historical) != 6 or {a["id"] for a in historical} != set(OFFICIAL_ARCHIVES):
        raise ValueError("exactly six historical official archives are required")
    old_records = []
    for artifact in historical:
        name = artifact["name"] + ".zip"
        path = safe_relative_path(Path(cfg["historical_archive_directory"]), name)
        expected = OFFICIAL_ARCHIVES[artifact["id"]]
        data = read_verified_bytes(path, expected_sha256=expected, max_bytes=MAX_FILE_BYTES)
        if artifact["digest"] != "sha256:" + expected or len(data) != artifact["size_in_bytes"]:
            raise ValueError("historical metadata differs from authenticated ZIP")
        files = _zip_files(data)
        hashes = _verify_inner_manifest(files, exact=artifact["id"] != 13106764292)
        payload["scientific/original-zips/" + name] = data
        old_records.append({"artifact_id": artifact["id"], "archive": name,
                            "sha256": expected, "bytes": len(data), "inner_hashes": len(hashes)})

    phase5 = cfg["phase5"]
    if phase5.get("status") != "completed" or phase5.get("run_attempt") != 1:
        raise ValueError("final ATUAL requires completed unique Phase 5, attempt 1")
    data = read_verified_bytes(Path(phase5["archive_path"]), expected_sha256=phase5["sha256"], max_bytes=MAX_FILE_BYTES)
    f5_files = _zip_files(data)
    _verify_inner_manifest(f5_files, exact=True)
    protocol = json.loads(f5_files["protocol.json"])
    status = json.loads(f5_files["status.json"])
    execution = json.loads(f5_files["execution.json"])
    rows = list(csv.DictReader(StringIO(f5_files["episodes.csv"].decode("utf-8-sig"))))
    keys = {(r["scenario_id"], r["environment_seed"], r["policy_id"]) for r in rows}
    expected = {(s["id"], str(seed), p) for s in protocol["scenarios"]
                for seed in range(500001, 500051) for p in protocol["policies"]}
    protocol_content = {k: v for k, v in protocol.items() if k != "approved_protocol_sha256"}
    protocol_digest = _sha(json.dumps(protocol_content, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    steps = [int(row["steps"]) for row in rows]
    if (not status.get("complete") or len(rows) != 32200 or len(keys) != 32200 or keys != expected
            or status.get("episodes_written") != 32200 or not status.get("weights_frozen_checked_at_end")
            or status.get("steps_used", 6440001) > 6440000
            or sum(steps) != status.get("steps_used") or any(not 1 <= n <= 200 for n in steps)
            or protocol_digest != APPROVED_PROTOCOL
            or protocol.get("approved_protocol_sha256") != APPROVED_PROTOCOL
            or execution.get("evaluator_sha") != APPROVED_EXECUTOR
            or str(execution.get("github_run_id")) != str(phase5["run_id"])
            or str(execution.get("github_run_attempt")) != "1" or not execution.get("no_training")):
        raise ValueError("Phase 5 identity/matrix/frozen result is not the approved complete campaign")
    if execution.get("frozen_weights_before") != status.get("frozen_weights_reference"):
        raise ValueError("Phase 5 frozen-weight references disagree")
    phase5_name = f"phase5-run-{phase5['run_id']}-attempt-1.zip"
    payload["scientific/original-zips/" + phase5_name] = data
    _gather(Path(cfg["analysis_directory"]), "scientific/final-analysis", payload)
    _gather(Path(cfg["evidence_directory"]), "evidence", payload)

    out.mkdir(parents=True, exist_ok=False)
    staging = out / "staging"
    staging.mkdir()
    for commit, label in ((APPROVED_SOURCE, "approved"), (head, "current")):
        target = staging / f"source-{label}-{commit}.tar"
        _git(repo, "archive", "--format=tar", "--output=" + str(target.resolve()), commit)
        payload["source/" + target.name] = _read(target)
    bundle = staging / "full-source-history.bundle"
    _git(repo, "bundle", "create", str(bundle.resolve()), "--all")
    verification = _git(repo, "bundle", "verify", str(bundle.resolve()))
    payload["source/full-source-history.bundle"] = _read(bundle)
    payload["evidence/git-bundle-verification.txt"] = (verification + "\n").encode("utf-8")
    for name in ("scripts/build_current_package.py", "arena/__init__.py", "arena/shared/__init__.py",
                 "arena/shared/integrity.py", "arena/shared/preservation.py"):
        payload["recovery-tools/" + name] = _read(repo / name)
    payload["RECOVERY.md"] = _read(repo / "docs/CURRENT_PACKAGE.md")
    inventory = {
        "schema_version": 1, "mission": cfg["mission"], "repository": "Myrroth/NOVA-Neural-Arena",
        "created_utc": datetime.now(timezone.utc).isoformat(), "approved_source_commit": APPROVED_SOURCE,
        "final_source_commit": head, "scientific_executor_commit": APPROVED_EXECUTOR,
        "approved_protocol_sha256": APPROVED_PROTOCOL, "historical_archives": old_records,
        "phase5": {k: v for k, v in phase5.items() if k != "archive_path"},
        "limitations": cfg.get("limitations", []), "no_models_modified": True,
        "packaging_executed_episodes": 0,
        "cloud_verification": "See the external upload/readback receipt; this archive cannot attest its own upload.",
        "payload": {name: {"bytes": len(value), "sha256": _sha(value)} for name, value in sorted(payload.items())},
    }
    payload["INVENTORY.json"] = _json(inventory)
    if sum(map(len, payload.values())) > MAX_PACKAGE_BYTES:
        raise ValueError("selected package payload exceeds size limit")
    payload["SHA256SUMS"] = "".join(f"{_sha(value)}  {name}\n" for name, value in sorted(payload.items())).encode("utf-8")
    package = out / PACKAGE_NAME
    with zipfile.ZipFile(package, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, value in sorted(payload.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, value, compresslevel=9)
    digest = _sha(package.read_bytes())
    receipt = verify_package(package, digest, extract_to=out / "recovery-verification")
    receipt.update({"package": str(package.resolve()), "source_commit": head,
                    "remote_roundtrip_verified": False, "scientific_archives": 7})
    (out / (PACKAGE_NAME + ".sha256")).write_text(digest + "  " + PACKAGE_NAME + "\n", encoding="ascii")
    (out / "package-receipt.json").write_bytes(_json(receipt))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--extract-to", type=Path)
    args = parser.parse_args()
    if args.verify:
        if not args.expected_sha256 or args.inputs or args.out:
            parser.error("verification requires --expected-sha256, without --inputs/--out")
        result = verify_package(args.verify, validate_sha256(args.expected_sha256), extract_to=args.extract_to)
    else:
        if not args.inputs or not args.out or args.expected_sha256 or args.extract_to:
            parser.error("building requires --inputs and a NEW --out")
        result = build_package(args.inputs, args.out)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
