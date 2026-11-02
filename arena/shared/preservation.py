"""Offline recovery of the six original, byte-for-byte scientific archives.

No network, checkpoint deserialization, games, or modification of source ZIPs.
The committed GitHub metadata catalog anchors their SHA-256 digests.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
import stat
import zipfile

from arena.shared.integrity import (
    read_verified_bytes, safe_relative_path, verify_checksum_manifest, write_new_json,
)

OFFICIAL_ARCHIVES = {
    13101471957: "f8e3ea85e2e7a01a2522ab24d7e96fa024d0232dec71f0f3171af25441ec1a74",
    13101072100: "3bc28cc078a38323539090284c80976f53351cd894a10882a2d7138e52359642",
    13100988601: "0c2dcde1586a3be8e3c55b3ff67c67d4f56dca53012b96b0d690db517ad54b48",
    13106764292: "cb74f81d59c3eeb37111998fa977346af381e3013c3116c6065e4e1569eed4cf",
    13108893868: "75c12daa2f5b0a126eed3d0ca8b72c9938d22f937e6911f27a5b5b5dcc83c2c4",
    13110740130: "dc4bb9640b5e54ce3dcc53ea7b766c880097810ca37ec4e0532549dd75dc9289",
}
EPISODE_COUNTS = {13101471957: 80, 13101072100: 80, 13100988601: 80,
                  13106764292: 350, 13108893868: 1050, 13110740130: 4200}


def extract_verified_zip(source: Path, destination: Path, *, expected_sha256: str) -> None:
    """Validate all members before creating an exclusive destination directory."""
    data = read_verified_bytes(source, expected_sha256=expected_sha256, max_bytes=16 * 1024 * 1024)
    with zipfile.ZipFile(BytesIO(data)) as archive:
        members = archive.infolist()
        if len(members) > 1000 or sum(m.file_size for m in members) > 64 * 1024 * 1024:
            raise ValueError("archive exceeds preservation size limits")
        names: set[str] = set()
        for member in members:
            # Official archives are flat regular files; never extract links/devices.
            if member.is_dir() or "/" in member.filename:
                raise ValueError("expected a flat archive of regular files")
            safe_relative_path(destination, member.filename)
            mode = member.external_attr >> 16
            if stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                raise ValueError("archive contains a non-regular member")
            if member.filename.casefold() in names:
                raise ValueError("archive contains colliding member names")
            names.add(member.filename.casefold())
            if member.flag_bits & 1:
                raise ValueError("encrypted archive member")
        destination.mkdir(parents=True, exist_ok=False)
        for member in members:
            with safe_relative_path(destination, member.filename).open("xb") as output:
                output.write(archive.read(member))  # zipfile verifies CRC as well.


def recover_archives(zip_directory: Path, destination: Path, metadata_path: Path) -> dict:
    """Recover verified copies, never overwriting an earlier recovery."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    artifacts = [artifact for run in metadata.values() for artifact in run["artifacts"]]
    if {a["id"] for a in artifacts} != set(OFFICIAL_ARCHIVES) or len(artifacts) != 6:
        raise ValueError("catalog must contain exactly the six original archives")
    # Validate every outer archive before creating recovered data.
    for artifact in artifacts:
        expected = OFFICIAL_ARCHIVES[artifact["id"]]
        if artifact["digest"] != "sha256:" + expected:
            raise ValueError("catalog differs from the pinned original archive digest")
        archive_path = safe_relative_path(zip_directory, artifact["name"] + ".zip")
        data = read_verified_bytes(archive_path, expected_sha256=expected)
        if len(data) != artifact["size_in_bytes"]:
            raise ValueError("catalog/archive size mismatch")
    destination.mkdir(parents=True, exist_ok=False)
    inventory = {"schema_version": 1, "repository": "Myrroth/NOVA-Neural-Arena",
                 "no_training": True, "no_new_episodes": True, "artifacts": []}
    for artifact in artifacts:
        archive_path = zip_directory / (artifact["name"] + ".zip")
        folder = destination / str(artifact["id"])
        extract_verified_zip(archive_path, folder, expected_sha256=OFFICIAL_ARCHIVES[artifact["id"]])
        members = {p.name for p in folder.iterdir()}
        # Phase 2's original manifest authenticates only the three result files.
        # Its other files are still anchored by the verified outer ZIP digest.
        expected_members = ({"episodes.csv", "config.json", "summary.json"}
                            if artifact["id"] == 13106764292 else members - {"SHA256SUMS"})
        listed = verify_checksum_manifest(folder, expected_files=expected_members)
        with (folder / "episodes.csv").open(encoding="utf-8", newline="") as handle:
            episodes = list(csv.DictReader(handle))
        if len(episodes) != EPISODE_COUNTS[artifact["id"]]:
            raise ValueError("historical episode count differs from official record")
        inventory["artifacts"].append({
            "artifact_id": artifact["id"], "run_id": artifact["workflow_run"]["id"],
            "archive": archive_path.name, "archive_sha256": OFFICIAL_ARCHIVES[artifact["id"]],
            "archive_bytes": artifact["size_in_bytes"], "internal_hashes": listed,
            "outer_zip_anchored_additional_hashes": {
                name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                for name in sorted(members - set(listed) - {"SHA256SUMS"})},
            "recorded_training_or_evaluation_rows": len(episodes),
            "expires_at_actions": artifact["expires_at"],
        })
    write_new_json(destination / "inventory.json", inventory)
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zips", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=Path("docs/history/actions-artifacts-metadata.json"))
    args = parser.parse_args()
    result = recover_archives(args.zips, args.out, args.metadata)
    print(json.dumps({"archives_verified": len(result["artifacts"]),
                      "internal_hashes_verified": sum(len(a["internal_hashes"]) for a in result["artifacts"]),
                      "inventory": str(args.out / "inventory.json")}, indent=2))


if __name__ == "__main__":
    main()
