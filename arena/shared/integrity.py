"""Small integrity primitives for new experiments; historical code is unchanged.

Expected digests must come from a trusted, independent catalog. A checksum file
distributed with an archive alone detects corruption, not deliberate replacement.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any


def _is_link_or_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        details = path.lstat()
    except FileNotFoundError:
        return False
    # pathlib.is_junction is unavailable on official Python 3.11. Inspect Windows
    # attributes as well, so a junction cannot bypass the relative-path boundary.
    return bool(getattr(details, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def validate_sha256(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError("expected a SHA-256 hexadecimal digest")
    return value.lower()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_path(root: str | Path, name: str) -> Path:
    """Resolve a portable artifact member, rejecting traversal and link escapes."""
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or "\x00" in name or name.startswith("/")):
        raise ValueError("unsafe artifact path")
    parts = name.split("/")
    if any(part in ("", ".", "..") or part.endswith((" ", ".")) for part in parts):
        raise ValueError("unsafe artifact path component")
    # Windows treats these as devices even with an extension.
    reserved = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{i}" for prefix in ("COM", "LPT") for i in range(1, 10)
    }
    if any(part.split(".")[0].upper() in reserved for part in parts):
        raise ValueError("reserved artifact path component")
    base = Path(root)
    if _is_link_or_reparse(base):
        raise ValueError("artifact root must not be a symbolic link")
    path = base
    for part in PurePosixPath(name).parts:
        path = path / part
        if _is_link_or_reparse(path):
            raise ValueError("artifact path must not contain symbolic links or junctions")
    if not path.resolve().is_relative_to(base.resolve()):
        raise ValueError("artifact path escapes its root")
    return path


def read_verified_bytes(path: str | Path, *, expected_sha256: str,
                        max_bytes: int = 64 * 1024 * 1024) -> bytes:
    """Read bounded bytes once, then verify exactly what the caller will consume."""
    expected = validate_sha256(expected_sha256)
    path = Path(path)
    if _is_link_or_reparse(path) or not path.is_file():
        raise ValueError("missing or unsafe input file")
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    with path.open("rb") as handle:
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("input exceeds byte limit")
    if not hmac.compare_digest(hashlib.sha256(data).hexdigest(), expected):
        raise ValueError("SHA-256 mismatch")
    return data


def verify_checksum_manifest(directory: str | Path, *, manifest: str = "SHA256SUMS",
                             expected_files: set[str] | frozenset[str] | None = None) -> dict[str, str]:
    """Verify strict GNU-style checksums, including duplicate/case path rejection.

    This verifies members listed in the manifest. Use ``expected_files`` to require
    an exact known set; additional files in the directory are not authenticated.
    """
    manifest_path = safe_relative_path(directory, manifest)
    if not manifest_path.is_file() or manifest_path.stat().st_size > 1024 * 1024:
        raise ValueError("missing or oversized checksum manifest")
    listed: dict[str, str] = {}
    folded: set[str] = set()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-fA-F]{64})  (.+)", line)
        if not match:
            raise ValueError("invalid checksum manifest entry")
        digest, name = match.groups()
        if name.casefold() in folded:
            raise ValueError("duplicate or case-colliding checksum path")
        path = safe_relative_path(directory, name)
        if not path.is_file():
            raise ValueError("checksum member is not a regular file")
        if not hmac.compare_digest(sha256_file(path), digest.lower()):
            raise ValueError(f"SHA-256 mismatch: {name}")
        listed[name] = digest.lower()
        folded.add(name.casefold())
    if not listed:
        raise ValueError("empty checksum manifest")
    if expected_files is not None and set(listed) != set(expected_files):
        raise ValueError("checksum member set differs from expected files")
    return listed


def write_new_json(path: str | Path, value: Any) -> None:
    """Create one JSON file exclusively; never replace a historical output."""
    path = Path(path)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
