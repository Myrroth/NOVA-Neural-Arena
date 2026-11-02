import hashlib
from pathlib import Path
import zipfile

import pytest

from arena.shared.preservation import extract_verified_zip


def make_zip(tmp_path, members):
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in members:
            archive.writestr(name, value)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("names", [["../escape"], ["C:escape"], ["file", "FILE"],
                                  ["NUL.txt"], ["folder/file"], ["file."]])
def test_reject_unsafe_members_before_writing(tmp_path, names):
    source, digest = make_zip(tmp_path, [(name, b"data") for name in names])
    destination = tmp_path / "recovered"
    with pytest.raises(ValueError):
        extract_verified_zip(source, destination, expected_sha256=digest)
    assert not destination.exists()


def test_no_overwrite_and_wrong_hash(tmp_path):
    source, digest = make_zip(tmp_path, [("model.pt", b"original")])
    destination = tmp_path / "recovered"
    with pytest.raises(ValueError, match="SHA-256"):
        extract_verified_zip(source, destination, expected_sha256="0" * 64)
    assert not destination.exists()
    extract_verified_zip(source, destination, expected_sha256=digest)
    with pytest.raises(FileExistsError):
        extract_verified_zip(source, destination, expected_sha256=digest)
    assert (destination / "model.pt").read_bytes() == b"original"
