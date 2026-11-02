"""Packaging verification is standard-library only and never executes a model."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
import zipfile

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_current_package.py"
spec = importlib.util.spec_from_file_location("build_current_package", SCRIPT)
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


class CurrentPackageVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("NNA_TEST_TMP"))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, *, mutate=None):
        files = {"source/example.txt": b"source snapshot\n", "evidence/decision.json": b"{}\n"}
        inventory = {"schema_version": 1, "mission": "NNA-FECHAMENTO-001",
                     "payload": {n: {"bytes": len(d), "sha256": hashlib.sha256(d).hexdigest()}
                                 for n, d in files.items()}}
        files["INVENTORY.json"] = json.dumps(inventory).encode()
        files["SHA256SUMS"] = "".join(
            f"{hashlib.sha256(d).hexdigest()}  {n}\n" for n, d in files.items()).encode()
        if mutate:
            mutate(files)
        path = self.root / "fixture.zip"
        with zipfile.ZipFile(path, "w") as output:
            for name, data in files.items():
                output.writestr(name, data)
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_complete_manifest_and_fresh_recovery_no_overwrite(self):
        path, digest = self.fixture()
        destination = self.root / "recovered"
        result = package.verify_package(path, digest, extract_to=destination)
        self.assertEqual(result["members_verified"], 3)
        self.assertTrue(result["local_recovery_verified"])
        self.assertEqual((destination / "source/example.txt").read_bytes(), b"source snapshot\n")
        with self.assertRaises(FileExistsError):
            package.verify_package(path, digest, extract_to=destination)

    def test_outer_tampering_rejected_before_recovery(self):
        path, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            package.verify_package(path, "0" * 64, extract_to=self.root / "bad")
        self.assertFalse((self.root / "bad").exists())

    def test_unlisted_or_corrupt_payload_rejected(self):
        for mutation in (lambda d: d.update({"extra.txt": b"unlisted"}),
                         lambda d: d.update({"source/example.txt": b"tampered"})):
            with self.subTest(mutation=mutation):
                path, digest = self.fixture(mutate=mutation)
                with self.assertRaises(ValueError):
                    package.verify_package(path, digest)

    def test_traversal_member_rejected_before_extraction(self):
        path, digest = self.fixture(mutate=lambda d: d.update({"../escape.txt": b"bad"}))
        with self.assertRaises(ValueError):
            package.verify_package(path, digest, extract_to=self.root / "bad")
        self.assertFalse((self.root / "bad").exists())

    def test_selected_text_cannot_contain_signed_download_or_token(self):
        for text in (b'https://example.org/raw?sig=abc', b'ghp_' + b'a' * 40):
            with self.assertRaisesRegex(ValueError, "credential-like"):
                package._check_text(text, "evidence.json")
        package._check_text(b'https://api.github.com/repos/a/b/actions/artifacts/1/zip', "evidence.json")


if __name__ == "__main__":
    unittest.main()
