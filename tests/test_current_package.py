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

    def preflight(self):
        primary = self.root / "run.json"
        primary.write_text(json.dumps({"id": 123, "conclusion": "failure", "episodes_executed": 0}))
        decision = self.root / "decision.md"
        decision.write_text("Fechamento por impedimento antes de qualquer episódio.\n", encoding="utf-8")
        report = self.root / "chat4.md"
        report.write_text("Relatório Chat 4: zero episódios, nenhuma inferência científica.\n", encoding="utf-8")
        document = lambda p: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        return {
            "status": "not_executed_preflight_failure", "episodes_executed": 0,
            "reservation_created": False, "reason": "Preflight failed before reservation.",
            "decision_url": "https://github.com/Myrroth/NOVA-Neural-Arena/issues/6#issuecomment-1",
            "failed_runs": [{"run_id": 123, "run_attempt": 1, "conclusion": "failure",
                             "episodes_executed": 0, "reservation_created": False,
                             "url": "https://github.com/Myrroth/NOVA-Neural-Arena/actions/runs/123",
                             "documents": [document(primary)]}],
            "decision_document": document(decision), "chat4_report": document(report),
        }

    def fixture(self, *, mutate=None, unavailable=False, inventory_mutate=None, extra=None):
        files = {"source/example.txt": b"source snapshot\n", "evidence/decision.json": b"{}\n"}
        state = {}
        if unavailable:
            record = package._add_unexecuted_phase5(self.preflight(), files)
            for i in range(6):
                files[f"scientific/original-zips/historical-fixture-{i}.zip"] = b"verification fixture only"
            state = {"mission_state": "engineering_closed_science_unavailable", "scientific_archives": 6,
                     "scientific_evaluation_available": False, "phase5": record}
        files.update(extra or {})
        inventory = {"schema_version": 1, "mission": "NNA-FECHAMENTO-001",
                     **state,
                     "payload": {n: {"bytes": len(d), "sha256": hashlib.sha256(d).hexdigest()}
                                 for n, d in files.items()}}
        if inventory_mutate:
            inventory_mutate(inventory)
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

    def test_preflight_closure_requires_real_hashed_documents_and_no_scientific_archive(self):
        facts = self.preflight()
        payload = {}
        record = package._add_unexecuted_phase5(facts, payload)
        self.assertIs(record["complete"], False)
        self.assertEqual(record["episodes_executed"], 0)
        self.assertIs(record["reservation_created"], False)
        self.assertEqual(package._phase5_state(record), ("engineering_closed_science_unavailable", 6))
        self.assertFalse(any(name.endswith(".zip") or name.endswith("episodes.csv") for name in payload))
        for doc in (record["decision_document"], record["chat4_report"], *record["failed_runs"][0]["documents"]):
            self.assertEqual(hashlib.sha256(payload[doc["member"]]).hexdigest(), doc["sha256"])
            self.assertEqual(len(payload[doc["member"]]), doc["bytes"])
        facts["chat4_report"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            package._add_unexecuted_phase5(facts, {})

    def test_preflight_refuses_incompatible_status_values_and_undocumented_runs(self):
        base = self.preflight()
        changes = (
            {"status": "completed"}, {"episodes_executed": 1}, {"episodes_executed": False},
            {"reservation_created": True}, {"complete": True}, {"archive_path": "fake.zip"},
            {"artifact_id": 1}, {"sha256": "0" * 64}, {"decision_url": ""}, {"decision_url": None}, {"reason": ""},
            {"failed_runs": []}, {"decision_document": None}, {"chat4_report": None},
        )
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                package._add_unexecuted_phase5(dict(base, **change), {})
        for change in ({"episodes_executed": 1}, {"reservation_created": True},
                       {"conclusion": "success"}, {"run_attempt": 2}, {"run_attempt": True}, {"documents": []}):
            facts = dict(base, failed_runs=[dict(base["failed_runs"][0], **change)])
            with self.subTest(run_change=change), self.assertRaises(ValueError):
                package._add_unexecuted_phase5(facts, {})

    def test_completed_label_cannot_disguise_zero_or_partial_results(self):
        for episodes in (0, 1, 32199, False):
            with self.subTest(episodes=episodes), self.assertRaisesRegex(ValueError, "zero or partial"):
                package._phase5_state({"status": "completed", "episodes_executed": episodes})
        self.assertEqual(package._phase5_state({"status": "completed", "episodes_executed": 32200}),
                         ("engineering_and_science_closed", 7))

    def test_verified_inventory_and_receipt_preserve_science_unavailable(self):
        path, digest = self.fixture(unavailable=True)
        receipt = package.verify_package(path, digest)
        self.assertEqual(receipt["mission_state"], "engineering_closed_science_unavailable")
        self.assertEqual(receipt["scientific_archives"], 6)
        self.assertIs(receipt["scientific_evaluation_available"], False)
        for key, wrong in (("mission_state", "engineering_and_science_closed"),
                           ("scientific_archives", 7), ("scientific_evaluation_available", True)):
            with self.subTest(key=key):
                path, digest = self.fixture(unavailable=True, inventory_mutate=lambda d: d.update({key: wrong}))
                with self.assertRaisesRegex(ValueError, "classification"):
                    package.verify_package(path, digest)

    def test_unavailable_package_rejects_fabricated_episode_table(self):
        path, digest = self.fixture(unavailable=True, extra={"scientific/final-analysis/episodes.csv": b"fake"})
        with self.assertRaisesRegex(ValueError, "episode CSV"):
            package.verify_package(path, digest)


if __name__ == "__main__":
    unittest.main()
