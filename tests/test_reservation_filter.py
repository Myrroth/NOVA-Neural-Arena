"""Execute the workflow's actual jq selector against reservation and prose."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest

WORKFLOW = Path(__file__).resolve().parents[1] / '.github/workflows/nna-phase5-once.yml'


class ReservationFilterTests(unittest.TestCase):
    def test_only_effective_reservations_block_a_new_campaign(self):
        jq = shutil.which('jq')
        if jq is None:
            self.skipTest('jq is required on the Ubuntu CI runner; no network or episodes')
        workflow = WORKFLOW.read_text(encoding='utf-8')
        query = re.search(r"--paginate --jq '([^']+)'", workflow).group(1)
        marker = 'NNA-FECHAMENTO-001:PHASE5-RESERVED'
        mentions = [
            {'id': 1, 'body': 'Audit: no reservation ' + marker + ' was found.'},
            {'id': 2, 'body': '## Decision\nExample: `' + marker + ' | immutable run ...`'},
            {'id': 3, 'body': None},
        ]
        real = {'id': 4, 'body': marker + ' | immutable run 41390000000 attempt 1 | protocol '
                + 'a' * 64 + ' | evaluator ' + 'b' * 40 + ' | no reruns, even on failure'}
        for comments, expected in ((mentions, []), (mentions + [real], ['4'])):
            with self.subTest(real_reservation=bool(expected)):
                result = subprocess.run([jq, '-r', query], input=json.dumps(comments),
                                        text=True, capture_output=True, check=True, timeout=10)
                self.assertEqual(result.stdout.splitlines(), expected)
