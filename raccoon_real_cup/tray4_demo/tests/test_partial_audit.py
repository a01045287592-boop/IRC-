import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from review import audit


class PartialAuditTests(unittest.TestCase):
    def inspect(self, partial, **updates):
        meta = dict(status='complete', success='user_confirmed', object_count=1,
                    target='plastic_1', objects=[dict(id='plastic_1', success='user_confirmed')])
        meta.update(updates)
        with tempfile.TemporaryDirectory() as td:
            with patch('review.read_json', side_effect=lambda p: meta if p.name == 'metadata.json' else {}):
                return audit(Path(td), partial=partial)['errors']

    def test_partial_confirmation_does_not_require_four_objects(self):
        errors = self.inspect(True)
        self.assertNotIn('target success confirmation required', errors)
        self.assertNotIn('four segments required', errors)
        self.assertIn('empty RGB or steps', errors)
        self.assertIn('one segment required', errors)

    def test_full_audit_remains_strict(self):
        self.assertIn('four object success confirmations required', self.inspect(False))

    def test_partial_does_not_accept_wrong_target_or_failed_run(self):
        self.assertIn('target success confirmation required', self.inspect(True, target='glass_1'))
        self.assertIn('invalid partial target/object_count', self.inspect(True, object_count=4))
        self.assertIn('full episode not confirmed successful', self.inspect(True, status='aborted'))


if __name__ == '__main__':
    unittest.main()
