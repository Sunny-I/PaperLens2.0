import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from pipeline.demo_papers import load_demo_papers


class SampleTests(unittest.TestCase):
    def test_reads_actual_pdf_bytes_and_ignores_other_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'B.PDF').write_bytes(b'user second pdf')
            (root / 'A.pdf').write_bytes(b'user first pdf')
            (root / 'notes.txt').write_text('ignore')
            with patch('pipeline.demo_papers.SAMPLE_DIR', root):
                self.assertEqual(load_demo_papers(), [('A.pdf', b'user first pdf'), ('B.PDF', b'user second pdf')])

    def test_empty_folder_reports_error_instead_of_inventing_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('pipeline.demo_papers.SAMPLE_DIR', Path(folder)), self.assertRaisesRegex(ValueError, 'No PDF'):
                load_demo_papers()
