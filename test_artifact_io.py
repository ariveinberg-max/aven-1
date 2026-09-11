import tempfile
import unittest
from pathlib import Path
from artifact_io import atomic_write, atomic_json, file_sha256
from train import valid_token_cache


class AtomicArtifactTests(unittest.TestCase):
    def test_failed_writer_preserves_previous_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'latest.pt'
            path.write_bytes(b'known-good')
            def fail(output):
                output.write(b'incomplete')
                raise RuntimeError('disk write failed')
            with self.assertRaises(RuntimeError):
                atomic_write(path, fail)
            self.assertEqual(path.read_bytes(), b'known-good')
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])

    def test_cache_corruption_even_at_same_size_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tokens.bin'
            path.write_bytes(b'\x01\x00\x02\x00')
            atomic_json(path.with_suffix('.json'), dict(bytes=4, tokenizer_sha256='identity', sha256=file_sha256(path)))
            self.assertTrue(valid_token_cache(path, 'identity'))
            self.assertFalse(valid_token_cache(path, 'other'))
            path.write_bytes(b'\x03\x00\x02\x00')
            self.assertFalse(valid_token_cache(path, 'identity'))
            path.with_suffix('.json').write_text('broken')
            self.assertFalse(valid_token_cache(path, 'identity'))


    def test_checkpoint_hash_and_load_use_same_open_snapshot(self):
        import torch
        from unittest.mock import patch
        from artifact_io import load_checkpoint_snapshot
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'latest.pt'
            torch.save({'step': 1}, path)
            expected = file_sha256(path)
            original_load = torch.load
            def replace_then_load(source, **kwargs):
                replacement = path.with_name('new.pt')
                torch.save({'step': 2}, replacement)
                replacement.replace(path)
                return original_load(source, **kwargs)
            with patch.object(torch, 'load', side_effect=replace_then_load):
                saved, fingerprint = load_checkpoint_snapshot(path, expected)
            self.assertEqual(saved['step'], 1)
            self.assertEqual(fingerprint, expected)
            self.assertEqual(torch.load(path, weights_only=True)['step'], 2)


    def test_new_report_is_atomic_and_cannot_replace_an_existing_one(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.json'
            atomic_json(path, {'result': 1}, overwrite=False)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                atomic_json(path, {'result': 2}, overwrite=False)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])
            with self.assertRaises(ValueError):
                atomic_json(Path(directory)/'invalid.json', {'result': float('nan')}, overwrite=False)
            self.assertFalse((Path(directory)/'invalid.json').exists())
