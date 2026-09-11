import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from brain import Brain, Config
from tokenizer import Tokenizer
from checkpoint_bundle import export_bundle, verify_bundle
from artifact_io import file_sha256


class BundleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.destination = self.root / 'export'
        model = Brain(Config(width=16, layers=1, heads=2, context=16))
        tok = Tokenizer()
        tok.save(self.source / 'tokenizer.json')
        self.saved = dict(config=vars(model.config), model=model.state_dict(), step=3,
                          stage='finetune', tokenizer_sha256=tok.fingerprint(), optimizer={})
        torch.save(self.saved, self.source / 'latest.pt')

    def test_roundtrip_and_corruption(self):
        before = file_sha256(self.source / 'latest.pt')
        manifest = export_bundle(self.source, self.destination)
        self.assertEqual(manifest, verify_bundle(self.destination))
        self.assertEqual(manifest['checkpoint']['step'], 3)
        self.assertEqual(before, file_sha256(self.source / 'latest.pt'))
        with (self.destination / 'latest.pt').open('ab') as file:
            file.write(b'corruption')
        with self.assertRaisesRegex(ValueError, 'integrity'):
            verify_bundle(self.destination)

    def test_existing_destination_never_overwritten(self):
        self.destination.mkdir()
        sentinel = self.destination / 'keep'
        sentinel.write_text('important')
        with self.assertRaises(ValueError):
            export_bundle(self.source, self.destination)
        self.assertEqual(sentinel.read_text(), 'important')

    def test_mismatched_tokenizer_cleans_partial_export(self):
        self.saved['tokenizer_sha256'] = 'wrong'
        torch.save(self.saved, self.source / 'latest.pt')
        with self.assertRaisesRegex(ValueError, 'merge rules'):
            export_bundle(self.source, self.destination)
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.root.glob('.export-*')), [])

    def test_atomic_checkpoint_replacement_preserves_open_snapshot(self):
        import shutil
        copy = shutil.copyfileobj
        def replace_after_open(source, destination, **kwargs):
            newer = self.source / 'new.pt'
            torch.save(dict(self.saved, step=4), newer)
            newer.replace(self.source / 'latest.pt')
            return copy(source, destination, **kwargs)
        with patch('checkpoint_bundle.shutil.copyfileobj', side_effect=replace_after_open):
            result = export_bundle(self.source, self.destination)
        self.assertEqual(result['checkpoint']['step'], 3)
        self.assertEqual(torch.load(self.source / 'latest.pt', weights_only=True)['step'], 4)
