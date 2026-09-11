import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from tokenizer import Tokenizer
from train import encode_corpus_to_cache, batch
from response_mask import build_response_mask
from brain import Brain, Config
from tracking import prediction_metrics


class ResponseMaskTests(unittest.TestCase):
    def test_labels_and_alignment_across_encoding_chunks(self):
        text = '### Instruction:\nName a color 🌱\n\n### Response:\nBlue.\n<|end|>\n\n'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus, cache, mask = [root / x for x in ['corpus.txt', 'tokens.bin', 'mask.bin']]
            corpus.write_text(text * 3)
            for vocab in [256, 275]:
                tok = Tokenizer()
                tok.train((text * 3).encode(), vocab)
                count = encode_corpus_to_cache(corpus, tok, cache, chunk_size=17)
                info = build_response_mask(corpus, cache, tok, mask)
                self.assertEqual(info['tokens'], count)
                ids = np.fromfile(cache, dtype=np.uint16)
                labels = np.fromfile(mask, dtype=np.uint8)
                output = b''.join(tok.vocab[int(token)] for token, label in zip(ids, labels) if label)
                self.assertNotIn(b'Instruction', output)
                self.assertNotIn(b'Name a color', output)
                self.assertIn(b'Blue.', output)
                self.assertIn(b'<|end|>', output)
                self.assertEqual(len(labels), count)

    def test_masked_targets_have_no_direct_loss_gradient(self):
        model = Brain(Config(width=16, heads=2, layers=1, context=8))
        ids = torch.arange(8).unsqueeze(0)
        targets = ids.clone()
        targets[:, :4] = -100
        logits, loss, _ = model(ids, targets)
        logits.retain_grad()
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[:, :4], torch.zeros_like(logits.grad[:, :4])))
        self.assertGreater(logits.grad[:, 4:].abs().sum().item(), 0)

    def test_batch_masks_next_token_not_input_position(self):
        data = np.arange(100, dtype=np.uint16)
        mask = (data % 2 == 0).astype(np.uint8)
        x, y = batch(data, 4, 8, 'cpu', target_mask=mask)
        self.assertTrue(torch.equal(y == -100, (x + 1) % 2 != 0))
        self.assertTrue(torch.equal(y[y != -100], (x + 1)[y != -100]))
        with self.assertRaisesRegex(ValueError, 'Could not sample'):
            batch(data, 1, 4, 'cpu', target_mask=np.zeros(100, dtype=np.uint8))

    def test_metrics_ignore_prompt_targets(self):
        logits = torch.tensor([[[9., 0.], [0., 9.]]])
        result = prediction_metrics(logits, torch.tensor([[-100, 1]]))
        self.assertEqual(result['accuracy'], 1.)
        with self.assertRaises(ValueError):
            prediction_metrics(logits, torch.full((1, 2), -100))

    def test_plain_text_rejected_without_partial_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus, cache, mask = [root / x for x in ['corpus.txt', 'tokens.bin', 'mask.bin']]
            corpus.write_text('plain text only')
            tok = Tokenizer()
            encode_corpus_to_cache(corpus, tok, cache)
            with self.assertRaisesRegex(ValueError, 'No response targets'):
                build_response_mask(corpus, cache, tok, mask)
            self.assertFalse(mask.exists())
