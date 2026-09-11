import unittest
import torch
from rng_state import capture_rng, restore_rng


class RNGTests(unittest.TestCase):
    def test_cpu_sampling_replays_exactly(self):
        previous = torch.get_rng_state()
        try:
            torch.manual_seed(47)
            state = capture_rng('cpu')
            expected = torch.rand(20)
            self.assertEqual(restore_rng({'rng_state': state}, 'cpu'), 'full_backend_state')
            self.assertTrue(torch.equal(torch.rand(20), expected))
        finally:
            torch.set_rng_state(previous)

    def test_legacy_and_backend_change_are_explicit(self):
        state = capture_rng('cpu')
        self.assertEqual(restore_rng({'rng': state['cpu']}, 'cpu'), 'legacy_cpu_only')
        self.assertEqual(restore_rng({'rng_state': state}, 'cuda'), 'cpu_restored_backend_changed')

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable')
    def test_mps_random_state_replays(self):
        previous = capture_rng('mps')
        try:
            state = capture_rng('mps')
            expected = torch.rand(20, device='mps').cpu()
            restore_rng({'rng_state': state}, 'mps')
            self.assertTrue(torch.equal(torch.rand(20, device='mps').cpu(), expected))
        finally:
            restore_rng({'rng_state': previous}, 'mps')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_random_state_replays(self):
        previous = capture_rng('cuda')
        try:
            state = capture_rng('cuda')
            expected = torch.rand(20, device='cuda').cpu()
            restore_rng({'rng_state': state}, 'cuda')
            self.assertTrue(torch.equal(torch.rand(20, device='cuda').cpu(), expected))
        finally:
            restore_rng({'rng_state': previous}, 'cuda')
