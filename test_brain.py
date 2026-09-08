import unittest
import torch
from brain import Brain, Config
from train import batch

class BrainTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        torch.set_num_threads(2)
        self.model = Brain(Config(width=32, layers=2, heads=4, context=16))

    def test_future_tokens_do_not_change_past_predictions(self):
        a = torch.randint(256, (1, 16))
        b = a.clone()
        b[:, 8:] = (b[:, 8:]+1)%256
        self.assertTrue(torch.allclose(self.model(a)[0][:, :8], self.model(b)[0][:, :8], atol=1e-6))

    def test_training_reduces_loss(self):
        x = torch.tensor([[97,98]*8])
        y = torch.tensor([[98,97]*8])
        initial = self.model(x,y)[1].item()
        opt = torch.optim.AdamW(self.model.parameters(), lr=.01)
        for _ in range(30):
            opt.zero_grad()
            loss = self.model(x,y)[1]
            loss.backward()
            opt.step()
        self.assertLess(self.model(x,y)[1].item(), initial*.3)

    def test_targets_are_next_bytes(self):
        x,y=batch(torch.arange(100),4,16,'cpu')
        self.assertTrue(torch.equal(x+1,y))

    def test_generation_handles_unicode_and_returns_measured_activity(self):
        text, activity = self.model.generate('Hello 🌱',count=3)
        self.assertTrue(text.startswith('Hello 🌱'))
        self.assertEqual(len(activity),2)
        self.assertEqual(len(activity[0]),32)
        self.assertTrue(all(v >= 0 for layer in activity for v in layer))

if __name__ == '__main__':
    unittest.main()
