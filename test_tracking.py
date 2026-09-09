import math
import unittest
import torch
from tracking import prediction_metrics, memory_metrics, diagnostics
from brain import Brain, Config

class TrackingTests(unittest.TestCase):
    def test_accuracy_and_entropy(self):
        logits = torch.tensor([[[9., 0., 0.], [0., 9., 0.]]])
        values = prediction_metrics(logits, torch.tensor([[0, 2]]))
        self.assertEqual(values['accuracy'], .5)
        self.assertEqual(values['top5_accuracy'], 1.)
        self.assertLess(values['entropy_nats'], .01)
        uniform = prediction_metrics(torch.zeros(1, 2, 3), torch.tensor([[0, 1]]))
        self.assertAlmostEqual(uniform['entropy_nats'], math.log(3), places=5)

    def test_diagnostics_do_not_change_model_or_gradients(self):
        m=Brain(Config(width=32, heads=4, layers=1, context=8))
        x=torch.randint(256,(1,8)); m(x,x)[1].backward()
        before=[p.grad.clone() for p in m.parameters()]
        values=diagnostics(m)
        self.assertTrue(all(math.isfinite(v) and v>=0 for v in values.values()))
        self.assertTrue(all(torch.equal(a,p.grad) for a,p in zip(before,m.parameters())))

    def test_peak_memory(self):
        self.assertGreater(memory_metrics('cpu')['memory/process_peak_rss_mb'],0)

if __name__=='__main__': unittest.main()
