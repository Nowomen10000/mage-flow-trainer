"""SinkGD: Sinkhorn normalization, zero-gradient safety, SDNQ master weights, and trainer wiring."""
import copy
import unittest

import torch

from trainer.training.config import OptimizerConfig
from trainer.training.optim import build_optimizer, estimate_optimizer_bytes
from trainer.training.sinkgd import sr_sinkhorn


def make(params, **cfg_kwargs):
    cfg = OptimizerConfig(kind='sinkgd', **cfg_kwargs)
    return build_optimizer(params, cfg)


class SinkhornMathTests(unittest.TestCase):
    def test_row_and_column_norms(self):
        torch.manual_seed(0)
        g = torch.randn(48, 96)
        x = sr_sinkhorn(g, iters=5)
        m, n = g.shape
        # The last step is the column step, so columns are exact and rows have converged.
        torch.testing.assert_close(x.norm(dim=0), torch.full((n,), m ** 0.5), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(x.norm(dim=1), torch.full((m,), n ** 0.5), atol=5e-2, rtol=5e-2)
        self.assertAlmostEqual(x.norm().item() / (m * n) ** 0.5, 1.0, places=2)

    def test_zero_gradient_rows_stay_zero(self):
        # Zero-initialized LoRA B gives A an all-zero gradient on the first step.
        self.assertTrue(torch.equal(sr_sinkhorn(torch.zeros(8, 64)), torch.zeros(8, 64)))
        g = torch.randn(8, 64)
        g[3] = 0
        x = sr_sinkhorn(g)
        self.assertTrue(torch.isfinite(x).all())
        self.assertTrue(torch.equal(x[3], torch.zeros(64)))

    def test_non_matrix_shapes(self):
        v = sr_sinkhorn(torch.randn(100))
        self.assertAlmostEqual(v.pow(2).mean().item(), 1.0, places=4)
        k = torch.randn(8, 4, 3, 3)
        out = sr_sinkhorn(k)
        self.assertEqual(out.shape, k.shape)
        self.assertAlmostEqual(out.pow(2).mean().item(), 1.0, delta=0.05)


class SinkGDOptimizerTests(unittest.TestCase):
    def test_update_is_not_clipped_and_state_is_empty(self):
        torch.manual_seed(0)
        w = torch.nn.Parameter(torch.zeros(8, 16))
        b = torch.nn.Parameter(torch.zeros(8))
        k = torch.nn.Parameter(torch.zeros(4, 2, 3, 3))
        opt = make([{'params': [w, b, k], 'lr': 1.0, 'weight_decay': 0.0}])
        grads = [torch.randn_like(p) for p in (w, b, k)]
        for p, g in zip((w, b, k), grads):
            p.grad = g.clone()
        opt.step()
        # lr=1 and RMS-1 updates: nothing may be clamped to +-1 as SDNQ's default "clip" mode would.
        torch.testing.assert_close(w.detach(), -sr_sinkhorn(grads[0]))
        self.assertGreater(w.abs().max().item(), 1.0)
        for p in (w, b, k):
            self.assertEqual(set(opt.state[p]), {'step'})
        self.assertEqual(estimate_optimizer_bytes(10 ** 6, OptimizerConfig(kind='sinkgd')), 0)

    def test_constructor_and_iteration_validation(self):
        from trainer.training.sinkgd import SinkGD
        p = torch.nn.Parameter(torch.zeros(4, 4))
        for params in ([p], (p,), iter([p]), iter([{'params': [p]}])):
            opt = SinkGD(params, sinkhorn_iterations=3)
            self.assertEqual(opt.param_groups[0]['sinkhorn_iterations'], 3)
        for params in ([], iter([])):
            with self.assertRaisesRegex(ValueError, 'empty'):
                SinkGD(params)
        for value in (0, -1, 1.5, True):
            with self.assertRaisesRegex(ValueError, 'sinkhorn_iterations'):
                OptimizerConfig(kind='sinkgd', sinkhorn_iterations=value)
        with self.assertRaisesRegex(ValueError, 'only applies'):
            OptimizerConfig(kind='adamw', sinkhorn_iterations=3)
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            SinkGD([p], typo=True)
        opt = make([{'params': [p]}], sinkhorn_iterations=3)
        self.assertEqual(opt.param_groups[0]['sinkhorn_iterations'], 3)

    def test_bf16_stochastic_rounding_moves_tiny_updates(self):
        torch.manual_seed(0)
        w = torch.nn.Parameter(torch.ones(64, 64, dtype=torch.bfloat16))
        opt = make([{'params': [w], 'lr': 1e-5}])
        for _ in range(50):
            w.grad = torch.randn(64, 64, dtype=torch.bfloat16)
            opt.step()
        # 1e-5 is far below half a bf16 ulp at 1.0 (~2e-3); round-to-nearest would leave all ones.
        self.assertFalse(torch.equal(w, torch.ones_like(w)))
        self.assertTrue(torch.isfinite(w.float()).all())

    def test_group_keys_and_validation(self):
        w = torch.nn.Parameter(torch.zeros(4, 4))
        opt = make([{'params': [w], 'lr': 1e-3}])
        self.assertEqual(opt.param_groups[0]['sinkhorn_iterations'], 5)
        self.assertEqual(opt.param_groups[0]['final_norm_mode'], 'none')
        from trainer.training.sinkgd import SinkGD
        with self.assertRaisesRegex(ValueError, 'sinkhorn_iterations'):
            SinkGD([{'params': [w], 'lr': 1e-3}], sinkhorn_iterations=0)


class SinkGDSDNQTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The trainer installs this before SDNQ training quantization. Stock SDNQ's integer
        # stochastic rounding is biased toward the nearest code and erases small updates.
        from trainer.training.sdnq_rounding import enable_unbiased_integer_rounding
        enable_unbiased_integer_rounding()

    def _sdnq_param(self, shape=(64, 128)):
        from sdnq.training import SDNQTensor
        w = torch.randn(*shape)
        return torch.nn.Parameter(SDNQTensor.from_float(
            w, weights_dtype='int8', group_size=32, use_stochastic_rounding=True))

    def test_sdnq_master_weight_trains_and_stays_quantized(self):
        from sdnq.training import SDNQTensor
        torch.manual_seed(0)
        p = self._sdnq_param()
        before = p.detach().dequantize(dtype=torch.float32).clone()
        opt = make([{'params': [p], 'lr': 1e-3}])
        g = torch.randn(64, 128)
        for _ in range(20):
            p.grad = g.clone()
            opt.step()
            opt.zero_grad()
        after = p.detach().dequantize(dtype=torch.float32)
        self.assertIsInstance(p.data, SDNQTensor)
        self.assertTrue(torch.isfinite(after).all())
        # Moves against the gradient on average (signed displacement along -sign(g)).
        moved = ((after - before) * (-g.sign())).mean().item()
        # Nominal movement is ~0.8 * lr * steps = 0.016; require at least half of it.
        self.assertGreater(moved, 0.008)

    def test_sdnq_with_kahan_and_quantized_buffers(self):
        torch.manual_seed(0)
        for quantize in (False, True):
            with self.subTest(quantize_state=quantize):
                p = self._sdnq_param((128, 256))
                opt = make([{'params': [p], 'lr': 1e-4}], use_kahan=True, quantize_state=quantize)
                for _ in range(3):
                    p.grad = torch.randn(128, 256)
                    opt.step()
                    opt.zero_grad()
                self.assertIn('kahan_buffer', opt.state[p])
                self.assertTrue(torch.isfinite(p.detach().dequantize(dtype=torch.float32)).all())
                self.assertGreater(estimate_optimizer_bytes(10 ** 6, OptimizerConfig(
                    kind='sinkgd', use_kahan=True)), 0)

    def test_state_dict_roundtrip(self):
        torch.manual_seed(0)
        b = torch.nn.Parameter(torch.zeros(16, 16))
        opt = make([{'params': [b], 'lr': 1e-3}], use_kahan=True)
        b.grad = torch.randn(16, 16)
        opt.step()
        saved = copy.deepcopy(opt.state_dict())
        c = torch.nn.Parameter(b.detach().clone())
        resumed = make([{'params': [c], 'lr': 1e-3}], use_kahan=True)
        resumed.load_state_dict(saved)
        g = torch.randn(16, 16)
        for p, o in ((b, opt), (c, resumed)):
            p.grad = g.clone()
            o.step()
        torch.testing.assert_close(b, c, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
