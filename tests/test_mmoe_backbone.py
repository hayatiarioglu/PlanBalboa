"""
AetherForecaster-X v35.2 Multi-Asset Master
Katman 2: Detach-Coupling MMoE Backbone Test Suite (Faz 3.3 / Adım 3.3.1 & 3.3.2)
"""

import unittest
import torch

from aether.signal.mmoe_backbone import DetachCoupledPredictionHeads, MMoEBackboneModule, MMoEOutput


class TestDetachCoupledPredictionHeads(unittest.TestCase):
    """
    Adım 3.3.1, 3.3.2, 3.5.1 & 3.5.2: MMoE Shared Expert, Gating ve Detach-Coupling Test Paneli.
    """

    def setUp(self):
        self.trunk_dim = 32
        self.hybrid_vol_dim = 4
        self.heads = DetachCoupledPredictionHeads(
            trunk_dim=self.trunk_dim,
            hybrid_vol_dim=self.hybrid_vol_dim,
            head_a_hidden_dim=16,
            head_b_hidden_dim=16
        )
        self.mmoe_model = MMoEBackboneModule(
            input_dim=12,
            hybrid_vol_dim=4,
            num_experts=4,
            expert_hidden_dim=16,
            expert_output_dim=32
        )

    def test_detach_coupling_forward_shapes(self):
        """Forward pass çıktı tensor boyutlarının doğrulanması."""
        batch_size = 8
        n_assets = 10
        
        h_i = torch.randn(batch_size, n_assets, self.trunk_dim)
        v_hybrid = torch.randn(batch_size, n_assets, self.hybrid_vol_dim)

        logit_a, logit_b = self.heads(h_i, v_hybrid)

        self.assertEqual(logit_a.shape, (batch_size, n_assets, 1))
        self.assertEqual(logit_b.shape, (batch_size, n_assets, 1))

    def test_gradient_leakage_detach(self):
        """
        Adım 3.3.2 (Kalite Testi): Head B kayıp fonksiyonunun Head A ağırlıklarına gradyan sızdırmadığının (Gradient Leakage)
        .detach() mekanizması ile kesin olarak kanıtlanması.
        """
        batch_size = 4
        n_assets = 5
        h_i = torch.randn(batch_size, n_assets, self.trunk_dim, requires_grad=True)
        v_hybrid = torch.randn(batch_size, n_assets, self.hybrid_vol_dim)

        # 1. Forward Pass
        logit_a, logit_b = self.heads(h_i, v_hybrid)

        # 2. SADECE Head B kaybından backward çalıştır
        loss_b = torch.sum(logit_b)
        loss_b.backward()

        # 3. Head A ağı parametrelerinde gradyanlar kesinlikle Hiç Oluşmamalı (None) veya 0.0 olmalıdır!
        for name, param in self.heads.head_a_net.named_parameters():
            if param.grad is not None:
                self.assertTrue(
                    torch.all(param.grad == 0.0).item(),
                    f"Head A parametresine ({name}) Head B kayıp gradyanı sızdı! Detach başarısız!"
                )
            else:
                self.assertIsNone(param.grad)

    def test_mmoe_gating_weights_and_experts(self):
        """
        Adım 3.5.1 & 3.5.2 (Kalite Testi): MMoE Shared Expert çıktıları ve Kapılama (Gating)
        ağırlıklarının Softmax toplamının 1.0 (Sum = 1.0) olduğunun doğrulanması.
        """
        batch_size = 4
        n_assets = 6
        x = torch.randn(batch_size, n_assets, 12)
        v_hybrid = torch.randn(batch_size, n_assets, 4)

        out = self.mmoe_model(x, v_hybrid)

        self.assertIsInstance(out, MMoEOutput)
        self.assertEqual(out.logit_head_a.shape, (batch_size, n_assets, 1))
        self.assertEqual(out.logit_head_b.shape, (batch_size, n_assets, 1))

        # Gate A ve Gate B ağırlıklarının her varlık için toplamı 1.0 olmalıdır
        gate_a_sum = torch.sum(out.gate_a_weights, dim=-1)
        gate_b_sum = torch.sum(out.gate_b_weights, dim=-1)

        self.assertTrue(torch.allclose(gate_a_sum, torch.ones_like(gate_a_sum), atol=1e-5))
        self.assertTrue(torch.allclose(gate_b_sum, torch.ones_like(gate_b_sum), atol=1e-5))


if __name__ == "__main__":
    unittest.main()

