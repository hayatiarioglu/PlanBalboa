"""
AetherForecaster-X v35.2 Multi-Asset Master
Katman 2: Sprint 3 Entegrasyon ve Zero-NaN Kalite Gate Test Paneli (Faz 3.6 / Adım 3.6.1 & 3.6.2)
"""

import unittest
import torch
import torch.optim as optim
import numpy as np
import pandas as pd

from aether.signal.signal_orchestrator import SignalPipelineOrchestrator, SignalPipelineResult


class TestSprint3QualityGate(unittest.TestCase):
    """
    Sprint 3 Kalite Gate Test Paneli: Katman 2 Nöral Ağ Mimarisi Bütünsel Doğrulaması.
    """

    def setUp(self):
        self.input_dim = 16
        self.hybrid_vol_dim = 4
        self.n_assets = 20

        self.orchestrator = SignalPipelineOrchestrator(
            input_dim=self.input_dim,
            hybrid_vol_dim=self.hybrid_vol_dim,
            num_experts=4,
            expert_hidden_dim=32,
            expert_output_dim=32,
            head_a_hidden_dim=32,
            head_b_hidden_dim=32,
            tau_0=1.0,
            tau_min=0.10,
            alpha_decay=0.05,
            eta_delist=1.0,
            top_k=5
        )

    def test_sprint3_pipeline_integration(self):
        """
        Adım 3.6.1: Katman 2 Nöral Ağ bileşenlerinin bütünsel entegrasyon testi.
        """
        x = torch.randn(self.n_assets, self.input_dim)
        v_hybrid = torch.randn(self.n_assets, self.hybrid_vol_dim)
        returns = pd.Series(np.linspace(-0.15, 0.25, self.n_assets))

        res = self.orchestrator(x, returns, v_hybrid, epoch=0)

        self.assertIsInstance(res, SignalPipelineResult)
        self.assertFalse(torch.isnan(res.total_loss).item())
        self.assertEqual(res.current_tau, 1.0)
        self.assertEqual(res.targets.shape[0], self.n_assets)

    def test_sprint3_zero_nan_quality_gate_and_convergence(self):
        """
        Adım 3.6.2 (Kalite Gate Onayı): 50 Epochluk eğitim döngüsünde kriz simülasyonu,
        düşen sıcaklık (tau -> 0.10) ve delist hisseler mevcudiyetinde 0 ADET NaN elde edildiğinin
        ve kaybın başarıyla düştüğünün (konverjans) kanıtlanması.
        """
        optimizer = optim.Adam(self.orchestrator.parameters(), lr=1e-3)
        
        # Simülasyon Verisi: 20 hisse (2 adedi delist: -100% getiri)
        x = torch.randn(self.n_assets, self.input_dim)
        v_hybrid = torch.randn(self.n_assets, self.hybrid_vol_dim)
        
        healthy_ret = np.linspace(-0.10, 0.15, 18)
        delist_ret = np.array([-1.00, -1.00])
        returns = pd.Series(np.concatenate([healthy_ret, delist_ret]))
        delist_flags = pd.Series([False]*18 + [True]*2)

        losses = []

        for ep in range(50):
            optimizer.zero_grad()
            res = self.orchestrator(x, returns, v_hybrid, epoch=ep, delisted_flags=delist_flags)

            loss = res.total_loss
            
            # ZERO-NAN GUARANTEE: Her epochta Loss NaN veya Inf üretemez!
            self.assertFalse(torch.isnan(loss).item(), f"Epoch {ep}'de NaN Loss tespit edildi!")
            self.assertFalse(torch.isinf(loss).item(), f"Epoch {ep}'de Inf Loss tespit edildi!")

            loss.backward()

            # ZERO-NAN GUARANTEE: Her epochta gradyanlar NaN veya Inf üretemez!
            for name, param in self.orchestrator.named_parameters():
                if param.grad is not None:
                    self.assertFalse(
                        torch.isnan(param.grad).any().item(),
                        f"Epoch {ep}'de {name} parametresinde NaN gradyan!"
                    )

            optimizer.step()
            losses.append(loss.item())

        # Konverjans Kontrolü: 50. Epoch kaybı 0. Epoch kaybından daha düşük olmalıdır
        self.assertLess(losses[-1], losses[0], f"Eğitim konverje olmadı! L_start: {losses[0]}, L_end: {losses[-1]}")


if __name__ == "__main__":
    unittest.main()
