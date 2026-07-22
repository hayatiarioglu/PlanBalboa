"""
AetherForecaster-X v35.2 Multi-Asset Master
Katman 2: Scheduled Temperature Decay Test Suite (Faz 3.4 / Adım 3.4.1 & 3.4.2)
"""

import unittest
import torch

from aether.signal.temperature_decay import ScheduledTemperatureDecayEngine, TemperatureDecayResult
from aether.signal.soft_ndcg_loss import ClampedScheduledSoftNDCGLoss


class TestScheduledTemperatureDecayEngine(unittest.TestCase):
    """
    Adım 3.4.1 ve 3.4.2: Scheduled Temperature Decay ve Sıfırlanmayan Gradyan Kalite Testi.
    """

    def setUp(self):
        self.decay_engine = ScheduledTemperatureDecayEngine(tau_0=1.0, tau_min=0.10, alpha=0.05)
        self.soft_ndcg_loss = ClampedScheduledSoftNDCGLoss(top_k=5, tau_min=0.10, clamp_val=10.0)

    def test_temperature_decay_decay_formula(self):
        """
        Adım 3.4.1: tau(e) = tau_min + (tau_0 - tau_min) * exp(-alpha * e) üstel sönümleme doğrulaması.
        """
        res_0 = self.decay_engine.get_tau(epoch=0)
        self.assertAlmostEqual(res_0.tau, 1.0, places=4)

        res_10 = self.decay_engine.get_tau(epoch=10)
        self.assertLess(res_10.tau, 1.0)
        self.assertGreater(res_10.tau, 0.10)

        # 100. epochta tau_min (0.10) tabanına yaklaşılmalıdır (tau(100) ~ 0.106)
        res_100 = self.decay_engine.get_tau(epoch=100)
        self.assertAlmostEqual(res_100.tau, 0.106, places=2)
        self.assertGreaterEqual(res_100.tau, 0.10)

    def test_temperature_decay_invalid_inputs(self):
        """Geçersiz tau_min (< 0.10) veya negatif epoch verildiğinde ValueError fırlatılması."""
        with self.assertRaises(ValueError):
            ScheduledTemperatureDecayEngine(tau_0=1.0, tau_min=0.01)

        with self.assertRaises(ValueError):
            self.decay_engine.get_tau(epoch=-5)

    def test_scheduled_temperature_decay(self):
        """
        Adım 3.4.2 (Kalite Testi): Epoch adımlarında sıcaklık düşse dahi (tau -> 0.10)
        gradyanların sıfırlanmadığını (gradient vanishing yaşanmadığını) doğrulayan test.
        """
        y_true = torch.tensor([0.20, 0.10, -0.05, -0.15])
        
        for ep in [0, 20, 50, 100]:
            y_pred = torch.tensor([1.5, 0.8, -0.2, -1.0], requires_grad=True)
            tau_res = self.decay_engine.get_tau(epoch=ep)
            
            loss = self.soft_ndcg_loss(y_pred, y_true, tau=tau_res.tau_tensor)
            loss.backward()

            # Gradyan normu sıfırdan büyük ve sonlu olmalıdır (Zero Gradient Vanishing)
            grad_norm = float(torch.norm(y_pred.grad))
            self.assertGreater(grad_norm, 1e-6, f"Epoch {ep} için gradyan sıfırlandı!")
            self.assertFalse(torch.isnan(y_pred.grad).any().item())


if __name__ == "__main__":
    unittest.main()
