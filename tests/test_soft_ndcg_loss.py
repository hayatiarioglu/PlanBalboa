"""
AetherForecaster-X v35.2 Multi-Asset Master
Katman 2: Clamped Scheduled Soft-NDCG Loss Test Suite (Faz 3.2 / Adım 3.2.1)
"""

import unittest
import torch
import numpy as np

from aether.signal.soft_ndcg_loss import (
    EpsilonProtectedPositiveIDCG,
    DelistSigmoidPenalty,
    ClampedLogitDifference,
    ClampedScheduledSoftNDCGLoss
)


class TestEpsilonProtectedPositiveIDCG(unittest.TestCase):
    """
    Adım 3.2.1, 3.2.2, 3.2.3 & 3.2.4: Soft-NDCG Kayıp Katmanları ve Zero-NaN Kalite Testleri.
    """

    def setUp(self):
        self.idcg_layer = EpsilonProtectedPositiveIDCG(top_k=5, eps=1e-6)
        self.delist_penalty_layer = DelistSigmoidPenalty(eta=1.5)
        self.logit_diff_layer = ClampedLogitDifference(tau_min=0.10, clamp_val=10.0)
        self.soft_ndcg_loss = ClampedScheduledSoftNDCGLoss(top_k=5, tau_min=0.10, clamp_val=10.0, eta_delist=1.0)

    def test_positive_idcg_normal_market(self):
        """Normal piyasa şartlarında (y_i > 0 hedefler mevcut) IDCG hesabı."""
        y_true = torch.tensor([0.20, 0.15, 0.05, -0.02, -0.10])
        idcg_val = self.idcg_layer(y_true)

        self.assertGreater(idcg_val.item(), 1e-6)
        self.assertFalse(torch.isnan(idcg_val).item())

    def test_positive_idcg_crisis_market_zero_nan_guard(self):
        """
        Kriz haftasında (tüm y_i <= 0) IDCG'nin 0 vermeyip tam 1e-6 epsilon kalkanıyla
        sıfıra bölmeyi engellemesi.
        """
        y_crisis = torch.tensor([-0.15, -0.25, -0.30, -0.05, -0.50])
        idcg_crisis = self.idcg_layer(y_crisis)

        # IDCG tam 1e-6 eşiğinde kilitlenmelidir
        self.assertAlmostEqual(idcg_crisis.item(), 1e-6, places=8)
        self.assertFalse(torch.isnan(idcg_crisis).item())

    def test_positive_idcg_batch_processing(self):
        """2D Batch tensor girdisinde (batch_size, n_assets) doğru shape ve epsilon koruması."""
        y_batch = torch.tensor([
            [0.10, 0.05, 0.0, -0.05],   # Normal hafta
            [-0.10, -0.20, -0.30, -0.40] # Kriz haftası
        ])
        idcg_batch = self.idcg_layer(y_batch)

        self.assertEqual(idcg_batch.shape, (2, 1))
        self.assertGreater(idcg_batch[0].item(), 1e-6)
        self.assertAlmostEqual(idcg_batch[1].item(), 1e-6, places=8)

    def test_delist_sigmoid_penalty_calculation(self):
        """
        Adım 3.2.2: Delist hisselere eklemeli Sigmoid cezasının doğrulanması.
        Model delist hisseye yüksek skor (y_pred = 5.0 -> sigmoid ~ 0.993) verirse ceza yüksek olmalıdır.
        """
        y_pred = torch.tensor([2.0, 0.5, 5.0, -3.0])  # 3. hisseye yüksek tahmin verilmiş
        delist_mask = torch.tensor([False, False, True, False])

        penalty = self.delist_penalty_layer(y_pred, delist_mask)
        
        # sigmoid(5.0) ~ 0.9933, eta = 1.5 => penalty ~ 1.5 * 0.9933 = 1.49
        expected_penalty = 1.5 * torch.sigmoid(torch.tensor(5.0)).item()
        self.assertAlmostEqual(penalty.item(), expected_penalty, places=4)

    def test_delist_sigmoid_penalty_no_delist(self):
        """Delist hisse olmadığında ceza 0.0 dönmelidir."""
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        delist_mask = torch.tensor([False, False, False])

        penalty = self.delist_penalty_layer(y_pred, delist_mask)
        self.assertEqual(penalty.item(), 0.0)

    def test_clamped_logit_difference_overflow_protection(self):
        """
        Adım 3.2.3: Aşırı yüksek logit farklarında [-10.0, +10.0] kırpmasının yapılması
        ve float overflow'un engellenmesi.
        """
        y_pred = torch.tensor([100.0, -100.0, 0.0])
        diff_matrix = self.logit_diff_layer(y_pred, tau=0.01)  # Çok düşük tau

        self.assertLessEqual(torch.max(diff_matrix).item(), 10.0)
        self.assertGreaterEqual(torch.min(diff_matrix).item(), -10.0)
        self.assertFalse(torch.isnan(diff_matrix).any().item())

    def test_soft_ndcg_zero_nan(self):
        """
        Adım 3.2.4 (Kalite Testi): Aşırı kriz simülasyonları, 0.01 tau, -100.0 logitler ve
        delist hisseler mevcudiyetinde tam 0 ADET NaN loss ve gradyan elde edildiğinin kanıtlanması.
        """
        y_pred = torch.tensor([50.0, -50.0, 100.0, -200.0], requires_grad=True)
        y_true_crisis = torch.tensor([-0.40, -0.60, -1.00, -0.80]) # Kriz ortamı
        delist_mask = torch.tensor([False, False, True, False])      # 3. hisse batan/delist

        loss = self.soft_ndcg_loss(y_pred, y_true_crisis, delisted_mask=delist_mask, tau=0.05)
        loss.backward()

        # Loss ve gradyanlar kesinlikle NaN veya Inf üretmemelidir
        self.assertFalse(torch.isnan(loss).item())
        self.assertFalse(torch.isinf(loss).item())
        self.assertIsNotNone(y_pred.grad)
        self.assertFalse(torch.isnan(y_pred.grad).any().item())


if __name__ == "__main__":
    unittest.main()


