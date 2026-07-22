"""
AetherForecaster-X v35.2 Multi-Asset Master
Katman 3: Valör-Cezalı Bounded Convex QP Solver Testleri

Faz 2.3: Alpha Signal Engine Birim Testleri.
"""

import unittest
import numpy as np
import pandas as pd

from aether.optimization.alpha_signal import AlphaSignalEngine, AlphaSignalResult


class TestAlphaSignalEngine(unittest.TestCase):

    def setUp(self):
        self.engine = AlphaSignalEngine(
            epsilon=1e-6,
            target_volatility_scale=1.0,
            tefas_alpha_multiplier=0.8
        )
        self.raw_signals = pd.Series({
            "GARAN": 0.05,
            "THYAO": 0.12,
            "ASELS": -0.02,
            "FON1": 0.01
        })
        self.asset_types = {
            "GARAN": "EQUITY",
            "THYAO": "EQUITY",
            "ASELS": "EQUITY",
            "FON1": "TEFAS"
        }

    def test_alpha_signal_processing(self):
        """
        Test 1: AlphaSignalEngine sınıfının ham sinyalleri kabul edip Z-Score normalizasyonu
        ve TEFAS sönümlemesi uyguladığını doğrular.
        """
        res = self.engine.process_alpha_signals(self.raw_signals, self.asset_types)

        self.assertIsInstance(res, AlphaSignalResult)
        self.assertEqual(len(res.normalized_alpha), 4)
        
        # Z-Score normalizasyonu sonucu ortalama ~ 0.0 ve std ~ 1.0 olmalı
        self.assertAlmostEqual(float(res.normalized_alpha.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(res.normalized_alpha.std(ddof=0)), 1.0, places=4)

        # q_vector = -scaled_alpha ilişkisini kontrol et
        pd.testing.assert_series_equal(res.q_vector, -res.scaled_alpha)

    def test_tefas_alpha_multiplier(self):
        """
        Test 2: TEFAS fonlarının alpha sinyalinin 0.8 katsayısı ile sönümlendiğini doğrular.
        """
        res = self.engine.process_alpha_signals(self.raw_signals, self.asset_types)

        # FON1 sinyali normalized_alpha * 0.8 olmalı
        expected_fon1_scaled = res.normalized_alpha["FON1"] * 0.8
        self.assertAlmostEqual(res.scaled_alpha["FON1"], expected_fon1_scaled, places=6)

    def test_scaling_and_tefas_multiplier(self):
        """
        Test 2: Z-Score sonrası target_volatility_scale ve TEFAS çarpanlarının (tefas_alpha_multiplier) 
        doğru uygulandığını kontrol eder.
        """
        raw = pd.Series({"GARAN": 0.05, "THYAO": 0.02, "FON1": 0.04, "FON2": 0.06})
        asset_types = {"GARAN": "EQUITY", "THYAO": "EQUITY", "FON1": "TEFAS_FREE", "FON2": "TEFAS_LIQUID"}
        
        res = self.engine.process_alpha_signals(raw, asset_types=asset_types)
        
        # Q Vektörü negatif alpha olmalıdır
        self.assertTrue((res.q_vector == -res.scaled_alpha).all())
        
        # Z-Score'lar
        mean_val = raw.mean()
        std_val = raw.std(ddof=0)
        
        z_fon1 = (0.04 - mean_val) / (std_val + 1e-6)
        expected_fon1 = z_fon1 * 1.0 * 0.8  # TEFAS çarpanı uygulanmalı (target_volatility_scale=1.0)
        
        z_fon2 = (0.06 - mean_val) / (std_val + 1e-6)
        expected_fon2 = z_fon2 * 1.0 * 0.8  # TEFAS çarpanı uygulanmalı (target_volatility_scale=1.0)
        
        self.assertAlmostEqual(res.scaled_alpha["FON1"], expected_fon1, places=6)
        self.assertAlmostEqual(res.scaled_alpha["FON2"], expected_fon2, places=6)

    def test_constant_signals_handling(self):
        """
        Test 3: Tüm sinyaller eşit olduğunda (std = 0) ZeroDivisionError olmadan
        sıfırlanmış normalizasyon üretildiğini doğrular.
        """
        flat_signals = pd.Series({"GARAN": 0.05, "THYAO": 0.05, "ASELS": 0.05})
        res = self.engine.process_alpha_signals(flat_signals)

        self.assertTrue((res.normalized_alpha == 0.0).all())
        self.assertTrue((res.q_vector == 0.0).all())

    def test_nan_handling(self):
        """
        Test 4: NaN/Inf içeren sinyallerin güvenli bir şekilde medyan ile doldurulduğunu doğrular.
        Ayrıca, sadece Inf'lerden oluşan bir matriste 0.0'a fallback yapıldığını test eder.
        """
        nan_signals = pd.Series({"GARAN": 0.05, "THYAO": np.nan, "ASELS": np.inf, "FON1": 0.01})
        res = self.engine.process_alpha_signals(nan_signals)

        self.assertFalse(res.raw_alpha.isna().any())
        self.assertFalse(np.isinf(res.raw_alpha.values).any())

        # Sadece Inf ve NaN içeren aşırı ekstrem senaryo
        all_inf_signals = pd.Series({"GARAN": np.inf, "THYAO": np.nan, "ASELS": -np.inf})
        res_extreme = self.engine.process_alpha_signals(all_inf_signals)
        
        self.assertFalse(res_extreme.raw_alpha.isna().any())
        self.assertFalse(np.isinf(res_extreme.raw_alpha.values).any())
        self.assertTrue((res_extreme.raw_alpha == 0.0).all())


if __name__ == "__main__":
    unittest.main()
