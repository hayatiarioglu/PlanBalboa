"""
Sprint 1 / Faz 1.5 / Adım 1.5.3 Kalite Testi:
test_tefas_nav_fallback()
Eksik NAV verisi durumunda EWMA Fallback ve Volatilite Taban Zırhının
devreye girdiğini kanıtlayan entegrasyon ve kalite gate testi.
"""

import unittest
from datetime import datetime, timedelta
import numpy as np

from aether.data.pit_snapshot import AssetType, DataPoint, PITDataSnapshot
from aether.data.hybrid_volatility import HybridVolatilityEngine, VolatilityResult


class TestTEFASNavFallbackGate(unittest.TestCase):

    def setUp(self):
        self.engine = HybridVolatilityEngine(egarch_weight=0.5, min_volatility_floor=1e-4)

    def test_tefas_nav_fallback(self):
        """
        Eksik veya yetersiz TEFAS NAV verisi durumunda EWMA ve Floor Fallback mekanizmalarının
        anında devreye girdiğini kanıtlayan Kalite Gate testi.
        """
        ts_base = datetime(2026, 7, 1)

        # 1. SENARYO A: Yalnızca 1 NAV Kaydı Bulunan Yeni TEFAS Fonu (Aşırı Veri Eksikliği)
        single_point = [DataPoint("NEW_FREE_FUND", AssetType.TEFAS_FREE, ts_base, price=2.50)]
        res_a = self.engine.compute_tefas_nav_volatility("NEW_FREE_FUND", AssetType.TEFAS_FREE, single_point)

        self.assertTrue(res_a.is_fallback, "1 NAV verili fonda is_fallback = True olmalıdır!")
        self.assertEqual(res_a.volatility, 1e-4, "Veri olmadığında volatilite min_volatility_floor=1e-4 olmalıdır!")
        self.assertEqual(res_a.parkinson_vol, 0.0)

        # 2. SENARYO B: Yalnızca 4 NAV Kaydı (Kısmi Eksik Veri < 10 gün)
        partial_points = [
            DataPoint("PARTIAL_FUND", AssetType.TEFAS_FREE, ts_base + timedelta(days=i), price=2.50 + i * 0.01)
            for i in range(4)
        ]
        res_b = self.engine.compute_tefas_nav_volatility("PARTIAL_FUND", AssetType.TEFAS_FREE, partial_points)

        self.assertTrue(res_b.is_fallback, "10 günden az NAV verisi olan fonda is_fallback = True olmalıdır!")
        self.assertGreater(res_b.volatility, 0.0)

        # 3. SENARYO C: TEFAS Likit Fonu (Para Piyasası) Veri Eksikliği Zırhı (%0.5 Yıllık Taban)
        liquid_single_point = [DataPoint("PPF_NEW", AssetType.TEFAS_LIQUID, ts_base, price=1.10)]
        res_c = self.engine.compute_tefas_nav_volatility("PPF_NEW", AssetType.TEFAS_LIQUID, liquid_single_point)

        self.assertTrue(res_c.is_fallback)
        self.assertEqual(res_c.volatility, 0.005, "TEFAS Likit fonlarda veri yoksa %0.5 (0.005) taban kullanılmalıdır!")

        # 4. SENARYO D: Tam Veri Seti (30 Günlük NAV - Fallback Tetiklenmemeli)
        full_points = [
            DataPoint("FULL_FUND", AssetType.TEFAS_FREE, ts_base + timedelta(days=i), price=2.50 + (np.sin(i) * 0.02))
            for i in range(30)
        ]
        res_d = self.engine.compute_tefas_nav_volatility("FULL_FUND", AssetType.TEFAS_FREE, full_points)

        self.assertFalse(res_d.is_fallback, "30 günlük tam veride is_fallback = False olmalıdır!")
        self.assertGreater(res_d.volatility, 1e-4)

        # 5. MULTI-ASSET SNAPSHOT ÜZERİNDE FALLBACK DENETİMİ
        snapshot = PITDataSnapshot(ts_base, ts_base)
        snapshot.tefas_free_funds["NEW_FREE_FUND"] = single_point
        snapshot.tefas_liquid_funds["PPF_NEW"] = liquid_single_point

        vol_results = self.engine.compute_snapshot_volatilities(snapshot)

        self.assertIn("NEW_FREE_FUND", vol_results)
        self.assertIn("PPF_NEW", vol_results)
        self.assertTrue(vol_results["NEW_FREE_FUND"].is_fallback)
        self.assertTrue(vol_results["PPF_NEW"].is_fallback)


if __name__ == "__main__":
    unittest.main()
