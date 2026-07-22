"""
Sprint 1 / Faz 1.5 / Adım 1.5.1 Birim Testi:
HybridVolatilityEngine EGARCH(1,1) + Parkinson Volatilite Tensörü Testleri
"""

import unittest
from datetime import datetime, timedelta
import numpy as np
from aether.data.pit_snapshot import AssetType, DataPoint
from aether.data.hybrid_volatility import HybridVolatilityEngine, VolatilityResult


class TestHybridVolatilityEngine(unittest.TestCase):

    def setUp(self):
        self.engine = HybridVolatilityEngine(egarch_weight=0.5, min_volatility_floor=1e-4)

    def test_parkinson_volatility(self):
        """Parkinson High-Low volatilite formülünün doğrulanması."""
        highs = np.array([105.0, 108.0, 110.0, 106.0])
        lows = np.array([100.0, 101.0, 102.0, 99.0])

        park_vol = self.engine.compute_parkinson_volatility(highs, lows)
        self.assertGreater(park_vol, 0.0)
        self.assertGreaterEqual(park_vol, 1e-4)

    def test_egarch_volatility(self):
        """EGARCH(1,1) / EWMA koşullu volatilite hesabı."""
        log_rets = np.array([0.01, -0.02, 0.015, -0.03, 0.005, -0.01])
        egarch_vol = self.engine.compute_egarch_volatility(log_rets)
        self.assertGreater(egarch_vol, 0.0)
        self.assertGreaterEqual(egarch_vol, 1e-4)

    def test_intraday_hybrid_volatility_combination(self):
        """Hisse ve BYF'ler için EGARCH(1,1) + Parkinson V_t^hybrid kombinasyonu."""
        start_date = datetime(2026, 7, 1)
        thyao_points = []

        # 10 günlük BIST bar verisi
        for d in range(10):
            ts_open = start_date + timedelta(days=d, hours=10)
            ts_close = start_date + timedelta(days=d, hours=17, minutes=30)
            base_p = 300.0 + d * 2.0
            thyao_points.append(DataPoint("THYAO", AssetType.EQUITY, ts_open, price=base_p, high_price=base_p * 1.03, low_price=base_p * 0.98))
            thyao_points.append(DataPoint("THYAO", AssetType.EQUITY, ts_close, price=base_p * 1.01, high_price=base_p * 1.03, low_price=base_p * 0.98))

        res = self.engine.compute_intraday_hybrid_volatility("THYAO", AssetType.EQUITY, thyao_points)

        self.assertEqual(res.symbol, "THYAO")
        self.assertEqual(res.asset_type, AssetType.EQUITY)
        self.assertFalse(res.is_fallback)

        # V_t^hybrid = 0.5 * egarch + 0.5 * parkinson
        expected_hybrid = 0.5 * res.egarch_vol + 0.5 * res.parkinson_vol
        self.assertAlmostEqual(res.volatility, expected_hybrid, places=6)

    def test_volatility_flooring(self):
        """Sıfır hareketli fiyat serilerinde 1e-4 volatilite floor garantisi."""
        ts = datetime(2026, 7, 17)
        flat_points = [
            DataPoint("FLAT_ETF", AssetType.ETF, ts, price=100.0, high_price=100.0, low_price=100.0),
            DataPoint("FLAT_ETF", AssetType.ETF, ts + timedelta(hours=1), price=100.0, high_price=100.0, low_price=100.0)
        ]

        res = self.engine.compute_intraday_hybrid_volatility("FLAT_ETF", AssetType.ETF, flat_points)
        self.assertEqual(res.volatility, 1e-4)

    def test_tefas_nav_volatility(self):
        """TEFAS fonlarında 30 günlük hareketli NAV standart sapması ve EWMA hesabı."""
        start_date = datetime(2026, 6, 1)
        tly_points = []
        for d in range(15):
            ts = start_date + timedelta(days=d)
            p = 2.0 + (d * 0.005) + (np.sin(d) * 0.002)
            tly_points.append(DataPoint("TLY", AssetType.TEFAS_FREE, ts, price=p))

        res_tly = self.engine.compute_tefas_nav_volatility("TLY", AssetType.TEFAS_FREE, tly_points, rolling_days=30)

        self.assertEqual(res_tly.symbol, "TLY")
        self.assertEqual(res_tly.asset_type, AssetType.TEFAS_FREE)
        self.assertEqual(res_tly.parkinson_vol, 0.0, "TEFAS fonlarında Parkinson volatilite 0.0 olmalıdır!")
        self.assertGreater(res_tly.volatility, 0.0)


if __name__ == "__main__":
    unittest.main()
