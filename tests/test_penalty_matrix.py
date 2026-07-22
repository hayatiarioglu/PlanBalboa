"""
AetherForecaster-X v35.2 Multi-Asset Master
Katman 3: Valör-Cezalı Bounded Convex QP Solver Testleri

Faz 2.1: Sermaye ve Valör Cezası Matrisleri birim testleri.
"""

import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from aether.data.pit_snapshot import AssetType, DataPoint, PITDataSnapshot
from aether.data.auction_ratio import AuctionRatioResult
from aether.data.hybrid_volatility import VolatilityResult
from aether.optimization.penalty import PenaltyMatrixEngine


class TestPenaltyMatrixEngine(unittest.TestCase):

    def setUp(self):
        self.engine = PenaltyMatrixEngine(
            gamma_0=0.01,
            beta_valor=0.001,
            adv_window_days=20,
            epsilon=1e-6
        )
        self.snapshot_time = datetime(2026, 7, 17, 17, 30)
        self.tefas_cutoff = datetime(2026, 7, 16, 23, 59, 59)
        self.snapshot = PITDataSnapshot(self.snapshot_time, self.tefas_cutoff)
        
        self.base_time = datetime(2026, 7, 10, 10, 0)
        
        # Test senaryosu verileri
        self.points_equity_zero = [DataPoint("ILLIQUID_EQ", AssetType.EQUITY, self.base_time, 10.0, volume=0.0, settlement_days=2)]
        self.points_equity_liquid = [DataPoint("LIQUID_EQ", AssetType.EQUITY, self.base_time, 100.0, volume=10000.0, settlement_days=2)]
        self.points_tefas_free = [DataPoint("TEFAS_FREE", AssetType.TEFAS_FREE, self.base_time, 1.5, volume=0.0, settlement_days=1)]
        self.points_tefas_liquid = [DataPoint("TEFAS_LIQUID", AssetType.TEFAS_LIQUID, self.base_time, 2.0, volume=0.0, settlement_days=0)]

        self.snapshot.equities["ILLIQUID_EQ"] = self.points_equity_zero
        self.snapshot.equities["LIQUID_EQ"] = self.points_equity_liquid
        self.snapshot.tefas_free_funds["TEFAS_FREE"] = self.points_tefas_free
        self.snapshot.tefas_liquid_funds["TEFAS_LIQUID"] = self.points_tefas_liquid

        self.auction_ratios = {
            "ILLIQUID_EQ": AuctionRatioResult("ILLIQUID_EQ", AssetType.EQUITY, 1.0, 1),
            "LIQUID_EQ": AuctionRatioResult("LIQUID_EQ", AssetType.EQUITY, 0.2, 1),
            "TEFAS_FREE": AuctionRatioResult("TEFAS_FREE", AssetType.TEFAS_FREE, 1.0, 1),
            "TEFAS_LIQUID": AuctionRatioResult("TEFAS_LIQUID", AssetType.TEFAS_LIQUID, 1.0, 1)
        }
        
        self.volatilities = {
            "ILLIQUID_EQ": VolatilityResult("ILLIQUID_EQ", AssetType.EQUITY, 0.40, 0.4, 0.4),
            "LIQUID_EQ": VolatilityResult("LIQUID_EQ", AssetType.EQUITY, 0.30, 0.3, 0.3),
            "TEFAS_FREE": VolatilityResult("TEFAS_FREE", AssetType.TEFAS_FREE, 0.10, 0.0, 0.1),
            "TEFAS_LIQUID": VolatilityResult("TEFAS_LIQUID", AssetType.TEFAS_LIQUID, 0.05, 0.0, 0.05)
        }

    def test_adv_effective_zero_armor(self):
        """
        Test 1: Hacimsiz (0.0 hacim) hisselerde epsilon = 1e-6 zırhının
        ZeroDivisionError oluşturmadığını ve matrisin NaN üretmediğini doğrular.
        """
        total_capital = 1_000_000.0  # 1 Milyon TL
        res = self.engine.compute_penalty_matrices(
            self.snapshot, self.auction_ratios, self.volatilities, total_capital
        )
        
        # DataFrame içerisinde NaN var mı?
        self.assertFalse(res.d_impact.isnull().values.any())
        self.assertFalse(res.d_settlement.isnull().values.any())
        
        # ILLIQUID_EQ için ADV_effective = 1e-6 olmalı
        self.assertAlmostEqual(res.adv_effective["ILLIQUID_EQ"], 1e-6)
        
        # Etkisi aşırı büyük bir sayı olmalı (ZeroDivision'dan kurtarılmış ancak fahiş ceza)
        impact = res.d_impact.loc["ILLIQUID_EQ", "ILLIQUID_EQ"]
        self.assertTrue(impact > 1e10)

    def test_tefas_impact_zero(self):
        """
        Test 2: TEFAS fonlarında D_impact = 0.0 olduğunu doğrular.
        """
        res = self.engine.compute_penalty_matrices(
            self.snapshot, self.auction_ratios, self.volatilities, total_capital=1e6
        )
        self.assertEqual(res.d_impact.loc["TEFAS_FREE", "TEFAS_FREE"], 0.0)
        self.assertEqual(res.d_impact.loc["TEFAS_LIQUID", "TEFAS_LIQUID"], 0.0)

    def test_settlement_penalty_hierarchy(self):
        """
        Test 3: T+2 valörlü hisselerin T+1 (TEFAS Free) ve T+0 (TEFAS Liquid)
        fonlardan daha büyük bir D_settlement ürettiğini doğrular.
        """
        res = self.engine.compute_penalty_matrices(
            self.snapshot, self.auction_ratios, self.volatilities, total_capital=1e6
        )
        d_set = res.d_settlement
        
        val_t2 = d_set.loc["LIQUID_EQ", "LIQUID_EQ"]
        val_t1 = d_set.loc["TEFAS_FREE", "TEFAS_FREE"]
        val_t0 = d_set.loc["TEFAS_LIQUID", "TEFAS_LIQUID"]
        
        self.assertEqual(val_t0, 0.0)
        self.assertTrue(val_t2 > val_t1 > val_t0)
        
        # Kesin matematik: beta * settlement_days
        expected_t2 = 0.001 * 2
        expected_t1 = 0.001 * 1
        self.assertAlmostEqual(val_t2, expected_t2)
        self.assertAlmostEqual(val_t1, expected_t1)

    def test_quadratic_capital_impact(self):
        """
        Test 4: Sermaye W_total yükseldikçe D_impact cezasının kuadratik (W^2) arttığını teyit eder.
        """
        cap1 = 1_000_000.0  # 1 Milyon TL
        cap2 = 2_000_000.0  # 2 Milyon TL (2 katı)
        
        res1 = self.engine.compute_penalty_matrices(
            self.snapshot, self.auction_ratios, self.volatilities, cap1
        )
        res2 = self.engine.compute_penalty_matrices(
            self.snapshot, self.auction_ratios, self.volatilities, cap2
        )
        
        impact1 = res1.d_impact.loc["LIQUID_EQ", "LIQUID_EQ"]
        impact2 = res2.d_impact.loc["LIQUID_EQ", "LIQUID_EQ"]
        
        # Sermaye 2 katına çıktığında, impact (W^2) 4 katına çıkmalıdır.
        self.assertAlmostEqual(impact2, impact1 * 4.0, places=5)


if __name__ == "__main__":
    unittest.main()
