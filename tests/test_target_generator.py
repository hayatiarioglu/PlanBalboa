"""
AetherForecaster-X v35.2 Multi-Asset Master
Katman 2: Robust Winsorized Scaler & Target Generator Test Suite (Faz 3.1 / Adım 3.1.1)
"""

import unittest
import numpy as np
import pandas as pd

from aether.signal.target_generator import (
    RobustWinsorizedScaler,
    WinsorizedScalerResult,
    DelistDecouplingTargetGenerator,
    TargetGeneratorResult
)


class TestRobustWinsorizedScaler(unittest.TestCase):
    """
    Katman 2 Robust Winsorized Scaler Birim Test Paneli.
    """

    def setUp(self):
        self.scaler = RobustWinsorizedScaler(lower_quantile=0.025, upper_quantile=0.975)
        self.target_gen = DelistDecouplingTargetGenerator(scaler=self.scaler)

    def test_winsorized_scaler_basic(self):
        """
        Adım 3.1.1: Haftalık getirilerde alt (%2.5) ve üst (%97.5) kırpmaların doğrulanması.
        """
        # 100 elemanlı sıralı veri
        raw_data = np.linspace(-0.50, 0.50, 100)
        res = self.scaler.fit_transform(raw_data)

        self.assertIsInstance(res, WinsorizedScalerResult)
        self.assertEqual(len(res.winsorized_returns), 100)

        # En alt ve en üst değerler quantile sınırlarında kalmalıdır
        self.assertAlmostEqual(res.winsorized_returns[0], res.lower_quantile_val, places=4)
        self.assertAlmostEqual(res.winsorized_returns[-1], res.upper_quantile_val, places=4)
        
        # Gerçekleşen kırpma sayıları
        self.assertGreater(res.n_clipped_lower, 0)
        self.assertGreater(res.n_clipped_upper, 0)

    def test_winsorized_scaler_pandas_series(self):
        """Pandas Series olarak veri iletildiğinde index'in korunması."""
        idx = [f"SYM_{i}" for i in range(100)]
        ser = pd.Series(np.linspace(-0.20, 0.20, 100), index=idx)

        res = self.scaler.fit_transform(ser)
        self.assertIsInstance(res.winsorized_returns, pd.Series)
        self.assertTrue(res.winsorized_returns.index.equals(ser.index))

    def test_winsorized_scaler_nan_inf_protection(self):
        """NaN veya Inf verisinde sistemin çökmeyip zırhlanması."""
        bad_data = np.array([np.nan, -np.inf, -0.10, 0.05, 0.12, np.inf, 0.20])
        res = self.scaler.fit_transform(bad_data)

        self.assertTrue(np.all(np.isfinite(res.winsorized_returns)))

    def test_winsorized_scaler_invalid_quantiles(self):
        """Geçersiz quantile dilimlerinde ValueError fırlatılması."""
        with self.assertRaises(ValueError):
            RobustWinsorizedScaler(lower_quantile=0.80, upper_quantile=0.20)

    def test_delist_decoupling_basic(self):
        """
        Adım 3.1.2: Delist hisselerin (-100%) ayrıştırılması ve özel ceza atanması.
        """
        symbols = ["THYAO", "GARAN", "AKBNK", "DELIST_STOCK"]
        returns = pd.Series([0.15, 0.05, -0.05, -1.00], index=symbols)

        res = self.target_gen.generate_targets(returns)

        self.assertIsInstance(res, TargetGeneratorResult)
        self.assertTrue(res.delisted_mask["DELIST_STOCK"])
        self.assertFalse(res.delisted_mask["THYAO"])
        self.assertEqual(res.combined_targets["DELIST_STOCK"], -2.0)

    def test_winsorized_target_generator(self):
        """
        Adım 3.1.3 (Kalite Testi): Delist hisseler dahil olduğunda dahi sağlam hisseler
        arasında hedef sıkışması (target compression) yaşanmadığının kanıtlanması.
        """
        symbols = [f"SYM_{i}" for i in range(20)] + ["BANKRUPT_1", "BANKRUPT_2"]
        # 20 sağlam hisse (-%10 ile +%15 arası), 2 delist hisse (-%100)
        healthy_ret = np.linspace(-0.10, 0.15, 20)
        delist_ret = np.array([-1.00, -1.00])
        all_returns = pd.Series(np.concatenate([healthy_ret, delist_ret]), index=symbols)

        res = self.target_gen.generate_targets(all_returns)

        # 1. Delist hisselerin hedef genişliğini daraltmadığı doğrulanmalıdır
        self.assertGreater(res.healthy_spread, 0.20, f"Hedef sıkışması tespit edildi! Spread: {res.healthy_spread}")

        # 2. Delist hisseler tam -2.0 cezasını almalıdır
        self.assertEqual(res.combined_targets["BANKRUPT_1"], -2.0)
        self.assertEqual(res.combined_targets["BANKRUPT_2"], -2.0)


if __name__ == "__main__":
    unittest.main()

