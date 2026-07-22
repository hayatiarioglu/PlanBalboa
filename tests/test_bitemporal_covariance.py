"""
Sprint 1 / Faz 1.3 / Adım 1.3.1 Birim Testi:
BitemporalCovarianceBuilder Frekans Eşleme ve Getiri Senkronizasyon Testleri
"""

import unittest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from aether.data.pit_snapshot import AssetType, DataPoint, PITDataSnapshot
from aether.data.bitemporal_covariance import BitemporalCovarianceBuilder, BitemporalCovarianceResult


class TestBitemporalCovarianceBuilder(unittest.TestCase):

    def setUp(self):
        self.builder = BitemporalCovarianceBuilder(bars_per_day=35)

    def test_compute_intraday_log_returns(self):
        """15 dakikalık DataPoint fiyatlarından logaritmik bar getirilerinin hesabı."""
        ts0 = datetime(2026, 7, 17, 10, 0)
        ts1 = datetime(2026, 7, 17, 10, 15)

        points = [
            DataPoint("THYAO", AssetType.EQUITY, ts0, price=100.0),
            DataPoint("THYAO", AssetType.EQUITY, ts1, price=105.0),
        ]
        s_15m = self.builder.compute_intraday_log_returns(points)
        self.assertEqual(len(s_15m), 1)
        expected_log_ret = np.log(105.0 / 100.0)
        self.assertAlmostEqual(s_15m.iloc[0], expected_log_ret, places=6)

    def test_compute_daily_returns_from_intraday(self):
        """15 dakikalık bar fiyatlarından günlük kapanış log getirilerinin hesabı."""
        day1_ts = datetime(2026, 7, 16, 17, 30)
        day2_ts = datetime(2026, 7, 17, 17, 30)

        points = [
            DataPoint("THYAO", AssetType.EQUITY, day1_ts, price=300.0),
            DataPoint("THYAO", AssetType.EQUITY, day2_ts, price=315.0),
        ]
        s_daily = self.builder.compute_daily_returns_from_intraday(points)
        self.assertEqual(len(s_daily), 1)
        expected_daily_ret = np.log(315.0 / 300.0)
        self.assertAlmostEqual(s_daily.iloc[0], expected_daily_ret, places=6)

    def test_build_bitemporal_returns_multi_asset(self):
        """Multi-Asset PIT snapshot üzerinde bitemporal getirilerin boyut eşleşmesi."""
        ts0 = datetime(2026, 7, 16, 17, 30)
        ts1 = datetime(2026, 7, 17, 17, 30)

        tefas_ts0 = datetime(2026, 7, 15, 23, 59, 59)
        tefas_ts1 = datetime(2026, 7, 16, 23, 59, 59)

        snapshot = PITDataSnapshot(snapshot_time=ts1, tefas_cutoff_date=tefas_ts1)

        # Equities
        snapshot.equities["THYAO"] = [
            DataPoint("THYAO", AssetType.EQUITY, ts0, price=300.0),
            DataPoint("THYAO", AssetType.EQUITY, ts1, price=310.0),
        ]
        # ETFs
        snapshot.etfs["GLDTR"] = [
            DataPoint("GLDTR", AssetType.ETF, ts0, price=180.0),
            DataPoint("GLDTR", AssetType.ETF, ts1, price=185.0),
        ]
        # TEFAS Free
        snapshot.tefas_free_funds["TLY"] = [
            DataPoint("TLY", AssetType.TEFAS_FREE, tefas_ts0, price=2.40),
            DataPoint("TLY", AssetType.TEFAS_FREE, tefas_ts1, price=2.45),
        ]
        # TEFAS Liquid
        snapshot.tefas_liquid_funds["PPF"] = [
            DataPoint("PPF", AssetType.TEFAS_LIQUID, tefas_ts0, price=1.10),
            DataPoint("PPF", AssetType.TEFAS_LIQUID, tefas_ts1, price=1.12),
        ]

        res = self.builder.build_bitemporal_returns(snapshot)

        self.assertEqual(len(res.symbols), 4)
        self.assertEqual(set(res.symbols), {"THYAO", "GLDTR", "TLY", "PPF"})
        self.assertEqual(set(res.intraday_symbols), {"THYAO", "GLDTR"})
        self.assertEqual(set(res.daily_symbols), {"TLY", "PPF"})

        # Intraday DataFrame sadece Hisse ve BYF içerir
        self.assertEqual(set(res.intraday_returns_df.columns), {"THYAO", "GLDTR"})

        # Daily DataFrame tüm sembolleri içerir (resample & senkronize edilmiş)
        self.assertEqual(set(res.daily_returns_df.columns), {"THYAO", "GLDTR", "TLY", "PPF"})

    def test_compute_newey_west_kappa(self):
        """Newey-West Bartlett kernel sönümleme katsayısının (kappa_effective) hesaplanması."""
        # 100 bar, %-20 otokorelasyonlu sentetik hisse getirisi
        np.random.seed(42)
        rets = np.random.normal(0, 0.01, 100)
        # Negatif otokorelasyon ekle (mean-reversion)
        for i in range(1, 100):
            rets[i] -= 0.3 * rets[i-1]

        df_intraday = pd.DataFrame({"THYAO": rets})
        kappa_eff = self.builder.compute_newey_west_kappa(df_intraday, max_lag=4, raw_n_bars=175.0)

        # Raw 175 yerine ~40-50 seviyesine sönümlenmeli
        self.assertLess(kappa_eff, 175.0)
        self.assertGreaterEqual(kappa_eff, 30.0)
        self.assertLessEqual(kappa_eff, 75.0)

    def test_build_weekly_bitemporal_covariance(self):
        """Hisse/BYF ve TEFAS fonlarının sönümlenmiş haftalık kovaryans matrisinin üretilmesi."""
        ts0 = datetime(2026, 7, 16, 17, 30)
        ts1 = datetime(2026, 7, 17, 17, 30)
        tefas_ts0 = datetime(2026, 7, 15, 23, 59, 59)
        tefas_ts1 = datetime(2026, 7, 16, 23, 59, 59)

        snapshot = PITDataSnapshot(snapshot_time=ts1, tefas_cutoff_date=tefas_ts1)
        snapshot.equities["THYAO"] = [DataPoint("THYAO", AssetType.EQUITY, ts0, 300.0), DataPoint("THYAO", AssetType.EQUITY, ts1, 310.0)]
        snapshot.tefas_free_funds["TLY"] = [DataPoint("TLY", AssetType.TEFAS_FREE, tefas_ts0, 2.40), DataPoint("TLY", AssetType.TEFAS_FREE, tefas_ts1, 2.45)]

        res = self.builder.build_bitemporal_returns(snapshot)
        cov_weekly, kappa = self.builder.build_weekly_bitemporal_covariance(res)

        self.assertEqual(cov_weekly.shape, (2, 2))
        self.assertEqual(list(cov_weekly.columns), ["THYAO", "TLY"])
        self.assertFalse(cov_weekly.isna().any().any())


if __name__ == "__main__":
    unittest.main()
