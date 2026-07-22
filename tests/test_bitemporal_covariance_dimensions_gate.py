"""
Sprint 1 / Faz 1.3 / Adım 1.3.3 Kalite Testi:
test_bitemporal_covariance_dimensions()
Blok matris birleşiminde boyut uyuşmazlığı, simetri bozulması ve NaN/Inf hatası
yaşanmadığını doğrulayan entegrasyon ve kalite gate testi.
"""

import unittest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from aether.data.pit_snapshot import AssetType, DataPoint, PITDataSnapshot
from aether.data.bitemporal_covariance import BitemporalCovarianceBuilder, BitemporalCovarianceResult


class TestBitemporalCovarianceDimensionsGate(unittest.TestCase):

    def setUp(self):
        self.builder = BitemporalCovarianceBuilder(bars_per_day=35)
        self.snapshot_time = datetime(2026, 7, 17, 17, 30, 0)
        self.tefas_cutoff = datetime(2026, 7, 16, 23, 59, 59)

    def test_bitemporal_covariance_dimensions(self):
        """
        8 farklı enstrümandan (3 Hisse + 2 BYF + 2 TEFAS Serbest + 1 TEFAS Likit)
        oluşan karma portföyde 8x8 kovaryans matrisinin boyut, simetri ve pozitiflik doğrulaması.
        """
        snapshot = PITDataSnapshot(self.snapshot_time, self.tefas_cutoff)
        start_date = datetime(2026, 6, 1)

        # 1. HİSSELER (3 Adet: THYAO, GARAN, EREGL)
        equities = ["THYAO", "GARAN", "EREGL"]
        for eq in equities:
            pts = []
            for d in range(15):
                ts0 = start_date + timedelta(days=d, hours=10)
                ts1 = start_date + timedelta(days=d, hours=17, minutes=30)
                price0 = 100.0 + np.random.uniform(-5, 5)
                price1 = price0 * (1.0 + np.random.uniform(-0.02, 0.02))
                pts.append(DataPoint(eq, AssetType.EQUITY, ts0, price0))
                pts.append(DataPoint(eq, AssetType.EQUITY, ts1, price1))
            snapshot.equities[eq] = pts

        # 2. BYF / ETF'LER (2 Adet: GLDTR, ZGOLD)
        etfs = ["GLDTR", "ZGOLD"]
        for etf in etfs:
            pts = []
            for d in range(15):
                ts0 = start_date + timedelta(days=d, hours=10)
                ts1 = start_date + timedelta(days=d, hours=17, minutes=30)
                price0 = 50.0 + np.random.uniform(-2, 2)
                price1 = price0 * (1.0 + np.random.uniform(-0.01, 0.01))
                pts.append(DataPoint(etf, AssetType.ETF, ts0, price0))
                pts.append(DataPoint(etf, AssetType.ETF, ts1, price1))
            snapshot.etfs[etf] = pts

        # 3. TEFAS SERBEST FONLAR (2 Adet: TLY, IPB)
        tefas_free = ["TLY", "IPB"]
        for tf in tefas_free:
            pts = []
            for d in range(15):
                ts = start_date + timedelta(days=d, hours=23, minutes=59)
                price = 2.0 + (d * 0.01) + np.random.uniform(-0.005, 0.005)
                pts.append(DataPoint(tf, AssetType.TEFAS_FREE, ts, price))
            snapshot.tefas_free_funds[tf] = pts

        # 4. TEFAS LİKİT FONLAR (1 Adet: PPF)
        snapshot.tefas_liquid_funds["PPF"] = [
            DataPoint("PPF", AssetType.TEFAS_LIQUID, start_date + timedelta(days=d, hours=23, minutes=59), 1.0 + d * 0.001)
            for d in range(15)
        ]

        # 5. GETİRİ Senkronizasyonu ve Haftalık Kovaryans Üretimi
        bitemporal_res = self.builder.build_bitemporal_returns(snapshot)
        cov_weekly, kappa_eff = self.builder.build_weekly_bitemporal_covariance(bitemporal_res)

        total_symbols_expected = 3 + 2 + 2 + 1  # 8 enstrüman

        # 1. BOYUT KONTROLÜ (Strict 8x8)
        self.assertEqual(cov_weekly.shape, (total_symbols_expected, total_symbols_expected))
        self.assertEqual(list(cov_weekly.index), bitemporal_res.symbols)
        self.assertEqual(list(cov_weekly.columns), bitemporal_res.symbols)

        # 2. SİMETRİ KONTROLÜ (Cov = Cov.T)
        matrix_vals = cov_weekly.values
        diff_from_transpose = np.max(np.abs(matrix_vals - matrix_vals.T))
        self.assertLess(diff_from_transpose, 1e-12, "Kovaryans matrisi kesinlikle simetrik olmalıdır!")

        # 3. NAN / INF HİÇBİR HÜCREDE OLMAMALI
        self.assertFalse(np.isnan(matrix_vals).any(), "Kovaryans matrisinde NaN tespit edildi!")
        self.assertFalse(np.isinf(matrix_vals).any(), "Kovaryans matrisinde Inf tespit edildi!")

        # 4. POZİTİF VARYANS KONTROLÜ (Köşegen Elemanlar Sigma_{i,i} >= 0)
        diag_vals = np.diag(matrix_vals)
        self.assertTrue((diag_vals >= 0.0).all(), "Kovaryans köşegeninde negatif varyans olamaz!")

        # 5. NEWEY-WEST SÖNÜMLEME SINIR KONTROLÜ (30 <= kappa <= 75)
        self.assertGreaterEqual(kappa_eff, 30.0)
        self.assertLessEqual(kappa_eff, 75.0)


if __name__ == "__main__":
    unittest.main()
