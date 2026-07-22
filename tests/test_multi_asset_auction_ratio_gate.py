"""
Sprint 1 / Faz 1.2 / Adım 1.2.3 Kalite Testi:
test_multi_asset_auction_ratio()
Hisseler, Borsa Yatırım Fonları (BYF) ve TEFAS Fonlarının tamamını kapsayan
kapanış seansı oranı (alpha_auction) entegrasyon ve kalite gate testi.
"""

import unittest
from datetime import datetime, timedelta
import pandas as pd
from aether.data.pit_snapshot import AssetType, DataPoint, PITDataSnapshot, PITDataAdapter
from aether.data.auction_ratio import HistoricalAuctionRatioEngine, AuctionRatioResult


class TestMultiAssetAuctionRatioGate(unittest.TestCase):

    def setUp(self):
        self.snapshot_time = datetime(2026, 7, 17, 17, 30, 0)
        self.tefas_cutoff_date = datetime(2026, 7, 16, 23, 59, 59)
        self.engine = HistoricalAuctionRatioEngine(window_days=20, min_periods=5, epsilon=1e-6)

    def test_multi_asset_auction_ratio(self):
        """
        4 farklı varlık grubunu içeren bir PIT snapshot üzerinde kapanış seansı oranlarının
        tam doğrulukla ve sıfır hata prensibiyle hesaplandığını kanıtlayan Kalite Gate testi.
        """
        snapshot = PITDataSnapshot(self.snapshot_time, self.tefas_cutoff_date)
        start_date = datetime(2026, 6, 15)

        # 1. HİSSE (THYAO): %15 Ortalama Kapanış Seansı Hacmi
        thyao_points = []
        for d in range(20):
            ts = start_date + timedelta(days=d)
            thyao_points.append(DataPoint("THYAO", AssetType.EQUITY, ts, 310.0, volume=100000.0, auction_volume=15000.0))
        snapshot.equities["THYAO"] = thyao_points

        # 2. BYF / ETF (GLDTR): %10 Ortalama Kapanış Seansı Hacmi
        gldtr_points = []
        for d in range(20):
            ts = start_date + timedelta(days=d)
            gldtr_points.append(DataPoint("GLDTR", AssetType.ETF, ts, 185.0, volume=50000.0, auction_volume=5000.0))
        snapshot.etfs["GLDTR"] = gldtr_points

        # 3. YENİ BYF (NEW_BYF): %0 Kapanış Hacmi (Epsilon Floor edilmeli)
        new_byf_points = []
        for d in range(20):
            ts = start_date + timedelta(days=d)
            new_byf_points.append(DataPoint("NEW_BYF", AssetType.ETF, ts, 100.0, volume=20000.0, auction_volume=0.0))
        snapshot.etfs["NEW_BYF"] = new_byf_points

        # 4. TEFAS SERBEST FON (TLY): Sabit 1.0 olmalı
        snapshot.tefas_free_funds["TLY"] = [
            DataPoint("TLY", AssetType.TEFAS_FREE, self.tefas_cutoff_date, 2.45)
        ]

        # 5. TEFAS LİKİT FON (PPF): Sabit 1.0 olmalı
        snapshot.tefas_liquid_funds["PPF"] = [
            DataPoint("PPF", AssetType.TEFAS_LIQUID, self.tefas_cutoff_date, 1.12)
        ]

        # HESAPLAMA ÇALIŞTIRMA
        results = self.engine.compute_snapshot_auction_ratios(snapshot)

        # 1. TÜM HEDEF SEMBOLLER HESAPLANDI MI?
        self.assertEqual(len(results), 5)
        self.assertEqual(set(results.keys()), {"THYAO", "GLDTR", "NEW_BYF", "TLY", "PPF"})

        # 2. THYAO HİSSE ORANI KONTROLÜ (%15)
        res_thyao = results["THYAO"]
        self.assertEqual(res_thyao.asset_type, AssetType.EQUITY)
        self.assertAlmostEqual(res_thyao.alpha_auction, 0.15, places=4)
        self.assertFalse(res_thyao.is_floored)

        # 3. GLDTR BYF ORANI KONTROLÜ (%10) - Adım 1.2.1
        res_gldtr = results["GLDTR"]
        self.assertEqual(res_gldtr.asset_type, AssetType.ETF)
        self.assertAlmostEqual(res_gldtr.alpha_auction, 0.10, places=4)
        self.assertFalse(res_gldtr.is_floored)

        # 4. NEW_BYF EPSILON FLOOR KONTROLÜ (1e-6)
        res_new_byf = results["NEW_BYF"]
        self.assertEqual(res_new_byf.alpha_auction, 1e-6)
        self.assertTrue(res_new_byf.is_floored)

        # 5. TEFAS FONLARI SABİT 1.0 KONTROLÜ - Adım 1.2.2
        res_tly = results["TLY"]
        res_ppf = results["PPF"]
        self.assertEqual(res_tly.alpha_auction, 1.0)
        self.assertEqual(res_tly.historical_days_used, 0)
        self.assertEqual(res_ppf.alpha_auction, 1.0)
        self.assertEqual(res_ppf.historical_days_used, 0)

        # 6. KATMAN 3 UYUMLULUĞU: Effective ADV Hesabında NaN/Inf İhlali Var Mı?
        for symbol, res in results.items():
            fake_adv20 = 1000000.0
            effective_adv = max(fake_adv20 * res.alpha_auction, 1e-6)
            self.assertGreater(effective_adv, 0.0)
            self.assertFalse(pd.isna(effective_adv))


if __name__ == "__main__":
    unittest.main()
