"""
Sprint 1 / Faz 1.2 / Adım 1.2.1 Birim Testi:
HistoricalAuctionRatioEngine BYF ve Hisse Kapanış Seansı Oranı (MA20) Testleri
"""

import unittest
from datetime import datetime, timedelta
from aether.data.pit_snapshot import AssetType, DataPoint, PITDataSnapshot
from aether.data.auction_ratio import HistoricalAuctionRatioEngine, AuctionRatioResult


class TestHistoricalAuctionRatioEngine(unittest.TestCase):

    def setUp(self):
        self.engine = HistoricalAuctionRatioEngine(window_days=20, min_periods=5, epsilon=1e-6)

    def test_etf_byf_auction_ratio_ma20(self):
        """BYF (Borsa Yatırım Fonu) için 20 günlük hareketli ortalama kapanış oranının hesaplanması."""
        start_date = datetime(2026, 6, 1)
        points = []

        # 25 günlük sentetik BYF (GLDTR) verisi: %10 kapanış seansı hacim oranı
        for i in range(25):
            current_date = start_date + timedelta(days=i)
            # Günde 5 bar (toplam hacim 100.000, kapanış seansı hacmi 10.000)
            for bar in range(5):
                ts = current_date.replace(hour=10 + bar, minute=0)
                is_auction = (bar == 4)
                auction_vol = 10000.0 if is_auction else 0.0
                points.append(DataPoint(
                    symbol="GLDTR",
                    asset_type=AssetType.ETF,
                    timestamp=ts,
                    price=185.0,
                    volume=20000.0,
                    auction_volume=auction_vol
                ))

        res = self.engine.compute_symbol_auction_ratio("GLDTR", AssetType.ETF, points)
        self.assertEqual(res.symbol, "GLDTR")
        self.assertEqual(res.asset_type, AssetType.ETF)
        self.assertAlmostEqual(res.alpha_auction, 0.10, places=4)
        self.assertEqual(res.historical_days_used, 20)
        self.assertFalse(res.is_floored)

    def test_zero_auction_volume_flooring(self):
        """Sıfır kapanış seansı hacminde alpha_auction'ın epsilon=1e-6 ile floor edilmesi."""
        start_date = datetime(2026, 6, 1)
        points = []

        for i in range(10):
            ts = start_date + timedelta(days=i)
            points.append(DataPoint(
                symbol="NEW_ETF",
                asset_type=AssetType.ETF,
                timestamp=ts,
                price=50.0,
                volume=10000.0,
                auction_volume=0.0
            ))

        res = self.engine.compute_symbol_auction_ratio("NEW_ETF", AssetType.ETF, points)
        self.assertEqual(res.alpha_auction, 1e-6)
        self.assertTrue(res.is_floored)

    def test_compute_snapshot_auction_ratios(self):
        """Snapshot konteyneri üzerindeki tüm varlık tiplerinde toplu hesaplama."""
        ts = datetime(2026, 7, 17, 17, 30)
        snapshot = PITDataSnapshot(snapshot_time=ts, tefas_cutoff_date=ts)

        # 1 Hisse + 1 BYF ekle
        snapshot.equities["THYAO"] = [
            DataPoint("THYAO", AssetType.EQUITY, ts, 310.0, volume=100.0, auction_volume=15.0)
        ]
        snapshot.etfs["ZGOLD"] = [
            DataPoint("ZGOLD", AssetType.ETF, ts, 95.0, volume=100.0, auction_volume=20.0)
        ]

    def test_tefas_funds_auction_ratio_constant_one(self):
        """TEFAS fonlarının gün içi kapanış seansı olmadığı için alpha_auction = 1.0 sabiti verilmesi."""
        ts = datetime(2026, 7, 16, 23, 59, 59)
        tly_points = [DataPoint("TLY", AssetType.TEFAS_FREE, ts, 2.45)]
        ppf_points = [DataPoint("PPF", AssetType.TEFAS_LIQUID, ts, 1.12)]

        res_tly = self.engine.compute_symbol_auction_ratio("TLY", AssetType.TEFAS_FREE, tly_points)
        res_ppf = self.engine.compute_symbol_auction_ratio("PPF", AssetType.TEFAS_LIQUID, ppf_points)

        self.assertEqual(res_tly.alpha_auction, 1.0)
        self.assertEqual(res_tly.historical_days_used, 0)
        self.assertFalse(res_tly.is_floored)

        self.assertEqual(res_ppf.alpha_auction, 1.0)
        self.assertEqual(res_ppf.historical_days_used, 0)
        self.assertFalse(res_ppf.is_floored)


if __name__ == "__main__":
    unittest.main()
