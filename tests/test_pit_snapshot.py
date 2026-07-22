"""
Sprint 1 / Faz 1.1 / Adım 1.1.1 Birim Testi:
PITDataSnapshot Multi-Asset Destegi ve Zamansal Kilitlenme Adaptörü
"""

import unittest
from datetime import datetime
import pandas as pd
from aether.data.pit_snapshot import (
    AssetType,
    DataPoint,
    PITDataSnapshot,
    PITDataAdapter,
)


class TestPITSnapshotMultiAsset(unittest.TestCase):

    def test_asset_type_properties(self):
        """Varlık tiplerinin intraday vs daily_nav niteliklerinin doğrulanması."""
        self.assertTrue(AssetType.EQUITY.is_intraday)
        self.assertFalse(AssetType.EQUITY.is_daily_nav)

        self.assertTrue(AssetType.ETF.is_intraday)
        self.assertFalse(AssetType.ETF.is_daily_nav)

        self.assertFalse(AssetType.TEFAS_FREE.is_intraday)
        self.assertTrue(AssetType.TEFAS_FREE.is_daily_nav)

        self.assertFalse(AssetType.TEFAS_LIQUID.is_intraday)
        self.assertTrue(AssetType.TEFAS_LIQUID.is_daily_nav)

    def test_data_point_validation(self):
        """DataPoint veri yapısındaki katı veri doğrulama kurallarının sınanması."""
        ts = datetime(2026, 7, 17, 17, 30)

        # Mükemmel veri noktası
        dp = DataPoint(
            symbol="THYAO",
            asset_type=AssetType.EQUITY,
            timestamp=ts,
            price=310.50,
            volume=150000.0,
            auction_volume=25000.0,
            settlement_days=2
        )
        self.assertEqual(dp.price, 310.50)
        self.assertEqual(dp.settlement_days, 2)

        # Fiyat <= 0 hatası
        with self.assertRaises(ValueError):
            DataPoint(symbol="THYAO", asset_type=AssetType.EQUITY, timestamp=ts, price=0.0)

        # Hacim < 0 hatası
        with self.assertRaises(ValueError):
            DataPoint(symbol="THYAO", asset_type=AssetType.EQUITY, timestamp=ts, price=10.0, volume=-5.0)

    def test_pit_snapshot_multi_asset_aggregation(self):
        """PITDataSnapshot konteynerinin 4 varlık sınıfını da başarıyla indekslemesi."""
        ts = datetime(2026, 7, 17, 17, 30)
        tefas_ts = datetime(2026, 7, 16, 23, 59, 59)

        snapshot = PITDataSnapshot(snapshot_time=ts, tefas_cutoff_date=tefas_ts)

        snapshot.equities["THYAO"] = [DataPoint("THYAO", AssetType.EQUITY, ts, 310.0)]
        snapshot.etfs["GLDTR"] = [DataPoint("GLDTR", AssetType.ETF, ts, 185.0)]
        snapshot.tefas_free_funds["TLY"] = [DataPoint("TLY", AssetType.TEFAS_FREE, tefas_ts, 2.45)]
        snapshot.tefas_liquid_funds["PPF"] = [DataPoint("PPF", AssetType.TEFAS_LIQUID, tefas_ts, 1.12)]

        self.assertEqual(snapshot.total_series_count(), 4)
        self.assertEqual(snapshot.get_all_symbols(), {"THYAO", "GLDTR", "TLY", "PPF"})

        self.assertEqual(snapshot.get_symbol_asset_type("THYAO"), AssetType.EQUITY)
        self.assertEqual(snapshot.get_symbol_asset_type("GLDTR"), AssetType.ETF)
        self.assertEqual(snapshot.get_symbol_asset_type("TLY"), AssetType.TEFAS_FREE)
        self.assertEqual(snapshot.get_symbol_asset_type("PPF"), AssetType.TEFAS_LIQUID)

        df = snapshot.to_dataframe()
        self.assertEqual(len(df), 4)
        self.assertEqual(set(df["symbol"]), {"THYAO", "GLDTR", "TLY", "PPF"})

    def test_pit_data_adapter_temporal_locking(self):
        """PITDataAdapter sınıfının Cuma 17:30 ve Perşembe 23:59 sızıntılarını kesmesi."""
        snapshot_time = datetime(2026, 7, 17, 17, 30, 0)      # Cuma 17:30
        tefas_cutoff_date = datetime(2026, 7, 16, 23, 59, 59)  # Perşembe 23:59:59

        raw_data = [
            # Geçerli Hisse (Cuma 17:30)
            {"symbol": "THYAO", "asset_type": "EQUITY", "timestamp": "2026-07-17 17:30:00", "price": 310.0, "volume": 100.0},
            # Sızan Hisse (Cuma 17:45 - ELETMELİ)
            {"symbol": "THYAO", "asset_type": "EQUITY", "timestamp": "2026-07-17 17:45:00", "price": 315.0, "volume": 100.0},
            # Geçerli BYF (Cuma 17:15)
            {"symbol": "GLDTR", "asset_type": "ETF", "timestamp": "2026-07-17 17:15:00", "price": 184.0, "volume": 50.0},
            # Geçerli TEFAS NAV (Perşembe 23:59)
            {"symbol": "TLY", "asset_type": "TEFAS_FREE", "timestamp": "2026-07-16 23:59:59", "price": 2.45, "volume": 0.0},
            # Sızan TEFAS NAV (Cuma 23:59 - ELETMELİ)
            {"symbol": "TLY", "asset_type": "TEFAS_FREE", "timestamp": "2026-07-17 23:59:59", "price": 2.50, "volume": 0.0},
        ]

        df_raw = pd.DataFrame(raw_data)
        adapter = PITDataAdapter(snapshot_time=snapshot_time, tefas_cutoff_date=tefas_cutoff_date)
        snapshot = adapter.build_snapshot(df_raw)

        # THYAO için sadece 17:30 barı kalmalı
        self.assertEqual(len(snapshot.equities["THYAO"]), 1)
        self.assertEqual(snapshot.equities["THYAO"][0].price, 310.0)

        # GLDTR için 17:15 barı kalmalı
        self.assertEqual(len(snapshot.etfs["GLDTR"]), 1)
        self.assertEqual(snapshot.etfs["GLDTR"][0].price, 184.0)

        # TLY için sadece Perşembe NAV'ı kalmalı
        self.assertEqual(len(snapshot.tefas_free_funds["TLY"]), 1)
        self.assertEqual(snapshot.tefas_free_funds["TLY"][0].price, 2.45)


if __name__ == "__main__":
    unittest.main()
