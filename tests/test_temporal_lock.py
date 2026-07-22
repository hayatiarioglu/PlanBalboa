"""
Sprint 1 / Faz 1.1 / Adım 1.1.2 Birim Testi:
TemporalLockEngine Zamansal Kilitlenme ve Gelecek Sızıntısı Engelleme Testleri
"""

import unittest
from datetime import datetime, timedelta
import pandas as pd
from aether.data.pit_snapshot import AssetType, DataPoint, PITDataSnapshot, PITDataAdapter
from aether.data.temporal_lock import TemporalLockEngine, TemporalViolationReport


class TestTemporalLockEngine(unittest.TestCase):

    def setUp(self):
        # Cuma 17:30:00.000000
        self.snapshot_time = datetime(2026, 7, 17, 17, 30, 0, 0)
        # Perşembe 23:59:59.999999
        self.tefas_cutoff_date = datetime(2026, 7, 16, 23, 59, 59, 999999)
        self.engine = TemporalLockEngine(self.snapshot_time, self.tefas_cutoff_date)

    def test_intraday_microsecond_boundary(self):
        """Hisse/BYF 15dk barlarının mikrosaniye seviyesinde kilitlenmesi."""
        # Tam sınır (Geçerli)
        self.assertTrue(self.engine.is_intraday_valid(datetime(2026, 7, 17, 17, 30, 0, 0)))

        # 1 mikrosaniye sonrası (İhlal)
        self.assertFalse(self.engine.is_intraday_valid(datetime(2026, 7, 17, 17, 30, 0, 1)))

        # 15 dakika sonrası (İhlal)
        self.assertFalse(self.engine.is_intraday_valid(datetime(2026, 7, 17, 17, 45, 0, 0)))

    def test_tefas_daily_nav_boundary(self):
        """TEFAS fonlarının Perşembe 23:59:59.999999 sınırında kilitlenmesi."""
        # Perşembe son saniye (Geçerli)
        self.assertTrue(self.engine.is_tefas_valid(datetime(2026, 7, 16, 23, 59, 59, 999999)))

        # Cuma ilk mikrosaniye (İhlal - Cuma NAV'ı henüz açıklanmadı)
        self.assertFalse(self.engine.is_tefas_valid(datetime(2026, 7, 17, 0, 0, 0, 0)))

    def test_filter_dataframe_audit_report(self):
        """TemporalLockEngine filter_dataframe metodunun denetim raporu üretmesi."""
        raw_data = [
            {"symbol": "THYAO", "asset_type": "EQUITY", "timestamp": "2026-07-17 17:30:00", "price": 310.0},
            {"symbol": "THYAO", "asset_type": "EQUITY", "timestamp": "2026-07-17 17:30:01", "price": 311.0}, # İhlal
            {"symbol": "GLDTR", "asset_type": "ETF", "timestamp": "2026-07-17 17:29:59", "price": 185.0},
            {"symbol": "TLY", "asset_type": "TEFAS_FREE", "timestamp": "2026-07-16 23:59:59", "price": 2.45},
            {"symbol": "TLY", "asset_type": "TEFAS_FREE", "timestamp": "2026-07-17 00:00:00", "price": 2.46}, # İhlal
        ]
        df_raw = pd.DataFrame(raw_data)
        clean_df, report = self.engine.filter_dataframe(df_raw)

        self.assertEqual(report.total_records_checked, 5)
        self.assertEqual(report.total_records_filtered, 2)
        self.assertEqual(report.intraday_violations, 1)
        self.assertEqual(report.tefas_violations, 1)
        self.assertTrue(report.has_violations)
        self.assertEqual(len(clean_df), 3)

    def test_validate_snapshot_exception(self):
        """Geçersiz zamanlı veri barındıran snapshot objesinin tespiti ve ValueError fırlatması."""
        snapshot = PITDataSnapshot(self.snapshot_time, self.tefas_cutoff_date)
        snapshot.equities["THYAO"] = [
            DataPoint("THYAO", AssetType.EQUITY, datetime(2026, 7, 17, 17, 35, 0), 312.0)
        ]

        with self.assertRaises(ValueError):
            self.engine.validate_snapshot(snapshot)


if __name__ == "__main__":
    unittest.main()
