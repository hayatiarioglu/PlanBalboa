"""
Sprint 1 / Faz 1.1 / Adım 1.1.3 Kalite Testi:
test_tefas_temporal_leakage()
Cuma günkü TEFAS NAV verilerinin henüz açıklanmadan sisteme sızmadığını kanıtlayan katı stres testi.
"""

import unittest
from datetime import datetime
import pandas as pd
from aether.data.pit_snapshot import AssetType, DataPoint, PITDataSnapshot, PITDataAdapter
from aether.data.temporal_lock import TemporalLockEngine, TemporalViolationReport


class TestTEFASTemporalLeakageGate(unittest.TestCase):

    def setUp(self):
        # Cuma 17:30 decision time
        self.snapshot_time = datetime(2026, 7, 17, 17, 30, 0, 0)
        # TEFAS Cutoff: Perşembe 23:59:59.999999 (son resmi açıklanan NAV)
        self.tefas_cutoff_date = datetime(2026, 7, 16, 23, 59, 59, 999999)
        self.adapter = PITDataAdapter(self.snapshot_time, self.tefas_cutoff_date)

    def test_tefas_temporal_leakage(self):
        """
        Cuma 17:30 itibarıyla henüz açıklanmamış Cuma/Cumartesi TEFAS NAV verilerinin
        sisteme sızmadığını ve %100 filtrelendiğini kanıtlayan stres testi.
        """
        # Sentetik Karma Veri Seti (Geçerli + Sızan TEFAS NAV'ları)
        raw_dataset = [
            # --- TEFAS SERBEST FONLAR (TLY, IPB) ---
            # Çarşamba NAV (Geçerli)
            {"symbol": "TLY", "asset_type": "TEFAS_FREE", "timestamp": "2026-07-15 23:59:59", "price": 2.40},
            # Perşembe NAV (Geçerli - Son Sınır)
            {"symbol": "TLY", "asset_type": "TEFAS_FREE", "timestamp": "2026-07-16 23:59:59", "price": 2.45},
            # Cuma Sabah Erken NAV (Sızıntı - SIFIR TOLERANS ELETMELİ)
            {"symbol": "TLY", "asset_type": "TEFAS_FREE", "timestamp": "2026-07-17 09:00:00", "price": 2.47},
            # Cuma 17:30 Anlık Tahmini NAV (Sızıntı - SIFIR TOLERANS ELETMELİ)
            {"symbol": "TLY", "asset_type": "TEFAS_FREE", "timestamp": "2026-07-17 17:30:00", "price": 2.49},
            # Cuma Gece Resmi Açıklanan NAV (Sızıntı - SIFIR TOLERANS ELETMELİ)
            {"symbol": "TLY", "asset_type": "TEFAS_FREE", "timestamp": "2026-07-17 23:59:59", "price": 2.50},

            # --- TEFAS LİKİT/PARA PİYASASI FONLARI (PPF) ---
            # Perşembe NAV (Geçerli)
            {"symbol": "PPF", "asset_type": "TEFAS_LIQUID", "timestamp": "2026-07-16 23:59:59", "price": 1.12},
            # Cuma NAV (Sızıntı - ELETMELİ)
            {"symbol": "PPF", "asset_type": "TEFAS_LIQUID", "timestamp": "2026-07-17 23:59:59", "price": 1.13},
            # Cumartesi Gecikmeli NAV (Sızıntı - ELETMELİ)
            {"symbol": "PPF", "asset_type": "TEFAS_LIQUID", "timestamp": "2026-07-18 08:00:00", "price": 1.14},

            # --- KONTROL GRUBU: BIST HİSSE & BYF (Geçerli Cuma Barları) ---
            {"symbol": "THYAO", "asset_type": "EQUITY", "timestamp": "2026-07-17 17:30:00", "price": 310.0},
            {"symbol": "GLDTR", "asset_type": "ETF", "timestamp": "2026-07-17 17:30:00", "price": 185.0},
        ]

        df_raw = pd.DataFrame(raw_dataset)

        # Audit raporuyla birlikte snapshot oluştur
        snapshot, report = self.adapter.build_snapshot_with_audit(df_raw)

        # 1. DENETİM RAPORU DOĞRULAMASI
        self.assertEqual(report.total_records_checked, 10)
        self.assertEqual(report.total_records_filtered, 5, "5 adet Cuma/Cumartesi TEFAS NAV sızıntısı filtrelenmeliydi!")
        self.assertEqual(report.tefas_violations, 5)
        self.assertEqual(report.intraday_violations, 0)
        self.assertTrue(report.has_violations)

        # 2. SNAPSHOT TEFAS İÇERİK DOĞRULAMASI
        tly_points = snapshot.tefas_free_funds["TLY"]
        ppf_points = snapshot.tefas_liquid_funds["PPF"]

        # TLY için sadece Çarşamba ve Perşembe NAV'ları kalmalı
        self.assertEqual(len(tly_points), 2)
        self.assertEqual(tly_points[-1].timestamp, datetime(2026, 7, 16, 23, 59, 59))
        self.assertEqual(tly_points[-1].price, 2.45)

        # PPF için sadece Perşembe NAV'ı kalmalı
        self.assertEqual(len(ppf_points), 1)
        self.assertEqual(ppf_points[0].timestamp, datetime(2026, 7, 16, 23, 59, 59))
        self.assertEqual(ppf_points[0].price, 1.12)

        # 3. KONTROL GRUBU (Hisse & BYF) KORUNDU MU?
        self.assertEqual(len(snapshot.equities["THYAO"]), 1)
        self.assertEqual(len(snapshot.etfs["GLDTR"]), 1)

        # 4. SNAPSHOT ZAMANSAL KONTROLÜ
        self.assertTrue(self.adapter.lock_engine.validate_snapshot(snapshot))


if __name__ == "__main__":
    unittest.main()
