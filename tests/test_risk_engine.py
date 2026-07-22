"""
Sprint 1 / Faz 1.6 / Adım 1.6.1 Birim Testi:
MultiAssetRiskEngine Katman 1 Master Risk Orkestratörü Testleri
"""

import unittest
from datetime import datetime, timedelta
import pandas as pd

from aether.data.pit_snapshot import AssetType
from aether.data.risk_engine import MultiAssetRiskEngine, MultiAssetRiskEngineOutput


class TestMultiAssetRiskEngine(unittest.TestCase):

    def setUp(self):
        self.risk_engine = MultiAssetRiskEngine()
        self.snapshot_time = datetime(2026, 7, 17, 17, 30, 0)
        self.tefas_cutoff_date = datetime(2026, 7, 16, 23, 59, 59)

    def test_process_raw_data_multi_asset_pipeline(self):
        """Bütün varlık sınıflarının RiskEngine çatısı altında uçtan uca çalışması."""
        raw_dataset = [
            # THYAO (EQUITY)
            {"symbol": "THYAO", "asset_type": "EQUITY", "timestamp": "2026-07-16 17:30:00", "price": 300.0, "volume": 100000.0, "auction_volume": 15000.0},
            {"symbol": "THYAO", "asset_type": "EQUITY", "timestamp": "2026-07-17 17:30:00", "price": 310.0, "volume": 120000.0, "auction_volume": 18000.0},
            # GLDTR (ETF)
            {"symbol": "GLDTR", "asset_type": "ETF", "timestamp": "2026-07-16 17:30:00", "price": 180.0, "volume": 50000.0, "auction_volume": 5000.0},
            {"symbol": "GLDTR", "asset_type": "ETF", "timestamp": "2026-07-17 17:30:00", "price": 185.0, "volume": 60000.0, "auction_volume": 6000.0},
            # TLY (TEFAS FREE)
            {"symbol": "TLY", "asset_type": "TEFAS_FREE", "timestamp": "2026-07-15 23:59:59", "price": 2.40},
            {"symbol": "TLY", "asset_type": "TEFAS_FREE", "timestamp": "2026-07-16 23:59:59", "price": 2.45},
            # PPF (TEFAS LIQUID)
            {"symbol": "PPF", "asset_type": "TEFAS_LIQUID", "timestamp": "2026-07-15 23:59:59", "price": 1.10},
            {"symbol": "PPF", "asset_type": "TEFAS_LIQUID", "timestamp": "2026-07-16 23:59:59", "price": 1.12},
        ]
        df_raw = pd.DataFrame(raw_dataset)

        output = self.risk_engine.process_raw_data(
            df_raw,
            snapshot_time=self.snapshot_time,
            tefas_cutoff_date=self.tefas_cutoff_date
        )

        # 1. SNAPSHOT & AUDIT REPORT
        self.assertIsNotNone(output.snapshot)
        self.assertIsNotNone(output.audit_report)
        self.assertEqual(output.audit_report.total_records_checked, 8)

        # 2. AUCTION RATIOS (GLDTR BYF + THYAO Hisse + TEFAS 1.0)
        self.assertEqual(set(output.auction_ratios.keys()), {"THYAO", "GLDTR", "TLY", "PPF"})
        self.assertEqual(output.auction_ratios["TLY"].alpha_auction, 1.0)
        self.assertEqual(output.auction_ratios["PPF"].alpha_auction, 1.0)

        # 3. VOLATILITIES (EGARCH + Parkinson & NAV Std + EWMA)
        self.assertEqual(set(output.volatilities.keys()), {"THYAO", "GLDTR", "TLY", "PPF"})
        self.assertGreater(output.volatilities["THYAO"].volatility, 0.0)
        self.assertEqual(output.volatilities["TLY"].parkinson_vol, 0.0)

        # 4. MÜHÜRLÜ KOVARYANS MATRİSİ (4x4, lambda_min >= 1e-6, PD)
        cov = output.covariance_matrix
        self.assertEqual(cov.shape, (4, 4))
        self.assertTrue(output.spectral_analysis.is_positive_definite)
        self.assertGreaterEqual(output.spectral_analysis.min_eigenvalue, 1e-6 - 1e-12)


if __name__ == "__main__":
    unittest.main()
