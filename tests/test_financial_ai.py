import os
import unittest
import pandas as pd
from financial_ai.database_vault import DatabaseVault
from financial_ai.background_scheduler import BackgroundScheduler
from financial_ai.telegram_bot import TelegramBotService, NOMINAL_PRICE_MAP

class TestFinancialAIPipeline(unittest.TestCase):

    def setUp(self):
        self.db_path = "data/test_financial_ai.db"
        self.db_vault = DatabaseVault(db_path=self.db_path)
        self.scheduler = BackgroundScheduler(db_vault=self.db_vault, data_path="data/bist_2016_2026_adjusted.parquet")
        self.bot = TelegramBotService(token="MOCK_TOKEN", chat_id="MOCK_CHAT", db_vault=self.db_vault)

    def tearDown(self):
        self.db_vault.close()
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_database_vault_watchlist(self):
        self.assertTrue(self.db_vault.add_to_watchlist("THYAO"))
        self.assertIn("THYAO", self.db_vault.get_watchlist())
        self.assertTrue(self.db_vault.remove_from_watchlist("THYAO"))
        self.assertNotIn("THYAO", self.db_vault.get_watchlist())

    def test_nominal_price_mapping(self):
        self.assertEqual(self.bot._get_nominal_price("THYAO", 80.0), NOMINAL_PRICE_MAP["THYAO"])
        self.assertEqual(self.bot._get_nominal_price("UNKNOWN_TICKER", 12.5), 12.5)

    def test_model_initialization_and_boot_sweep(self):
        self.scheduler.initialize_models()
        self.assertIsNotNone(self.scheduler.primary_m)
        self.assertIsNotNone(self.scheduler.meta_m)
        self.assertGreaterEqual(self.scheduler.df_processed["ticker"].nunique(), 100)

        last_sig = self.db_vault.get_last_signal("THYAO")
        self.assertIsNotNone(last_sig)
        self.assertIn("current_price", last_sig)
        self.assertIn("p_success", last_sig)

if __name__ == "__main__":
    unittest.main()
