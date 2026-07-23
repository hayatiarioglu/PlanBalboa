import os
import sys
import signal
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from time import sleep

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

from financial_ai.database_vault import DatabaseVault
from financial_ai.background_scheduler import BackgroundScheduler
from financial_ai.telegram_bot import TelegramBotService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("MainOrchestrator")


class DummyHealthCheckHandler(BaseHTTPRequestHandler):
    """Render Web Service Port Binding için Basit HTTP Healthcheck Sunucusu."""
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"OK - BIST100 DSS v13.1 Engine Active")

    def log_message(self, format, *args):
        return


def start_dummy_healthcheck_server():
    """Render'ın Port taramasını tatmin edecek 8080 / PORT HTTP sunucusu."""
    port = int(os.getenv("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), DummyHealthCheckHandler)
    logger.info(f"🌐 Render HealthCheck HTTP Sunucusu Port {port} Üzerinde Başlatıldı.")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()


def main():
    logger.info("==================================================")
    logger.info("BIST100 Sürüm 13.1 DSS Otonom Motor Başlatılıyor...")
    logger.info("==================================================")

    start_dummy_healthcheck_server()

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "MOCK_TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "MOCK_CHAT_ID")
    db_path = os.getenv("DB_PATH", "data/financial_ai.db")

    db_vault = DatabaseVault(db_path=db_path)
    telegram_bot = TelegramBotService(token=bot_token, chat_id=chat_id, db_vault=db_vault)

    scheduler_service = BackgroundScheduler(db_vault=db_vault)
    telegram_bot.scheduler = scheduler_service
    scheduler_service.telegram_bot = telegram_bot # Çift Yönlü Bağlantı

    init_thread = threading.Thread(target=scheduler_service.initialize_models, daemon=True)
    init_thread.start()

    try:
        logger.info("🚀 Tüm sistemler aktif. Telegram Bot Dinleyicisi başlatılıyor...")
        telegram_bot.start_bot()
    except Exception as e:
        logger.critical(f"Kritik Çalışma Zamanı Hatası: {e}")
    finally:
        logger.warning("⚠️ Servisler kapatılıyor. DatabaseVault temizleniyor...")
        db_vault.close()
        logger.info("✅ Tüm servisler güvenle kapatıldı.")


if __name__ == "__main__":
    main()
