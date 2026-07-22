import os
import sys
import signal
import logging
from time import sleep

# Guaranteed Directory Permissions Setup
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

from financial_ai.database_vault import DatabaseVault
from financial_ai.background_scheduler import BackgroundScheduler
from financial_ai.telegram_bot import TelegramBotService

# Logging Yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("MainOrchestrator")


def main():
    logger.info("==================================================")
    logger.info("BIST100 Sürüm 13.1 DSS Otonom Motor Başlatılıyor...")
    logger.info("==================================================")

    # Environment Variables Kontrolü
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "MOCK_TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "MOCK_CHAT_ID")
    db_path = os.getenv("DB_PATH", "data/financial_ai.db")

    # 1. Veritabanı Vault Katmanını Başlat
    logger.info(f"💾 Veritabanı başlatılıyor: {db_path}")
    db_vault = DatabaseVault(db_path=db_path)

    # 2. Telegram Bot Servisini Oluştur
    logger.info("🤖 Telegram Bot Servisi hazırlanıyor...")
    telegram_bot = TelegramBotService(token=bot_token, chat_id=chat_id, db_vault=db_vault)

    # 3. Background Scheduler'ı Oluştur ve Bot Köprüsünü Bağla
    logger.info("⏰ 7/24 Arka Plan Zamanlayıcısı yapılandırılıyor...")
    scheduler_service = BackgroundScheduler(
        db_vault=db_vault
    )
    scheduler_service.initialize_models()

    # Graceful Shutdown (Kibar Kapanma) Sinyal Yakalayıcıları
    def graceful_shutdown(signum, frame):
        logger.warning("⚠️ Kapanma sinyali alındı (SIGINT/SIGTERM). Servisler durduruluyor...")
        db_vault.close()
        logger.info("✅ Tüm servisler güvenle kapatıldı. Çıkış yapılıyor.")
        sys.exit(0)

    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    # 4. Telegram Bot Polling Döngüsünü Başlat (Ana Thread'i Kilitler)
    try:
        logger.info("🚀 Tüm sistemler aktif. Telegram Bot Dinleyicisi başlatılıyor...")
        telegram_bot.start_bot()
    except Exception as e:
        logger.critical(f"Kritik Çalışma Zamanı Hatası: {e}")
        graceful_shutdown(None, None)


if __name__ == "__main__":
    main()
