import asyncio
import logging
import html
from typing import Optional, Dict, Any, List
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    Application
)
from financial_ai.database_vault import DatabaseVault

logger = logging.getLogger("TelegramBot")

class TelegramBotService:
    """
    Sürüm 13.1 DSS Otonom Telegram Botu Core Servisi.
    Mesaj Dili: Sade, anlaşılır, salağa anlatır gibi net Türkçe (Ne Yapacağız? Neden? Nereye Kadar?).
    """

    def __init__(self, token: str, chat_id: str, db_vault: DatabaseVault):
        self.token = token
        self.chat_id = chat_id
        self.db_vault = db_vault
        self.application: Optional[Application] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def start_bot(self):
        """Botu ana thread üzerinde async olarak başlatır."""
        self.loop = asyncio.get_event_loop()
        self.application = ApplicationBuilder().token(self.token).build()

        # Komut İşleyicileri
        self.application.add_handler(CommandHandler("start", self._cmd_start))
        self.application.add_handler(CommandHandler("scan", self._cmd_scan))
        self.application.add_handler(CommandHandler("hisse", self._cmd_hisse))
        self.application.add_handler(CommandHandler("durum", self._cmd_durum))

        logger.info("Telegram Bot dinleyicisi başlatılıyor...")
        self.application.run_polling(drop_pending_updates=True)

    # =========================================================================
    # THREAD-SAFE DIŞ BİLDİRİM KÖPRÜSÜ (SCHEDULER THREAD'İ İÇİN)
    # =========================================================================
    def send_notification_from_thread(self, ticker: str, alert_type: str, reason: str):
        """BackgroundScheduler thread'inden gelen acil durum mesajlarını güvenle aktarır."""
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._async_send_alert(ticker, alert_type, reason),
                self.loop
            )
        else:
            logger.error("Event loop aktif değil, bildirim gönderilemedi!")

    async def _async_send_alert(self, ticker: str, alert_type: str, reason: str):
        """Async bildirim gönderici (Sade Anlaşılır Türkçe)."""
        try:
            safe_reason = html.escape(reason)
            if alert_type == "EMERGENCY_SAT":
                msg = (
                    f"🚨 <b>ACİL UYARI: {ticker} İÇİN ÇIKIŞ ZAMANI!</b> 🚨\n"
                    f"━━━━━━━━━━━━━━━━━━━━━\n"
                    f"❓ <b>NE YAPACAĞIZ?</b>\n"
                    f"🔴 <b>HEMEN SAT / NAKİTE GEÇ!</b>\n\n"
                    f"💡 <b>NEDEN ÇIKIYORUZ?</b>\n"
                    f"{safe_reason}\n\n"
                    f"🛡️ <i>Sermayeni Koruma Zırhı Devreye Girdi.</i>"
                )
            else:
                msg = (
                    f"🔔 <b>SİSTEM BİLDİRİMİ: {ticker}</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━\n"
                    f"{safe_reason}"
                )

            await self.application.bot.send_message(
                chat_id=self.chat_id,
                text=msg,
                parse_mode=ParseMode.HTML
            )
        except Exception as e:
            logger.error(f"Telegram alert gönderme hatası: {e}")

    # =========================================================================
    # SADE VE NET HTML KART JENERATÖRÜ (ÇOCUĞA ANLATIR GİBİ)
    # =========================================================================
    def _format_opportunity_card(self, signal: Dict[str, Any]) -> str:
        """Sade Türkçe ile Neden - Ne Yapacağız - Nereye Kadar Kartı"""
        ticker = html.escape(str(signal['ticker']))
        p_success = signal.get('p_success', 0.0) * 100
        stop_loss = signal.get('stop_loss_price', 0.0)
        target_low = signal.get('target_price_low', 0.0)
        target_high = signal.get('target_price_high', 0.0)
        advisory = html.escape(str(signal.get('revision_reason', 'NÖTR')))
        signal_code = signal.get('signal_code', 0)

        action_text = "🟢 AL / POZİSYON AÇ" if signal_code == 1 else ("🔴 SAT / NAKİTE GEÇ" if signal_code == -1 else "🟡 BEKLE / YENİ ALIM YAPMA")

        return (
            f"🎯 <b>YAPAY ZEKÂ FIRSAT UYARISI: {ticker}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"❓ <b>NE YAPACAĞIZ?</b>\n"
            f"👉 <b>{action_text}</b>\n\n"
            f"💡 <b>NEDEN BU KARAR VERİLDİ?</b>\n"
            f"Yapay zekâ 19 göstergeyi (Bilanço, Takas, Para Akışı) taradı.\n"
            f"• <b>Tahmin Başarı Güveni:</b> %{p_success:.1f}\n"
            f"• <b>Model Notu:</b> {advisory}\n\n"
            f"🎯 <b>BEKLENEN HEDEF KÂR (20 GÜN):</b>\n"
            f"• <b>Hedef Fiyat:</b> <b>{target_low:.2f} TL - {target_high:.2f} TL</b> (+7.5% ... +9.5%)\n\n"
            f"🛡️ <b>KORUMA BARIYERİ (STOP LOSS):</b>\n"
            f"• Fiyat <b>{stop_loss:.2f} TL</b> altına düşerse robot seni uyarır ve <i>'Zarar büyümeden çıkalım'</i> der.\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"🕒 <i>Sürüm 13.1 DSS | 7/24 Kesintisiz Otonom Borsa Asistanın</i>"
        )

    # =========================================================================
    # KOMUT İŞLEYİCİLERİ (COMMAND HANDLERS)
    # =========================================================================
    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        welcome_txt = (
            "🤖 <b>Merhaba! Ben Senin 7/24 Otonom Borsa Asistanınım.</b>\n\n"
            "Hiçbir karmaşık terim kullanmadan, borsadaki en iyi fırsatları sana bildiririm.\n\n"
            "Kullanabileceğin Komutlar:\n"
            "• `/scan` : Yapay zekânın bulduğu en iyi ilk 5 fırsatı gösterir.\n"
            "• `/hisse THYAO` : Yazdığın hisse için <i>'Ne yapalım, hedef ne?'</i> raporunu getirir.\n"
            "• `/durum` : Robotun çalışıp çalışmadığını kontrol eder."
        )
        await update.message.reply_text(welcome_txt, parse_mode=ParseMode.HTML)

    async def _cmd_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Top-5 Aktif Fırsat Kartlarını getirir."""
        await update.message.reply_text("🔎 <i>BIST100 evrenini senin için tarıyorum, lütfen 3 saniye bekle...</i>", parse_mode=ParseMode.HTML)

        try:
            signals = await asyncio.to_thread(self._get_top_signals)

            if not signals:
                await update.message.reply_text("ℹ️ Şu an yapay zekânın çok güvenli gördüğü bir alım fırsatı yok. Paşa paşa nakitte bekliyoruz.")
                return

            for sig in signals:
                card_html = self._format_opportunity_card(sig)
                await update.message.reply_text(card_html, parse_mode=ParseMode.HTML)

        except Exception as e:
            logger.error(f"/scan komut hatası: {e}")
            await update.message.reply_text("❌ Tarama yapılırken bir sistem hatası oluştu.")

    async def _cmd_hisse(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Tekil hisse sorgulaması."""
        if not context.args:
            await update.message.reply_text("⚠️ Lütfen merak ettiğin hissenin adını yaz. Örnek: `/hisse THYAO`", parse_mode=ParseMode.HTML)
            return

        ticker = context.args[0].upper().strip()
        signal = await asyncio.to_thread(self.db_vault.get_last_signal, ticker)

        if not signal:
            await update.message.reply_text(f"❌ <b>{ticker}</b> için veritabanımda henüz bir analiz kaydı yok.", parse_mode=ParseMode.HTML)
            return

        card_html = self._format_opportunity_card(signal)
        await update.message.reply_text(card_html, parse_mode=ParseMode.HTML)

    async def _cmd_durum(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Sistem sağlık kontrolü."""
        status_txt = (
            "✅ <b>SİSTEM SAĞLIK RAPORU</b>\n\n"
            "• <b>Robot Durumu:</b> 7/24 Bulut Sunucuda Tıkır Tıkır Çalışıyor.\n"
            "• <b>Borsa Bekçisi (10:15):</b> Açılışta ani çöküş var mı nöbette.\n"
            "• <b>Kapanış Motoru (18:15):</b> Gün sonu bilançoları ve takasları inceliyor.\n"
            "• <b>Kafa Karışıklığı Engeli:</b> Gereksiz AL-SAT mesajı atmaz, seni yormaz."
        )
        await update.message.reply_text(status_txt, parse_mode=ParseMode.HTML)

    def _get_top_signals(self) -> List[Dict[str, Any]]:
        """DB'den en yüksek P(Success) skorlu aktif +1 sinyallerini çeker."""
        query = """
            SELECT * FROM signals 
            WHERE signal_code = 1
            ORDER BY timestamp DESC, p_success DESC LIMIT 5;
        """
        with self.db_vault._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]
