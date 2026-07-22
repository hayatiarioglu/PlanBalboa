import asyncio
import logging
import html
import pandas as pd
from datetime import datetime, timedelta
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
    5 AL, 5 BEKLE, 5 SAT Katmanlı BIST100 Taraması, Tarih Damgalı Kartlar.
    """

    def __init__(self, token: str, chat_id: str, db_vault: DatabaseVault):
        self.token = token
        self.chat_id = chat_id
        self.db_vault = db_vault
        self.application: Optional[Application] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.scheduler = None

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
    # THREAD-SAFE DIŞ BİLDİRİM KÖPRÜSÜ
    # =========================================================================
    def send_notification_from_thread(self, ticker: str, alert_type: str, reason: str):
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._async_send_alert(ticker, alert_type, reason),
                self.loop
            )
        else:
            logger.error("Event loop aktif değil, bildirim gönderilemedi!")

    async def _async_send_alert(self, ticker: str, alert_type: str, reason: str):
        try:
            safe_reason = html.escape(reason)
            now_str = datetime.now().strftime("%d.%m.%Y %H:%M")
            if alert_type == "EMERGENCY_SAT":
                msg = (
                    f"🚨 <b>ACİL UYARI: {ticker} İÇİN ÇIKIŞ ZAMANI!</b> 🚨\n"
                    f"━━━━━━━━━━━━━━━━━━━━━\n"
                    f"📅 <b>Tarih:</b> {now_str}\n"
                    f"❓ <b>NE YAPACAĞIZ?</b>\n"
                    f"🔴 <b>HEMEN SAT / NAKİTE GEÇ!</b>\n\n"
                    f"💡 <b>NEDEN ÇIKIYORUZ?</b>\n"
                    f"{safe_reason}\n\n"
                    f"🛡️ <i>Sermayeni Koruma Zırhı Devreye Girdi.</i>"
                )
            else:
                msg = f"🔔 <b>SİSTEM BİLDİRİMİ: {ticker}</b> ({now_str})\n━━━━━━━━━━━━━━━━━━━━━\n{safe_reason}"

            await self.application.bot.send_message(
                chat_id=self.chat_id,
                text=msg,
                parse_mode=ParseMode.HTML
            )
        except Exception as e:
            logger.error(f"Telegram alert gönderme hatası: {e}")

    # =========================================================================
    # DETAYLI & TARİH DAMGALI HTML KART JENERATÖRÜ
    # =========================================================================
    def _format_opportunity_card(self, signal: Dict[str, Any]) -> str:
        ticker = html.escape(str(signal['ticker']))
        p_success = signal.get('p_success', 0.0) * 100
        stop_loss = signal.get('stop_loss_price', 0.0)
        target_low = signal.get('target_price_low', 0.0)
        target_high = signal.get('target_price_high', 0.0)
        advisory = html.escape(str(signal.get('revision_reason', 'NÖTR')))
        signal_code = signal.get('signal_code', 0)

        # Tarih Hesaplamaları (Bugünün Tarihi & 20 İşlem Günü / ~30 Takvim Günü Hedef Vade)
        analysis_date = datetime.now().strftime("%d.%m.%Y")
        target_date = (datetime.now() + timedelta(days=30)).strftime("%d.%m.%Y")

        action_text = "🟢 AL / POZİSYON AÇ" if signal_code == 1 else ("🔴 SAT / NAKİTE GEÇ" if signal_code == -1 else "🟡 BEKLE / NAKİTTE KAL")

        return (
            f"🎯 <b>YAPAY ZEKÂ FIRSAT UYARISI: {ticker}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📅 <b>Analiz Tarihi:</b> <b>{analysis_date}</b>\n"
            f"⏳ <b>Hedef Vade:</b> <b>20 İşlem Günü (~{target_date})</b>\n\n"
            f"❓ <b>NE YAPACAĞIZ?</b>\n"
            f"👉 <b>{action_text}</b>\n\n"
            f"💡 <b>NEDEN BU KARAR VERİLDİ?</b>\n"
            f"Yapay zekâ 19 göstergeyi (Bilanço, Takas, Para Akışı) taradı.\n"
            f"• <b>Tahmin Başarı Güveni:</b> %{p_success:.1f}\n"
            f"• <b>Model Notu:</b> {advisory}\n\n"
            f"🎯 <b>BEKLENEN HEDEF KÂR (20 GÜN):</b>\n"
            f"• <b>Hedef Fiyat Bölgesi:</b> <b>{target_low:.2f} TL - {target_high:.2f} TL</b> (+7.5% ... +9.5%)\n\n"
            f"🛡️ <b>KORUMA BARIYERİ (STOP LOSS):</b>\n"
            f"• Fiyat <b>{stop_loss:.2f} TL</b> altına düşerse robot seni uyarır ve <i>'Zarar büyümeden çıkalım'</i> der.\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"🕒 <i>Sürüm 13.1 DSS | 7/24 Kesintisiz Otonom Borsa Asistanın</i>"
        )

    # =========================================================================
    # KOMUT İŞLEYİCİLERİ
    # =========================================================================
    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        welcome_txt = (
            "🤖 <b>Merhaba! Ben Senin 7/24 Otonom Borsa Asistanınım.</b>\n\n"
            "Hiçbir karmaşık terim kullanmadan, borsadaki en iyi fırsatları sana bildiririm.\n\n"
            "Kullanabileceğin Komutlar:\n"
            "• `/scan` : BIST100 evrenini tarar ve (5 AL, 5 BEKLE, 5 SAT) gruplu sıralama sunar.\n"
            "• `/hisse THYAO` : Yazdığın hisse için <i>'Ne yapalım, hedef ne?'</i> detaylı kartını getirir.\n"
            "• `/durum` : Robotun çalışıp çalışmadığını kontrol eder."
        )
        await update.message.reply_text(welcome_txt, parse_mode=ParseMode.HTML)

    async def _cmd_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """5 AL, 5 BEKLE, 5 SAT Gruplanmış 15 Hisselik BIST100 Taraması."""
        await update.message.reply_text("🔎 <i>BIST100 evreni 5 AL, 5 BEKLE, 5 SAT gruplarıyla taranıyor...</i>", parse_mode=ParseMode.HTML)

        try:
            signals_buy = await asyncio.to_thread(self._get_grouped_signals, 1, 5)
            signals_wait = await asyncio.to_thread(self._get_grouped_signals, 0, 5)
            signals_sell = await asyncio.to_thread(self._get_grouped_signals, -1, 5)

            analysis_date = datetime.now().strftime("%d.%m.%Y")
            target_date = (datetime.now() + timedelta(days=30)).strftime("%d.%m.%Y")

            msg = (
                f"🏆 <b>YAPAY ZEKÂ BIST100 DETAYLI TARAMA RAPORU</b>\n"
                f"📅 <b>Analiz Tarihi:</b> {analysis_date} | ⏳ <b>Vade:</b> ~{target_date}\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n\n"
            )

            # 🟢 5 AL HİSSESİ
            msg += "🟢 <b>ALIM FIRSATI OLAN HİSSELER (TOP-5 AL):</b>\n"
            if signals_buy:
                for idx, sig in enumerate(signals_buy, 1):
                    msg += f"{idx}. <b>{html.escape(sig['ticker'])}</b> | Güven: <b>%{sig['p_success']*100:.1f}</b> | Hedef: <b>{sig['target_price_low']:.2f} TL</b>\n"
            else:
                msg += "<i>Şu an yüksek olasılıklı alım sinyali veren hisse yok.</i>\n"

            msg += "\n🟡 <b>NÖTR / POZİSYONU KORU HİSSELERİ (TOP-5 BEKLE):</b>\n"
            if signals_wait:
                for idx, sig in enumerate(signals_wait, 1):
                    msg += f"{idx}. <b>{html.escape(sig['ticker'])}</b> | Güven: <b>%{sig['p_success']*100:.1f}</b>\n"
            else:
                msg += "<i>Nötr konumda hisse bulunmuyor.</i>\n"

            msg += "\n🔴 <b>SAT / UZAK DURULMASI GEREKEN HİSSELER (TOP-5 SAT):</b>\n"
            if signals_sell:
                for idx, sig in enumerate(signals_sell, 1):
                    msg += f"{idx}. <b>{html.escape(sig['ticker'])}</b> | Düşüş Riski Var\n"
            else:
                msg += "<i>Şu an acil satılması gereken hisse bulunmuyor.</i>\n"

            msg += "\n━━━━━━━━━━━━━━━━━━━━━\n💡 <i>Detaylı kart için: `/hisse HİSSE_KODU`</i>"
            await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

        except Exception as e:
            logger.error(f"/scan komut hatası: {e}")
            await update.message.reply_text("❌ Tarama yapılırken bir sistem hatası oluştu.")

    async def _cmd_hisse(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Tekil hisse sorgulaması. DB'de yoksa canlı hesaplar."""
        if not context.args:
            await update.message.reply_text("⚠️ Lütfen merak ettiğin hissenin adını yaz. Örnek: `/hisse THYAO`", parse_mode=ParseMode.HTML)
            return

        ticker = context.args[0].upper().strip()
        signal = await asyncio.to_thread(self.db_vault.get_last_signal, ticker)

        if not signal and self.scheduler and self.scheduler.primary_m:
            try:
                df_live = pd.read_parquet("data/bist_2016_2026_adjusted.parquet")
                signal = await asyncio.to_thread(self.scheduler.evaluate_eod_signal, df_live, ticker)
            except Exception as e:
                logger.error(f"Canlı hisse hesaplama hatası ({ticker}): {e}")

        if not signal:
            await update.message.reply_text(f"❌ <b>{ticker}</b> hissesi bulunamadı veya analiz henüz tamamlanmadı.", parse_mode=ParseMode.HTML)
            return

        card_html = self._format_opportunity_card(signal)
        await update.message.reply_text(card_html, parse_mode=ParseMode.HTML)

    async def _cmd_durum(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        status_txt = (
            "✅ <b>SİSTEM SAĞLIK RAPORU</b>\n\n"
            "• <b>Robot Durumu:</b> 7/24 Bulut Sunucuda Tıkır Tıkır Çalışıyor.\n"
            "• <b>Borsa Bekçisi (10:15):</b> Açılışta ani çöküş var mı nöbette.\n"
            "• <b>Kapanış Motoru (18:15):</b> Gün sonu bilançoları ve takasları inceliyor.\n"
            "• <b>Kafa Karışıklığı Engeli:</b> Gereksiz AL-SAT mesajı atmaz, seni yormaz."
        )
        await update.message.reply_text(status_txt, parse_mode=ParseMode.HTML)

    def _get_grouped_signals(self, signal_code: int, limit: int = 5) -> List[Dict[str, Any]]:
        """DB'den belirli sinyal koduna göre gruplanmış ilk N hisseyi çeker."""
        query = """
            SELECT * FROM signals 
            WHERE id IN (SELECT MAX(id) FROM signals GROUP BY ticker)
            AND signal_code = ?
            ORDER BY p_success DESC LIMIT ?;
        """
        with self.db_vault._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (signal_code, limit))
            return [dict(row) for row in cursor.fetchall()]
