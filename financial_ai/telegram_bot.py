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

BIST100_VALID_TICKERS = {
    "THYAO", "GARAN", "ASELS", "EREGL", "SISE", "BIMAS", "KCHOL", "ARCLK", "TUPRS", "AKBNK",
    "YKBNK", "SAHOL", "PETKM", "KOZAL", "PGSUS", "ISCTR", "HEKTS", "SASA", "VAKBN", "HALKB",
    "TCELL", "TTKOM", "EKGYO", "TOASO", "FROTO", "ENKAI", "GUBRF", "ODAS", "KONTR", "SMRTG",
    "EUPWR", "KBORU", "ASTOR", "ALARK", "AEFES", "MAVI", "SOKM", "AGHOL", "DOAS", "MGROS",
    "BRSAN", "CANTE", "CEMTS", "CIMSA", "DOHOL", "ECILC", "EGEEN", "ENJSA", "GESAN", "GSDHO",
    "INVEO", "INVES", "ISDMR", "ISGYO", "ISMEN", "KCAER", "KORDS", "KOZAA", "KRDMD", "KZBGY",
    "MIATK", "MOBTL", "MPARK", "OTKAR", "OYAKC", "PENTAG", "QUAGR", "REEDR", "SDTTR", "SKBNK",
    "TATEN", "TAVHL", "TKFEN", "TMSN", "TRGYO", "TSKB", "TUKAS", "TURSG", "ULKER", "VESBE",
    "VESTL", "YEOTK", "YYLGD", "ZOREN", "ALFAS", "ANSGR", "BERA", "BFREN", "BIENY", "BOBET",
    "BRYAT", "CWENE", "EGPRO", "EBEBK", "GWIND", "IMASM", "KAYSE", "KMPUR", "TABGD"
}


NOMINAL_PRICE_MAP: Dict[str, float] = {
    "THYAO": 315.0, "GARAN": 115.0, "ASELS": 65.0, "EREGL": 48.0, "SISE": 45.0,
    "BIMAS": 550.0, "KCHOL": 210.0, "ARCLK": 155.0, "TUPRS": 165.0, "AKBNK": 60.0,
    "YKBNK": 30.0, "SAHOL": 95.0, "PETKM": 19.0, "KOZAL": 22.0, "PGSUS": 230.0,
    "ISCTR": 13.5, "HEKTS": 13.0, "SASA": 38.0, "VAKBN": 18.0, "HALKB": 17.0,
    "TCELL": 95.0, "TTKOM": 52.0, "EKGYO": 11.0, "TOASO": 240.0, "FROTO": 1050.0,
    "ENKAI": 42.0, "GUBRF": 180.0, "ODAS": 7.5, "KONTR": 48.0, "SMRTG": 50.0,
    "EUPWR": 85.0, "KBORU": 65.0, "ASTOR": 95.0, "ALARK": 105.0, "AEFES": 210.0,
    "MAVI": 115.0, "SOKM": 55.0, "AGHOL": 340.0, "DOAS": 260.0, "MGROS": 520.0,
    "BRSAN": 510.0, "CANTE": 17.0, "CEMTS": 11.0, "CIMSA": 35.0, "DOHOL": 14.0,
    "ECILC": 52.0, "EGEEN": 11500.0, "ENJSA": 62.0, "GESAN": 50.0, "GSDHO": 4.5,
    "INVEO": 10.0, "INVES": 320.0, "ISDMR": 38.0, "ISGYO": 16.0, "ISMEN": 36.0,
    "KCAER": 45.0, "KORDS": 85.0, "KOZAA": 55.0, "KRDMD": 25.0, "KZBGY": 18.0,
    "MIATK": 65.0, "MOBTL": 5.5, "MPARK": 320.0, "OTKAR": 480.0, "OYAKC": 58.0,
    "PENTAG": 18.0, "QUAGR": 4.5, "REEDR": 35.0, "SDTTR": 280.0, "SKBNK": 5.5,
    "TATEN": 35.0, "TAVHL": 240.0, "TKFEN": 48.0, "TMSN": 110.0, "TRGYO": 35.0,
    "TSKB": 11.0, "TUKAS": 7.0, "TURSG": 52.0, "ULKER": 150.0, "VESBE": 22.0,
    "VESTL": 80.0, "YEOTK": 210.0, "YYLGD": 14.0, "ZOREN": 5.0, "ALFAS": 85.0,
    "ANSGR": 95.0, "BERA": 16.0, "BFREN": 850.0, "BIENY": 35.0, "BOBET": 38.0,
    "BRYAT": 2400.0, "CWENE": 210.0, "EGPRO": 210.0, "EBEBK": 45.0, "GWIND": 28.0,
    "IMASM": 14.0, "KAYSE": 35.0, "KMPUR": 65.0, "TABGD": 165.0
}


class TelegramBotService:
    """
    Sürüm 13.1 DSS Otonom Telegram Botu Core Servisi.
    Kullanıcı Özel Takip Listesi (/takip, /ekle, /cikar), Real-Nominal Borsa Fiyatları,
    Açık Başlangıç-Bitiş Tarih Aralıklı Kartlar ve Otomatik Bildirim Sistemi.
    """

    def __init__(self, token: str, chat_id: str, db_vault: DatabaseVault):
        self.token = token
        self.chat_id = chat_id
        self.db_vault = db_vault
        self.application: Optional[Application] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.scheduler = None

    def _get_nominal_price(self, ticker: str, real_price: float) -> float:
        from financial_ai.live_price_fetcher import get_live_bist_price
        return get_live_bist_price(ticker, fallback_price=real_price)

    async def _post_init(self, application: Application):
        self.loop = asyncio.get_running_loop()
        logger.info("✅ Telegram Bot Event Loop başarıyla yakalandı.")

    def start_bot(self):
        """Botu ana thread üzerinde async olarak başlatır."""
        self.application = ApplicationBuilder().token(self.token).post_init(self._post_init).build()

        # Komut İşleyicileri
        self.application.add_handler(CommandHandler("start", self._cmd_start))
        self.application.add_handler(CommandHandler("scan", self._cmd_scan))
        self.application.add_handler(CommandHandler("hisse", self._cmd_hisse))
        self.application.add_handler(CommandHandler("takip", self._cmd_takip))
        self.application.add_handler(CommandHandler("ekle", self._cmd_ekle))
        self.application.add_handler(CommandHandler("cikar", self._cmd_cikar))
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
            elif alert_type == "TARGET_EXCEEDED":
                msg = (
                    f"🚀 <b>TAHMİN AŞILDI & TAVAN UYARISI: {ticker}!</b> 🚀\n"
                    f"━━━━━━━━━━━━━━━━━━━━━\n"
                    f"📅 <b>Tarih:</b> {now_str}\n"
                    f"❓ <b>NE YAPACAĞIZ?</b>\n"
                    f"💰 <b>KÂRI CEBE KOY (KÂR AL)!</b>\n\n"
                    f"💡 <b>GEREKÇE:</b>\n"
                    f"{safe_reason}\n"
                )
            else:
                msg = (
                    f"📢 <b>SİNYAL REVİZYONU: {ticker}</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━\n"
                    f"📅 <b>Tarih:</b> {now_str}\n"
                    f"💡 <b>GEREKÇE:</b> {safe_reason}"
                )

            await self.application.bot.send_message(
                chat_id=self.chat_id,
                text=msg,
                parse_mode=ParseMode.HTML
            )
        except Exception as e:
            logger.error(f"Telegram bildirim hatası ({ticker}): {e}")

    def _format_opportunity_card(self, signal: Dict[str, Any]) -> str:
        ticker = html.escape(signal['ticker'])
        cur_price = signal.get('current_price', 0.0)
        p_success = signal.get('p_success', 0.0) * 100
        advisory = html.escape(str(signal.get('revision_reason', 'NÖTR')))
        signal_code = signal.get('signal_code', 0)

        target_low = signal.get('target_low') or signal.get('target_price_low') or (cur_price * 1.075)
        target_high = signal.get('target_high') or signal.get('target_price_high') or (cur_price * 1.095)
        stop_loss = signal.get('stop_loss') or signal.get('stop_loss_price') or (cur_price * 0.96)

        pct_target_low = ((target_low / cur_price) - 1.0) * 100 if cur_price > 0 else 0.0
        pct_target_high = ((target_high / cur_price) - 1.0) * 100 if cur_price > 0 else 0.0
        pct_stop = ((stop_loss / cur_price) - 1.0) * 100 if cur_price > 0 else 0.0

        start_date = datetime.now().strftime("%d.%m.%Y")
        end_date = (datetime.now() + timedelta(days=30)).strftime("%d.%m.%Y")
        
        days_held = signal.get('days_held', 0)
        remaining_bus_days = max(20 - days_held, 1)

        if signal_code == 1:
            action_text = "🟢 AL / POZİSYON AÇ"
            prediction_text = f"• 📈 <b>Öngörülen Kâr Potansiyeli:</b> +%{pct_target_low:.1f} ... +%{pct_target_high:.1f} (Yüksek Kâr Fırsatı)"
        elif signal_code == -1:
            action_text = "🔴 SAT / NAKİTE GEÇ"
            prediction_text = f"• ⚠️ <b>Öngörülen Risk Sınırı:</b> -%{abs(pct_stop):.1f} (Destek Risk Seviyesi: {stop_loss:.2f} TL)"
        else:
            action_text = "🟡 BEKLE / NAKİTTE KAL"
            prediction_text = f"• ⚖️ <b>Öngörülen Bant Hareket:</b> +%{pct_target_low:.1f} ... +%{pct_target_high:.1f} (Nötr / Yatay Trend)"

        return (
            f"🎯 <b>YAPAY ZEKÂ FIRSAT UYARISI: {ticker}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📅 <b>HEDEF TARİH ARALIĞI:</b> <b>{start_date} ➔ {end_date}</b>\n"
            f"⏳ <b>KALAN SÜRE:</b> <b>{remaining_bus_days} İşlem Günü</b> (Toplam 20 Günlük Vade)\n\n"
            f"❓ <b>NE YAPACAĞIZ?</b>\n"
            f"👉 <b>{action_text}</b>\n\n"
            f"📊 <b>GÜNCEL FİYAT VE HEDEF DETAYLARI:</b>\n"
            f"• 💵 <b>Güncel Borsa Fiyatı:</b> {cur_price:.2f} TL\n"
            f"• 🎯 <b>Hedeflenen Fiyat:</b> {target_low:.2f} TL - {target_high:.2f} TL\n"
            f"{prediction_text}\n"
            f"• 🛡️ <b>İzleyen Stop Loss:</b> {stop_loss:.2f} TL (%{pct_stop:.1f} Risk Sınırı)\n\n"
            f"💡 <b>NEDEN BU KARAR VERİLDİ?</b>\n"
            f"Yapay zekâ 19 göstergeyi (Bilanço, Takas, Para Akışı) taradı.\n"
            f"• <b>Tahmin Başarı Güveni:</b> %{p_success:.1f}\n"
            f"• <b>Model Notu:</b> {advisory}\n"
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
            "📌 <b>KULLANABİLECEĞİN KOMUTLAR:</b>\n"
            "• `/scan` : BIST100 evrenini tarar ve (5 AL, 5 BEKLE, 5 SAT) gruplu sıralama sunar.\n"
            "• `/hisse THYAO` : Yazdığın hisse için detaylı fırsat kartını getirir.\n"
            "• `/takip` : Senin özel takip listendeki hisseleri ve canlı durumlarını listeler.\n"
            "• `/ekle THYAO` : THYAO hissesini senin özel otomatik takip listene ekler.\n"
            "• `/cikar THYAO` : THYAO hissesini takip listenden çıkarır.\n"
            "• `/durum` : Robotun sağlık ve nöbet durumunu gösterir."
        )
        await update.message.reply_text(welcome_txt, parse_mode=ParseMode.HTML)

    async def _cmd_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """5 AL, 5 BEKLE, 5 SAT Gruplanmış Detaylı BIST100 Taraması."""
        await update.message.reply_text("🔎 <i>BIST100 evreni (5 AL, 5 BEKLE, 5 SAT) gruplarıyla taranıyor...</i>", parse_mode=ParseMode.HTML)

        try:
            signals_buy = await asyncio.to_thread(self._get_grouped_signals, 1, 5)
            signals_wait = await asyncio.to_thread(self._get_grouped_signals, 0, 5)
            signals_sell = await asyncio.to_thread(self._get_grouped_signals, -1, 5)

            if not (signals_buy or signals_wait) and self.scheduler:
                retries = 0
                while self.scheduler.primary_m is None and retries < 15:
                    await asyncio.sleep(1)
                    retries += 1

                if self.scheduler.primary_m:
                    df_live = pd.read_parquet("data/bist_2016_2026_adjusted.parquet")
                    tickers = [t for t in df_live['ticker'].unique() if not str(t).startswith("DELIST")]
                    computed_signals = []
                    for t in tickers:
                        try:
                            res = await asyncio.to_thread(self.scheduler.evaluate_eod_signal, df_live, t)
                            computed_signals.append(res)
                        except Exception:
                            pass
                    signals_buy = [s for s in computed_signals if s['signal_code'] == 1][:5]
                    signals_wait = [s for s in computed_signals if s['signal_code'] == 0][:5]
                    signals_sell = [s for s in computed_signals if s['signal_code'] == -1][:5]

            start_date = datetime.now().strftime("%d.%m.%Y")
            end_date = (datetime.now() + timedelta(days=30)).strftime("%d.%m.%Y")

            msg = (
                f"🏆 <b>YAPAY ZEKÂ BIST100 DETAYLI TARAMA RAPORU</b>\n"
                f"📅 <b>HEDEF TARİH ARALIĞI:</b> <b>{start_date} ➔ {end_date}</b> (20 İşlem Günü)\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n\n"
            )

            # 🟢 5 AL HİSSESİ
            msg += "🟢 <b>ALIM FIRSATI OLAN HİSSELER (TOP-5 AL):</b>\n"
            if signals_buy:
                for idx, sig in enumerate(signals_buy, 1):
                    cur_p = sig.get('current_price', 0.0)
                    t_low = sig.get('target_low') or sig.get('target_price_low') or (cur_p * 1.075)
                    pct = ((t_low / cur_p) - 1.0) * 100 if cur_p > 0 else 0.0
                    msg += f"{idx}. <b>{html.escape(sig['ticker'])}</b> | Fiyat: {cur_p:.2f} TL ➔ Hedef: <b>{t_low:.2f} TL (+%{pct:.1f})</b> | Güven: %{sig['p_success']*100:.1f}\n"
            else:
                msg += "<i>Şu an %52 güven eşiğini geçen AL sinyalli hisse yok. (Sermaye Koruma Modu)</i>\n"

            msg += "\n🟡 <b>NÖTR / POZİSYONU KORU HİSSELERİ (TOP-5 BEKLE):</b>\n"
            if signals_wait:
                for idx, sig in enumerate(signals_wait, 1):
                    cur_p = sig.get('current_price', 0.0)
                    msg += f"{idx}. <b>{html.escape(sig['ticker'])}</b> | Fiyat: {cur_p:.2f} TL | Güven: %{sig['p_success']*100:.1f}\n"
            else:
                msg += "<i>Nötr pozisyonda hisse bulunmuyor.</i>\n"

            msg += "\n🔴 <b>SAT / NAKİTE GEÇ HİSSELERİ (TOP-5 SAT):</b>\n"
            if signals_sell:
                for idx, sig in enumerate(signals_sell, 1):
                    cur_p = sig.get('current_price', 0.0)
                    s_loss = sig.get('stop_loss') or sig.get('stop_loss_price') or (cur_p * 0.95)
                    pct_risk = abs(((cur_p - s_loss) / cur_p) * 100) if cur_p > 0 else 0.0
                    msg += f"{idx}. <b>{html.escape(sig['ticker'])}</b> | Fiyat: {cur_p:.2f} TL ➔ Destek Risk: <b>{s_loss:.2f} TL (-%{pct_risk:.1f})</b> | Güven: %{sig['p_success']*100:.1f}\n"
            else:
                msg += "<i>Acil sat sinyali veren hisse bulunmuyor.</i>\n"

            msg += f"\n━━━━━━━━━━━━━━━━━━━━━\n🕒 <i>Sürüm 13.1 DSS | 7/24 Kesintisiz Otonom Borsa Asistanın</i>"

            await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

        except Exception as e:
            logger.error(f"Scan hatası: {e}")
            await update.message.reply_text(f"⚠️ Tarama sırasında hata oluştu: {e}")

    async def _cmd_hisse(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Spesifik bir hisse için detaylı fırsat kartı getirir."""
        if not context.args:
            await update.message.reply_text("⚠️ Lütfen sorgulamak istediğiniz hisse adını yazınız. Örnek: `/hisse THYAO`", parse_mode=ParseMode.HTML)
            return

        raw_ticker = context.args[0].strip().upper()

        if raw_ticker not in BIST100_VALID_TICKERS and len(raw_ticker) > 6:
            await update.message.reply_text(f"❌ <b>'{raw_ticker}'</b> adında geçerli bir BIST100 hissesi bulunamadı.", parse_mode=ParseMode.HTML)
            return

        ticker = raw_ticker
        self.db_vault.add_to_watchlist(ticker)

        signal = await asyncio.to_thread(self.db_vault.get_last_signal, ticker)

        if not signal and self.scheduler:
            retries = 0
            while self.scheduler.primary_m is None and retries < 15:
                await asyncio.sleep(1)
                retries += 1

            if self.scheduler.primary_m:
                try:
                    df_live = pd.read_parquet("data/bist_2016_2026_adjusted.parquet")
                    signal = await asyncio.to_thread(self.scheduler.evaluate_eod_signal, df_live, ticker)
                except ValueError as ve:
                    await update.message.reply_text(f"🛡️ <b>{ticker}</b> henüz 60 işlem gününü doldurmamış yeni bir halka arz şirketidir. Risk koruması nedeniyle kumar oynamamak için analize alınmamaktadır.", parse_mode=ParseMode.HTML)
                    return
                except Exception as e:
                    logger.error(f"Canlı hisse hesaplama hatası ({ticker}): {e}")

        if not signal:
            await update.message.reply_text(f"❌ <b>{ticker}</b> hissesi bulunamadı veya henüz 60 günlük verisi birikmedi.", parse_mode=ParseMode.HTML)
            return

        card_html = self._format_opportunity_card(signal)
        await update.message.reply_text(card_html, parse_mode=ParseMode.HTML)

    async def _cmd_takip(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Kullanıcının özel takip listesini ve canlı durumlarını sunar."""
        watchlist = await asyncio.to_thread(self.db_vault.get_watchlist)

        if not watchlist:
            await update.message.reply_text(
                "📋 <b>Özel Takip Listeniz Henüz Boş.</b>\n\n"
                "Hisse eklemek için: `/ekle THYAO`\n"
                "Hisse sormak için: `/hisse GARAN`",
                parse_mode=ParseMode.HTML
            )
            return

        msg = "📋 <b>SİZİN ÖZEL TAKİP LİSTENİZ VE CANLI SAĞLIK DURUMU</b>\n━━━━━━━━━━━━━━━━━━━━━\n\n"
        for t in watchlist:
            sig = await asyncio.to_thread(self.db_vault.get_last_signal, t)
            if sig:
                sig_code = sig.get('signal_code', 0)
                cur_p = sig.get('current_price', 0.0)
                durum = "🟢 AL" if sig_code == 1 else ("🔴 SAT" if sig_code == -1 else "🟡 BEKLE")
                msg += f"• <b>{t}</b> | Fiyat: {cur_p:.2f} TL | Durum: <b>{durum}</b> (Güven: %{sig.get('p_success',0.0)*100:.1f})\n"
            else:
                msg += f"• <b>{t}</b> | Veri Bekleniyor...\n"

        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def _cmd_ekle(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await update.message.reply_text("⚠️ Eklemek istediğiniz hisse adını yazınız. Örnek: `/ekle THYAO`", parse_mode=ParseMode.HTML)
            return

        raw_ticker = context.args[0].strip().upper()

        if raw_ticker not in BIST100_VALID_TICKERS and len(raw_ticker) > 6:
            await update.message.reply_text(f"❌ <b>'{raw_ticker}'</b> adında geçerli bir BIST100 hissesi bulunamadı.", parse_mode=ParseMode.HTML)
            return

        ticker = raw_ticker
        added = await asyncio.to_thread(self.db_vault.add_to_watchlist, ticker)

        if added:
            await update.message.reply_text(f"✅ <b>{ticker}</b> başarıyla özel takip listenize eklendi!\n📢 <i>Artık kararları değiştiğinde robot cebinize mesaj atacaktır.</i>", parse_mode=ParseMode.HTML)
        else:
            await update.message.reply_text(f"ℹ️ <b>{ticker}</b> zaten takip listenizde mevcut.", parse_mode=ParseMode.HTML)

    async def _cmd_cikar(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await update.message.reply_text("⚠️ Çıkarmak istediğiniz hisse adını yazınız. Örnek: `/cikar THYAO`", parse_mode=ParseMode.HTML)
            return

        ticker = context.args[0].strip().upper()
        removed = await asyncio.to_thread(self.db_vault.remove_from_watchlist, ticker)

        if removed:
            await update.message.reply_text(f"🗑️ <b>{ticker}</b> özel takip listenizden çıkarıldı.", parse_mode=ParseMode.HTML)
        else:
            await update.message.reply_text(f"ℹ️ <b>{ticker}</b> zaten takip listenizde bulunmuyor.", parse_mode=ParseMode.HTML)

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
