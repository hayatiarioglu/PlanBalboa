import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

sys.path.append(os.getcwd())

from financial_ai.database_vault import DatabaseVault
from scripts.train_dual_horizon_models import ReTrainerVersion131

logger = logging.getLogger("BackgroundScheduler")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

class BackgroundScheduler:
    """
    SÜRÜM 13.1 DSS OTONOM ARKA PLAN ZAMANLAYICISI VE KARAR MOTORU.
    1. 10:15 Saf Matematiksel Fiyat Sentineli.
    2. 18:15 Gün Sonu 19-Modüllü EOD Analizi ve SHAP Sürücü Analizi.
    3. Cold-Start Fallback (P_{t-1} = P_t) ve Hysteresis Buffer.
    4. 9-Durumlu Eksiksiz Sinyal Konsolidasyon Matrisi.
    """

    CONSOLIDATION_MATRIX = {
        (1, 1): ("AGRESIF FIRSAT KARTI", "Tam Kapasite Boga Firsati"),
        (1, 0): ("MOMENTUM POZISYON", "Trend Guclu, Ek Alim Yapma"),
        (1, -1): ("TREND KORU / DUZELTME", "Pozisyonu Koru, Yeni Alim Yapma"),
        (0, 1): ("SCALPING FIRSATI", "5 Gunluk Kisa Vadeli Alim (Siki Stop)"),
        (0, 0): ("NOTR / NAKIT", "Islemsiz Bekle (Nakit)"),
        (0, -1): ("KISA VADELI BASKI", "Uzak Dur / Satis Baskisi Var"),
        (-1, 1): ("TEPKI YUKSELISI TUZAGI", "ALMA! Olu Kedi Sicramasi Riski"),
        (-1, 0): ("ZAYIF TREND / SAT", "Pozisyon Azalt / Cikis Arama"),
        (-1, -1): ("TAM NAKIT / DEVRE KESICI", "Tam Kapasite Cikis / Risk Yok")
    }

    def __init__(self, db_vault: DatabaseVault, data_path: str = "data/bist_2016_2026_adjusted.parquet"):
        self.db_vault = db_vault
        self.data_path = data_path
        self.retrainer = ReTrainerVersion131(data_path)
        self.primary_m, self.meta_m, self.metrics = None, None, None
        self.df_processed = None

    def initialize_models(self):
        """Modeli başlatır, ağırlıkları dondurur ve ilk açılış taramasını (Boot Sweep) yapar."""
        logger.info("[SCHEDULER] Sürüm 13.1 Modelleri Yükleniyor ve Eğitiliyor...")
        df_clean = self.retrainer.apply_anti_cheat_and_dynamic_features()
        df_labeled = self.retrainer.apply_triple_barrier_and_return_weighting(df_clean)
        self.primary_m, self.meta_m, self.metrics = self.retrainer.train_v131_models(df_labeled)
        self.df_processed = df_clean
        logger.info("[SCHEDULER] Modeller Eğitildi ve Kalibre Edildi.")

        # Açılış İlk Taraması (Boot Sweep) - Veritabanını Anında Doldurur
        logger.info("[SCHEDULER] Açılış Taraması (Boot Sweep) Başlatılıyor...")
        tickers = ["THYAO", "GARAN", "ASELS", "EREGL", "SISE", "BIMAS", "KCHOL", "ARCLK"]
        for t in tickers:
            try:
                self.evaluate_eod_signal(self.df_processed, t)
            except Exception as e:
                logger.error(f"[BOOT SWEEP ERROR] {t}: {e}")
        logger.info("[SCHEDULER] Açılış Taraması Tamamlandı. Veritabanı Hazır.")

    def evaluate_1015_gap_sentinel(self, df_live: pd.DataFrame, ticker: str) -> Optional[Dict[str, Any]]:
        """10:15 Saf Matematiksel Fiyat Bekçisi."""
        last_signal = self.db_vault.get_last_signal(ticker)
        if not last_signal:
            return None

        stop_loss_prev = last_signal["stop_loss_price"]
        ticker_df = df_live[df_live["ticker"] == ticker].sort_values("timestamp")
        if ticker_df.empty:
            return None

        latest_row = ticker_df.iloc[-1]
        p_1015 = latest_row["close"]
        prev_close = ticker_df.iloc[-2]["close"] if len(ticker_df) >= 2 else p_1015

        atr_20 = latest_row.get("atr20", p_1015 * 0.02)
        pct_change = (p_1015 - prev_close) / prev_close
        dynamic_stop_threshold = -max(1.8 * (atr_20 / prev_close), 0.04)

        if p_1015 <= stop_loss_prev or pct_change <= dynamic_stop_threshold:
            logger.warning(f"[10:15 GAP SENTINEL OVERRIDE] {ticker} - P_1015: {p_1015:.2f} TL (Stop: {stop_loss_prev:.2f} TL)")
            return {
                "ticker": ticker,
                "timestamp": latest_row["timestamp"],
                "override_type": "EMERGENCY_GAP_STOP",
                "price_1015": p_1015,
                "stop_loss_price": stop_loss_prev,
                "pct_change": pct_change,
                "reason": f"10:15 Anlık Fiyat ({p_1015:.2f} TL) Stop Loss ({stop_loss_prev:.2f} TL) Bariyerini Kırdı!"
            }
        return None

    def evaluate_eod_signal(self, df_live: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """18:15 EOD Analizi ve Sinyal Hesabı."""
        if self.df_processed is not None:
            df_processed = self.df_processed
        elif not any(c.endswith("_ranked") for c in df_live.columns):
            df_processed = self.retrainer.apply_anti_cheat_and_dynamic_features()
        else:
            df_processed = df_live

        ticker_df = df_processed[df_processed["ticker"] == ticker].sort_values("timestamp")
        if ticker_df.empty:
            raise ValueError(f"Ticker {ticker} dataset'te bulunamadı.")

        latest_row = ticker_df.iloc[-1]
        cur_price = latest_row["close"]

        feature_cols = [
            'pe_ratio_ranked', 'pb_ratio_ranked', 'ebitda_growth_ranked', 'roe_ranked', 
            'obi_ranked', 'akd_top5_share_ranked', 'net_money_flow_tl_ranked', 
            'z_score_price_ranked', 'z_score_volume_ranked'
        ]
        X_sample = pd.DataFrame([latest_row[feature_cols]])

        class_probs = self.primary_m.predict_proba(X_sample)[0]
        p_bearish, p_neutral, p_bullish = class_probs[0], class_probs[1], class_probs[2]
        p_success = self.meta_m.predict_proba(X_sample)[0, 1]

        last_signal = self.db_vault.get_last_signal(ticker)

        # Cold-Start Fallback
        p_success_prev = last_signal["p_success"] if last_signal and last_signal["p_success"] is not None else p_success
        last_state = last_signal["signal_code"] if last_signal and last_signal["signal_code"] is not None else 0

        target_low = cur_price * 1.075
        target_high = cur_price * 1.095
        stop_loss = cur_price * 0.96

        atr_20 = latest_row.get("atr20", cur_price * 0.02)
        prev_close = ticker_df.iloc[-2]["close"] if len(ticker_df) >= 2 else cur_price
        pct_change = (cur_price - prev_close) / prev_close
        dynamic_stop_threshold = -max(1.8 * (atr_20 / prev_close), 0.04)

        engine_a_sig = 1 if p_bullish > p_bearish and p_bullish >= 0.35 else (-1 if p_bearish > 0.45 else 0)
        engine_b_sig = 1 if p_success >= 0.54 else (-1 if p_success < 0.45 else 0)

        consolidated_signal_name, advisory = self.CONSOLIDATION_MATRIX.get((engine_b_sig, engine_a_sig), ("NOTR / NAKIT", "Islemsiz Bekle"))

        if pct_change <= dynamic_stop_threshold or p_success < 0.35 or (p_success < 0.45 and (last_signal is None or last_signal.get("days_held", 5) >= 5)):
            current_signal_code = -1 # SAT
            revision_reason = f"Acil Devre Kesici / Düsüs Trendi (P: %{p_success*100:.1f})"
        elif p_success >= 0.54 and p_success_prev >= 0.54:
            current_signal_code = 1 # AL
            revision_reason = f"2-Gunluk Teyitli Boga Trendi (P: %{p_success*100:.1f})"
        else:
            current_signal_code = last_state if last_state != 0 or last_signal is not None else (1 if p_success >= 0.54 else 0)
            revision_reason = "Hysteresis Bolgesi - Pozisyon Korundu"

        self.db_vault.execute_write_async(
            """
            INSERT INTO signals (ticker, signal_code, p_success, p_success_prev, stop_loss_price, target_price_low, target_price_high, engine_a_signal, engine_b_signal, revision_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ticker, current_signal_code, float(p_success), float(p_success_prev), float(stop_loss), float(target_low), float(target_high), int(engine_a_sig), int(engine_b_sig), revision_reason)
        )

        return {
            "ticker": ticker,
            "timestamp": latest_row["timestamp"],
            "current_price": cur_price,
            "signal_code": current_signal_code,
            "consolidated_name": consolidated_signal_name,
            "advisory": advisory,
            "p_success": p_success,
            "p_success_prev": p_success_prev,
            "target_low": target_low,
            "target_high": target_high,
            "stop_loss": stop_loss,
            "engine_a_signal": engine_a_sig,
            "engine_b_signal": engine_b_sig,
            "revision_reason": revision_reason
        }
