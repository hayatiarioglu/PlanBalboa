import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())

from scripts.train_dual_horizon_models import ReTrainerVersion131

def predict_tuprs_jan_2026():
    print("===================================================================================")
    print(" TUPRS (TÜPRAŞ) 2026 OCAK AYI OTONOM GELECEK TAHMİNİ (PURE FORWARD PREDICTION)")
    print("===================================================================================")

    # 1. Load Data and Run Version 13.1 Pipeline
    retrainer = ReTrainerVersion131()
    df_clean = retrainer.apply_anti_cheat_and_dynamic_features()
    df_labeled = retrainer.apply_triple_barrier_and_return_weighting(df_clean)
    primary_m, meta_m, metrics = retrainer.train_v131_models(df_labeled)

    # 2. Filter TUPRS data for 2026-01-01 (or first available day of Jan 2026)
    tuprs_jan_2026 = df_clean[(df_clean["ticker"] == "TUPRS") & (df_clean["timestamp"] >= "2026-01-01T00:00:00Z") & (df_clean["timestamp"] <= "2026-01-31T23:59:59Z")].sort_values("timestamp").reset_index(drop=True)

    if tuprs_jan_2026.empty:
        print("HATA: 2026 Ocak verisi bulunamadı.")
        return

    row = tuprs_jan_2026.iloc[0]
    t_str = row["timestamp"][:10]
    cur_price = row["close"]

    feature_cols = [c for c in tuprs_jan_2026.columns if c.endswith("_ranked")]
    X_sample = pd.DataFrame([row[feature_cols]])

    class_probs = primary_m.predict_proba(X_sample)[0]
    p_bearish, p_neutral, p_bullish = class_probs[0], class_probs[1], class_probs[2]
    p_success = meta_m.predict_proba(X_sample)[0, 1]

    # Targets (+7.5% to +9.5%) & Stop (-4.0%)
    target_low = cur_price * 1.075
    target_high = cur_price * 1.095
    stop_loss = cur_price * 0.96

    # Precise 3-Class Decision Logic:
    if p_bearish > 0.54:
        decision_signal = "SAT (-1)"
    elif p_success >= 0.54:
        decision_signal = "GUCULU AL (+1)"
    else:
        decision_signal = "BEKLE - NOTR (0)"

    print(f"\n===================================================================================")
    print(f" 🎯 TUPRS 2026 OCAK FİRSAT KARTI (OPPORTUNITY CARD)")
    print(f" -> Tahmin Tarihi: {t_str}")
    print(f" -> Mevcut Fiyat: {round(cur_price, 2)} TL")
    print(f" -> Model Tahmin Sinyali: {decision_signal}")
    print(f" -> Olasılık P(Success): %{p_success * 100:.1f}")
    print(f" -> Boğa P(+1) Olasılığı: %{p_bullish * 100:.1f}")
    print(f" -> Ayı P(-1) Olasılığı: %{p_bearish * 100:.1f}")
    print(f" -> Hedef Fiyat Aralığı: {round(target_low, 2)} TL - {round(target_high, 2)} TL (+%7.5 ... +%9.5)")
    print(f" -> Stop Loss Seviyesi: {round(stop_loss, 2)} TL (-%4.0)")
    print(f"===================================================================================")

if __name__ == "__main__":
    predict_tuprs_jan_2026()
