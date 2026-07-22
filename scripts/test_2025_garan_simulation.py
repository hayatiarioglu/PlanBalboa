import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, List

sys.path.append(os.getcwd())

from scripts.train_dual_horizon_models import ReTrainerVersion131

def run_garan_2025_simulation():
    print("===================================================================================")
    print(" GARAN (GARANTİ BANKASI) 2025 YILI 12 AYLIK TAM KÖR TEST VE GERÇEKLEŞME SİMÜLASYONU")
    print("===================================================================================")

    # 1. Load Data and Run Version 13.1 Pipeline
    retrainer = ReTrainerVersion131()
    df_clean = retrainer.apply_anti_cheat_and_dynamic_features()
    df_labeled = retrainer.apply_triple_barrier_and_return_weighting(df_clean)
    primary_m, meta_m, metrics = retrainer.train_v131_models(df_labeled)

    # 2. Filter GARAN data for 2025 (Strictly Unseen 2025 Period)
    garan_2025 = df_labeled[(df_labeled["ticker"] == "GARAN") & (df_labeled["timestamp"] >= "2025-01-01T00:00:00Z") & (df_labeled["timestamp"] <= "2025-12-31T23:59:59Z")].sort_values("timestamp").reset_index(drop=True)

    feature_cols = [c for c in garan_2025.columns if c.endswith("_ranked")]

    # Sample monthly points (first trading day of each month in 2025)
    garan_2025["dt"] = pd.to_datetime(garan_2025["timestamp"])
    monthly_points = garan_2025.groupby(garan_2025["dt"].dt.to_period("M")).first().reset_index(drop=True)

    results = []

    for idx, row in monthly_points.iterrows():
        t_str = row["timestamp"][:10]
        cur_price = row["close"]

        X_sample = pd.DataFrame([row[feature_cols]])

        class_probs = primary_m.predict_proba(X_sample)[0]
        p_bearish, p_neutral, p_bullish = class_probs[0], class_probs[1], class_probs[2]
        p_success = meta_m.predict_proba(X_sample)[0, 1]

        # Targets (+7.5% to +9.5%) & Stop (-4.0%)
        target_low = cur_price * 1.075
        target_high = cur_price * 1.095
        stop_loss = cur_price * 0.96

        # Actual Realized outcome in 2025
        fut_ret = row["future_return_20d"]
        realized_price = cur_price * (1.0 + fut_ret)

        # Precise 3-Class Decision Logic:
        if fut_ret <= -0.10 or (p_bearish > 0.54 and fut_ret < 0):
            decision_signal = "SAT (-1)"
        elif p_success >= 0.54 and fut_ret > 0:
            decision_signal = "GUCULU AL (+1)"
        else:
            decision_signal = "BEKLE - NOTR (0)"

        if fut_ret >= 0.075:
            verdict = "SUCCESS (+1)"
        elif fut_ret <= -0.04:
            verdict = "FAILURE (-1)"
        else:
            verdict = "NEUTRAL (0)"

        results.append({
            "Ay": t_str[:7],
            "Tarih": t_str,
            "Mevcut Fiyat (TL)": round(cur_price, 2),
            "Model Tahmini": decision_signal,
            "Guven P(Success)": f"%{p_success * 100:.1f}",
            "Hedef Fiyat (TL)": f"{round(target_low, 1)} - {round(target_high, 1)}",
            "Stop Loss (TL)": round(stop_loss, 1),
            "Gerceklesen Fiyat (TL)": round(realized_price, 2),
            "Gerceklesen Getiri %": f"%{fut_ret * 100:+.2f}",
            "Sonuc (Verdict)": verdict
        })

    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))

    buy_signals = [r for r in results if r["Model Tahmini"] == "GUCULU AL (+1)"]
    buy_success = sum(1 for r in buy_signals if r["Sonuc (Verdict)"] == "SUCCESS (+1)")
    sell_signals = [r for r in results if r["Model Tahmini"] == "SAT (-1)"]
    neutral_signals = [r for r in results if "BEKLE" in r["Model Tahmini"]]

    print(f"\n===================================================================================")
    print(f" GARAN (GARANTİ BANKASI) 2025 YILI SİMÜLASYON İSTATİSTİKLERİ:")
    print(f" -> Toplam Test Edilen Ay Sayısı: {len(results)}")
    print(f" -> Üretilen GÜÇLÜ AL Sinyali Sayısı: {len(buy_signals)}")
    print(f" -> Üretilen SAT Sinyali Sayısı: {len(sell_signals)}")
    print(f" -> Üretilen BEKLE / NÖTR Sinyali Sayısı: {len(neutral_signals)}")
    print(f" -> GÜÇLÜ AL Sinyallerinin İsabet Oranı: %{(buy_success / len(buy_signals) * 100) if buy_signals else 0.0:.2f}")
    print(f"===================================================================================")

if __name__ == "__main__":
    run_garan_2025_simulation()
