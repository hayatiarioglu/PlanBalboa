import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())

def verify_garan_jan_2026_outcome():
    print("===================================================================================")
    print(" GARAN 2026 OCAK TAHMINI GERCEKLESME DENETIMI (STRICT SINGLE-MONTH VERIFICATION)")
    print("===================================================================================")

    df = pd.read_parquet("data/bist_2016_2026_adjusted.parquet")

    # Fetch GARAN data for Jan 2026 -> Feb 2026 ONLY
    garan = df[(df["ticker"] == "GARAN") & (df["timestamp"] >= "2026-01-01T00:00:00Z") & (df["timestamp"] <= "2026-02-05T23:59:59Z")].sort_values("timestamp").reset_index(drop=True)

    if garan.empty:
        print("HATA: 2026 Ocak verisi bulunamadı.")
        return

    entry_row = garan.iloc[0]
    entry_price = entry_row["close"]
    entry_date = entry_row["timestamp"][:10]

    # Target Range: 2.72 TL - 2.77 TL
    target_low = entry_price * 1.075
    target_high = entry_price * 1.095
    stop_loss = entry_price * 0.96

    # Outcome at T+20 trading days (~20th row or end of Jan)
    outcome_row = garan.iloc[min(20, len(garan) - 1)]
    realized_price = outcome_row["close"]
    realized_date = outcome_row["timestamp"][:10]
    realized_return = (realized_price / entry_price) - 1.0

    if realized_return >= 0.075:
        verdict = "REALIZED_SUCCESS (+1) --- HEDEF KAR BOLGESINE TAM ULASILDI!"
    elif realized_return <= -0.04:
        verdict = "REALIZED_FAILURE (-1) --- STOP LOSS TETIKLENDI"
    else:
        verdict = "REALIZED_NEUTRAL (0) --- YATAY / BEKLEME"

    print(f"\n===================================================================================")
    print(f" GARAN 2026 OCAK GERCEKLESME DENETIM RAPORU")
    print(f" -> Islem Giris Tarihi: {entry_date}")
    print(f" -> Giris Fiyati: {round(entry_price, 2)} TL")
    print(f" -> Uretilen Model Tahmini: GUCULU AL (+1)")
    print(f" -> Beklenen Hedef Fiyat: {round(target_low, 2)} TL - {round(target_high, 2)} TL (+%7.5 ... +%9.5)")
    print(f" -> Stop Loss Seviyesi: {round(stop_loss, 2)} TL (-%4.0)")
    print(f" -----------------------------------------------------------------------------------")
    print(f" -> Gerceklesme Kontrol Tarihi: {realized_date}")
    print(f" -> Gerceklesen Borsa Fiyati: {round(realized_price, 2)} TL")
    print(f" -> Gerceklesen Getiri Orani: %{realized_return * 100:+.2f}")
    print(f" -> DENETIM SONUCU (VERDICT): {verdict}")
    print(f"===================================================================================")

if __name__ == "__main__":
    verify_garan_jan_2026_outcome()
