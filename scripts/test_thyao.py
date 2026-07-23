import os
import sys
import joblib
import pandas as pd
import numpy as np

sys.path.append(os.getcwd())

def run_thyao_report():
    df = pd.read_parquet("data/bist_2016_2026_adjusted.parquet")
    primary_model = joblib.load("models/primary_model.joblib")
    meta_model = joblib.load("models/meta_model.joblib")

    df["event_timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

    df["rolling_sma20"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    df["rolling_std20"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(20, min_periods=5).std())
    df["z_score_price"] = (df["close"] - df["rolling_sma20"]) / np.maximum(1e-5, df["rolling_std20"])
    
    df["rolling_vol_sma20"] = df.groupby("ticker")["volume"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    df["rolling_vol_std20"] = df.groupby("ticker")["volume"].transform(lambda x: x.rolling(20, min_periods=5).std())
    df["z_score_volume"] = (df["volume"] - df["rolling_vol_sma20"]) / np.maximum(1e-5, df["rolling_vol_std20"])

    def calc_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=5).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=5).mean()
        rs = gain / np.maximum(1e-5, loss)
        return 100 - (100 / (1 + rs))

    df["rsi20"] = df.groupby("ticker")["close"].transform(lambda x: calc_rsi(x, 20))

    features_to_rank = ["pe_ratio", "pb_ratio", "ebitda_growth", "roe", "obi", "akd_top5_share", "net_money_flow_tl", "z_score_price", "z_score_volume", "rsi20"]
    for feat in features_to_rank:
        df[f"{feat}_ranked"] = df.groupby("timestamp")[feat].transform(lambda x: (x.rank(method="min") - 1.0) / max(1.0, (len(x) - 1.0)))

    feature_cols = [c for c in df.columns if c.endswith("_ranked")]

    thyao_df = df[(df["ticker"] == "THYAO") & (df["timestamp"] >= "2025-01-01") & (df["timestamp"] <= "2026-07-01")].copy()
    thyao_df["month"] = pd.to_datetime(thyao_df["timestamp"]).dt.to_period("M")

    months = sorted(thyao_df["month"].unique())
    
    print("\n" + "="*75)
    print("THYAO (TÜRK HAVA YOLLARI) 2025 - 2026 AY AY MODEL KANIT RAPORU")
    print("="*75 + "\n")

    correct_predictions = 0

    for m in months:
        m_df = thyao_df[thyao_df["month"] == m].sort_values("timestamp")
        start_row = m_df.iloc[0]
        t_str = str(start_row["timestamp"])[:10]
        
        x_sample = pd.DataFrame([start_row[feature_cols]])
        p_price = float(start_row["close"])

        curr_idx = df[(df["ticker"] == "THYAO") & (df["timestamp"] == start_row["timestamp"])].index[0]
        future_idx = min(curr_idx + 20, len(df) - 1)
        future_price = float(df.iloc[future_idx]["close"])
        actual_return = (future_price - p_price) / p_price * 100.0

        p_class = primary_model.predict_proba(x_sample)[0]
        p_bull = p_class[2]
        p_bear = p_class[0]

        p_raw = float(meta_model.predict_proba(x_sample)[0, 1])
        p_succ = float(0.15 + (p_raw - 0.28) / (0.81 - 0.28 + 1e-5) * 0.70)

        sig = "[GREEN / AL]" if (p_succ >= 0.52 and p_bull > p_bear) else ("[RED / SAT]" if (p_succ < 0.40 or p_bear > 0.45) else "[YELLOW / BEKLE]")
        
        target_pct = max(6.5, (p_succ - 0.15) * 0.20 * 100 + 5.0)
        target_price = p_price * (1 + target_pct / 100.0)

        is_success = (sig == "[GREEN / AL]" and actual_return > 0) or (sig == "[RED / SAT]" and actual_return < 0) or (sig == "[YELLOW / BEKLE]" and abs(actual_return) < 4.0)
        if is_success:
            correct_predictions += 1
        
        status_str = "SUCCESS / ISABETLI" if is_success else "MISMATCH / SAPMA"

        print(f"AY: {m}")
        print(f"   • Baslangic Tarihi  : {t_str} | Baslangic Fiyati: {p_price:.2f} TL")
        print(f"   • Model Sinyali     : {sig} (Guven: %{p_succ*100:.1f})")
        print(f"   • Tahmin Edilen     : Hedef Fiyat {target_price:.2f} TL (+%{target_pct:.1f})")
        print(f"   • 20 Gun Sonra      : Gerceklesen Fiyat {future_price:.2f} TL (Net Degisim: %{actual_return:+.1f})")
        print(f"   • SONUC             : {status_str}\n")

    acc_pct = (correct_predictions / len(months)) * 100.0
    print("="*75)
    print(f"2025 - 2026 THYAO TOPLAM DOGRULUK ORANI: %{acc_pct:.1f} ({len(months)} Ayin {correct_predictions} Ayinda Tam İsabet)")
    print("="*75)

if __name__ == "__main__":
    run_thyao_report()
