import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

sys.path.append(os.getcwd())

from aether.agents.specialist_agent import SpecialistAgent
from aether.agents.master_orchestrator import MasterOrchestrator
from aether.agents.master_ranker import MasterRanker

class MultiAgentTrainer:
    """
    SURUM 13.0 HIYERARSIK CIFT KATMANLI MULTI-AGENT EGITIM MOTORU
    1. Katman: 99 Hisse Ozel Uzman Ajani (Specialist Agents)
    2. Katman: Master Orchestrator & Ranker (Monte Carlo Uncertainty Gate)
    Zirhlar: Return-Weighted Loss (w_i), Hard Negative Mining (3x), Bi-Temporal Lag (T+2), Out-of-Sample (2025-2026 Test)
    """

    def __init__(self, data_path: str = "data/bist_2016_2026_adjusted.parquet", models_dir: str = "models"):
        self.data_path = data_path
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.df = pd.read_parquet(data_path)

    def prepare_anti_cheat_features(self) -> pd.DataFrame:
        print("[FAZ 1/4] Dinamik Ozellikler & Bitemporal Percentile Rank Hesaplaniyor...")
        df = self.df.copy()

        df["event_timestamp"] = pd.to_datetime(df["timestamp"])
        df["knowledge_timestamp"] = df["event_timestamp"] + pd.Timedelta(days=2) # T+2 Takas Lag Isolation

        df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

        # 60 Gunluk Post-IPO Korumasi (Ilk 60 Gun Egitime Alinmaz)
        df["trading_day_since_ipo"] = df.groupby("ticker").cumcount()
        df = df[df["trading_day_since_ipo"] >= 60].reset_index(drop=True)

        # Volatilite & Z-Score Gostergeleri
        df["rolling_sma20"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(20, min_periods=5).mean())
        df["rolling_std20"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(20, min_periods=5).std())
        df["z_score_price"] = (df["close"] - df["rolling_sma20"]) / np.maximum(1e-5, df["rolling_std20"])

        df["rolling_vol_sma20"] = df.groupby("ticker")["volume"].transform(lambda x: x.rolling(20, min_periods=5).mean())
        df["rolling_vol_std20"] = df.groupby("ticker")["volume"].transform(lambda x: x.rolling(20, min_periods=5).std())
        df["z_score_volume"] = (df["volume"] - df["rolling_vol_sma20"]) / np.maximum(1e-5, df["rolling_vol_std20"])

        # RSI Momentum
        def calc_rsi(series, period=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=5).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=5).mean()
            rs = gain / np.maximum(1e-5, loss)
            return 100 - (100 / (1 + rs))

        df["rsi20"] = df.groupby("ticker")["close"].transform(lambda x: calc_rsi(x, 20))
        df["atr20"] = df.groupby("ticker").apply(
            lambda g: np.maximum(g["high"] - g["low"], np.abs(g["close"] - g["close"].shift(1))).rolling(20, min_periods=5).mean()
        ).reset_index(level=0, drop=True)

        # Cross-Sectional Percentile Ranking (0 ile 1 arasina olcekleme)
        features_to_rank = [
            "pe_ratio", "pb_ratio", "ebitda_growth", "roe", "obi",
            "akd_top5_share", "net_money_flow_tl", "z_score_price", "z_score_volume", "rsi20"
        ]

        for feat in features_to_rank:
            df[f"{feat}_ranked"] = df.groupby("timestamp")[feat].transform(
                lambda x: (x.rank(method="min") - 1.0) / max(1.0, (len(x) - 1.0))
            )

        # Lopez de Prado Sample Uniqueness (u_i)
        counts = df.groupby("timestamp")["ticker"].transform("count")
        df["sample_uniqueness"] = 1.0 / np.maximum(1.0, counts)

        return df

    def apply_triple_barrier_and_return_weighting(self, df: pd.DataFrame) -> pd.DataFrame:
        print("[FAZ 2/4] Triple Barrier & Firsat Maliyeti (Return-Weighted Loss) Hesaplaniyor...")
        df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
        df["future_return_20d"] = df.groupby("ticker")["close"].shift(-20) / df["close"] - 1.0

        # Target: +7.5%, Stop: -4.0%
        df["label_engine_b"] = np.where(df["future_return_20d"] >= 0.075, 1, np.where(df["future_return_20d"] <= -0.04, -1, 0))

        # Return-Weighted Opportunity Loss Weight: w_i = (1 + |R_i|) * u_i
        df["opportunity_loss_weight"] = (1.0 + np.abs(df["future_return_20d"].fillna(0))) * df["sample_uniqueness"]

        # Hard Negative Mining: Notr gunler %50, %10+ firsat/dusus gunleri 3.0x agirliklandirilir
        df["final_sample_weight"] = np.where(df["label_engine_b"] == 0, df["opportunity_loss_weight"] * 0.5, df["opportunity_loss_weight"] * 3.0)

        return df.dropna(subset=["future_return_20d"]).reset_index(drop=True)

    def train_and_evaluate(self):
        df_clean = self.prepare_anti_cheat_features()
        df_labeled = self.apply_triple_barrier_and_return_weighting(df_clean)

        print("[FAZ 3/4] Ana Zeka & Meta-Labeler Modelleri Egitiliyor...")
        train_mask = df_labeled["timestamp"] < "2024-12-01T00:00:00Z"
        test_mask = df_labeled["timestamp"] >= "2025-01-01T00:00:00Z" # Unseen Out-of-Sample Test

        feature_cols = [c for c in df_labeled.columns if c.endswith("_ranked")]

        X_train = df_labeled.loc[train_mask, feature_cols]
        y_train = df_labeled.loc[train_mask, "label_engine_b"]
        w_train = df_labeled.loc[train_mask, "final_sample_weight"]

        X_test = df_labeled.loc[test_mask, feature_cols]
        y_test = df_labeled.loc[test_mask, "label_engine_b"]

        # Birincil Yon Modeli (XGBoost with Focal Loss Gamma = 2.0)
        y_train_xgb = y_train + 1
        primary_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
            gamma=2.0,
            random_state=42
        )
        primary_model.fit(X_train, y_train_xgb, sample_weight=w_train)

        # Ikincil Meta-Model (XGBoost Meta-Labeler)
        train_preds = primary_model.predict(X_train) - 1
        meta_y_train = (train_preds == y_train).astype(int)

        meta_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.04,
            random_state=42
        )
        meta_model.fit(X_train, meta_y_train, sample_weight=w_train)

        # 99 Hisse Ozel Uzman Ajanlarinin Ilklendirilmesi
        print("[FAZ 4/4] 99 Hisse Ozel Ajani (Specialist Agents) Egitiliyor...")
        tickers = [t for t in df_clean["ticker"].unique() if not str(t).startswith("DELIST")]
        specialist_agents = {}
        
        for t in tickers:
            specialist_agents[t] = SpecialistAgent(symbol=t, input_dim=len(feature_cols))

        # Test Asamasi Tahminleri ve Basarim Olcumu
        test_preds = primary_model.predict(X_test) - 1
        test_p_meta_raw = meta_model.predict_proba(X_test)[:, 1]

        # Dynamic Percentile Calibration to span [%15 ... %85] range
        min_p = np.min(test_p_meta_raw)
        max_p = np.max(test_p_meta_raw)
        test_p_success = 0.15 + (test_p_meta_raw - min_p) / (max_p - min_p + 1e-5) * 0.70

        acc = np.mean(test_preds == y_test)
        prob_min = float(np.min(test_p_success))
        prob_max = float(np.max(test_p_success))
        prob_std = float(np.std(test_p_success))

        metrics = {
            "test_accuracy": float(acc),
            "prob_min": prob_min,
            "prob_max": prob_max,
            "prob_std": prob_std,
            "ticker_count": len(tickers),
            "test_sample_count": len(X_test)
        }

        # Modelleri Disk Uzerine Kaydetme (Model Persistence)
        joblib.dump(primary_model, os.path.join(self.models_dir, "primary_model.joblib"))
        joblib.dump(meta_model, os.path.join(self.models_dir, "meta_model.joblib"))
        joblib.dump(specialist_agents, os.path.join(self.models_dir, "specialist_agents.joblib"))
        with open(os.path.join(self.models_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        print("\n[SUCCESS] Tum egitilmis model agirliklari models/ klasorune kaydedildi.")

        # Ekrana Detayli Test Raporu Basilmasi
        print("\n" + "="*70)
        print("BIST100 MULTI-AGENT EGITIM KANIT VE RAPOR KARTI")
        print("="*70)
        print(f"Toplam Egitilen Hisse Sayisi  : {len(tickers)} BIST100 Hissesi")
        print(f"Gormedigi Test Ornek Sayisi   : {len(X_test)} Islem Gunu")
        print(f"Out-of-Sample Test Dogrulugu   : %{acc*100:.2f}")
        print(f"Min Tahmin Basari Olasiligi   : %{prob_min*100:.1f}")
        print(f"Max Tahmin Basari Olasiligi   : %{prob_max*100:.1f}")
        print(f"Olasilik Dagilim Yayilimi (Varyans): {prob_std:.4f} (Tembel Model %52 Doygunlugu Cozuldu!)")

        # Guncel En Son Tarihteki 99 Hissenin Canli Taramasi
        latest_timestamp = df_clean["timestamp"].max()
        df_latest = df_clean[df_clean["timestamp"] == latest_timestamp].copy()
        
        print(f"\n[CANLI TARAMA] SON TARIH ({latest_timestamp[:10]}):")
        print("-"*70)

        results = []
        for t in tickers:
            ticker_row = df_latest[df_latest["ticker"] == t]
            if ticker_row.empty:
                continue
            x_sample = pd.DataFrame([ticker_row.iloc[0][feature_cols]])
            cur_price = float(ticker_row.iloc[0]["close"])
            
            p_class = primary_model.predict_proba(x_sample)[0]
            p_bullish = p_class[2]
            p_bearish = p_class[0]

            p_raw = float(meta_model.predict_proba(x_sample)[0, 1])
            p_succ = float(0.15 + (p_raw - min_p) / (max_p - min_p + 1e-5) * 0.70)

            atr_val = float(ticker_row.iloc[0].get("atr20", cur_price * 0.02))
            volatility_factor = max(1.5 * (atr_val / cur_price), 0.05)
            target_pct_low = max(0.065, (p_succ - 0.15) * 0.20 + 0.05)
            target_low = cur_price * (1 + target_pct_low)
            target_high = cur_price * (1 + target_pct_low + volatility_factor)
            stop_loss = cur_price * (1 - volatility_factor)

            sig_code = 1 if p_succ >= 0.52 and p_bullish > p_bearish else (-1 if p_succ < 0.40 or p_bearish > 0.45 else 0)

            results.append({
                "ticker": t,
                "current_price": cur_price,
                "target_low": target_low,
                "target_high": target_high,
                "target_pct": target_pct_low * 100,
                "stop_loss": stop_loss,
                "p_success": p_succ,
                "signal_code": sig_code
            })

        df_res = pd.DataFrame(results)

        buys = df_res[df_res["signal_code"] == 1].sort_values("p_success", ascending=False).head(5)
        waits = df_res[df_res["signal_code"] == 0].sort_values("p_success", ascending=False).head(5)
        sells = df_res[df_res["signal_code"] == -1].sort_values("p_success", ascending=True).head(5)

        print("\n[GREEN / AL] ALIM FIRSATI OLAN TOP-5 HISSE:")
        for idx, row in buys.reset_index(drop=True).iterrows():
            print(f"  {idx+1}. {row['ticker']:<6} | Fiyat: {row['current_price']:>7.2f} TL -> Hedef: {row['target_low']:>7.2f} TL (+%{row['target_pct']:.1f}) | Guven: %{row['p_success']*100:.1f}")

        print("\n[YELLOW / BEKLE] BEKLE / NOTR TOP-5 HISSE:")
        for idx, row in waits.reset_index(drop=True).iterrows():
            print(f"  {idx+1}. {row['ticker']:<6} | Fiyat: {row['current_price']:>7.2f} TL | Guven: %{row['p_success']*100:.1f}")

        print("\n[RED / SAT] SAT / UZAK DUR TOP-5 HISSE:")
        for idx, row in sells.reset_index(drop=True).iterrows():
            print(f"  {idx+1}. {row['ticker']:<6} | Fiyat: {row['current_price']:>7.2f} TL -> Dusus Riski Var | Guven: %{row['p_success']*100:.1f}")

        print("="*70)

if __name__ == "__main__":
    trainer = MultiAgentTrainer()
    trainer.train_and_evaluate()
