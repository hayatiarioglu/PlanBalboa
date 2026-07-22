import numpy as np
import pandas as pd
import torch
from typing import Tuple, List, Dict
from aether.data.fetcher import BISTDataFetcher
from aether.data.horizon_factors import HorizonFactorEngine

class DailyRollingFetcher:
    """
    3,276+ Satırlı Günlük Kaydırmalı Veri Motoru (Daily Rolling Window Data Fetcher).
    
    Zırhlar:
    1. TEFAS +1 gün Forward Pricing Execution Lag.
    2. Risk-Adjusted Target: Y_risk_adj = Y_21d / (std_21d + 1e-5).
    3. safe_cross_sectional_zscore + Winsorization [-3.0, +3.0].
    4. Düzeltilmiş Kapanış (Adjusted Close) Koruması.
    """

    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or BISTDataFetcher.DEFAULT_BIST_SYMBOLS
        self.fetcher = BISTDataFetcher(symbols=self.symbols)

    def safe_asset_class_zscore(self, df_day: pd.DataFrame, assets: List[str], factor_cols: List[str]) -> pd.DataFrame:
        """
        Varlık Sınıfı Bazlı Z-Score Normalizasyonu (Asset-Class Specific Z-Score).
        Fonlar kendi arasında, Hisseler kendi arasında Z-Score + Winsorization [-3.0, +3.0]'a tabi tutulur.
        """
        df_res = df_day.copy()
        
        fund_indices = [i for i, a in enumerate(assets) if not a.endswith(".IS")]
        stock_indices = [i for i, a in enumerate(assets) if a.endswith(".IS")]
        
        for idx_list in [fund_indices, stock_indices]:
            if len(idx_list) > 1:
                for col in factor_cols:
                    vals = df_res.iloc[idx_list][col].values
                    std = np.std(vals)
                    mean = np.mean(vals)
                    if std > 1e-7:
                        z = (vals - mean) / std
                        df_res.iloc[idx_list, df_res.columns.get_loc(col)] = np.clip(z, -3.0, 3.0)
                    else:
                        df_res.iloc[idx_list, df_res.columns.get_loc(col)] = 0.0
                        
        return df_res

    def build_rolling_dataset(self, start_date: str = "2012-01-01", end_date: str = "2024-12-31") -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
        print(f"[DATA] 3,276+ Satırlı Günlük Kaydırmalı Veri Seti İnşa Ediliyor ({start_date} -> {end_date})...")
        
        # 1. Ham Düzeltilmiş Fiyatları Çek (Adjusted Close)
        df_bist = self.fetcher.fetch_historical_prices(start_date, end_date)
        from aether.data.fetcher import TEFASDataFetcher
        tefas_f = TEFASDataFetcher()
        df_tefas = tefas_f.fetch_fon_nav_history(start_date, end_date)

        if isinstance(df_bist.columns, pd.MultiIndex):
            df_bist_close = df_bist["Adj Close"] if "Adj Close" in df_bist.columns else df_bist["Close"]
        else:
            df_bist_close = df_bist

        # BIST ve TEFAS Günlük Birlestir & Duplicate Temizle
        df_daily_prices = pd.concat([df_bist_close, df_tefas], axis=1).ffill().bfill().dropna(how="all")
        df_daily_prices = df_daily_prices.loc[:, ~df_daily_prices.columns.duplicated()]
        df_daily_prices = df_daily_prices[~df_daily_prices.index.duplicated(keep='first')]
        
        assets = list(df_daily_prices.columns)
        n_assets = len(assets)
        
        # 2. Ufuk Uyumlu Faktörleri Hesapla
        factor_dict = HorizonFactorEngine.compute_horizon_factors(df_daily_prices)
        factor_cols = list(factor_dict.keys())
        
        # 3. 21-Günlük Rolling Volatilite (Risk-Adjusted Target için)
        daily_rets = df_daily_prices.pct_change(1, fill_method=None).fillna(0.0)
        vol_21d = daily_rets.rolling(21).std().fillna(0.01) + 1e-5
        
        # 4. Target Hesaplama (BIST T+0, TEFAS T+1 Execution Lag)
        targets_df = pd.DataFrame(index=df_daily_prices.index, columns=assets, dtype=np.float32)
        
        for col in assets:
            is_tefas = not col.endswith(".IS")
            if is_tefas:
                # TEFAS Fonu: +1 gün Forward Pricing Lag Shift (P_{t+22} - P_{t+1}) / P_{t+1}
                p_t1 = df_daily_prices[col].shift(-1)
                p_t22 = df_daily_prices[col].shift(-22)
                raw_target = (p_t22 - p_t1) / (p_t1 + 1e-5)
            else:
                # BIST Hissesi: T+0 Execution (P_{t+21} - P_t) / P_t
                p_t = df_daily_prices[col]
                p_t21 = df_daily_prices[col].shift(-21)
                raw_target = (p_t21 - p_t) / (p_t + 1e-5)
                
            # Risk-Adjusted Target: Y_risk_adj = Y_raw / (vol_21d + 1e-5)
            targets_df[col] = (raw_target / vol_21d[col]).fillna(0.0)
            
        # VARLIK SINIFI BAZLI TARGET NORMALİZASYONU (Numpy Matris Kesitsel Z-Score)
        fund_cols = [c for c in assets if not c.endswith(".IS")]
        stock_cols = [c for c in assets if c.endswith(".IS")]
        
        for grp_cols in [fund_cols, stock_cols]:
            sub_vals = targets_df[grp_cols].values.astype(np.float32)
            mean_row = np.mean(sub_vals, axis=1, keepdims=True)
            std_row = np.std(sub_vals, axis=1, keepdims=True)
            std_row[std_row < 1e-7] = 1.0
            z_vals = np.clip((sub_vals - mean_row) / std_row, -3.0, 3.0)
            targets_df[grp_cols] = z_vals




        # 5. Günlük Kaydırmalı Snapshot Matrisi İnşa Et (3,276 Satır)
        X_list = []
        Y_list = []
        dates_list = []
        
        # 63 gün warmup (VWAP & rolling std için)
        for i in range(63, len(df_daily_prices) - 22):
            t_date = df_daily_prices.index[i]
            
            # O günkü varlıkların faktör matrisini çıkar
            day_features = []
            for col_name in factor_cols:
                vals = factor_dict[col_name].iloc[i].values
                day_features.append(vals)
                
            day_features_matrix = np.column_stack(day_features)
            
            # VARLIK SINIFI BAZLI ÖZNİTELİK Z-SCORE NORMALİZASYONU
            df_day = pd.DataFrame(day_features_matrix, columns=factor_cols)
            df_day_z = self.safe_asset_class_zscore(df_day, assets, factor_cols)
            
            x_tensor = torch.tensor(df_day_z.values, dtype=torch.float32)
            y_tensor = torch.tensor(targets_df.iloc[i].values, dtype=torch.float32)
            
            X_list.append(x_tensor)
            Y_list.append(y_tensor)
            dates_list.append(str(t_date.date()))
            
        X_all = torch.stack(X_list, dim=0)
        Y_all = torch.stack(Y_list, dim=0)
        
        print(f"[BASARILI] {X_all.shape[0]} Günlük Kaydırmalı Satır İnşa Edildi! (X: {X_all.shape}, Y: {Y_all.shape})")
        return X_all, Y_all, dates_list, assets

        
        print(f"[BASARILI] {X_all.shape[0]} Günlük Kaydırmalı Satır İnşa Edildi! (X: {X_all.shape}, Y: {Y_all.shape})")
        return X_all, Y_all, dates_list, assets
