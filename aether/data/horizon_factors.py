import numpy as np
import pandas as pd

class HorizonFactorEngine:
    """
    1-Aylık Vadeli Ufuk Uyumlu Faktör Hesaplayıcı (Horizon-Aligned Quantitative Alpha Engine).
    
    1. RS_Sector: Göreceli Sektör İvmesi (Asset 21d Return / Sector Median 21d Return)
    2. Sharpe_Momentum: Volatilite Ayarlı Getiri (21d Mean Return / 21d Std Dev * sqrt(252))
    3. VWAP_Ratio: Hacimli Trend Gücü (21d VWAP / 63d VWAP)
    4. Macro_Spread: BIST100 21d Momentum / Benchmark Rates
    """

    @staticmethod
    def compute_horizon_factors(df_prices: pd.DataFrame, df_volumes: pd.DataFrame = None) -> pd.DataFrame:
        # 1. 21-Günlük (1-Aylık) Getiriler
        ret_21d = df_prices.pct_change(21, fill_method=None).fillna(0.0)
        
        # 2. Göreceli Sektör / Piyasa İvmesi (RS_Sector)
        sector_median = ret_21d.median(axis=1)
        rs_sector = ret_21d.sub(sector_median, axis=0)
        
        # 3. Sharpe Momentum (21d Rolling Sharpe Ratio)
        rolling_mean = df_prices.pct_change(1, fill_method=None).rolling(21).mean().fillna(0.0)
        rolling_std = df_prices.pct_change(1, fill_method=None).rolling(21).std().fillna(0.01) + 1e-5
        sharpe_momentum = (rolling_mean / rolling_std) * np.sqrt(252)
        
        # 4. VWAP Ratio (21d VWAP / 63d VWAP)
        if df_volumes is not None and not df_volumes.empty:
            pv = df_prices * df_volumes
            vwap_21 = pv.rolling(21).sum() / (df_volumes.rolling(21).sum() + 1e-5)
            vwap_63 = pv.rolling(63).sum() / (df_volumes.rolling(63).sum() + 1e-5)
            vwap_ratio = (vwap_21 / vwap_63) - 1.0
        else:
            p_ma21 = df_prices.rolling(21).mean()
            p_ma63 = df_prices.rolling(63).mean()
            vwap_ratio = (p_ma21 / p_ma63) - 1.0
            
        # 5. Macro Spread (Piyasa Genel Trend Gücü)
        mkt_trend = sector_median.to_frame(name="mkt_trend")
        macro_spread = pd.DataFrame(np.repeat(mkt_trend.values, df_prices.shape[1], axis=1), 
                                    index=df_prices.index, columns=df_prices.columns)
        
        # Faktörleri Sözlük Olarak Topla
        factors = {
            "ret_21d": ret_21d.fillna(0.0),
            "rs_sector": rs_sector.fillna(0.0),
            "sharpe_momentum": sharpe_momentum.fillna(0.0),
            "vwap_ratio": vwap_ratio.fillna(0.0),
            "macro_spread": macro_spread.fillna(0.0)
        }
        return factors
