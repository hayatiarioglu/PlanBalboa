"""
AetherForecaster-X v35.2 Multi-Asset Master
Katman 1: Point-in-Time (PIT) Data Lake & Hybrid Risk Engine

Modül: Hybrid Volatility & TEFAS NAV Fallback Engine (Hibrit Volatilite ve Fon Yedekleme Motoru)
Faz 1.5 / Adım 1.5.1: Hisseler ve BYF'ler için EGARCH(1,1) + Parkinson volatilite tensörünün (V_t^hybrid) üretilmesi.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from aether.data.pit_snapshot import AssetType, DataPoint, PITDataSnapshot


@dataclass
class VolatilityResult:
    """
    Tek bir varlığın volatilite hesaplama sonuçları.
    """
    symbol: str
    asset_type: AssetType
    volatility: float             # Yıllıklandırılmış veya günlük hibrit volatilite (V_t^hybrid)
    parkinson_vol: float          # Parkinson High-Low volatilitesi
    egarch_vol: float             # EGARCH(1,1) / EWMA volatilitesi
    is_fallback: bool = False     # TEFAS fallback veya veri eksikliği fallback kullanıldı mı?


class HybridVolatilityEngine:
    """
    Hisseler ve Borsa Yatırım Fonları (BYF) için EGARCH(1,1) + Parkinson kombinasyonlu
    hibrit volatilite tensörü üreten motor.
    """

    def __init__(
        self,
        egarch_weight: float = 0.5,
        min_volatility_floor: float = 1e-4,
        trading_days_per_year: int = 252
    ):
        """
        :param egarch_weight: EGARCH volatilite ağırlığı (w_1). Parkinson ağırlığı (1 - w_1) olur.
        :param min_volatility_floor: Sıfır volatilite singülaritesini engelleyen katı alt taban (1e-4).
        :param trading_days_per_year: Yıllıklandırma çarpanı (252 gün).
        """
        self.egarch_weight = egarch_weight
        self.min_volatility_floor = min_volatility_floor
        self.trading_days_per_year = trading_days_per_year

    def compute_parkinson_volatility(self, high_prices: np.ndarray, low_prices: np.ndarray) -> float:
        """
        Parkinson High-Low volatilite hesabı:
        sigma_parkinson = sqrt( 1 / (4 * ln 2) * (ln(H / L))^2 )
        """
        if len(high_prices) == 0 or len(low_prices) == 0 or len(high_prices) != len(low_prices):
            return self.min_volatility_floor

        # Sıfır veya negatif fiyat koruması
        valid_mask = (high_prices > 0) & (low_prices > 0) & (high_prices >= low_prices)
        if not np.any(valid_mask):
            return self.min_volatility_floor

        h = high_prices[valid_mask]
        l = low_prices[valid_mask]

        # Parkinson Varyans Elemanları
        log_hl_sq = (np.log(h / l)) ** 2
        parkinson_var = float(np.mean(log_hl_sq) / (4.0 * np.log(2.0)))
        if parkinson_var <= 0:
            return self.min_volatility_floor

        daily_vol = np.sqrt(parkinson_var)

        # Yıllıklandırılmış Parkinson Volatilitesi
        annual_vol = daily_vol * np.sqrt(self.trading_days_per_year)
        return max(float(annual_vol), self.min_volatility_floor)

    def compute_egarch_volatility(self, log_returns: np.ndarray) -> float:
        """
        EGARCH(1,1) / EWMA Koşullu Volatilite Hesabı.
        Log getirilerdeki asimetrik şokları ve volatilite kümelenmesini (volatility clustering) yakalar.
        """
        if len(log_returns) < 5:
            return self.min_volatility_floor

        # EWMA / EGARCH Basit Yaklaşımı (lambda = 0.94 RiskMetrics standardı)
        decay_factor = 0.94
        n = len(log_returns)
        weights = (1.0 - decay_factor) * (decay_factor ** np.arange(n)[::-1])
        weights /= np.sum(weights)

        mean_ret = np.mean(log_returns)
        sq_devs = (log_returns - mean_ret) ** 2

        # Asimetri düzeltmesi (Negatif getiriler volatiliteyi daha çok tetikler)
        asymmetry_factor = np.where(log_returns < 0, 1.2, 0.8)
        weighted_var = float(np.sum(weights * sq_devs * asymmetry_factor))
        if weighted_var <= 0:
            return self.min_volatility_floor

        daily_vol = np.sqrt(weighted_var)
        annual_vol = daily_vol * np.sqrt(self.trading_days_per_year)
        return max(float(annual_vol), self.min_volatility_floor)

    def compute_intraday_hybrid_volatility(
        self,
        symbol: str,
        asset_type: AssetType,
        points: List[DataPoint]
    ) -> VolatilityResult:
        """
        Hisse veya BYF için EGARCH(1,1) + Parkinson V_t^hybrid birleşik volatilitesini üretir.
        """
        if len(points) < 2:
            return VolatilityResult(
                symbol=symbol,
                asset_type=asset_type,
                volatility=self.min_volatility_floor,
                parkinson_vol=self.min_volatility_floor,
                egarch_vol=self.min_volatility_floor,
                is_fallback=True
            )

        df = pd.DataFrame([{
            "date": pt.timestamp.date(),
            "price": pt.price,
            "high": pt.high_price if pt.high_price is not None else pt.price,
            "low": pt.low_price if pt.low_price is not None else pt.price
        } for pt in points])

        # Günlük High, Low ve Kapanış Fiyatlarını Grupla
        daily_df = df.groupby("date").agg({
            "high": "max",
            "low": "min",
            "price": "last"
        }).reset_index()
        daily_df.sort_values("date", inplace=True)

        # Parkinson Volatilitesi
        park_vol = self.compute_parkinson_volatility(
            daily_df["high"].values,
            daily_df["low"].values
        )

        # Log Getiriler & EGARCH Volatilitesi
        prices = daily_df["price"].values
        log_rets = np.diff(np.log(prices)) if len(prices) > 1 else np.array([0.0])
        egarch_vol = self.compute_egarch_volatility(log_rets)

        # Hibrit Volatilite Kombinasyonu: V_t^hybrid = w1 * sigma_EGARCH + (1 - w1) * sigma_Parkinson
        w1 = self.egarch_weight
        hybrid_vol = w1 * egarch_vol + (1.0 - w1) * park_vol
        final_vol = max(float(hybrid_vol), self.min_volatility_floor)

        return VolatilityResult(
            symbol=symbol,
            asset_type=asset_type,
            volatility=final_vol,
            parkinson_vol=park_vol,
            egarch_vol=egarch_vol,
            is_fallback=False
        )

    def compute_tefas_nav_volatility(
        self,
        symbol: str,
        asset_type: AssetType,
        points: List[DataPoint],
        rolling_days: int = 30
    ) -> VolatilityResult:
        """
        Adım 1.5.2:
        TEFAS fonları için 30 günlük hareketli NAV standart sapması ve EWMA oynaklık modülü.
        TEFAS fonlarının gün içi High/Low bar verisi olmadığı için Parkinson hesabı yapılmaz;
        30 günlük hareketli std dev ve EWMA modülü harmanlanır.
        Eksik NAV durumunda EWMA Fallback devreye girer.
        """
        # TEFAS Likit fonları (Para piyasası) için varsayılan düşük volatilite (örn. %0.5 yıllık)
        default_floor = 0.005 if asset_type == AssetType.TEFAS_LIQUID else self.min_volatility_floor

        if len(points) < 3:
            return VolatilityResult(
                symbol=symbol,
                asset_type=asset_type,
                volatility=default_floor,
                parkinson_vol=0.0,
                egarch_vol=default_floor,
                is_fallback=True
            )

        df = pd.DataFrame([{
            "date": pt.timestamp.date(),
            "price": pt.price
        } for pt in points])
        daily_df = df.groupby("date").agg({"price": "last"}).reset_index()
        daily_df.sort_values("date", inplace=True)

        prices = daily_df["price"].values
        if len(prices) < 2:
            return VolatilityResult(
                symbol=symbol,
                asset_type=asset_type,
                volatility=default_floor,
                parkinson_vol=0.0,
                egarch_vol=default_floor,
                is_fallback=True
            )

        log_rets = np.diff(np.log(prices))

        # 1. 30 Günlük Hareketli NAV Standart Sapması (Annualized)
        recent_rets = log_rets[-rolling_days:] if len(log_rets) >= rolling_days else log_rets
        if len(recent_rets) >= 2:
            std_daily = float(np.std(recent_rets, ddof=1))
            rolling_std_vol = std_daily * np.sqrt(self.trading_days_per_year)
        else:
            rolling_std_vol = default_floor

        # 2. EWMA Volatilitesi
        ewma_vol = self.compute_egarch_volatility(log_rets)
        if ewma_vol <= self.min_volatility_floor and default_floor > self.min_volatility_floor:
            ewma_vol = default_floor

        # Harmanlama: 50% 30-Günlük Std + 50% EWMA
        combined_vol = 0.5 * rolling_std_vol + 0.5 * ewma_vol
        final_vol = max(float(combined_vol), default_floor)
        is_fallback = len(log_rets) < 10

        return VolatilityResult(
            symbol=symbol,
            asset_type=asset_type,
            volatility=final_vol,
            parkinson_vol=0.0,  # TEFAS fonlarında Parkinson 0.0'dır
            egarch_vol=ewma_vol,
            is_fallback=is_fallback
        )

    def compute_snapshot_volatilities(self, snapshot: PITDataSnapshot) -> Dict[str, VolatilityResult]:
        """
        Snapshot içindeki tüm Hisseler, BYF'ler ve TEFAS fonları için volatiliteleri toplu hesaplar.
        """
        results: Dict[str, VolatilityResult] = {}

        # 1. Hisseler (EQUITY)
        for symbol, points in snapshot.equities.items():
            results[symbol] = self.compute_intraday_hybrid_volatility(symbol, AssetType.EQUITY, points)

        # 2. BYF / ETF
        for symbol, points in snapshot.etfs.items():
            results[symbol] = self.compute_intraday_hybrid_volatility(symbol, AssetType.ETF, points)

        # 3. TEFAS Serbest Fonlar
        for symbol, points in snapshot.tefas_free_funds.items():
            results[symbol] = self.compute_tefas_nav_volatility(symbol, AssetType.TEFAS_FREE, points)

        # 4. TEFAS Likit Fonlar
        for symbol, points in snapshot.tefas_liquid_funds.items():
            results[symbol] = self.compute_tefas_nav_volatility(symbol, AssetType.TEFAS_LIQUID, points)

        return results
