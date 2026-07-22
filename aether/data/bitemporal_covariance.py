"""
AetherForecaster-X v35.2 Multi-Asset Master
Katman 1: Point-in-Time (PIT) Data Lake & Hybrid Risk Engine

Modül: Bitemporal Covariance Builder (Frekans Uyumlaştırmalı Kovaryans Motoru)
Faz 1.3 / Adım 1.3.1: 15 dakikalık hisse/BYF getirileri ile günlük resample edilmiş
TEFAS fon getirilerinin frekans eşlemesini yapan BitemporalCovarianceBuilder sınıfı.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd

from aether.data.pit_snapshot import AssetType, DataPoint, PITDataSnapshot


@dataclass
class BitemporalCovarianceResult:
    """
    Frekans eşlemesi yapılmış ham bitemporal getiriler ve blok matris haritası.
    """
    symbols: List[str]
    asset_types: List[AssetType]
    intraday_symbols: List[str]
    daily_symbols: List[str]
    intraday_returns_df: pd.DataFrame  # 15-dk bar getirileri (Hisse & BYF)
    daily_returns_df: pd.DataFrame     # Senkronize edilmiş günlük getiriler (Tüm varlıklar)
    block_indices: Dict[str, Tuple[int, int]] = field(default_factory=dict)


class BitemporalCovarianceBuilder:
    """
    15 dakikalık yüksek frekanslı hisse/BYF getirileri ile günlük TEFAS NAV getirilerini
    boyut uyuşmazlığı ve frekans karmaşası yaşanmadan senkronize eden builder sınıfı.
    """

    def __init__(self, bars_per_day: int = 35):
        """
        :param bars_per_day: Bir BIST işlem günündeki 15 dakikalık bar sayısı (7 saat * 4 = 35 bar).
        """
        self.bars_per_day = bars_per_day

    def compute_intraday_log_returns(self, points: List[DataPoint]) -> pd.Series:
        """
        15 dakikalık DataPoint fiyat serisinden logaritmik bar getirilerini r_t = ln(P_t / P_{t-1}) hesaplar.
        """
        if len(points) < 2:
            return pd.Series(dtype=float)

        df = pd.DataFrame([{"timestamp": pt.timestamp, "price": pt.price} for pt in points])
        df.sort_values("timestamp", inplace=True)
        df["log_return"] = np.log(df["price"] / df["price"].shift(1))
        df.dropna(subset=["log_return"], inplace=True)
        return pd.Series(data=df["log_return"].values, index=df["timestamp"])

    def compute_daily_returns_from_intraday(self, points: List[DataPoint]) -> pd.Series:
        """
        15 dakikalık bar fiyatlarından günlük kapanış fiyatlarını çıkarır ve günlük log getiriyi hesaplar.
        """
        if len(points) < 2:
            return pd.Series(dtype=float)

        df = pd.DataFrame([{
            "date": pt.timestamp.date(),
            "timestamp": pt.timestamp,
            "price": pt.price
        } for pt in points])

        # Her günün son fiyatını (günlük kapanış) al
        daily_closes = df.groupby("date").agg({"price": "last", "timestamp": "last"}).reset_index()
        daily_closes.sort_values("date", inplace=True)
        daily_closes["daily_log_return"] = np.log(daily_closes["price"] / daily_closes["price"].shift(1))
        daily_closes.dropna(subset=["daily_log_return"], inplace=True)
        return pd.Series(data=daily_closes["daily_log_return"].values, index=daily_closes["date"])

    def compute_tefas_daily_log_returns(self, points: List[DataPoint]) -> pd.Series:
        """
        TEFAS fonlarının günlük NAV fiyat serisinden günlük logaritmik getiriyi hesaplar.
        """
        if len(points) < 2:
            return pd.Series(dtype=float)

        df = pd.DataFrame([{
            "date": pt.timestamp.date(),
            "price": pt.price
        } for pt in points])
        df = df.groupby("date").agg({"price": "last"}).reset_index()
        df.sort_values("date", inplace=True)
        df["daily_log_return"] = np.log(df["price"] / df["price"].shift(1))
        df.dropna(subset=["daily_log_return"], inplace=True)
        return pd.Series(data=df["daily_log_return"].values, index=df["date"])

    def build_bitemporal_returns(self, snapshot: PITDataSnapshot) -> BitemporalCovarianceResult:
        """
        Snapshot içindeki Hisseler, BYF'ler ve TEFAS fonlarının getirilerini toplar,
        hem 15-dk hem günlük senkronize getiri DataFrame'lerini ve blok haritasını inşa eder.
        """
        all_symbols: List[str] = []
        all_asset_types: List[AssetType] = []
        intraday_symbols: List[str] = []
        daily_symbols: List[str] = []

        intraday_series_dict: Dict[str, pd.Series] = {}
        daily_series_dict: Dict[str, pd.Series] = {}

        # 1. HİSSELER (EQUITY - Intraday)
        for symbol, points in snapshot.equities.items():
            all_symbols.append(symbol)
            all_asset_types.append(AssetType.EQUITY)
            intraday_symbols.append(symbol)
            
            s_15m = self.compute_intraday_log_returns(points)
            if len(s_15m) > 0:
                intraday_series_dict[symbol] = s_15m
            s_daily = self.compute_daily_returns_from_intraday(points)
            if len(s_daily) > 0:
                daily_series_dict[symbol] = s_daily

        # 2. BYF / ETF (ETF - Intraday)
        for symbol, points in snapshot.etfs.items():
            all_symbols.append(symbol)
            all_asset_types.append(AssetType.ETF)
            intraday_symbols.append(symbol)

            s_15m = self.compute_intraday_log_returns(points)
            if len(s_15m) > 0:
                intraday_series_dict[symbol] = s_15m
            s_daily = self.compute_daily_returns_from_intraday(points)
            if len(s_daily) > 0:
                daily_series_dict[symbol] = s_daily

        # 3. TEFAS SERBEST FONLAR (TEFAS_FREE - Daily NAV)
        for symbol, points in snapshot.tefas_free_funds.items():
            all_symbols.append(symbol)
            all_asset_types.append(AssetType.TEFAS_FREE)
            daily_symbols.append(symbol)

            s_daily = self.compute_tefas_daily_log_returns(points)
            if len(s_daily) > 0:
                daily_series_dict[symbol] = s_daily

        # 4. TEFAS LİKİT FONLAR (TEFAS_LIQUID - Daily NAV)
        for symbol, points in snapshot.tefas_liquid_funds.items():
            all_symbols.append(symbol)
            all_asset_types.append(AssetType.TEFAS_LIQUID)
            daily_symbols.append(symbol)

            s_daily = self.compute_tefas_daily_log_returns(points)
            if len(s_daily) > 0:
                daily_series_dict[symbol] = s_daily

        # DataFrame birleştirmeleri
        intraday_df = pd.DataFrame(intraday_series_dict) if intraday_series_dict else pd.DataFrame()
        daily_df = pd.DataFrame(daily_series_dict) if daily_series_dict else pd.DataFrame()

        # Eksik sütunları (örn. <2 veri noktası olanları) güvenli şekilde ekle
        for sym in intraday_symbols:
            if sym not in intraday_df.columns:
                intraday_df[sym] = 0.0

        for sym in all_symbols:
            if sym not in daily_df.columns:
                daily_df[sym] = 0.0

        # NaN doldurma (0.0 nötr getiri ile)
        intraday_df = intraday_df.fillna(0.0)
        daily_df = daily_df.fillna(0.0)

        # Blok indeks haritası
        block_indices: Dict[str, Tuple[int, int]] = {}
        for idx, sym in enumerate(all_symbols):
            block_indices[sym] = (idx, idx)

        return BitemporalCovarianceResult(
            symbols=all_symbols,
            asset_types=all_asset_types,
            intraday_symbols=intraday_symbols,
            daily_symbols=daily_symbols,
            intraday_returns_df=intraday_df,
            daily_returns_df=daily_df,
            block_indices=block_indices
        )

    def compute_newey_west_kappa(
        self,
        intraday_returns_df: pd.DataFrame,
        max_lag: int = 4,
        raw_n_bars: float = 175.0
    ) -> float:
        """
        Newey-West Bartlett Kernel ile otokorelasyon sönümleme katsayısını (kappa_effective) hesaplar.
        15 dakikalık barlarda negatif otokorelasyon (mean-reversion) bulunduğundan,
        kappa_effective raw N=175 yerine ~40-50 seviyesine sönümlenir.
        """
        if len(intraday_returns_df) < max_lag + 2 or intraday_returns_df.shape[1] == 0:
            return 45.0  # Güvenli sönümlenmiş varsayılan katsayı

        autocorr_sum = 0.0
        cols_count = 0

        for col in intraday_returns_df.columns:
            series = intraday_returns_df[col]
            if series.std() < 1e-8:
                continue

            cols_count += 1
            for k in range(1, max_lag + 1):
                rho_k = series.autocorr(lag=k)
                if not np.isnan(rho_k):
                    w_k = 1.0 - (k / (max_lag + 1))  # Bartlett kernel katsayısı
                    autocorr_sum += w_k * rho_k

        if cols_count == 0:
            return 45.0

        avg_kernel_autocorr = autocorr_sum / cols_count
        kappa_eff = raw_n_bars * (1.0 + 2.0 * avg_kernel_autocorr)

        # Güvenlik Sınırları: 15-dk barlar için [30.0, 75.0] aralığında mühürlenir
        return float(np.clip(kappa_eff, 30.0, 75.0))

    def build_weekly_bitemporal_covariance(
        self,
        bitemporal_res: BitemporalCovarianceResult,
        max_lag: int = 4,
        daily_scaling_factor: float = 5.0
    ) -> Tuple[pd.DataFrame, float]:
        """
        Adım 1.3.2:
        Hisse ve BYF bloğuna Newey-West sönümlemesini (kappa_effective ~40-50) uygular;
        TEFAS bloğunu ve çapraz bloğu haftalık ölçek katsayısıyla (x5) ölçekleyip
        birleşik haftalık kovaryans matrisini üretir.
        """
        symbols = bitemporal_res.symbols
        n = len(symbols)
        if n == 0:
            return pd.DataFrame(index=symbols, columns=symbols), 45.0

        # 1. Newey-West sönümleme katsayısı hesabı
        kappa_eff = self.compute_newey_west_kappa(
            bitemporal_res.intraday_returns_df,
            max_lag=max_lag
        )

        # 2. 15-dk bar kovaryans matrisi (Hisse & BYF) -> Haftalık ölçekleme (kappa_eff ile)
        intraday_syms = bitemporal_res.intraday_symbols
        cov_matrix = pd.DataFrame(0.0, index=symbols, columns=symbols)

        if len(intraday_syms) > 0 and len(bitemporal_res.intraday_returns_df) > 1:
            cov_15m = bitemporal_res.intraday_returns_df[intraday_syms].cov()
            cov_intraday_weekly = cov_15m * kappa_eff
            cov_matrix.loc[intraday_syms, intraday_syms] = cov_intraday_weekly

        # 3. Günlük senkronize verilerden TEFAS bloğu ve Çapraz bloklar -> Haftalık ölçekleme (x5 ile)
        daily_df = bitemporal_res.daily_returns_df
        daily_syms = bitemporal_res.daily_symbols

        if len(daily_df) > 1:
            cov_daily_sample = daily_df.cov() * daily_scaling_factor

            # TEFAS Bloğu (TEFAS x TEFAS)
            if len(daily_syms) > 0:
                cov_matrix.loc[daily_syms, daily_syms] = cov_daily_sample.loc[daily_syms, daily_syms]

            # Çapraz Bloklar (Hisse/BYF x TEFAS)
            if len(intraday_syms) > 0 and len(daily_syms) > 0:
                cov_matrix.loc[intraday_syms, daily_syms] = cov_daily_sample.loc[intraday_syms, daily_syms]
                cov_matrix.loc[daily_syms, intraday_syms] = cov_daily_sample.loc[daily_syms, intraday_syms]

        return cov_matrix, kappa_eff
