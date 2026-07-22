"""
AetherForecaster-X v35.2 Multi-Asset Master
Katman 1: Point-in-Time (PIT) Data Lake & Hybrid Risk Engine

Modül: Bitemporal Historical Auction Ratio Engine (Kapanış Seansı Oran Motoru)
Faz 1.2 / Adım 1.2.1: BYF'ler ve Hisseler için alpha_auction = MA20(alpha_historical) hesabı.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from aether.data.pit_snapshot import AssetType, DataPoint, PITDataSnapshot


@dataclass
class AuctionRatioResult:
    """
    Varlık bazlı hesaplanmış kapanış seansı oran sonuçları.
    """
    symbol: str
    asset_type: AssetType
    alpha_auction: float
    historical_days_used: int
    is_floored: bool = False


class HistoricalAuctionRatioEngine:
    """
    Hisseler ve Borsa Yatırım Fonları (BYF) için geçmiş 20 günlük kapanış seansı hacim
    oranını (MA20) hesaplayan, gelecek sızıntısını engelleyen ve sıfıra bölmeyi flooreden motor.
    """

    def __init__(self, window_days: int = 20, min_periods: int = 5, epsilon: float = 1e-6):
        """
        :param window_days: Hareketli ortalama pencere gün sayısı (Varsayılan: 20 gün).
        :param min_periods: Hesaplama için gereken minimum geçmiş gün sayısı.
        :param epsilon: Sıfıra bölme patlamasını engelleyen minimum alpha tabanı.
        """
        self.window_days = window_days
        self.min_periods = min_periods
        self.epsilon = epsilon

    def calculate_daily_ratios(self, points: List[DataPoint]) -> pd.DataFrame:
        """
        Bir varlığa ait DataPoint serisinden günlük toplam hacim ve kapanış seansı hacmini gruplar,
        günlük alpha_daily = auction_volume / total_volume oranını çıkarır.
        """
        if not points:
            return pd.DataFrame(columns=["date", "total_volume", "auction_volume", "daily_alpha"])

        records = []
        for pt in points:
            records.append({
                "date": pt.timestamp.date(),
                "volume": pt.volume,
                "auction_volume": pt.auction_volume if pt.auction_volume is not None else 0.0
            })

        df = pd.DataFrame(records)
        daily_df = df.groupby("date").agg({
            "volume": "sum",
            "auction_volume": "sum"
        }).reset_index()

        # Günlük oran hesabı (Sıfır toplam hacim korumasıyla)
        daily_df["daily_alpha"] = np.where(
            daily_df["volume"] > 0,
            daily_df["auction_volume"] / daily_df["volume"],
            0.0
        )
        # Günlük oranlar [0.0, 1.0] aralığına hapsedilir
        daily_df["daily_alpha"] = np.clip(daily_df["daily_alpha"], 0.0, 1.0)
        return daily_df

    TEFAS_AUCTION_RATIO_CONSTANT: float = 1.0

    def compute_symbol_auction_ratio(self, symbol: str, asset_type: AssetType, points: List[DataPoint]) -> AuctionRatioResult:
        """
        Tek bir enstrüman (Hisse, BYF veya TEFAS Fonu) için MA20 kapanış seansı oranını hesaplar.
        Adım 1.2.2: TEFAS fonları gün içi kapanış seansına tabi olmadığı için katsayı sabit 1.0'dır.
        """
        # TEFAS fonları gün içi kapanış seansına tabi değildir -> Sabit alpha_auction = 1.0
        if asset_type.is_daily_nav:
            return AuctionRatioResult(
                symbol=symbol,
                asset_type=asset_type,
                alpha_auction=self.TEFAS_AUCTION_RATIO_CONSTANT,
                historical_days_used=0,
                is_floored=False
            )

        daily_df = self.calculate_daily_ratios(points)
        if len(daily_df) == 0:
            return AuctionRatioResult(
                symbol=symbol,
                asset_type=asset_type,
                alpha_auction=self.epsilon,
                historical_days_used=0,
                is_floored=True
            )

        # Gelecek sızıntısını engellemek için strictly tamamlanmış geçmiş günlerin MA20'si alınır
        recent_daily_alphas = daily_df["daily_alpha"].tail(self.window_days).values
        days_used = len(recent_daily_alphas)

        if days_used < self.min_periods:
            raw_alpha = float(np.mean(recent_daily_alphas)) if days_used > 0 else 0.0
        else:
            raw_alpha = float(np.mean(recent_daily_alphas))

        # Epsilon tabanlaması (Sıfıra bölme zırhı)
        is_floored = raw_alpha < self.epsilon
        final_alpha = max(raw_alpha, self.epsilon)

        return AuctionRatioResult(
            symbol=symbol,
            asset_type=asset_type,
            alpha_auction=final_alpha,
            historical_days_used=days_used,
            is_floored=is_floored
        )

    def compute_snapshot_auction_ratios(self, snapshot: PITDataSnapshot) -> Dict[str, AuctionRatioResult]:
        """
        PITDataSnapshot içerisindeki Hisseler, BYF'ler ve TEFAS fonlarının tamamı için
        alpha_auction katsayılarını toplu olarak hesaplar.
        """
        results: Dict[str, AuctionRatioResult] = {}

        # 1. Hisseler (EQUITY)
        for symbol, points in snapshot.equities.items():
            results[symbol] = self.compute_symbol_auction_ratio(symbol, AssetType.EQUITY, points)

        # 2. BYF / Borsa Yatırım Fonları (ETF) - Adım 1.2.1
        for symbol, points in snapshot.etfs.items():
            results[symbol] = self.compute_symbol_auction_ratio(symbol, AssetType.ETF, points)

        # 3. TEFAS Serbest Fonlar
        for symbol, points in snapshot.tefas_free_funds.items():
            results[symbol] = self.compute_symbol_auction_ratio(symbol, AssetType.TEFAS_FREE, points)

        # 4. TEFAS Likit Fonlar
        for symbol, points in snapshot.tefas_liquid_funds.items():
            results[symbol] = self.compute_symbol_auction_ratio(symbol, AssetType.TEFAS_LIQUID, points)

        return results
