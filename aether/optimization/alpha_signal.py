"""
AetherForecaster-X v35.2 Multi-Asset Master
Katman 3: Valör-Cezalı Bounded Convex QP Solver

Modül: Alpha Signal Engine (Beklenen Getiri ve Sinyal İşleme Motoru)
Faz 2.3 Adım 2.3.1: Ham Alpha / Momentum / MMoE tahmin sinyallerini kabul eden,
z-score normalizasyonu ve varlık sınıflarına göre ölçekleme (scaling) yapan temel motor.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Union
import numpy as np
import pandas as pd


@dataclass
class AlphaSignalResult:
    """
    Alpha sinyal işleme sonuçlarını ve metriklerini tutan veri sınıfı.
    """
    raw_alpha: pd.Series
    normalized_alpha: pd.Series
    scaled_alpha: pd.Series
    mean_alpha: float
    std_alpha: float
    q_vector: pd.Series  # QP Solver linear term: q = -scaled_alpha


class AlphaSignalEngine:
    """
    Beklenen Getiri Vektörü (Alpha) İşleme Motoru.
    - Ham model tahminlerini (Head A / Head B / Momentum) kabul eder.
    - Z-Score Normalizasyonu uygular: mu_norm = (mu - mean) / (std + eps)
    - Varlık sınıflarına (Hisse / TEFAS Fon) göre dinamik risk/volatilite ölçeklemesi yapar.
    - QP Solver amaç fonksiyonu için q = -scaled_alpha vektörünü üretir.
    """

    def __init__(
        self,
        epsilon: float = 1e-6,
        target_volatility_scale: float = 1.0,
        tefas_alpha_multiplier: float = 0.8
    ):
        """
        :param epsilon: Sıfıra bölme zırhı (1e-6).
        :param target_volatility_scale: Sinyal genliğini ölçekleyen temel çarpan.
        :param tefas_alpha_multiplier: TEFAS fonları için alpha sönümleme katsayısı (düşük volatilite beklentisi).
        """
        self.epsilon = epsilon
        self.target_volatility_scale = target_volatility_scale
        self.tefas_alpha_multiplier = tefas_alpha_multiplier

    def process_alpha_signals(
        self,
        raw_signals: Union[pd.Series, Dict[str, float]],
        asset_types: Optional[Dict[str, str]] = None
    ) -> AlphaSignalResult:
        """
        Adım 2.3.1 - 2.3.3: Ham alpha sinyallerini işler.
        
        :param raw_signals: Varlık bazlı ham alpha / beklenen getiri sinyalleri (pd.Series veya Dict).
        :param asset_types: Varlık türü eşleştirmesi (ör: {'GARAN': 'EQUITY', 'FON1': 'TEFAS'}).
        :return: AlphaSignalResult nesnesi.
        """
        if isinstance(raw_signals, dict):
            raw_series = pd.Series(raw_signals)
        elif isinstance(raw_signals, pd.Series):
            raw_series = raw_signals.copy()
        else:
            raise TypeError("raw_signals bir pd.Series veya dict olmalıdır!")

        if raw_series.empty:
            raise ValueError("raw_signals boş olamaz!")

        # NaN / Inf kontrolü ve temizliği (Sadece sonlu değerlerle median hesaplanır)
        if raw_series.isna().any() or np.isinf(raw_series.values).any():
            finite_series = raw_series[np.isfinite(raw_series)]
            if not finite_series.empty:
                median_val = float(finite_series.median())
            else:
                median_val = 0.0
            
            raw_series = raw_series.replace([np.inf, -np.inf], np.nan).fillna(median_val)

        # 1. İstatistiksel Hesaplamalar
        mean_val = float(raw_series.mean())
        std_val = float(raw_series.std(ddof=0))

        # 2. Z-Score Normalizasyonu: mu_norm = (mu - mean) / (std + eps)
        if std_val < self.epsilon:
            # Sabit sinyal durumu (tüm elemanlar eşitse std ~ 0)
            normalized_series = pd.Series(0.0, index=raw_series.index)
        else:
            normalized_series = (raw_series - mean_val) / (std_val + self.epsilon)

        # 3. Varlık Sınıfı Scaling ve Volatilite Hedeflemesi
        scaled_series = normalized_series * self.target_volatility_scale

        if asset_types is not None:
            scaling_factors = []
            for symbol in raw_series.index:
                atype = str(asset_types.get(symbol, "EQUITY")).upper()
                if atype.startswith("TEFAS"):
                    scaling_factors.append(self.tefas_alpha_multiplier)
                else:
                    scaling_factors.append(1.0)
            scaled_series = scaled_series * pd.Series(scaling_factors, index=raw_series.index)

        # 4. QP Solver Linear Term: q = -scaled_alpha (Maksimum getiri için min -w^T mu)
        q_vector = -scaled_series

        return AlphaSignalResult(
            raw_alpha=raw_series,
            normalized_alpha=normalized_series,
            scaled_alpha=scaled_series,
            mean_alpha=mean_val,
            std_alpha=std_val,
            q_vector=q_vector
        )
