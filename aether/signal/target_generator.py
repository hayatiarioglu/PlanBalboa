"""
AetherForecaster-X v35.2 Multi-Asset Master
Katman 2: Clamped Scheduled Soft-NDCG & Detach-Coupled MMoE

Modül: Robust Winsorized Scaler & Target Generator
Faz 3.1 / Adım 3.1.1: Haftalık getirilerde aşırı uç değerlerin (outliers) hedef getiri dağılımını
bozmasını engelleyen, alt ve üst %2.5'lik (q_0.025, q_0.975) dilimleri kırpan Winsorized normalizasyon katmanı.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class WinsorizedScalerResult:
    """
    Winsorized ölçekleme çıktıları ve istatistiksel rapor nesnesi.
    """
    raw_returns: Union[pd.Series, np.ndarray]
    winsorized_returns: Union[pd.Series, np.ndarray]
    lower_quantile_val: float
    upper_quantile_val: float
    n_clipped_lower: int
    n_clipped_upper: int


class RobustWinsorizedScaler:
    """
    Katman 2 Robust Winsorized Dönüştürücü.
    - Haftalık getiri serilerinde alt (%2.5) ve üst (%97.5) kilit dilim değerlerini tespit eder.
    - Aşırı spekülatif tavan/taban fiyat hareketlerinin nöral ağ gradyanlarını saptırmasını engeller.
    """

    def __init__(
        self,
        lower_quantile: float = 0.025,
        upper_quantile: float = 0.975,
        epsilon: float = 1e-8
    ):
        """
        :param lower_quantile: Alt kırpma yüzdesi (Varsayılan %2.5 -> 0.025).
        :param upper_quantile: Üst kırpma yüzdesi (Varsayılan %97.5 -> 0.975).
        :param epsilon: Sayısal kararlılık toleransı.
        """
        if not (0.0 <= lower_quantile < upper_quantile <= 1.0):
            raise ValueError(f"Geçersiz quantile dilimleri! 0 <= lower ({lower_quantile}) < upper ({upper_quantile}) <= 1 olmalıdır.")
        
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.epsilon = epsilon

    def fit_transform(
        self,
        returns: Union[pd.Series, np.ndarray, List[float]]
    ) -> WinsorizedScalerResult:
        """
        Haftalık getirileri kabul eder, alt ve üst %2.5 dilimleri hesaplar ve veriyi Winsorize eder.

        :param returns: Ham haftalık getiri serisi (pd.Series, np.ndarray veya List[float])
        :return: WinsorizedScalerResult nesnesi
        """
        if isinstance(returns, list):
            arr = np.asarray(returns, dtype=np.float64)
            is_series = False
            index = None
        elif isinstance(returns, pd.Series):
            arr = returns.to_numpy(dtype=np.float64)
            is_series = True
            index = returns.index
        elif isinstance(returns, np.ndarray):
            arr = returns.astype(np.float64)
            is_series = False
            index = None
        else:
            raise TypeError("returns nesnesi pd.Series, np.ndarray veya List[float] olmalıdır!")

        if arr.size == 0:
            raise ValueError("Getiri verisi boş olamaz!")

        # NaN / Inf Temizleme Zırhı
        finite_mask = np.isfinite(arr)
        if not np.all(finite_mask):
            finite_vals = arr[finite_mask]
            fill_val = float(np.median(finite_vals)) if finite_vals.size > 0 else 0.0
            arr = np.where(finite_mask, arr, fill_val)

        # Quantile Hesaplamaları (%2.5 ve %97.5)
        q_lower = float(np.quantile(arr, self.lower_quantile))
        q_upper = float(np.quantile(arr, self.upper_quantile))

        # Kırpılacak eleman sayılarının tespiti
        n_lower = int(np.sum(arr < q_lower))
        n_upper = int(np.sum(arr > q_upper))

        # Kırpma (Clipping) Operasyonu
        winsorized_arr = np.clip(arr, a_min=q_lower, a_max=q_upper)

        if is_series:
            raw_out = pd.Series(arr, index=index)
            winsorized_out = pd.Series(winsorized_arr, index=index)
        else:
            raw_out = arr
            winsorized_out = winsorized_arr

        return WinsorizedScalerResult(
            raw_returns=raw_out,
            winsorized_returns=winsorized_out,
            lower_quantile_val=q_lower,
            upper_quantile_val=q_upper,
            n_clipped_lower=n_lower,
            n_clipped_upper=n_upper
        )


@dataclass
class TargetGeneratorResult:
    """
    Delist Ayrıştırma ve Hedef Üretim Sonuç Raporu.
    """
    combined_targets: Union[pd.Series, np.ndarray]    # Delist hisseler dahil nihai hedef vektörü
    healthy_targets: Union[pd.Series, np.ndarray]     # Sadece sağlam hisselerin ölçeklenmiş hedefleri
    delisted_mask: Union[pd.Series, np.ndarray]      # Delist hisse maskesi (True = Delist)
    healthy_spread: float                            # Sağlam hisselerin hedef genişliği (Max - Min)
    decoupled_delist_penalty: float                  # Delist hisselere verilen ayrılmış ceza değeri


class DelistDecouplingTargetGenerator:
    """
    Adım 3.1.2: Delist/İflas hisselerin (-100%) getirisinin sağlam hisseleri sıkıştırmasını engelleyen ayrıştırma katmanı.
    - Delist hisseleri (-%90 altı getiri veya explicit delist maskesi) normalizasyon havuzundan izole eder.
    - RobustWinsorizedScaler ile hedefleri sadece sağlam hisselere uygular.
    - Delist hisselere ayrıştırılmış ceza hedefi (ör: -2.0) atayarak hedef sıkışmasını (target compression) %100 engeller.
    """

    def __init__(
        self,
        scaler: Optional[RobustWinsorizedScaler] = None,
        delist_return_threshold: float = -0.90,
        delist_target_penalty: float = -2.0
    ):
        """
        :param scaler: Kırpma için RobustWinsorizedScaler nesnesi.
        :param delist_return_threshold: Otomatik delist/iflas getiri sınırı (%-90).
        :param delist_target_penalty: Delist hisselere verilecek ayrıştırılmış ceza hedef değeri.
        """
        self.scaler = scaler or RobustWinsorizedScaler()
        self.delist_return_threshold = delist_return_threshold
        self.delist_target_penalty = delist_target_penalty

    def generate_targets(
        self,
        returns: Union[pd.Series, np.ndarray],
        delisted_flags: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> TargetGeneratorResult:
        """
        Delist ve sağlam hisseleri ayrıştırarak hedef getiri matrisini üretir.

        :param returns: Varlıkların haftalık ham getiri serisi.
        :param delisted_flags: Özel delist bayrağı (True/1 = Delist, False/0 = Sağlam).
        :return: TargetGeneratorResult nesnesi.
        """
        if isinstance(returns, pd.Series):
            ret_ser = returns.copy()
            is_series = True
            idx = returns.index
            ret_arr = ret_ser.to_numpy(dtype=np.float64)
        elif isinstance(returns, np.ndarray):
            ret_arr = returns.astype(np.float64)
            is_series = False
            idx = None
        else:
            raise TypeError("returns nesnesi pd.Series veya np.ndarray olmalıdır!")

        if ret_arr.size == 0:
            raise ValueError("Getiri verisi boş olamaz!")

        # 1. Delist Maskesi Tespiti
        if delisted_flags is not None:
            if isinstance(delisted_flags, pd.Series):
                d_mask_arr = delisted_flags.to_numpy(dtype=bool)
            else:
                d_mask_arr = np.asarray(delisted_flags, dtype=bool)
        else:
            # Otomatik eşik kontrolü (Getiri <= -0.90 -> Delist)
            d_mask_arr = ret_arr <= self.delist_return_threshold

        healthy_mask_arr = ~d_mask_arr

        # Eğer hiç sağlam hisse yoksa (kriz/tüm evren çökmüşse)
        if not np.any(healthy_mask_arr):
            combined = np.full_like(ret_arr, fill_value=self.delist_target_penalty)
            if is_series:
                c_out = pd.Series(combined, index=idx)
                h_out = pd.Series([], index=[])
                d_out = pd.Series(d_mask_arr, index=idx)
            else:
                c_out = combined
                h_out = np.array([])
                d_out = d_mask_arr

            return TargetGeneratorResult(
                combined_targets=c_out,
                healthy_targets=h_out,
                delisted_mask=d_out,
                healthy_spread=0.0,
                decoupled_delist_penalty=self.delist_target_penalty
            )

        # 2. Sadece Sağlam Hisseler Üzerinde Winsorized Scaling
        healthy_returns = ret_arr[healthy_mask_arr]
        scaler_res = self.scaler.fit_transform(healthy_returns)
        healthy_scaled = scaler_res.winsorized_returns

        if isinstance(healthy_scaled, pd.Series):
            healthy_scaled_arr = healthy_scaled.to_numpy()
        else:
            healthy_scaled_arr = healthy_scaled

        # 3. Birleştirilmiş Hedef Vektörü İnşası
        combined_arr = np.zeros_like(ret_arr, dtype=np.float64)
        combined_arr[healthy_mask_arr] = healthy_scaled_arr
        combined_arr[d_mask_arr] = self.delist_target_penalty

        # Sağlam hisse hedef genişliği (Spread)
        spread = float(np.max(healthy_scaled_arr) - np.min(healthy_scaled_arr))

        if is_series:
            combined_out = pd.Series(combined_arr, index=idx)
            healthy_out = pd.Series(healthy_scaled_arr, index=idx[healthy_mask_arr])
            delist_mask_out = pd.Series(d_mask_arr, index=idx)
        else:
            combined_out = combined_arr
            healthy_out = healthy_scaled_arr
            delist_mask_out = d_mask_arr

        return TargetGeneratorResult(
            combined_targets=combined_out,
            healthy_targets=healthy_out,
            delisted_mask=delist_mask_out,
            healthy_spread=spread,
            decoupled_delist_penalty=self.delist_target_penalty
        )

