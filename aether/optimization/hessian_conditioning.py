"""
AetherForecaster-X v35.2 Multi-Asset Master
Katman 3: Valör-Cezalı Bounded Convex QP Solver

Modül: Hessian Conditioning Engine (Hessian Koşullandırma ve Düzenlileştirme Motoru)
Faz 2.2: QP Solver amaç fonksiyonundaki Hessian matrisinin (P = 2 * (Sigma + D_total))
koşul sayısını (Condition Number) kappa(P) <= 10^4 seviyesinde sabitleyen ve
aşırı volatilite/ceza değerlerini Soft-Clipping + Tikhonov düzenlileştirmesi ile dizginleyen modül.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class HessianConditioningResult:
    """
    Koşullandırılmış Hessian matrisi ve spektral teşhis raporu.
    """
    p_conditioned: pd.DataFrame
    p_raw: pd.DataFrame
    kappa_raw: float
    kappa_conditioned: float
    epsilon_tikhonov: float
    theta_threshold: float
    lambda_min: float
    lambda_max: float
    is_positive_definite: bool


class HessianConditioningEngine:
    """
    Hessian Matrisi Soft-Clipping ve Tikhonov Düzeltme Motoru.
    - D_total / D_impact ceza matrisini Sigma_weekly spektral tavanı ile kırpar (Soft-Clipping).
    - Condition Number kappa(P) > 10^4 durumunda dinamik Tikhonov ekleyerek (P + eps * I)
      matrisi QP Solver için optimal konveks alana çeker.
    """

    def __init__(
        self,
        theta: float = 100.0,
        max_condition_number: float = 1e4,
        min_eigenvalue_floor: float = 1e-6
    ):
        """
        :param theta: Covariance max özdeğerinin kaç katına kadar ceza kabul edileceği (Varsayılan 100.0).
        :param max_condition_number: Hedef maksimum durum sayısı kappa(P) <= 10^4.
        :param min_eigenvalue_floor: Mutlak minimum pozitif tanımlılık tabanı (1e-6).
        """
        self.theta = theta
        self.max_condition_number = max_condition_number
        self.min_eigenvalue_floor = min_eigenvalue_floor

    def compute_conditioned_hessian(
        self,
        sigma_weekly: pd.DataFrame,
        d_total: pd.DataFrame
    ) -> HessianConditioningResult:
        if sigma_weekly.empty or d_total.empty:
            raise ValueError("Sigma ve D_total matrisleri boş olamaz!")

        # 1. Hizalama ve Veri Bütünlüğü Kontrolü (Alignment & NaN Check)
        symbols = list(sigma_weekly.index)
        n = len(symbols)

        # Sütun ve satırların birebir aynı olmasını garantiye al
        d_total_aligned = d_total.reindex(index=symbols, columns=symbols).fillna(0.0)

        # NaN veya Inf içeren kovaryans matrisi linalg.eigh'yi çökertecektir.
        if sigma_weekly.isna().any().any() or np.isinf(sigma_weekly.values).any():
            raise ValueError("Sigma matrisi NaN veya Inf içeremez. Bu durum linalg hesaplamalarını çökertir!")

        # 2. Sigma_weekly spektral analizi (Maksimum özdeğer hesabı)
        sigma_vals = (sigma_weekly.values + sigma_weekly.values.T) / 2.0
        eigvals_sigma = np.linalg.eigvalsh(sigma_vals)
        lambda_max_sigma = max(float(np.max(eigvals_sigma)), self.min_eigenvalue_floor)

        # 3. Soft-Clipping Tavanı: threshold = theta * lambda_max_sigma
        theta_threshold = self.theta * lambda_max_sigma

        # D_total ceza matrisini Soft-Clipping ile dizginleme
        d_total_vals = d_total_aligned.values.copy()
        d_clipped_vals = np.minimum(d_total_vals, theta_threshold)
        d_clipped = pd.DataFrame(d_clipped_vals, index=symbols, columns=symbols)

        # 3. Ham Hessian Kurulumu: P_raw = 2 * (Sigma_weekly + D_clipped)
        p_raw_matrix = 2.0 * (sigma_vals + d_clipped_vals)
        p_raw_df = pd.DataFrame(p_raw_matrix, index=symbols, columns=symbols)

        # 4. P_raw Özdeğer Analizi
        eigvals_p, eigvecs_p = np.linalg.eigh(p_raw_matrix)
        lambda_min_raw = float(np.min(eigvals_p))
        lambda_max_raw = float(np.max(eigvals_p))

        kappa_raw = lambda_max_raw / max(lambda_min_raw, 1e-16) if lambda_min_raw > 0 else float("inf")

        # 5. Dinamik Tikhonov Düzeltmesi (Adım 2.2.3)
        # Goal: (lambda_max + eps) / (lambda_min + eps) <= kappa_max
        # => lambda_max + eps <= kappa_max * lambda_min + kappa_max * eps
        # => eps * (kappa_max - 1) >= lambda_max - kappa_max * lambda_min
        # => eps >= (lambda_max - kappa_max * lambda_min) / (kappa_max - 1)

        eps_tikhonov = 0.0

        # Eğer lambda_min <= min_floor ise önce tabana çek
        if lambda_min_raw < self.min_eigenvalue_floor:
            eps_floor = self.min_eigenvalue_floor - lambda_min_raw
            eps_tikhonov = max(eps_tikhonov, eps_floor)

        # Condition number 10^4 sınırını aşıyorsa Tikhonov shift hesapla
        current_min = lambda_min_raw + eps_tikhonov
        current_max = lambda_max_raw + eps_tikhonov
        current_kappa = current_max / max(current_min, 1e-16)

        if current_kappa > self.max_condition_number:
            needed_eps = (current_max - self.max_condition_number * current_min) / (self.max_condition_number - 1.0)
            if needed_eps > 0:
                eps_tikhonov += float(needed_eps)

        # 6. P_conditioned Rekonstrüksiyonu: P_raw + eps_tikhonov * I
        p_conditioned_matrix = p_raw_matrix + eps_tikhonov * np.eye(n)
        p_conditioned_df = pd.DataFrame(p_conditioned_matrix, index=symbols, columns=symbols)

        # Nihai Spektral Kontrol
        eigvals_cond = np.linalg.eigvalsh(p_conditioned_matrix)
        lambda_min_cond = float(np.min(eigvals_cond))
        lambda_max_cond = float(np.max(eigvals_cond))
        kappa_cond = lambda_max_cond / max(lambda_min_cond, 1e-16)
        is_pd = lambda_min_cond >= (self.min_eigenvalue_floor - 1e-9)

        return HessianConditioningResult(
            p_conditioned=p_conditioned_df,
            p_raw=p_raw_df,
            kappa_raw=kappa_raw,
            kappa_conditioned=kappa_cond,
            epsilon_tikhonov=eps_tikhonov,
            theta_threshold=theta_threshold,
            lambda_min=lambda_min_cond,
            lambda_max=lambda_max_cond,
            is_positive_definite=is_pd
        )
