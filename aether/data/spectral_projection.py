"""
AetherForecaster-X v35.2 Multi-Asset Master
Katman 1: Point-in-Time (PIT) Data Lake & Hybrid Risk Engine

Modül: Higham Spectral Positive Definite Projection Engine (Spektral Pozitif Tanımlılık Yansıtma Motoru)
Faz 1.4 / Adım 1.4.1: N = N_hisse + N_BYF + N_TEFAS elemanlı hibrit kovaryans matrisinin
özdeğer ayrıştırması (eigenvalue decomposition) ve spektral teşhisi.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class SpectralAnalysisResult:
    """
    Kovaryans matrisinin özdeğer ayrıştırma ve spektral durum teşhis raporu.
    """
    eigenvalues: np.ndarray          # Küçükten büyüğe sıralı özdeğerler (lambda_1 <= ... <= lambda_N)
    eigenvectors: np.ndarray         # Dikgen özvektör matrisi V (V^T * V = I)
    min_eigenvalue: float            # Minimum özdeğer lambda_min
    max_eigenvalue: float            # Maksimum özdeğer lambda_max
    condition_number: float          # Durum sayısı kappa = lambda_max / max(lambda_min, 1e-16)
    is_positive_definite: bool       # lambda_min > 1e-8 ise True
    negative_eigenvalues_count: int  # lambda_i <= 0 olan özdeğer sayısı


class HighamSpectralProjectionEngine:
    """
    $N = N_{\text{hisse}} + N_{\text{BYF}} + N_{\text{TEFAS}}$ boyutlu hibrit kovaryans matrislerinin
    özdeğer spektrumunu analiz eden ve Higham en yakın pozitif tanımlı (Nearest Positive Definite)
    matris projeksiyonunu gerçekleştiren motor.
    """

    def __init__(self, min_eigenvalue_floor: float = 1e-6, max_iterations: int = 100):
        """
        :param min_eigenvalue_floor: QP Solver ve Cholesky patlamasını önleyen katı alt taban (delta = 1e-6).
        :param max_iterations: Alternatif Projeksiyon (Dykstra/Higham) algoritması maksimum iterasyon sayısı.
        """
        self.min_eigenvalue_floor = min_eigenvalue_floor
        self.max_iterations = max_iterations

    def decompose_covariance_matrix(self, cov_matrix: pd.DataFrame) -> SpectralAnalysisResult:
        """
        Adım 1.4.1:
        Sistemdeki tüm varlık tiplerini içeren kovaryans matrisinin simetrik özdeğer ayrıştırmasını (np.linalg.eigh) yapar.
        Özdeğerler ve özvektör matrisi elde edilir.
        """
        if cov_matrix.empty or cov_matrix.shape[0] != cov_matrix.shape[1]:
            raise ValueError("Kovaryans matrisi kare ve dolu olmalıdır!")

        vals = cov_matrix.values
        # Matris simetrisini garanti et: A_sym = (A + A^T) / 2
        sym_matrix = (vals + vals.T) / 2.0

        # Simetrik özdeğer ayrıştırması (Her zaman gerçek özdeğerler döndürür)
        eigenvalues, eigenvectors = np.linalg.eigh(sym_matrix)

        # Özdeğerler küçükten büyüğe sıralanır
        sort_indices = np.argsort(eigenvalues)
        sorted_eigenvalues = eigenvalues[sort_indices]
        sorted_eigenvectors = eigenvectors[:, sort_indices]

        lambda_min = float(sorted_eigenvalues[0])
        lambda_max = float(sorted_eigenvalues[-1])

        # Durum sayısı hesabı (Condition Number)
        cond_num = lambda_max / max(lambda_min, 1e-16)
        is_pd = lambda_min >= (self.min_eigenvalue_floor - 1e-10)
        neg_count = int(np.sum(sorted_eigenvalues <= 0.0))

        return SpectralAnalysisResult(
            eigenvalues=sorted_eigenvalues,
            eigenvectors=sorted_eigenvectors,
            min_eigenvalue=lambda_min,
            max_eigenvalue=lambda_max,
            condition_number=cond_num,
            is_positive_definite=is_pd,
            negative_eigenvalues_count=neg_count
        )

    def project_nearest_positive_definite(
        self,
        cov_matrix: pd.DataFrame,
        epsilon: Optional[float] = None
    ) -> Tuple[pd.DataFrame, SpectralAnalysisResult]:
        """
        Adım 1.4.2:
        Farklı frekanslardan gelen gürültülerin oluşturduğu negatif özdeğerleri strictly
        epsilon = 1e-6 seviyesinde tabanlar ve mühürlü pozitif tanımlı kovaryans matrisini üretir:
        Sigma_sealed = V * max(Lambda, 1e-6 * I) * V^T
        """
        if epsilon is None:
            epsilon = self.min_eigenvalue_floor

        analysis = self.decompose_covariance_matrix(cov_matrix)
        V = analysis.eigenvectors
        lambdas = analysis.eigenvalues

        # Özdeğerleri epsilon = 1e-6 seviyesinde tabanla (Yuvarlama hassasiyet payı ile)
        target_epsilon = max(epsilon, 1e-6)
        floored_lambdas = np.maximum(lambdas, target_epsilon + 1e-10)

        # Projeksiyon matrisi rekonstrüksiyonu: Sigma_proj = V * diag(floored_lambdas) * V^T
        proj_matrix = V @ np.diag(floored_lambdas) @ V.T

        # Matris simetrisini mühürle: (A + A^T) / 2
        sealed_matrix = (proj_matrix + proj_matrix.T) / 2.0

        sealed_df = pd.DataFrame(
            sealed_matrix,
            index=cov_matrix.index,
            columns=cov_matrix.columns
        )

        # Son teşhis analizini yenile
        post_analysis = self.decompose_covariance_matrix(sealed_df)
        return sealed_df, post_analysis
