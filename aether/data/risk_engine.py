"""
AetherForecaster-X v35.2 Multi-Asset Master
Katman 1: Point-in-Time (PIT) Data Lake & Hybrid Risk Engine

Modül: Multi-Asset Master Risk Engine Orchestrator (Katman 1 Birleşik Risk Motoru)
Faz 1.6 / Adım 1.6.1: Bütün varlık sınıflarının (Hisse, BYF, TEFAS Serbest, TEFAS Likit)
RiskEngine çatısı altında birleştirilmesi.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from aether.data.pit_snapshot import AssetType, PITDataSnapshot, PITDataAdapter
from aether.data.temporal_lock import TemporalLockEngine, TemporalViolationReport
from aether.data.auction_ratio import HistoricalAuctionRatioEngine, AuctionRatioResult
from aether.data.bitemporal_covariance import BitemporalCovarianceBuilder, BitemporalCovarianceResult
from aether.data.spectral_projection import HighamSpectralProjectionEngine, SpectralAnalysisResult
from aether.data.hybrid_volatility import HybridVolatilityEngine, VolatilityResult


@dataclass
class MultiAssetRiskEngineOutput:
    """
    Katman 1 birleşik risk motorunun tüm çıktı paketini temsil eden mühürlü yapı.
    """
    snapshot: PITDataSnapshot
    audit_report: TemporalViolationReport
    auction_ratios: Dict[str, AuctionRatioResult]
    volatilities: Dict[str, VolatilityResult]
    covariance_matrix: pd.DataFrame  # Higham projeksiyonlu, lambda_min >= 1e-6 mühürlü kovaryans
    spectral_analysis: SpectralAnalysisResult
    effective_kappa: float


class MultiAssetRiskEngine:
    """
    Hisseler (BIST), Borsa Yatırım Fonları (BYF), TEFAS Serbest Fonlar ve TEFAS Likit Fonların
    tamamını tek bir boru hattında (pipeline) birleştiren master risk orkestratörü.
    """

    def __init__(
        self,
        window_days_auction: int = 20,
        min_eigenvalue_floor: float = 1e-6,
        egarch_weight: float = 0.5,
        min_volatility_floor: float = 1e-4
    ):
        self.auction_engine = HistoricalAuctionRatioEngine(window_days=window_days_auction, epsilon=min_eigenvalue_floor)
        self.covariance_builder = BitemporalCovarianceBuilder(bars_per_day=35)
        self.spectral_engine = HighamSpectralProjectionEngine(min_eigenvalue_floor=min_eigenvalue_floor)
        self.volatility_engine = HybridVolatilityEngine(egarch_weight=egarch_weight, min_volatility_floor=min_volatility_floor)

    def process_raw_data(
        self,
        raw_data_df: pd.DataFrame,
        snapshot_time: datetime,
        tefas_cutoff_date: datetime
    ) -> MultiAssetRiskEngineOutput:
        """
        Adım 1.6.1:
        Bütün varlık sınıflarını zamansal kilitlenme, oran hesabı, bitemporal kovaryans,
        Higham spektral projeksiyon ve volatilite tensörü adımlarından tek seferde geçirir.
        """
        # 1. ZAMANSAL KİLİTLENME VE SNAPSHOT OLUŞTURMA
        adapter = PITDataAdapter(snapshot_time=snapshot_time, tefas_cutoff_date=tefas_cutoff_date)
        snapshot, audit_report = adapter.build_snapshot_with_audit(raw_data_df)

        # 2. BİTEMPORAL HISTORICAL AUCTION RATIO HESABI (MA20 / Constant 1.0)
        auction_ratios = self.auction_engine.compute_snapshot_auction_ratios(snapshot)

        # 3. HYBRID VOLATILITY (EGARCH + Parkinson / 30d NAV Std + EWMA)
        volatilities = self.volatility_engine.compute_snapshot_volatilities(snapshot)

        # 4. FREKANS EŞLEŞTİRMELİ BİTEMPORAL GETİRİLER VE KOVARYANS
        bitemporal_res = self.covariance_builder.build_bitemporal_returns(snapshot)
        raw_cov_weekly, kappa_eff = self.covariance_builder.build_weekly_bitemporal_covariance(bitemporal_res)

        # 5. HIGHAM SPECTRAL NEAREST POSITIVE DEFINITE PROJECTION (lambda_min >= 1e-6)
        if not raw_cov_weekly.empty and raw_cov_weekly.shape[0] > 0:
            sealed_cov, spectral_analysis = self.spectral_engine.project_nearest_positive_definite(raw_cov_weekly)
        else:
            sealed_cov = raw_cov_weekly
            spectral_analysis = SpectralAnalysisResult(
                eigenvalues=np.array([]),
                eigenvectors=np.array([[]]),
                min_eigenvalue=1e-6,
                max_eigenvalue=1e-6,
                condition_number=1.0,
                is_positive_definite=True,
                negative_eigenvalues_count=0
            )

        return MultiAssetRiskEngineOutput(
            snapshot=snapshot,
            audit_report=audit_report,
            auction_ratios=auction_ratios,
            volatilities=volatilities,
            covariance_matrix=sealed_cov,
            spectral_analysis=spectral_analysis,
            effective_kappa=kappa_eff
        )
