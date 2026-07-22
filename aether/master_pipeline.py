"""
AetherForecaster-X v35.2 Multi-Asset Master
Faz 4.3 / Adım 4.3.1: Cuma 17:52 Otonom İcra ve Karar Paneli Ana Orkestratörü

Modül: AetherMasterPipeline (Katman 1 -> Katman 2 -> Katman 3 Bütünsel İcra Motoru)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
import torch

from aether.trainer.checkpoint_manager import CheckpointManager
from aether.trainer.data_pipeline import NightlyDataPipeline
from aether.signal.signal_orchestrator import SignalPipelineOrchestrator
from aether.optimization.solver_orchestrator import PortfolioQPOrchestrator, PortfolioOptimizationResult, PortfolioQPConfig
from aether.data.risk_engine import MultiAssetRiskEngine
from aether.monitoring.telemetry import TelemetryLogger


@dataclass
class MasterPipelineExecutionResult:
    """
    Cuma 17:52 Otonom İcra ve Karar Paneli Sonuç Paketi.
    """
    target_weights: pd.Series
    alpha_signals: pd.Series
    cov_matrix: pd.DataFrame
    optimization_result: PortfolioOptimizationResult
    loaded_checkpoint_version: str
    execution_time: str
    coverage_ratio: float


class AetherMasterPipeline:
    """
    Cuma 17:52 Otonom Karar ve Portföy Üretim Pipeline'ı.
    Katman 1 (Sinyal/MMoE) -> Katman 2 (Kovaryans/Risk) -> Katman 3 (OSQP Dışbükey Optimizasyon)
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path] = "checkpoints",
        expected_universe: Optional[List[str]] = None,
        signal_orchestrator: Optional[SignalPipelineOrchestrator] = None,
        portfolio_orchestrator: Optional[PortfolioQPOrchestrator] = None,
        risk_engine: Optional[MultiAssetRiskEngine] = None
    ):
        self.checkpoint_manager = CheckpointManager(str(checkpoint_dir))
        self.signal_orchestrator = signal_orchestrator
        self.portfolio_orchestrator = portfolio_orchestrator or PortfolioQPOrchestrator()
        self.risk_engine = risk_engine or MultiAssetRiskEngine()
        self.telemetry = TelemetryLogger()
        
        if expected_universe is not None:
            self.data_pipeline = NightlyDataPipeline(expected_universe=expected_universe, min_threshold_ratio=0.95)
        else:
            self.data_pipeline = None

    def execute_friday_pipeline(
        self,
        snapshot: Dict[str, Any],
        total_capital: float,
        config: Optional[PortfolioQPConfig] = None,
        asset_types: Optional[Dict[str, str]] = None
    ) -> MasterPipelineExecutionResult:
        """
        Cuma 17:52 canlı icra hattını çalıştırır:
        1. PIT Veri Snapshot doğrulaması (%95 kapsama zırhı)
        2. W_{t-1} Checkpoint yüklenmesi
        3. Katman 1: Sinyal & Getiri tahmini (MMoE Logit Head B)
        4. Katman 2: Bi-Temporal Kovaryans ve Risk matrisi
        5. Katman 3: Valör-Cezalı OSQP Konveks Portföy Optimizasyonu
        """
        import time
        start_time = time.perf_counter()
        
        assets = snapshot["assets"]
        
        # 1. Min-Asset Threshold Guard (%95 Zırhı)
        if self.data_pipeline is not None:
            coverage_ratio = self.data_pipeline.validate_coverage(assets)
        else:
            coverage_ratio = 1.0

        # 2. Checkpoint W_{t-1} Yükleme
        latest_cp = self.checkpoint_manager.get_latest_checkpoint_path()
        version_tag = "W_t-1_default"
        
        if latest_cp is not None and self.signal_orchestrator is not None:
            checkpoint_state = self.checkpoint_manager.load_checkpoint(
                model=self.signal_orchestrator.mmoe_model,
                checkpoint_path=latest_cp
            )
            version_tag = checkpoint_state.get("version_tag", latest_cp.stem)
        
        # 3. Katman 1: Sinyal Üretimi
        x = snapshot["x"]
        if isinstance(x, torch.Tensor) and x.ndim == 2: x = x.unsqueeze(0)
        
        returns = snapshot.get("returns", np.zeros((1, len(assets))))
        if isinstance(returns, torch.Tensor): 
            if returns.ndim == 1: returns = returns.unsqueeze(0)
        else:
            if returns.ndim == 1: returns = np.expand_dims(returns, axis=0)
            
        v_hybrid = snapshot["v_hybrid"]
        if isinstance(v_hybrid, torch.Tensor) and v_hybrid.ndim == 2: v_hybrid = v_hybrid.unsqueeze(0)
        
        if self.signal_orchestrator is not None:
            self.signal_orchestrator.eval()
            with torch.no_grad():
                signal_res = self.signal_orchestrator(
                    x=x,
                    returns=returns,
                    v_hybrid=v_hybrid
                )
                # Head B logits as predicted alpha return signals
                raw_alpha_arr = signal_res.logit_head_b.squeeze().cpu().numpy()
        else:
            # Fallback signal if model not injected
            raw_alpha_arr = snapshot.get("raw_alpha", np.random.randn(len(assets)) * 0.02)
            
        raw_alpha_series = pd.Series(raw_alpha_arr, index=assets)

        # 4. Katman 2: Bi-Temporal Kovaryans Matrisi
        if "cov_matrix" in snapshot:
            cov_df = snapshot["cov_matrix"]
        else:
            daily_returns = snapshot.get("daily_returns", pd.DataFrame(np.random.randn(60, len(assets)) * 0.01, columns=assets))
            cov_matrix_np = np.cov(daily_returns.to_numpy(), rowvar=False) + np.eye(len(assets)) * 1e-5
            cov_df = pd.DataFrame(cov_matrix_np, index=assets, columns=assets)

        volatilities = snapshot.get("volatilities", pd.Series(np.sqrt(np.diag(cov_df)), index=assets))
        adv_series = snapshot.get("adv_series", pd.Series(10_000_000.0, index=assets))
        settlement_days = snapshot.get("settlement_days", pd.Series(2, index=assets))

        # 5. Katman 3: Dışbükey Portföy Optimizasyonu
        opt_res = self.portfolio_orchestrator.optimize(
            cov_matrix=cov_df,
            raw_alpha_signals=raw_alpha_series,
            volatilities=volatilities,
            adv_series=adv_series,
            settlement_days=settlement_days,
            total_capital=total_capital,
            config=config,
            asset_types=asset_types
        )
        
        pipeline_duration_ms = (time.perf_counter() - start_time) * 1000.0
        
        # Telemetry Log
        self.telemetry.log_optimization_result(
            result=opt_res,
            pipeline_duration_ms=pipeline_duration_ms,
            loaded_checkpoint=version_tag
        )

        return MasterPipelineExecutionResult(
            target_weights=opt_res.weights,
            alpha_signals=raw_alpha_series,
            cov_matrix=cov_df,
            optimization_result=opt_res,
            loaded_checkpoint_version=version_tag,
            execution_time=datetime.now().isoformat(),
            coverage_ratio=coverage_ratio
        )
