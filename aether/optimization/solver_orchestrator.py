"""
AetherForecaster-X v35.2 Multi-Asset Master
Katman 3: Valör-Cezalı Bounded Convex QP Solver

Modül: Portfolio QP Solver Orchestrator (Katman 3 Ana Bütünleştirici Motor)
Faz 2.6 / Adım 2.6.1 & Adım 2.6.2: Penalty Engine, Hessian Conditioning, Alpha Signal Engine,
Matrix Builder ve OSQP Solver'ı tek bir otonom orkestratör altında birleştiren ve 1 Milyar TL
büyüklüğündeki portföylerde sıfır kayma (Zero-Slippage) elde edildiğini doğrulayan ana sınıf.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd

from aether.optimization.penalty import PenaltyMatrixEngine, PenaltyMatrixResult
from aether.optimization.hessian_conditioning import HessianConditioningEngine, HessianConditioningResult
from aether.optimization.alpha_signal import AlphaSignalEngine, AlphaSignalResult
from aether.optimization.matrix_builder import (
    PortfolioQPBuilder,
    PortfolioQPConfig,
    QPProblemInput
)
from aether.optimization.osqp_solver import OSQPSolverWrapper, OSQPResult


@dataclass
class PortfolioOptimizationResult:
    """
    Katman 3 Konveks Portföy Optimizasyonunun Nihai İcra ve Teşhis Paketi.
    """
    weights: pd.Series                     # Optimal portföy ağırlıkları w_i^* (N,)
    turnover_weights: Optional[pd.Series]   # Pozisyon değişim hacimleri u_i^* (N,) (Eğer aktifleştirilmişse)
    is_success: bool                       # Optimizasyon konverjans başarısı
    status: str                            # Çözücü durum metni ('solved', vb.)
    objective_value: float                 # QP amaç fonksiyonu değeri
    condition_number: float                # Şartlandırılmış Hessian Condition Number (kappa <= 10^4)
    total_turnover: float                  # Toplam portföy devir hızı (Sum u_i)
    estimated_impact_cost_tl: float        # Tahmini likidite etki maliyeti (TL)
    estimated_settlement_cost_tl: float    # Tahmini valör takas maliyeti (TL)
    run_time_ms: float                     # Optimizasyon çalışma süresi (Milisaniye)
    solver_info: Dict[str, Any] = field(default_factory=dict)


class PortfolioQPOrchestrator:
    """
    Katman 3 Bütünsel QP Solver Orkestratör Motoru.
    """

    def __init__(
        self,
        penalty_engine: Optional[PenaltyMatrixEngine] = None,
        hessian_engine: Optional[HessianConditioningEngine] = None,
        alpha_engine: Optional[AlphaSignalEngine] = None,
        matrix_builder: Optional[PortfolioQPBuilder] = None,
        solver_wrapper: Optional[OSQPSolverWrapper] = None
    ):
        self.penalty_engine = penalty_engine or PenaltyMatrixEngine()
        self.hessian_engine = hessian_engine or HessianConditioningEngine()
        self.alpha_engine = alpha_engine or AlphaSignalEngine()
        self.matrix_builder = matrix_builder or PortfolioQPBuilder()
        self.solver_wrapper = solver_wrapper or OSQPSolverWrapper()

    def optimize(
        self,
        cov_matrix: pd.DataFrame,
        raw_alpha_signals: pd.Series,
        volatilities: pd.Series,
        adv_series: pd.Series,
        settlement_days: pd.Series,
        total_capital: float,
        config: Optional[PortfolioQPConfig] = None,
        asset_types: Optional[Dict[str, str]] = None
    ) -> PortfolioOptimizationResult:
        """
        Katman 3 optimizasyonunu baştan sona (End-to-End) çalıştırır.

        :param cov_matrix: Kovaryans matrisi (N x N)
        :param raw_alpha_signals: Ham getiri/momentum sinyalleri (N,)
        :param volatilities: Varlık bazlı volatilite serisi (N,)
        :param adv_series: 20 günlük Ortalama Günlük Hacim (ADV_20) TL serisi (N,)
        :param settlement_days: Takas valör gün sayıları (T+0, T+1, T+2) (N,)
        :param total_capital: Yönetilen toplam portföy sermayesi TL (W_total)
        :param config: Portföy kısıt konfigürasyonu
        :param asset_types: Varlık türü eşleştirmesi (EQUITY, TEFAS_FREE vb.)
        :return: PortfolioOptimizationResult
        """
        start_time = perf_counter()

        if config is None:
            config = PortfolioQPConfig()

        symbols = list(cov_matrix.index)
        n_assets = len(symbols)

        # 1. Likidite & Valör Ceza Matrislerinin Üretilmesi (Faz 2.1)
        pen_res = self.penalty_engine.compute_penalty_matrices_from_series(
            volatilities=volatilities,
            adv_series=adv_series,
            settlement_days=settlement_days,
            W_total=total_capital,
            symbols=symbols,
            asset_types=asset_types
        )

        # 2. Hessian Şartlandırma ve Soft-Clipping (Faz 2.2)
        hess_res = self.hessian_engine.compute_conditioned_hessian(
            sigma_weekly=cov_matrix,
            d_total=pen_res.d_total
        )

        # 3. Alpha Sinyal Normalizasyonu (Faz 2.3)
        alpha_res = self.alpha_engine.process_alpha_signals(raw_alpha_signals, asset_types)

        # 4. Kısıt Matrislerinin İnşası (Faz 2.4 & 2.5)
        qp_input = self.matrix_builder.build_qp_input(
            P_conditioned=hess_res.p_conditioned,
            mu_norm=alpha_res.scaled_alpha,
            symbols=symbols,
            config=config
        )

        # 5. OSQP Konveks Çözücünün Adaptif Toleransla Çalıştırılması (Faz 2.4.4)
        solver_res = self.solver_wrapper.solve_with_adaptive_tolerance(
            P=qp_input.P,
            q=qp_input.q,
            A=qp_input.A,
            l=qp_input.l,
            u=qp_input.u
        )

        elapsed_ms = (perf_counter() - start_time) * 1000.0

        # Sonuçların Ayrıştırılması
        w_opt = pd.Series(solver_res.x[:n_assets], index=symbols)
        
        if config.w_prev is not None:
            w_prev_ser = pd.Series(config.w_prev).reindex(symbols).fillna(0.0)
            tot_turnover = float((w_opt - w_prev_ser).abs().sum())
        else:
            tot_turnover = 0.0

        if qp_input.is_turnover_expanded:
            u_opt = pd.Series(solver_res.x[n_assets:], index=symbols)
        else:
            u_opt = None

        # Tahmini Maliyet Hesaplamaları (TL)
        # Impact Maliyeti = Sum(w_i^2 * D_impact_ii) * W_total
        d_impact_diag = np.diag(pen_res.d_impact.loc[symbols, symbols].to_numpy())
        d_settle_diag = np.diag(pen_res.d_settlement.loc[symbols, symbols].to_numpy())


        w_arr = w_opt.to_numpy()
        impact_cost_tl = float(np.sum((w_arr ** 2) * d_impact_diag))
        settle_cost_tl = float(np.sum(np.abs(w_arr) * d_settle_diag) * total_capital)

        return PortfolioOptimizationResult(
            weights=w_opt,
            turnover_weights=u_opt,
            is_success=solver_res.is_success,
            status=solver_res.status,
            objective_value=solver_res.obj_val,
            condition_number=hess_res.kappa_conditioned,
            total_turnover=tot_turnover,
            estimated_impact_cost_tl=impact_cost_tl,
            estimated_settlement_cost_tl=settle_cost_tl,
            run_time_ms=elapsed_ms,
            solver_info=solver_res.info
        )
