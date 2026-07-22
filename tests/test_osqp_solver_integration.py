"""
AetherForecaster-X v35.2 Multi-Asset Master
Katman 3: Valör-Cezalı Bounded Convex QP Solver

Birim ve Entegrasyon Testi: OSQP Solver Stability & Stress Test Suite (Adım 2.4.5)
Farklı piyasa rejimleri (Normal, Kriz/Yüksek Volatilite, Illikit Şok) ve karmaşık kısıt setleri altında
optimizasyon kararlılığının uçtan uca doğrulanması.
"""

import unittest
import numpy as np
import pandas as pd

from aether.optimization.hessian_conditioning import HessianConditioningEngine
from aether.optimization.alpha_signal import AlphaSignalEngine
from aether.optimization.matrix_builder import (
    PortfolioQPBuilder,
    PortfolioQPConfig,
    GroupConstraint
)
from aether.optimization.osqp_solver import OSQPSolverWrapper


class TestOSQPSolverIntegration(unittest.TestCase):
    """
    Adım 2.4.5: Farklı kısıt setleri ve piyasa rejimleri altında OSQP Çözücü Kararlılık Testleri.
    """

    def setUp(self):
        # 10 varlıklı evren (BIST Hisse + TEFAS Fonları)
        self.symbols = [
            "GARAN", "THYAO", "EREGL", "AKBNK", "KCHOL",
            "TEFAS_ALC", "TEFAS_ZPL", "TEFAS_TCD", "TEFAS_MAC", "TEFAS_YAZ"
        ]
        self.n_assets = len(self.symbols)

        self.asset_types = {sym: "TEFAS" if sym.startswith("TEFAS") else "EQUITY" for sym in self.symbols}
        self.hessian_engine = HessianConditioningEngine(theta=100.0, min_eigenvalue_floor=1e-6)
        self.alpha_engine = AlphaSignalEngine()
        self.builder = PortfolioQPBuilder()
        self.solver = OSQPSolverWrapper(eps_abs=1e-5, eps_rel=1e-5, verbose=False)

    def _generate_synthetic_cov(self, base_vol: float = 0.20, noise: float = 0.05) -> pd.DataFrame:
        """Sentetik pozitif tanımlı kovaryans matrisi üretir."""
        np.random.seed(42)
        rnd = np.random.randn(self.n_assets, self.n_assets)
        cov = np.dot(rnd, rnd.T) * (base_vol ** 2) / self.n_assets
        np.fill_diagonal(cov, (base_vol + np.random.uniform(0, noise, size=self.n_assets)) ** 2)
        return pd.DataFrame(cov, index=self.symbols, columns=self.symbols)

    def _create_diag_df(self, values: list[float]) -> pd.DataFrame:
        """Diagonal DataFrame üretici."""
        return pd.DataFrame(np.diag(values), index=self.symbols, columns=self.symbols)

    def test_normal_market_regime_stability(self):
        """
        1. Normal Piyasa Rejimi Kararlılık Testi.
        Standart volatilite (%20) ve likidite koşullarında %100 bütçe tam yatırılmışlığı.
        """
        cov_df = self._generate_synthetic_cov(base_vol=0.20)
        raw_alpha = pd.Series([0.15, 0.12, 0.10, 0.08, 0.05, 0.04, 0.03, 0.02, 0.01, 0.00], index=self.symbols)
        
        # Likidite ve Valör ceza matrisleri
        D_impact = self._create_diag_df([0.001] * 5 + [0.0] * 5)
        D_settlement = self._create_diag_df([0.002] * 5 + [0.001] * 5)

        # 2. Hessian Conditioning
        hess_res = self.hessian_engine.compute_conditioned_hessian(
            sigma_weekly=cov_df,
            d_total=D_impact + D_settlement
        )

        # 3. Alpha Sinyali
        alpha_res = self.alpha_engine.process_alpha_signals(raw_alpha, self.asset_types)

        # 4. Kısıt Yapılandırması (Long-only, hisse başı maks %25)
        config = PortfolioQPConfig(
            target_total_weight=1.0,
            min_asset_weight=0.0,
            max_asset_weight=0.25,
            allow_short=False
        )

        qp_input = self.builder.build_qp_input(hess_res.p_conditioned, alpha_res.scaled_alpha, self.symbols, config)
        res = self.solver.solve(qp_input.P, qp_input.q, qp_input.A, qp_input.l, qp_input.u)

        self.assertTrue(res.is_success)
        self.assertEqual(res.status, "solved")
        self.assertAlmostEqual(float(np.sum(res.x)), 1.0, places=4)
        self.assertTrue(np.all(res.x >= -1e-5))
        self.assertTrue(np.all(res.x <= 0.25 + 1e-5))

    def test_high_volatility_crisis_regime_stability(self):
        """
        2. Yüksek Volatilite / Kriz Rejimi Kararlılık Testi.
        Aşırı volatilite (%80), illikit şok (yüksek impact cezası) ve yüksek condition number altında konverjans.
        """
        cov_df = self._generate_synthetic_cov(base_vol=0.80, noise=0.25)
        raw_alpha = pd.Series([0.50, -0.20, 0.30, -0.40, 0.10, 0.05, 0.02, -0.10, 0.15, 0.00], index=self.symbols)
        
        # Yüksek illikitlik şoku
        D_impact = self._create_diag_df([0.50] * 5 + [0.0] * 5)
        D_settlement = self._create_diag_df([0.05] * 10)

        hess_res = self.hessian_engine.compute_conditioned_hessian(
            sigma_weekly=cov_df,
            d_total=D_impact + D_settlement
        )

        # Condition number capping (kappa <= 10^4) devreye girmiş olmalı
        self.assertLessEqual(hess_res.kappa_conditioned, 1e4 + 1e-2)

        alpha_res = self.alpha_engine.process_alpha_signals(raw_alpha, self.asset_types)

        config = PortfolioQPConfig(
            target_total_weight=1.0,
            min_asset_weight=0.0,
            max_asset_weight=0.30,
            allow_short=False
        )

        qp_input = self.builder.build_qp_input(hess_res.p_conditioned, alpha_res.scaled_alpha, self.symbols, config)
        res = self.solver.solve_with_adaptive_tolerance(
            qp_input.P, qp_input.q, qp_input.A, qp_input.l, qp_input.u,
            target_eps_abs=1e-6, fallback_eps_abs=1e-4
        )

        self.assertTrue(res.is_success)
        self.assertAlmostEqual(float(np.sum(res.x)), 1.0, places=4)
        self.assertTrue(np.all(res.x >= -1e-4))

    def test_overlapping_group_constraints_regime(self):
        """
        3. Karmaşık / Çakışan Grup Kısıtları Rejimi Testi.
        TEFAS Fonları toplamı %10 - %30 arası, BIST Hisse toplamı %70 - %90 arası.
        """
        cov_df = self._generate_synthetic_cov(base_vol=0.25)
        raw_alpha = pd.Series([0.20, 0.15, 0.18, 0.22, 0.10, 0.08, 0.05, 0.06, 0.04, 0.02], index=self.symbols)
        
        D_impact = self._create_diag_df([0.005] * 10)
        D_settlement = self._create_diag_df([0.001] * 10)

        hess_res = self.hessian_engine.compute_conditioned_hessian(
            sigma_weekly=cov_df,
            d_total=D_impact + D_settlement
        )

        alpha_res = self.alpha_engine.process_alpha_signals(raw_alpha, self.asset_types)

        # Grup kısıtları:
        # BIST Hisseleri (İlk 5): Min %70, Max %90
        # TEFAS Fonları (Son 5): Min %10, Max %30
        group_bist = GroupConstraint(
            group_name="BIST_Equities",
            symbols=["GARAN", "THYAO", "EREGL", "AKBNK", "KCHOL"],
            min_weight=0.70,
            max_weight=0.90
        )
        group_tefas = GroupConstraint(
            group_name="TEFAS_Funds",
            symbols=["TEFAS_ALC", "TEFAS_ZPL", "TEFAS_TCD", "TEFAS_MAC", "TEFAS_YAZ"],
            min_weight=0.10,
            max_weight=0.30
        )

        config = PortfolioQPConfig(
            target_total_weight=1.0,
            min_asset_weight=0.0,
            max_asset_weight=0.35,
            group_constraints=[group_bist, group_tefas]
        )

        qp_input = self.builder.build_qp_input(hess_res.p_conditioned, alpha_res.scaled_alpha, self.symbols, config)
        res = self.solver.solve(qp_input.P, qp_input.q, qp_input.A, qp_input.l, qp_input.u)

        self.assertTrue(res.is_success)
        
        # BIST toplam ağırlığı kontrolü [0.70, 0.90]
        bist_weight_sum = np.sum(res.x[:5])
        self.assertGreaterEqual(bist_weight_sum, 0.70 - 1e-4)
        self.assertLessEqual(bist_weight_sum, 0.90 + 1e-4)

        # TEFAS toplam ağırlığı kontrolü [0.10, 0.30]
        tefas_weight_sum = np.sum(res.x[5:])
        self.assertGreaterEqual(tefas_weight_sum, 0.10 - 1e-4)
        self.assertLessEqual(tefas_weight_sum, 0.30 + 1e-4)

    def test_tight_borderline_simplex_regime(self):
        """
        4. Sınır / Dar Simpleks Rejimi Testi.
        10 varlık için her bir varlığın maks ağırlığı tam %10 (10 * 0.10 = 1.0 bütçeyi tam karşılar).
        Çözümün tam eşitlik noktasında (w_i = 0.10, ∀i) sabitlenmesi.
        """
        cov_df = self._generate_synthetic_cov(base_vol=0.15)
        raw_alpha = pd.Series([0.10] * self.n_assets, index=self.symbols)
        
        D_impact = self._create_diag_df([0.001] * 10)
        D_settlement = self._create_diag_df([0.001] * 10)

        hess_res = self.hessian_engine.compute_conditioned_hessian(
            sigma_weekly=cov_df,
            d_total=D_impact + D_settlement
        )

        alpha_res = self.alpha_engine.process_alpha_signals(raw_alpha, self.asset_types)

        # Her varlık maks %10 -> Tek bir geçerli nokta var: [0.10, 0.10, ..., 0.10]
        config = PortfolioQPConfig(
            target_total_weight=1.0,
            min_asset_weight=0.0,
            max_asset_weight=0.10
        )

        qp_input = self.builder.build_qp_input(hess_res.p_conditioned, alpha_res.scaled_alpha, self.symbols, config)
        res = self.solver.solve(qp_input.P, qp_input.q, qp_input.A, qp_input.l, qp_input.u)

        self.assertTrue(res.is_success)
        self.assertTrue(np.allclose(res.x, 0.10, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
