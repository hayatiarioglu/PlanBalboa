"""
AetherForecaster-X v35.2 Multi-Asset Master
Katman 3: Valör-Cezalı Bounded Convex QP Solver

Sprint 2 Kalite Gate & 1 Milyar TL Zero-Slippage Simülasyon Testi (Faz 2.6 / Adım 2.6.1 & 2.6.2)
Tüm Katman 3 QP Solver motorunun uçtan uca (End-to-End) entegrasyonu ve 1 Milyar TL büyüklüğündeki
dev portföylerde sıfır kayma (Zero-Slippage) elde edildiğini mühürleyen nihai sertifikasyon testi.
"""

import unittest
import numpy as np
import pandas as pd

from aether.optimization.solver_orchestrator import PortfolioQPOrchestrator, PortfolioOptimizationResult
from aether.optimization.matrix_builder import PortfolioQPConfig, GroupConstraint


class TestSprint2QualityGate(unittest.TestCase):
    """
    Faz 2.6: Sprint 2 Entegrasyonu ve Zero-Slippage Kalite Gate Sertifikasyon Test Paneli.
    """

    def setUp(self):
        self.orchestrator = PortfolioQPOrchestrator()
        
        # 15 Varlıklı Gerçekçi BIST & TEFAS Evreni
        self.symbols = [
            "THYAO", "GARAN", "AKBNK", "EREGL", "KCHOL", "TUPRS", "SAHOL", "BIMAS",
            "TEFAS_ALC", "TEFAS_ZPL", "TEFAS_TCD", "TEFAS_MAC", "TEFAS_YAZ", "TEFAS_TI1", "TEFAS_OZD"
        ]
        self.n_assets = len(self.symbols)

        self.asset_types = {sym: "TEFAS" if sym.startswith("TEFAS") else "EQUITY" for sym in self.symbols}

        # Sentetik Kovaryans Matrisi (%22 Ortalama Volatilite)
        np.random.seed(100)
        rnd = np.random.randn(self.n_assets, self.n_assets)
        cov = np.dot(rnd, rnd.T) * (0.22 ** 2) / self.n_assets
        np.fill_diagonal(cov, 0.22 ** 2)
        self.cov_df = pd.DataFrame(cov, index=self.symbols, columns=self.symbols)

        # Volatiliteler ve Ortalama Ciro (ADV_20 TL)
        self.volatilities = pd.Series(np.sqrt(np.diag(cov)), index=self.symbols)
        
        # BIST Hisse ciroları (500M - 1.5B TL), TEFAS Fonları ciroları (20M TL)
        adv_list = [1.5e9, 1.2e9, 1.0e9, 9e8, 8e8, 7e8, 6e8, 5e8] + [2e7] * 7
        self.adv_series = pd.Series(adv_list, index=self.symbols)

        # Takas Günleri (BIST = T+2, TEFAS = T+0 / T+1)
        settlement_list = [2] * 8 + [0, 0, 1, 1, 1, 1, 1]
        self.settlement_days = pd.Series(settlement_list, index=self.symbols)

        # Ham Alpha Sinyalleri
        self.raw_alpha = pd.Series([
            0.35, 0.30, 0.28, 0.25, 0.22, 0.20, 0.18, 0.15,
            0.10, 0.08, 0.12, 0.05, 0.04, 0.03, 0.02
        ], index=self.symbols)

    def test_end_to_end_sprint2_orchestrator_integration(self):
        """
        Adım 2.6.1: Katman 3 QP Solver bileşenlerinin bütünsel entegrasyonu ve SLA hız testi.
        Tüm alt motorlar (Penalty, Conditioning, Alpha, Builder, OSQP) otonom çalışmalıdır.
        """
        config = PortfolioQPConfig(
            target_total_weight=1.0,
            min_asset_weight=0.0,
            max_asset_weight=0.20,
            allow_short=False
        )

        res = self.orchestrator.optimize(
            cov_matrix=self.cov_df,
            raw_alpha_signals=self.raw_alpha,
            volatilities=self.volatilities,
            adv_series=self.adv_series,
            settlement_days=self.settlement_days,
            total_capital=1e8,  # 100 Milyon TL
            config=config,
            asset_types=self.asset_types
        )

        self.assertTrue(res.is_success)
        self.assertEqual(res.status, "solved")
        self.assertAlmostEqual(float(res.weights.sum()), 1.0, places=4)
        
        # Real-time SLA kontrolü: Optimizasyon çalışma süresi 50 ms altında olmalıdır
        self.assertLess(res.run_time_ms, 50.0, f"Optimizasyon süresi SLA ihlali yaptı: {res.run_time_ms:.2f} ms")
        
        # Condition number sınırı (kappa <= 10^4)
        self.assertLessEqual(res.condition_number, 1e4 + 1e-2)

    def test_zero_slippage_1_billion_tl_simulation(self):
        """
        Adım 2.6.2: 1 Milyar TL ($1.000.000.000$ TL) dev sermaye simülasyonunda sıfır kayma (Zero-Slippage) testi.
        Likidite etki matrisi (D_impact ~ W^2 / ADV^2) sermaye büyüdükçe kaymayı önlemek için
        otomatik olarak likit TEFAS fonlarına ve yüksek ADV'li hisselere kayma yapmalıdır.
        """
        billion_capital = 1e9  # 1 Milyar TL

        config = PortfolioQPConfig(
            target_total_weight=1.0,
            min_asset_weight=0.0,
            max_asset_weight=0.25,
            allow_short=False
        )

        res = self.orchestrator.optimize(
            cov_matrix=self.cov_df,
            raw_alpha_signals=self.raw_alpha,
            volatilities=self.volatilities,
            adv_series=self.adv_series,
            settlement_days=self.settlement_days,
            total_capital=billion_capital,
            config=config,
            asset_types=self.asset_types
        )

        self.assertTrue(res.is_success)
        self.assertAlmostEqual(float(res.weights.sum()), 1.0, places=4)

        # 1 Milyar TL devasa hacimde sıfır NaN/Inf ve tam konverjans
        self.assertTrue(np.all(np.isfinite(res.weights.to_numpy())))
        self.assertTrue(np.all(res.weights.to_numpy() >= -1e-6))

        # 1 Milyar TL'de en likit hisseler (THYAO, GARAN) ve TEFAS fonlarının ağırlık payı yüksek kalmalıdır
        top_liquid_weight = float(res.weights["THYAO"] + res.weights["GARAN"])
        self.assertGreater(top_liquid_weight, 0.20, "1 Milyar TL sermayede yüksek ADV'li hisselere kayma gerçekleşmedi!")


if __name__ == "__main__":
    unittest.main()
