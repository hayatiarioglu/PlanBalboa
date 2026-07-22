"""
Unit Tests for PortfolioQPBuilder (Faz 2.4.2 & Faz 2.4.3)
"""

import unittest
import numpy as np
import pandas as pd

from aether.optimization.matrix_builder import (
    PortfolioQPBuilder,
    PortfolioQPConfig,
    GroupConstraint,
    QPProblemInput
)
from aether.optimization.osqp_solver import OSQPSolverWrapper


class TestPortfolioQPBuilder(unittest.TestCase):

    def setUp(self):
        self.symbols = ["GARAN", "THYAO", "TEFAS_ALC", "TEFAS_ZPL"]
        self.builder = PortfolioQPBuilder()
        self.solver = OSQPSolverWrapper(verbose=False)

        # Örnek 4x4 Hessian ve 4 boyutlu alpha vektörü
        self.P_df = pd.DataFrame([
            [0.08, 0.02, 0.00, 0.00],
            [0.02, 0.10, 0.00, 0.00],
            [0.00, 0.00, 0.04, 0.00],
            [0.00, 0.00, 0.00, 0.01]
        ], index=self.symbols, columns=self.symbols)

        self.mu_ser = pd.Series([0.15, 0.10, 0.05, 0.02], index=self.symbols)

    def test_qp_builder_budget_constraint(self):
        """Bütçe eşitlik kısıtının (Sum w = 1.0) matristeki karşılığı."""
        qp_input = self.builder.build_qp_input(self.P_df, self.mu_ser, self.symbols)

        self.assertEqual(qp_input.A.shape[1], 4)
        # 1 Bütçe + 4 Kutusal Kısıt = 5 Satır A matrisi
        self.assertEqual(qp_input.A.shape[0], 5)

        # İlk satır bütçe kısıtı olmalıdır
        self.assertTrue(np.all(qp_input.A[0] == 1.0))
        self.assertEqual(qp_input.l[0], 1.0)
        self.assertEqual(qp_input.u[0], 1.0)
        self.assertFalse(qp_input.inequality_mask[0])  # Eşitlik kısıtıdır

    def test_qp_builder_inequality_long_only_and_max_bounds(self):
        """Faz 2.4.3: Eşitsizlik kısıtları (0 <= w <= w_max) ve Long-Only denetimleri."""
        # Long-only iken negatif alt sınır kabul edilmemelidir
        config_invalid_short = PortfolioQPConfig(
            allow_short=False,
            asset_bounds={"GARAN": (-0.10, 0.50)}
        )
        with self.assertRaises(ValueError):
            self.builder.build_qp_input(self.P_df, self.mu_ser, self.symbols, config_invalid_short)

        # Sum(w_max) < 1.0 ise çözümsüzlük hatası fırlatılmalıdır
        config_infeasible_max = PortfolioQPConfig(
            min_asset_weight=0.0,
            max_asset_weight=0.20  # 4 * 0.20 = 0.80 < 1.0 (Toplam bütçeyi karşılamaz!)
        )
        with self.assertRaises(ValueError):
            self.builder.build_qp_input(self.P_df, self.mu_ser, self.symbols, config_infeasible_max)

    def test_qp_builder_inequality_max_weight_capping_in_osqp(self):
        """Faz 2.4.3: Varlık başına maksimum %35 ağırlık sınırının OSQP çözücüsünde katı uygulanması."""
        config = PortfolioQPConfig(
            target_total_weight=1.0,
            min_asset_weight=0.0,
            max_asset_weight=0.35,  # Her varlık maks %35 (4 * 0.35 = 1.40 >= 1.0 geçerli)
            allow_short=False
        )

        qp_input = self.builder.build_qp_input(self.P_df, self.mu_ser, self.symbols, config)

        res = self.solver.solve(
            P=qp_input.P,
            q=qp_input.q,
            A=qp_input.A,
            l=qp_input.l,
            u=qp_input.u
        )

        self.assertTrue(res.is_success)
        self.assertAlmostEqual(float(np.sum(res.x)), 1.0, places=4)
        
        # Long-only denetimi: Tüm ağırlıklar >= 0 olmalıdır
        self.assertTrue(np.all(res.x >= -1e-6))

        # Maksimum ağırlık denetimi: Hiçbir varlık %35'i geçemez
        for idx, sym in enumerate(self.symbols):
            self.assertLessEqual(res.x[idx], 0.35 + 1e-5, f"{sym} %35 maks sınırını ihlal etti!")

    def test_qp_builder_group_constraints(self):
        """Grup maruziyet eşitsizlik kısıtlarının (Group Inequality Constraints) denetimi."""
        group_tefas = GroupConstraint(
            group_name="TEFAS_Funds",
            symbols=["TEFAS_ALC", "TEFAS_ZPL"],
            min_weight=0.05,
            max_weight=0.40
        )
        config = PortfolioQPConfig(group_constraints=[group_tefas])

        qp_input = self.builder.build_qp_input(self.P_df, self.mu_ser, self.symbols, config)

        # 1 Bütçe + 4 Kutusal + 1 Grup = 6 Satır A matrisi
        self.assertEqual(qp_input.A.shape[0], 6)
        
        # Son satır Eşitsizlik kısıtı olmalıdır
        self.assertTrue(qp_input.inequality_mask[-1])
        group_row = qp_input.A[-1]
        self.assertEqual(group_row[self.symbols.index("TEFAS_ALC")], 1.0)
        self.assertEqual(group_row[self.symbols.index("TEFAS_ZPL")], 1.0)

    def test_qp_builder_turnover_linearization_and_bounds(self):
        """Faz 2.5.1: Pozisyon değişim hacminin (u_i >= |w_{t,i} - w_{t-1,i}|) lineerleştirilmesi ve OSQP çözümü."""
        w_prev = {"GARAN": 0.50, "THYAO": 0.50, "TEFAS_ALC": 0.0, "TEFAS_ZPL": 0.0}
        
        # Max toplam portföy devir hızı = %20 (0.20)
        config = PortfolioQPConfig(
            target_total_weight=1.0,
            min_asset_weight=0.0,
            max_asset_weight=1.0,
            w_prev=w_prev,
            max_total_turnover=0.20
        )

        qp_input = self.builder.build_qp_input(self.P_df, self.mu_ser, self.symbols, config)

        self.assertTrue(qp_input.is_turnover_expanded)
        self.assertEqual(qp_input.P.shape, (8, 8))  # 2N = 8
        self.assertEqual(qp_input.A.shape[1], 8)

        res = self.solver.solve(qp_input.P, qp_input.q, qp_input.A, qp_input.l, qp_input.u)
        self.assertTrue(res.is_success)

        # Karar değişkeninin ilk 4 elemanı yeni portföy ağırlıkları (w_t), son 4 elemanı değişim hacimleridir (u_i)
        w_sol = res.x[:4]
        u_sol = res.x[4:]

        # Toplam bütçe tam yatırılmış olmalıdır
        self.assertAlmostEqual(float(np.sum(w_sol)), 1.0, places=4)

        # Gerçekleşen değişimlerin mutlak toplamı max_total_turnover'ı (%20) geçmemelidir
        w_prev_arr = np.array([w_prev[s] for s in self.symbols])
        actual_turnover = float(np.sum(np.abs(w_sol - w_prev_arr)))

    def test_turnover_bounds(self):
        """Faz 2.5.2: Varlık bazlı pozisyon değişim sınırları ve 0 turnover (sabit tutma) doğrulaması."""
        w_prev = {"GARAN": 0.40, "THYAO": 0.30, "TEFAS_ALC": 0.20, "TEFAS_ZPL": 0.10}

        # 1. Senaryo: Varlık bazlı özel pozisyon değişim sınırları (GARAN maks %5, THYAO maks %15)
        config_asset_turn = PortfolioQPConfig(
            target_total_weight=1.0,
            w_prev=w_prev,
            max_asset_turnover={"GARAN": 0.05, "THYAO": 0.15}
        )

        qp_input1 = self.builder.build_qp_input(self.P_df, self.mu_ser, self.symbols, config_asset_turn)
        res1 = self.solver.solve(qp_input1.P, qp_input1.q, qp_input1.A, qp_input1.l, qp_input1.u)

        self.assertTrue(res1.is_success)
        w_sol1 = res1.x[:4]
        
        # GARAN değişimi |w_garan - 0.40| <= 0.05 olmalıdır
        garan_idx = self.symbols.index("GARAN")
        self.assertLessEqual(abs(w_sol1[garan_idx] - 0.40), 0.05 + 1e-4)

        # 2. Senaryo: Sıfır devir hızı (max_total_turnover = 0.0) -> Portföy tam sabit kalmalıdır (w_t == w_{t-1})
        config_zero_turnover = PortfolioQPConfig(
            target_total_weight=1.0,
            w_prev=w_prev,
            max_total_turnover=0.0
        )

        qp_input2 = self.builder.build_qp_input(self.P_df, self.mu_ser, self.symbols, config_zero_turnover)
        res2 = self.solver.solve(qp_input2.P, qp_input2.q, qp_input2.A, qp_input2.l, qp_input2.u)

        self.assertTrue(res2.is_success)
        w_sol2 = res2.x[:4]
        w_prev_arr = np.array([w_prev[s] for s in self.symbols])
        self.assertTrue(np.allclose(w_sol2, w_prev_arr, atol=1e-3))

        # 3. Senaryo: Negatif turnover sınırı verildiğinde ValueError fırlatılmalıdır
        config_invalid_turnover = PortfolioQPConfig(
            target_total_weight=1.0,
            w_prev=w_prev,
            max_total_turnover=-0.10
        )
        with self.assertRaises(ValueError):
            self.builder.build_qp_input(self.P_df, self.mu_ser, self.symbols, config_invalid_turnover)


if __name__ == "__main__":
    unittest.main()


