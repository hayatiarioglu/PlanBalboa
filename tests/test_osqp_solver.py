"""
Unit Tests for OSQPSolverWrapper (Faz 2.4.1)
"""

import unittest
import numpy as np
import pandas as pd
from scipy import sparse

from aether.optimization.osqp_solver import OSQPSolverWrapper, OSQPResult


class TestOSQPSolverWrapper(unittest.TestCase):

    def setUp(self):
        self.wrapper = OSQPSolverWrapper(verbose=False, eps_abs=1e-6, eps_rel=1e-6)

    def test_osqp_basic_optimization(self):
        """Basit 2 varlıklı portföy optimizasyonu testi."""
        # Hessian P = 2 * [[0.04, 0.01], [0.01, 0.09]]
        # Alpha q = [-0.10, -0.05] (Minimizing -mu is maximizing mu)
        P = np.array([[0.08, 0.02], [0.02, 0.18]])
        q = np.array([-0.10, -0.05])

        # Kısıtlar:
        # 1) Toplam ağırlık = 1.0  (1.0 <= x1 + x2 <= 1.0)
        # 2) Kutusal kısıtlar (0.0 <= x1 <= 1.0, 0.0 <= x2 <= 1.0)
        A = np.array([
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        l = np.array([1.0, 0.0, 0.0])
        u = np.array([1.0, 1.0, 1.0])

        res = self.wrapper.solve(P, q, A, l, u)

        self.assertTrue(res.is_success)
        self.assertEqual(res.status, "solved")
        self.assertAlmostEqual(np.sum(res.x), 1.0, places=4)
        self.assertTrue(np.all(res.x >= -1e-5))
        self.assertTrue(np.all(res.x <= 1.0 + 1e-5))

    def test_osqp_pandas_input(self):
        """Pandas DataFrame ve Series tipinde girdiler ile sınama."""
        assets = ["GARAN", "THYAO", "TEFAS_ALC"]
        P_df = pd.DataFrame([
            [0.10, 0.02, 0.00],
            [0.02, 0.12, 0.00],
            [0.00, 0.00, 0.01]
        ], index=assets, columns=assets)

        q_ser = pd.Series([-0.15, -0.12, -0.03], index=assets)

        # Bütçe eşitlik kısıtı: sum(x) = 1.0
        A_df = pd.DataFrame([[1.0, 1.0, 1.0]], columns=assets)
        l_ser = pd.Series([1.0])
        u_ser = pd.Series([1.0])

        res = self.wrapper.solve(P_df, q_ser, A_df, l_ser, u_ser)

        self.assertTrue(res.is_success)
        self.assertEqual(len(res.x), 3)
        self.assertAlmostEqual(float(np.sum(res.x)), 1.0, places=4)

    def test_osqp_nan_inf_protection(self):
        """Girdilerde NaN veya Inf olduğunda ValueError fırlatıldığının kanıtlanması."""
        P = np.array([[1.0, 0.0], [0.0, 1.0]])
        q = np.array([np.nan, 0.1])
        A = np.eye(2)
        l = np.array([0.0, 0.0])
        u = np.array([1.0, 1.0])

        with self.assertRaises(ValueError):
            self.wrapper.solve(P, q, A, l, u)

        q_inf = np.array([np.inf, 0.1])
        with self.assertRaises(ValueError):
            self.wrapper.solve(P, q_inf, A, l, u)

    def test_osqp_dimension_mismatch(self):
        """Boyut uyumsuzluklarında ValueError fırlatılması."""
        P = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # Kare değil
        q = np.array([0.1, 0.2])
        A = np.eye(2)
        l = np.array([0.0, 0.0])
        u = np.array([1.0, 1.0])

        with self.assertRaises(ValueError):
            self.wrapper.solve(P, q, A, l, u)

    def test_osqp_invalid_bounds(self):
        """l > u durumunda ValueError fırlatılması."""
        P = np.eye(2)
        q = np.array([0.1, 0.2])
        A = np.eye(2)
        l = np.array([2.0, 0.0])  # l[0] = 2.0 > u[0] = 1.0
        u = np.array([1.0, 1.0])

        with self.assertRaises(ValueError):
            self.wrapper.solve(P, q, A, l, u)

    def test_osqp_infeasible_problem(self):
        """Çözümsüz (infeasible) kısıtlarda is_success=False ve durum tespiti."""
        P = np.eye(2)
        q = np.array([0.0, 0.0])
        # x1 + x2 >= 2.0  ve  x1 + x2 <= 1.0 (Çelişkili)
        A = np.array([[1.0, 1.0], [1.0, 1.0]])
        l = np.array([2.0, -np.inf])
        u = np.array([np.inf, 1.0])

        res = self.wrapper.solve(P, q, A, l, u)
        self.assertFalse(res.is_success)
        self.assertIn("infeasible", res.status.lower())

    def test_osqp_persistent_setup_update_and_warm_start(self):
        """Faz 2.4.4: Persistent setup, vector update (q, l, u), update_settings ve warm_start testi."""
        P = np.array([[0.08, 0.02], [0.02, 0.18]])
        q = np.array([-0.10, -0.05])
        A = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        l = np.array([1.0, 0.0, 0.0])
        u = np.array([1.0, 1.0, 1.0])

        # Setup
        self.wrapper.setup(P, q, A, l, u)
        res1 = self.wrapper.solve_prepared()
        self.assertTrue(res1.is_success)

        # Settings Update (eps_abs, eps_rel)
        self.wrapper.update_settings(eps_abs=1e-7, eps_rel=1e-7)
        self.assertEqual(self.wrapper.eps_abs, 1e-7)

        # Warm start ile başlangıç noktası besleme
        self.wrapper.warm_start(x=res1.x)

        # Vector update (Alpha sinyali q güncellendi)
        q_new = np.array([-0.05, -0.20])
        self.wrapper.update_vectors(q=q_new)

        res2 = self.wrapper.solve_prepared()
        self.assertTrue(res2.is_success)
        # Yeni q sinyaline göre x[1] (ikinci varlık) daha yüksek ağırlık almalıdır
        self.assertGreater(res2.x[1], res1.x[1])

    def test_osqp_solve_with_adaptive_tolerance(self):
        """Faz 2.4.4: Adaptif tolerans fallback mekanizması testi."""
        P = np.array([[0.10, 0.00], [0.00, 0.10]])
        q = np.array([-0.10, -0.10])
        A = np.array([[1.0, 1.0]])
        l = np.array([1.0])
        u = np.array([1.0])

        res = self.wrapper.solve_with_adaptive_tolerance(
            P, q, A, l, u,
            target_eps_abs=1e-8,
            target_eps_rel=1e-8,
            fallback_eps_abs=1e-4,
            fallback_eps_rel=1e-4
        )
        self.assertTrue(res.is_success)


if __name__ == "__main__":
    unittest.main()

