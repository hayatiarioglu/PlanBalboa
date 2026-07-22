"""
AetherForecaster-X v35.2 Multi-Asset Master
Katman 3: Valör-Cezalı Bounded Convex QP Solver Testleri

Faz 2.2: Hessian Soft-Clipping ve Tikhonov Düzeltmesi (Conditioning) Birim Testleri.
"""

import unittest
import numpy as np
import pandas as pd

from aether.optimization.hessian_conditioning import HessianConditioningEngine, HessianConditioningResult


class TestHessianConditioningEngine(unittest.TestCase):

    def setUp(self):
        self.engine = HessianConditioningEngine(
            theta=100.0,
            max_condition_number=1e4,
            min_eigenvalue_floor=1e-6
        )
        self.symbols = ["GARAN", "THYAO", "ASELS", "FON1"]
        n = len(self.symbols)

        # Temel 4x4 Kovaryans Matrisi (Sigma_weekly)
        cov_values = np.array([
            [0.0004, 0.0001, 0.0001, 0.00005],
            [0.0001, 0.0005, 0.0002, 0.00005],
            [0.0001, 0.0002, 0.0006, 0.0001],
            [0.00005, 0.00005, 0.0001, 0.0002]
        ])
        self.sigma_weekly = pd.DataFrame(cov_values, index=self.symbols, columns=self.symbols)

    def test_soft_clipping_threshold(self):
        """
        Test 1: Aşırı yüksek D_total ceza değerlerinin theta * lambda_max(Sigma) ile
        Soft-Clipping süzgecinden geçirilip sınırladığını doğrular.
        """
        # Aşırı büyük ceza değerleri (100.0 gibi - Kovaryans yanında devasa)
        d_values = np.diag([0.01, 50.0, 500.0, 0.001])
        d_total = pd.DataFrame(d_values, index=self.symbols, columns=self.symbols)

        res = self.engine.compute_conditioned_hessian(self.sigma_weekly, d_total)

        # Sigma'nın max özdeğerini bulalım
        eigvals_sigma = np.linalg.eigvalsh(self.sigma_weekly.values)
        lambda_max_sigma = float(np.max(eigvals_sigma))
        expected_threshold = 100.0 * lambda_max_sigma

        self.assertAlmostEqual(res.theta_threshold, expected_threshold, places=6)
        
        # P_raw içerisindeki diyagonaller Soft-Clipping sınırı aşılmayacak şekilde kurulmalı
        # P = 2 * (Sigma + D_clipped) => D_clipped <= theta_threshold
        # P_raw[2,2] <= 2 * (Sigma[2,2] + theta_threshold)
        max_p22 = 2.0 * (self.sigma_weekly.iloc[2, 2] + expected_threshold)
        self.assertTrue(res.p_raw.iloc[2, 2] <= max_p22 + 1e-9)

    def test_condition_number_bound(self):
        """
        Test 2: D_total ceza matrisi ve kucuk ozdegerler yuzunden Condition Number kappa(P) > 10^5 seviyelerine
        firlasa bile, Tikhonov düzeltmesi ile strictly kappa(P_conditioned) <= 10^4 yapildigini dogrular.
        """
        # Kötü koşullanmış kovaryans matrisi (lambda_max = 0.01, lambda_min = 1e-6)
        ill_cov = np.diag([0.01, 0.001, 0.0001, 1e-6])
        sigma_ill = pd.DataFrame(ill_cov, index=self.symbols, columns=self.symbols)

        # D_total Soft-Clipping tavanına yakın değerler ekler
        d_values = np.diag([1.0, 0.0, 0.0, 0.0])
        d_total = pd.DataFrame(d_values, index=self.symbols, columns=self.symbols)

        res = self.engine.compute_conditioned_hessian(sigma_ill, d_total)

        # Koşullandırılmış matrisin durum sayısı <= 10^4 olmalı
        self.assertTrue(res.kappa_conditioned <= 1e4 + 1e-3, f"Kappa unexpected: {res.kappa_conditioned}")
        self.assertTrue(res.epsilon_tikhonov > 0.0, "Tikhonov düzenlileştirmesi devreye girmeliydi!")

    def test_positive_definiteness_guarantee(self):
        """
        Test 3: P_conditioned matrisinin en küçük özdeğerinin strictly >= 1e-6
        olduğunu ve Pozitif Tanımlı (is_positive_definite = True) olduğunu kanıtlar.
        """
        d_values = np.diag([0.001, 0.002, 0.005, 0.0005])
        d_total = pd.DataFrame(d_values, index=self.symbols, columns=self.symbols)

        res = self.engine.compute_conditioned_hessian(self.sigma_weekly, d_total)

        self.assertTrue(res.is_positive_definite)
        self.assertTrue(res.lambda_min >= 1e-6)

    def test_identity_when_well_conditioned(self):
        """
        Test 4: Zaten iyi koşullanmış (kappa < 10^4) bir matriste Tikhonov eklenmediğini (eps ~= 0.0) doğrular.
        """
        d_values = np.diag([0.0001, 0.0002, 0.0001, 0.0001])
        d_total = pd.DataFrame(d_values, index=self.symbols, columns=self.symbols)

        res = self.engine.compute_conditioned_hessian(self.sigma_weekly, d_total)

        self.assertAlmostEqual(res.epsilon_tikhonov, 0.0, places=7)
        self.assertTrue(res.kappa_conditioned <= 1e4)


    def test_alignment_and_nan_protection(self):
        """
        Test 5: Sigma ve D_total matrislerindeki index kaymalarının doğru hizalandığını ve 
        Sigma içinde NaN geldiğinde sistemin patlamak yerine kontrollü hata fırlattığını doğrular.
        """
        # Kaymış indexlere sahip D_total
        d_values = np.diag([0.001, 0.002])
        d_misaligned = pd.DataFrame(d_values, index=["THYAO", "FON1"], columns=["THYAO", "FON1"])
        
        # Hata fırlatmadan çalışmalı, eksik indexler (GARAN, ASELS) 0.0 ile dolmalı
        res = self.engine.compute_conditioned_hessian(self.sigma_weekly, d_misaligned)
        self.assertEqual(res.p_conditioned.shape, (4, 4))
        
        # Sigma içinde NaN varsa ValueError fırlatmalı
        sigma_nan = self.sigma_weekly.copy()
        sigma_nan.iloc[0, 1] = np.nan
        with self.assertRaises(ValueError):
            self.engine.compute_conditioned_hessian(sigma_nan, d_misaligned)


if __name__ == "__main__":
    unittest.main()
