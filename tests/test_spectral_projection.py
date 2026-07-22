"""
Sprint 1 / Faz 1.4 / Adım 1.4.1 Birim Testi:
HighamSpectralProjectionEngine Özdeğer Ayrıştırma (Eigenvalue Decomposition) Testleri
"""

import unittest
import numpy as np
import pandas as pd
from aether.data.spectral_projection import HighamSpectralProjectionEngine, SpectralAnalysisResult


class TestSpectralProjectionEngine(unittest.TestCase):

    def setUp(self):
        self.engine = HighamSpectralProjectionEngine(min_eigenvalue_floor=1e-6)

    def test_decompose_positive_definite_covariance(self):
        """Pozitif tanımlı kovaryans matrisinin özdeğer ayrıştırması."""
        # 3x3 Pozitif Tanımlı Matris
        symbols = ["THYAO", "GLDTR", "TLY"]
        cov_data = np.array([
            [0.04, 0.01, 0.002],
            [0.01, 0.02, 0.001],
            [0.002, 0.001, 0.005]
        ])
        cov_df = pd.DataFrame(cov_data, index=symbols, columns=symbols)

        res = self.engine.decompose_covariance_matrix(cov_df)

        self.assertEqual(len(res.eigenvalues), 3)
        self.assertGreater(res.min_eigenvalue, 0.0)
        self.assertTrue(res.is_positive_definite)
        self.assertEqual(res.negative_eigenvalues_count, 0)
        self.assertGreater(res.condition_number, 1.0)

        # Reconstitution Testi: V * Lambda * V^T = Cov
        V = res.eigenvectors
        L = np.diag(res.eigenvalues)
        reconstructed = V @ L @ V.T
        np.testing.assert_allclose(reconstructed, cov_data, atol=1e-10)

    def test_detect_non_positive_definite_matrix(self):
        """Negatif özdeğere sahip bozuk/non-PD matrisin tespit edilmesi."""
        symbols = ["THYAO", "GARAN", "GLDTR"]
        # Mükemmel kolineer/bozuk kovaryans matrisi (Negatif özdeğer içeren)
        bad_cov_data = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 4.0],
            [3.0, 4.0, 1.0]
        ])
        cov_df = pd.DataFrame(bad_cov_data, index=symbols, columns=symbols)

    def test_project_nearest_positive_definite(self):
        """Negatif özdeğerli bozuk matrisin Higham spektral projeksiyonla epsilon=1e-6 seviyesinde mühürlenmesi."""
        symbols = ["THYAO", "GARAN", "GLDTR"]
        bad_cov_data = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 4.0],
            [3.0, 4.0, 1.0]
        ])
        cov_df = pd.DataFrame(bad_cov_data, index=symbols, columns=symbols)

        # Higham Projeksiyonunu çalıştır
        sealed_df, post_analysis = self.engine.project_nearest_positive_definite(cov_df, epsilon=1e-6)

        # 1. Minimum özdeğer strictly >= 1e-6 olmalı
        self.assertGreaterEqual(post_analysis.min_eigenvalue, 1e-6 - 1e-12)
        self.assertTrue(post_analysis.is_positive_definite)
        self.assertEqual(post_analysis.negative_eigenvalues_count, 0)

        # 2. Cholesky Ayrıştırması Hatasız Başarılı Olmalı!
        try:
            L = np.linalg.cholesky(sealed_df.values)
            cholesky_success = True
        except np.linalg.LinAlgError:
            cholesky_success = False

        self.assertTrue(cholesky_success, "Mühürlü kovaryans matrisi Cholesky ayrıştırmasından başarıyla geçmelidir!")


if __name__ == "__main__":
    unittest.main()
