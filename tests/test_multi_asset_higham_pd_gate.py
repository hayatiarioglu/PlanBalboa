"""
Sprint 1 / Faz 1.4 / Adım 1.4.3 Kalite Testi:
test_multi_asset_higham_pd()
Karma varlık kovaryans matrisinde lambda_min >= 1e-6 pozitif tanımlılık garantisini,
Cholesky faktörizasyonu sıfır hatayı ve Katman 3 QP Solver korumasını doğrulayan Kalite Gate testi.
"""

import unittest
import numpy as np
import pandas as pd

from aether.data.spectral_projection import HighamSpectralProjectionEngine, SpectralAnalysisResult


class TestMultiAssetHighamPDGate(unittest.TestCase):

    def setUp(self):
        self.engine = HighamSpectralProjectionEngine(min_eigenvalue_floor=1e-6)

    def test_multi_asset_higham_pd(self):
        """
        6 Farklı enstrümandan (2 Hisse + 2 BYF + 1 TEFAS Serbest + 1 TEFAS Likit) oluşan,
        kasıtlı olarak kolineerlik ve negatif özdeğer verilerek bozulmuş matrisin
        Higham projeksiyonuyla %100 pozitif tanımlı (lambda_min >= 1e-6) yapıldığının kanıtlanması.
        """
        symbols = ["THYAO", "GARAN", "GLDTR", "ZGOLD", "TLY", "PPF"]
        n = len(symbols)

        # 1. Kasıtlı Bozuk/Non-PD Kovaryans Matrisi Oluşturma (Negatif özdeğerli)
        np.random.seed(123)
        # Rank-deficient matris üret: 6x2 rastgele faktör * 2x6
        factors = np.random.normal(0, 1, (n, 2))
        corrupted_cov = factors @ factors.T
        # Çapraz bloğa negatif gürültü enjekte et (Negatif özdeğer garantisi için)
        corrupted_cov[0, 4] = -5.0
        corrupted_cov[4, 0] = -5.0
        corrupted_cov[1, 5] = 8.0
        corrupted_cov[5, 1] = 8.0

        corrupted_df = pd.DataFrame(corrupted_cov, index=symbols, columns=symbols)

        # Ham matrisin non-PD olduğunu doğrula
        raw_analysis = self.engine.decompose_covariance_matrix(corrupted_df)
        self.assertLess(raw_analysis.min_eigenvalue, 0.0, "Ham matris kasıtlı olarak negatif özdeğer içermelidir!")
        self.assertFalse(raw_analysis.is_positive_definite)

        # 2. Higham Spektral Projeksiyonunu Çalıştır
        sealed_df, post_analysis = self.engine.project_nearest_positive_definite(corrupted_df, epsilon=1e-6)

        # 3. POZİTİF TANIMLILIK GARANTİSİ (lambda_min >= 1e-6)
        self.assertGreaterEqual(
            post_analysis.min_eigenvalue,
            1e-6 - 1e-12,
            "Mühürlenmiş matrisin minimum özdeğeri strictly >= 1e-6 olmalıdır!"
        )
        self.assertTrue(post_analysis.is_positive_definite)
        self.assertEqual(post_analysis.negative_eigenvalues_count, 0)

        # 4. CHOLESKY FAKTORİZASYONU TESTİ (PyTorch ve Katman 3 QP Solver Zırhı)
        try:
            L = np.linalg.cholesky(sealed_df.values)
            cholesky_success = True
        except np.linalg.LinAlgError:
            cholesky_success = False

        self.assertTrue(cholesky_success, "Cholesky ayrıştırması (L * L^T) sıfır hatayla tamamlanmalıdır!")

        # 5. BOYUT VE ŞEMA KORUMASI
        self.assertEqual(sealed_df.shape, (6, 6))
        self.assertEqual(list(sealed_df.index), symbols)
        self.assertEqual(list(sealed_df.columns), symbols)


if __name__ == "__main__":
    unittest.main()
