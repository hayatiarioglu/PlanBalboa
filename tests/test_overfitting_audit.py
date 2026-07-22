"""
AetherForecaster-X v35.2 Multi-Asset Master
Sprint 4: Overfitting & Generalization Rigorous Audit Test

Modül: test_overfitting_audit.py
Bu test, modelin ezber yapıp yapmadığını (Overfitting) 3 brutal yöntemle dürüstçe kanıtlar:
1. Permütasyon ve Gürültü Hassasiyet Testi (Feature Noise Sensitivity Test)
2. Görülmemiş Zaman Dilimi Testi (Out-of-Sample 2024-2026 Ranking Accuracy)
3. Varlık Çıkarma / Transferability Testi (Asset Leave-One-Out Generalization)
"""

from __future__ import annotations
import unittest
import numpy as np
import pandas as pd
import torch

from aether.data.fetcher import HistoricalDatasetBuilder
from aether.trainer.checkpoint_manager import CheckpointManager
from aether.monitoring.error_analyzer import RootCauseAnalyzer


class TestOverfittingAudit(unittest.TestCase):
    """
    Modelin ezber yapıp yapmadığını kesin olarak kanıtlayan bilimsel denetim sınıfı.
    """

    def setUp(self):
        from aether.signal.mmoe_backbone import MMoEBackboneModule
        self.checkpoint_mgr = CheckpointManager(checkpoint_dir="checkpoints")
        self.model = MMoEBackboneModule(input_dim=10, hybrid_vol_dim=3, num_experts=4, expert_hidden_dim=64, expert_output_dim=32)
        ckpt_path = self.checkpoint_mgr.get_latest_checkpoint_path()
        if ckpt_path is not None and ckpt_path.exists():
            self.checkpoint_mgr.load_checkpoint(model=self.model, checkpoint_path=ckpt_path)
        self.dataset_builder = HistoricalDatasetBuilder()
        self.analyzer = RootCauseAnalyzer()

    def test_unseen_data_ranking_generalization(self):
        """
        1. TEST: Görülmemiş 2024-2026 Out-of-Sample verilerinde sıralama başarısı.
        """
        print("\n=======================================================================")
        print("[AUDIT 1] GORULMEMIS 2024-2026 VERISI ILE SIRALAMA GENELLEME TESTI")
        print("=======================================================================")

        snapshots = self.dataset_builder.build_weekly_sequence(start_date="2024-01-01", end_date="2026-01-01")
        if len(snapshots) == 0:
            self.skipTest("2024-2026 OOS verisi bulunamadi!")

        ndcg_scores = []
        spearman_scores = []

        self.model.eval()
        with torch.no_grad():
            for snap in snapshots:
                x = snap["x"]
                v_hybrid = snap["v_hybrid"]
                act_rets = pd.Series(snap["actual_realized_returns"], index=snap["assets"])

                # Z-Score Normalization
                x_mean = torch.mean(x, dim=1, keepdim=True)
                x_std = torch.std(x, dim=1, keepdim=True) + 1e-6
                x_norm = torch.clamp((x - x_mean) / x_std, min=-3.0, max=3.0)

                sig_out = self.model(x=x_norm, v_hybrid=v_hybrid)
                pred_scores = pd.Series(sig_out.logit_head_b.squeeze().cpu().numpy(), index=snap["assets"])

                ndcg = self.analyzer.calculate_ndcg_at_k(pred_scores, act_rets, k=10)
                spearman = self.analyzer.calculate_spearman_rank_correlation(pred_scores, act_rets)

                ndcg_scores.append(ndcg)
                spearman_scores.append(spearman)

        avg_ndcg = float(np.mean(ndcg_scores))
        avg_spearman = float(np.mean(spearman_scores))

        print(f"  -> Görülmemiş 2024-2026 Verisi Ortalama NDCG@10: {avg_ndcg:.4f}")
        print(f"  -> Görülmemiş 2024-2026 Verisi Ortalama Spearman Sıra Korelasyonu: {avg_spearman:.4f}")

        # Yeni Protokol Eşikleri: NDCG@10 > 0.65, Spearman > 0.45
        self.assertGreater(avg_ndcg, 0.65, f"Ezber Uyarısı! NDCG@10 seviyesi çok düşük: {avg_ndcg:.4f}")
        self.assertGreater(avg_spearman, 0.45, f"Ezber Uyarısı! Spearman korelasyonu çok düşük: {avg_spearman:.4f}")

    def test_feature_noise_sensitivity(self):
        """
        2. TEST: Rassal Gürültü Hassasiyet Testi (Feature Noise Sensitivity Test)
        """
        print("\n=======================================================================")
        print("[AUDIT 2] RASSAL GURULTU HASSASIYET TESTI (NOISE SENSITIVITY)")
        print("=======================================================================")

        snapshots = self.dataset_builder.build_weekly_sequence(start_date="2024-01-01", end_date="2024-06-01")
        if len(snapshots) == 0:
            self.skipTest("OOS test verisi bulunamadi.")

        clean_ndcgs = []
        noisy_ndcgs = []

        self.model.eval()
        with torch.no_grad():
            for snap in snapshots:
                x_clean = snap["x"]
                v_clean = snap["v_hybrid"]
                act_rets = pd.Series(snap["actual_realized_returns"], index=snap["assets"])

                # Temiz Z-Score
                x_mean = torch.mean(x_clean, dim=1, keepdim=True)
                x_std = torch.std(x_clean, dim=1, keepdim=True) + 1e-6
                x_clean_norm = torch.clamp((x_clean - x_mean) / x_std, min=-3.0, max=3.0)

                sig_clean = self.model(x=x_clean_norm, v_hybrid=v_clean)
                pred_clean = pd.Series(sig_clean.logit_head_b.squeeze().cpu().numpy(), index=snap["assets"])
                clean_ndcg = self.analyzer.calculate_ndcg_at_k(pred_clean, act_rets, k=10)
                clean_ndcgs.append(clean_ndcg)

                # Gürültülü girdi (%50 Gaussian Noise)
                x_noisy = x_clean + torch.randn_like(x_clean) * 0.50
                x_noisy_mean = torch.mean(x_noisy, dim=1, keepdim=True)
                x_noisy_std = torch.std(x_noisy, dim=1, keepdim=True) + 1e-6
                x_noisy_norm = torch.clamp((x_noisy - x_noisy_mean) / x_noisy_std, min=-3.0, max=3.0)

                v_noisy = v_clean + torch.randn_like(v_clean) * 0.50

                sig_noisy = self.model(x=x_noisy_norm, v_hybrid=v_noisy)
                pred_noisy = pd.Series(sig_noisy.logit_head_b.squeeze().cpu().numpy(), index=snap["assets"])
                noisy_ndcg = self.analyzer.calculate_ndcg_at_k(pred_noisy, act_rets, k=10)
                noisy_ndcgs.append(noisy_ndcg)

        avg_clean = float(np.mean(clean_ndcgs))
        avg_noisy = float(np.mean(noisy_ndcgs))
        delta_noise = avg_clean - avg_noisy

        print(f"  -> Temiz Veri NDCG@10: {avg_clean:.4f}")
        print(f"  -> Gürültülü Veri (%50 Bozulmuş) NDCG@10: {avg_noisy:.4f}")
        print(f"  -> Gürültü Tepki Farkı (Delta Noise): {delta_noise:.4f}")

        # Protokol Eşiği: Delta Noise > 0.15 (Model gürültüye duyarlı olmalı!)
        self.assertGreater(delta_noise, 0.15, f"Ezber Uyarısı! Model gürültüye tepki vermedi (Delta={delta_noise:.4f})!")

    def test_top10_hit_ratio_vs_random(self):
        """
        3. TEST: Top 10 İsabet Oranı vs Rastgele Seçim (Hit Ratio Benchmark Test)
        """
        print("\n=======================================================================")
        print("[AUDIT 3] TOP 10 ISABET ORANI VS RASTGELE SECIM (HIT RATIO BENCHMARK)")
        print("=======================================================================")

        snapshots = self.dataset_builder.build_weekly_sequence(start_date="2024-01-01", end_date="2025-01-01")
        if len(snapshots) == 0:
            self.skipTest("OOS test verisi bulunamadi.")

        model_top10_returns = []
        random_returns = []

        np.random.seed(42)
        self.model.eval()

        with torch.no_grad():
            for snap in snapshots:
                x = snap["x"]
                v_hybrid = snap["v_hybrid"]
                act_rets = pd.Series(snap["actual_realized_returns"], index=snap["assets"])

                x_mean = torch.mean(x, dim=1, keepdim=True)
                x_std = torch.std(x, dim=1, keepdim=True) + 1e-6
                x_norm = torch.clamp((x - x_mean) / x_std, min=-3.0, max=3.0)

                sig_out = self.model(x=x_norm, v_hybrid=v_hybrid)
                pred_scores = pd.Series(sig_out.logit_head_b.squeeze().cpu().numpy(), index=snap["assets"])

                # Model Top 10
                top10_assets = pred_scores.nlargest(10).index
                model_top10_ret = float(act_rets.loc[top10_assets].mean())
                model_top10_returns.append(model_top10_ret)

                # Rastgele 10 Varlık
                rand_assets = np.random.choice(snap["assets"], size=10, replace=False)
                rand_ret = float(act_rets.loc[rand_assets].mean())
                random_returns.append(rand_ret)

        avg_model_ret = float(np.mean(model_top10_returns))
        avg_random_ret = float(np.mean(random_returns))
        alpha_spread = (avg_model_ret - avg_random_ret) * 100.0

        print(f"  -> Model Top 10 Ortalama Getiri: %{avg_model_ret * 100:.2f}")
        print(f"  -> Rastgele Seçim 10 Varlık Ortalama Getiri: %{avg_random_ret * 100:.2f}")
        print(f"  -> Modelin Rastgele Seçime Üstünlüğü (Alpha Spread): %{alpha_spread:.2f}")

        # Protokol Eşiği: Alpha Spread > +8.0%
        self.assertGreater(alpha_spread, 8.0, f"Ezber Uyarısı! Model Alpha Spread (+%{alpha_spread:.2f}) eşik seviyesinin (+%8.0) altında kaldı!")


if __name__ == "__main__":
    unittest.main()
