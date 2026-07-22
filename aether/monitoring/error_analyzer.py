"""
AetherForecaster-X v35.2 Multi-Asset Master
Katman 4: Kök Neden Hata Analiz Motoru (Error & Root-Cause Analyzer)

Modül: error_analyzer.py
Bu modül, modelin ürettiği Top 10 Yükselen ve Top 10 Düşen sıralama listelerinin
"Dedektifi ve Hakimi" olarak çalışır. Hataları 3 süzgeç üzerinden teşhis eder:
1. Sistemik Piyasa Şoku Filtresi (Volatilite Sıçraması / Market Shock)
2. Sektörel Sapma Filtresi (Sector-level Noise)
3. Gerçek Sıralama Hatası (ΔRank, NDCG@10, Spearman Rank Correlation)

Ayrıca, etiket zamanlama uyumsuzluğunu engellemek için Haftalık Trajektör Sapmasını (ΔTrajectory)
hesaplar ve Clamped Diagnosis Weight üreterek train_real_model.py fine-tuning adımına iletir.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import scipy.stats as stats


@dataclass
class RootCauseDiagnosisResult:
    """
    error_analyzer.py Tarafından Üretilen Kök Neden Teşhis Paketi.
    """
    total_error_score: float                     # Toplam sapma miktarı (e_t)
    market_shock_weight: float                   # Sistemik piyasa çöküşü/şok payı (0.0 - 1.0)
    sector_noise_weight: float                   # Sektörel gürültü payı (0.0 - 1.0)
    ranking_error_weight: float                  # Modelin gerçek sıralama hatası payı (0.0 - 1.0)
    ndcg_at_10: float                            # Top 10 Sıralama Başarısı (0.0 - 1.0)
    spearman_correlation: float                  # Spearman Sıra Korelasyonu (-1.0 - +1.0)
    trajectory_deviation: float                  # Haftalık Trajektör Sapması (ΔTrajectory)
    diagnosis_weight: float                      # İletilecek ham teşhis katsayısı
    effective_learning_rate: float               # Clamped Dinamik Öğrenme Oranı (10^-6 <= η <= 10^-4)
    diagnosis_summary: str                       # İnsan tarafından okunabilir kök neden özeti


class RootCauseAnalyzer:
    """
    Top 10 Yükselen ve Top 10 Düşen Sinyal Listesinin Kök Neden Teşhis Motoru.
    """

    def __init__(
        self,
        base_learning_rate: float = 1e-4,
        min_learning_rate: float = 1e-6,
        max_learning_rate: float = 1e-4,
        market_shock_vol_threshold: float = 0.04  # %4 üstü haftalık volatilite = piyasa şoku
    ):
        self.base_learning_rate = base_learning_rate
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.market_shock_vol_threshold = market_shock_vol_threshold

    def calculate_ndcg_at_k(
        self,
        predicted_scores: pd.Series,
        actual_returns: pd.Series,
        k: int = 10
    ) -> float:
        """
        Top k eleman için Normalized Discounted Cumulative Gain (NDCG@k) hesaplar.
        """
        if len(predicted_scores) == 0 or len(actual_returns) == 0:
            return 0.0

        common_assets = predicted_scores.index.intersection(actual_returns.index)
        if len(common_assets) == 0:
            return 0.0

        pred = predicted_scores.loc[common_assets]
        act = actual_returns.loc[common_assets]

        # Top k tahmin edileni al
        top_k_pred_assets = pred.nlargest(min(k, len(pred))).index

        # Relevance score = gerçekleşen getiri (pozitif kırpılmış)
        rel_pred = act.loc[top_k_pred_assets].to_numpy()
        rel_pred = np.maximum(rel_pred, 0.0)

        # DCG hesaplama
        discounts = np.log2(np.arange(2, len(rel_pred) + 2))
        dcg = np.sum((2**rel_pred - 1) / discounts)

        # Ideal DCG hesaplama
        rel_ideal = act.nlargest(min(k, len(act))).to_numpy()
        rel_ideal = np.maximum(rel_ideal, 0.0)
        idcg = np.sum((2**rel_ideal - 1) / np.log2(np.arange(2, len(rel_ideal) + 2)))

        if idcg <= 1e-8:
            return 1.0  # Piyasa tamamen düşüyorsa 0 bölmeyi engelle

        return float(dcg / idcg)

    def calculate_spearman_rank_correlation(
        self,
        predicted_scores: pd.Series,
        actual_returns: pd.Series
    ) -> float:
        """
        Tahmin sıralaması ile gerçekleşen getiriler arasındaki Spearman Sıra Korelasyonunu hesaplar.
        """
        common_assets = predicted_scores.index.intersection(actual_returns.index)
        if len(common_assets) < 3:
            return 0.0

        corr, _ = stats.spearmanr(
            predicted_scores.loc[common_assets].to_numpy(),
            actual_returns.loc[common_assets].to_numpy()
        )
        return float(corr) if not np.isnan(corr) else 0.0

    def analyze_root_cause(
        self,
        predicted_scores: pd.Series,             # Model sıralama skorları (Head A)
        predicted_returns: pd.Series,            # Model % getiri tahminleri (Head B)
        actual_returns: pd.Series,               # Gerçekleşen % getiriler
        market_index_return: float,              # BIST100 / Piyasa endeksi haftalık getirisi
        market_hybrid_volatility: float,         # V_t^hybrid piyasa volatilite seviyesi
        sector_mapping: Optional[Dict[str, str]] = None # Varlık -> Sektör haritası
    ) -> RootCauseDiagnosisResult:
        """
        Tahmin ile gerçekleşen arasındaki hatayı 3 filtre üzerinden teşhis eder.
        """
        # Yinelenen Sembolleri Temizle (Deduplicate)
        predicted_scores = predicted_scores[~predicted_scores.index.duplicated(keep='first')]
        if predicted_returns is not None:
            predicted_returns = predicted_returns[~predicted_returns.index.duplicated(keep='first')]
        actual_returns = actual_returns[~actual_returns.index.duplicated(keep='first')]

        common_assets = list(set(predicted_scores.index).intersection(set(actual_returns.index)))
        if not common_assets:
            return RootCauseDiagnosisResult(
                total_error_score=1.0,
                market_shock_weight=0.5,
                sector_noise_weight=0.5,
                ranking_error_weight=0.0,
                ndcg_at_10=0.0,
                spearman_correlation=0.0,
                trajectory_deviation=1.0,
                diagnosis_weight=0.0,
                effective_learning_rate=self.min_learning_rate,
                diagnosis_summary="Veri kesişimi bulunamadı."
            )

        pred_scores = predicted_scores.loc[common_assets]
        pred_rets = predicted_returns.reindex(common_assets).fillna(0.0) if predicted_returns is not None else pred_scores
        act_rets = actual_returns.loc[common_assets]


        # 1. Toplam Sapma (e_t) ve Trajektör Sapması (ΔTrajectory)
        raw_errors = np.abs(pred_rets - act_rets)
        total_error_score = float(np.mean(raw_errors))
        trajectory_deviation = float(np.std(act_rets - pred_rets))

        # 2. Metrikler: NDCG@10 ve Spearman Correlation
        ndcg_10 = self.calculate_ndcg_at_k(pred_scores, act_rets, k=10)
        spearman_corr = self.calculate_spearman_rank_correlation(pred_scores, act_rets)

        # 3. Teşhis 1: Sistemik Piyasa Şoku (Haftalık & 5 Günlük Trajektör Kontrolü)
        is_market_shock = (
            abs(market_index_return) > 0.03 or
            market_hybrid_volatility > self.market_shock_vol_threshold
        )
        if is_market_shock:
            market_shock_weight = min(1.0, abs(market_index_return) / 0.05 + market_hybrid_volatility / 0.06)
        else:
            market_shock_weight = 0.10

        # 4. Teşhis 2: Sektörel Sapma
        sector_noise_weight = 0.20
        if sector_mapping is not None:
            sector_errors = []
            df = pd.DataFrame({'act': act_rets, 'pred': pred_rets, 'sector': pd.Series(sector_mapping)})
            for _, grp in df.groupby('sector'):
                if len(grp) > 1:
                    sector_errors.append(np.mean(np.abs(grp['act'] - grp['pred'])))
            if len(sector_errors) > 0 and np.std(sector_errors) > 0.02:
                sector_noise_weight = 0.50

        # 5. Teşhis 3: Gerçek Sıralama Hatası (ΔRank)
        ranking_error_weight = max(0.0, 1.0 - (market_shock_weight * 0.5 + sector_noise_weight * 0.3))
        ranking_error_weight *= (1.0 - max(0.0, spearman_corr))

        # 6. Diagnosis Weight Hesabı
        raw_diagnosis_weight = float(ranking_error_weight)

        # Clamped Learning Rate: 10^-6 <= η_effective <= 10^-4
        effective_lr = float(np.clip(
            self.base_learning_rate * raw_diagnosis_weight,
            self.min_learning_rate,
            self.max_learning_rate
        ))

        # Teşhis Özeti
        summary = (
            f"Kök Neden Teşhisi: Piyasa Şoku=%{market_shock_weight*100:.1f}, "
            f"Sektör Gürültüsü=%{sector_noise_weight*100:.1f}, "
            f"Gerçek Sıralama Hatası=%{ranking_error_weight*100:.1f} | "
            f"NDCG@10={ndcg_10:.2f}, Spearman={spearman_corr:.2f} | "
            f"Dinamik LR={effective_lr:.2e}"
        )

        return RootCauseDiagnosisResult(
            total_error_score=total_error_score,
            market_shock_weight=float(market_shock_weight),
            sector_noise_weight=float(sector_noise_weight),
            ranking_error_weight=float(ranking_error_weight),
            ndcg_at_10=ndcg_10,
            spearman_correlation=spearman_corr,
            trajectory_deviation=trajectory_deviation,
            diagnosis_weight=raw_diagnosis_weight,
            effective_learning_rate=effective_lr,
            diagnosis_summary=summary
        )

