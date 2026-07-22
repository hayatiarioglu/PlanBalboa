"""
AetherForecaster-X v35.2 Multi-Asset Master
Katman 2: Clamped Scheduled Soft-NDCG & Detach-Coupled MMoE

Modül: Clamped Scheduled Soft-NDCG Loss Function
Faz 3.2 / Adım 3.2.1: Kriz haftalarında y_i > 0 hisse kalmadığında (tüm piyasa çöktüğünde)
sıfıra bölmeyi (division by zero) engelleyen Epsilon Korumalı Positive IDCG katmanı.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class EpsilonProtectedPositiveIDCG(nn.Module):
    """
    Adım 3.2.1: Epsilon Korumalı Positive IDCG Katmanı.
    Piyasa çöküşlerinde (kriz haftaları) y_i > 0 olan hisse kalmadığında IDCG = 0 olup
    Soft-NDCG kaybının 0/0 -> NaN vermesini 10^-6 epsilon kalkanı ile engeller.
    """

    def __init__(
        self,
        top_k: Optional[int] = None,
        eps: float = 1e-6
    ):
        """
        :param top_k: NDCG hesaplanacak üst k varlık sayısı (None ise tüm varlıklar).
        :param eps: Sıfıra bölme zırhı (10^-6).
        """
        super().__init__()
        self.top_k = top_k
        self.eps = eps

    def forward(
        self,
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Gerçek hedef getiri/relevance vektöründen ideal DCG (IDCG_pos) değerini hesaplar.

        :param y_true: Shape (batch_size, n_assets) veya (n_assets,) gerçek hedefler.
        :return: Shape (batch_size, 1) veya (1,) epsilon korumalı IDCG_pos tensorü.
        """
        if y_true.dim() == 1:
            y_true_2d = y_true.unsqueeze(0)
            is_1d = True
        else:
            y_true_2d = y_true
            is_1d = False

        batch_size, n_assets = y_true_2d.shape
        k = self.top_k if self.top_k is not None and 0 < self.top_k <= n_assets else n_assets

        # Sadece pozitif relevance (y_i > 0) dikkate alınır
        y_pos = torch.clamp(y_true_2d, min=0.0)

        # Azalan sırada sırala (Ideal Ranking)
        sorted_y, _ = torch.sort(y_pos, descending=True, dim=-1)
        topk_y = sorted_y[:, :k]

        # Gain = 2^{y_i} - 1
        gains = torch.pow(2.0, topk_y) - 1.0

        # Discount = log_2(1 + i)  (1-indexed position: i = 1, 2, ..., k)
        ranks = torch.arange(1, k + 1, device=y_true.device, dtype=y_true.dtype)
        discounts = torch.log2(ranks + 1.0)

        # IDCG = Sum(Gains / Discounts)
        idcg_raw = torch.sum(gains / discounts, dim=-1, keepdim=True)

        # Epsilon Koruması: IDCG_pos = max(IDCG_raw, 10^-6)
        idcg_protected = torch.clamp(idcg_raw, min=self.eps)

        if is_1d:
            return idcg_protected.squeeze(0)
        return idcg_protected


class DelistSigmoidPenalty(nn.Module):
    """
    Adım 3.2.2: Delist hisselere eklemeli Sigmoid cezası ekleme katmanı.
    L_delist = eta * sum_{d in Delist} sigmoid(hat{y}_d)

    Model delist/iflas eden hisselere yüksek tahmin skoru (logit) verirse,
    sigmoid(hat{y}_d) -> 1 yaklaşır ve modele ağır bir ek ceza uygulayarak skoru düşürmeye zorlar.
    """

    def __init__(self, eta: float = 1.0):
        """
        :param eta: Delist ceza katsayısı (Ağırlık katsayısı eta > 0).
        """
        super().__init__()
        if eta <= 0:
            raise ValueError(f"Geçersiz eta katsayısı! eta > 0 olmalıdır: {eta}")
        self.eta = eta

    def forward(
        self,
        y_pred: torch.Tensor,
        delisted_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        :param y_pred: Model tahmin skorları (logits), shape (batch_size, n_assets) veya (n_assets,)
        :param delisted_mask: Delist hisse maskesi (True/1 = Delist), aynı shape'te bool/float tensor.
        :return: Skalar tensor delist sigmoid ceza kaybı.
        """
        if y_pred.shape != delisted_mask.shape:
            raise ValueError(f"y_pred boyutu ({y_pred.shape}), delisted_mask ({delisted_mask.shape}) ile uyuşmuyor!")

        mask_bool = delisted_mask.to(dtype=torch.bool)
        
        if not torch.any(mask_bool):
            # Delist hisse yoksa ceza = 0.0
            return torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)

        # Sigmoid dönüşümü: sigma(hat{y}_d) = 1 / (1 + exp(-hat{y}_d))
        sigmoid_preds = torch.sigmoid(y_pred)
        
        # Sadece delist hisselerin sigmoid cezalarını topla
        delist_sigmoids = torch.masked_select(sigmoid_preds, mask_bool)
        
        # L_delist = eta * sum(sigmoid(hat{y}_d))
        penalty_loss = self.eta * torch.sum(delist_sigmoids)
        
        return penalty_loss


class ClampedLogitDifference(nn.Module):
    """
    Adım 3.2.3: Plackett-Luce float overflow zırhı olan Clamped Logit Difference katmanı.
    Delta_{ij} = torch.clamp( (hat{y}_i - hat{y}_j) / max(tau, tau_min), min=-10.0, max=+10.0 )
    """

    def __init__(
        self,
        tau_min: float = 0.10,
        clamp_val: float = 10.0
    ):
        """
        :param tau_min: Minimum sıcaklık sınırı (0.10).
        :param clamp_val: Maksimum ve minimum kırpma genliği (+-10.0).
        """
        super().__init__()
        self.tau_min = tau_min
        self.clamp_val = clamp_val

    def forward(
        self,
        y_pred: torch.Tensor,
        tau: Union[float, torch.Tensor] = 1.0
    ) -> torch.Tensor:
        """
        :param y_pred: Model tahmin skorları (batch_size, n_assets) veya (n_assets,)
        :param tau: Sıcaklık katsayısı (tau >= tau_min)
        :return: (batch_size, n_assets, n_assets) veya (n_assets, n_assets) logit farkları matrisi Delta_{ij}
        """
        if y_pred.dim() == 1:
            y_pred_2d = y_pred.unsqueeze(0)
            is_1d = True
        else:
            y_pred_2d = y_pred
            is_1d = False

        if isinstance(tau, torch.Tensor):
            tau_val = torch.clamp(tau, min=self.tau_min)
        else:
            tau_val = max(float(tau), self.tau_min)

        # Pairwise Logit Farkı: Delta_{ij} = y_i - y_j (n_assets x n_assets)
        diff_matrix = y_pred_2d.unsqueeze(2) - y_pred_2d.unsqueeze(1)

        # Temperature Scaling & Clamping (-10.0, +10.0)
        scaled_diff = diff_matrix / tau_val
        clamped_diff = torch.clamp(scaled_diff, min=-self.clamp_val, max=self.clamp_val)

        if is_1d:
            return clamped_diff.squeeze(0)
        return clamped_diff


class ClampedScheduledSoftNDCGLoss(nn.Module):
    """
    Katman 2 Ana Kayıp Fonksiyonu: Clamped Scheduled Soft-NDCG Loss.
    - Epsilon Korumalı Positive IDCG (Adım 3.2.1)
    - Delist Sigmoid Cezası (Adım 3.2.2)
    - Clamped Logit Difference (Adım 3.2.3)
    bileşenlerini birleştirir ve kriz koşullarında tam 0 adet NaN üretme garantisi verir.
    """

    def __init__(
        self,
        top_k: Optional[int] = None,
        tau_min: float = 0.10,
        clamp_val: float = 10.0,
        eta_delist: float = 1.0,
        eps: float = 1e-6
    ):
        super().__init__()
        self.idcg_layer = EpsilonProtectedPositiveIDCG(top_k=top_k, eps=eps)
        self.delist_penalty_layer = DelistSigmoidPenalty(eta=eta_delist)
        self.logit_diff_layer = ClampedLogitDifference(tau_min=tau_min, clamp_val=clamp_val)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        delisted_mask: Optional[torch.Tensor] = None,
        tau: Union[float, torch.Tensor] = 0.05,

        telemetry_logger: Optional[Any] = None,
        iteration: int = 0,
        stage: str = "train"
    ) -> torch.Tensor:
        """
        Soft-NDCG Kayıp Değerini Hesaplar.

        :param y_pred: Model tahmin skorları (batch_size, n_assets) veya (n_assets,)
        :param y_true: Gerçek hedefler aynı shape'te.
        :param delisted_mask: Delist hisse maskesi (True/1 = Delist).
        :param tau: Dinamik sıcaklık (temperature).
        :param telemetry_logger: Telemetri kancası (opsiyonel).
        :param iteration: Telemetri için iterasyon sayısı.
        :param stage: Telemetri için aşama (train/val/test).
        :return: Skalar Loss tensorü.
        """
        if y_pred.dim() == 3 and y_pred.shape[-1] == 1:
            y_pred = y_pred.squeeze(-1)
        if y_true.dim() == 3 and y_true.shape[-1] == 1:
            y_true = y_true.squeeze(-1)

        if y_pred.dim() == 1:
            y_pred_2d = y_pred.unsqueeze(0)
            y_true_2d = y_true.unsqueeze(0)
            d_mask_2d = delisted_mask.unsqueeze(0) if delisted_mask is not None else None
        else:
            y_pred_2d = y_pred
            y_true_2d = y_true
            d_mask_2d = delisted_mask


        # 0. Cross-Sectional Logit Z-Score Normalizasyonu (Ölü Gradyan Kalkanı):
        # Logit farklarının sigmoid türevinde kilitlenmesini engeller.
        y_pred_mean = torch.mean(y_pred_2d, dim=-1, keepdim=True)
        y_pred_std = torch.std(y_pred_2d, dim=-1, keepdim=True, correction=0) + 1e-5
        y_pred_norm = (y_pred_2d - y_pred_mean) / y_pred_std

        # 1. Clamped Logit Differences Delta_{ij}
        diff_matrix = self.logit_diff_layer(y_pred_norm, tau=tau)


        # 2. Soft-Ranks R_i = 1 + Sum_{j != i} sigmoid( (hat{y}_j - hat{y}_i) / tau )
        # Note: diff_matrix = (y_i - y_j) / tau => (y_j - y_i) / tau = -diff_matrix
        # Diyagonalde j==i için sigmoid(0) = 0.5 döner. 0.5 çıkararak self-comparison (kendiyle kıyas) etkisini tam sıfırlıyoruz.
        soft_ranks = 1.0 + torch.sum(torch.sigmoid(-diff_matrix), dim=-1) - 0.5

        # 3. Cross-Sectional Relevance Grades y_rel (0.0 .. 5.0)
        # Ham getirileri enine kesitte 0-5 dereceye dönüştürür (Relevance grade mapping)
        ranks_pct = torch.argsort(torch.argsort(y_true_2d, dim=-1), dim=-1).to(dtype=y_true_2d.dtype) / max(1, y_true_2d.shape[-1] - 1)
        y_rel = ranks_pct * 5.0

        # --- 1. YÜKSELEN TOP 3 İÇİN CEZA / ÖDÜL HESABI ---
        top3_up_boost = 1.0 + 4.0 * torch.exp(-0.80 * (soft_ranks - 1.0) ** 2)
        gains_up = torch.pow(2.0, y_rel) - 1.0
        discounts_up = torch.log2(soft_ranks + 1.0)
        soft_dcg_up = torch.sum((gains_up / discounts_up) * top3_up_boost, dim=-1, keepdim=True)
        idcg_pos_up = self.idcg_layer(y_rel)
        soft_ndcg_up = soft_dcg_up / idcg_pos_up
        ndcg_loss_up = torch.mean(1.0 - soft_ndcg_up)

        # --- 2. DÜŞEN TOP 3 İÇİN CEZA / ÖDÜL HESABI (Ters Sıralama) ---
        # Ters sıralamada en çok düşenler üst sıraya geçer
        rev_soft_ranks = 1.0 + torch.sum(torch.sigmoid(diff_matrix), dim=-1) - 0.5
        rev_y_rel = (1.0 - ranks_pct) * 5.0
        top3_down_boost = 1.0 + 4.0 * torch.exp(-0.80 * (rev_soft_ranks - 1.0) ** 2)
        gains_down = torch.pow(2.0, rev_y_rel) - 1.0
        discounts_down = torch.log2(rev_soft_ranks + 1.0)
        soft_dcg_down = torch.sum((gains_down / discounts_down) * top3_down_boost, dim=-1, keepdim=True)
        idcg_pos_down = self.idcg_layer(rev_y_rel)
        soft_ndcg_down = soft_dcg_down / idcg_pos_down
        ndcg_loss_down = torch.mean(1.0 - soft_ndcg_down)

        # Toplam Çift Yönlü Sıralama Kaybı (Hem Yükselen Hem Düşen Top 3 Ağır Cezalı)
        ndcg_loss = 0.50 * (ndcg_loss_up + ndcg_loss_down)


        # 6. Delist Sigmoid Penalty
        if d_mask_2d is not None:
            delist_loss = self.delist_penalty_layer(y_pred_2d, d_mask_2d)
        else:
            delist_loss = 0.0

        # 7. Sınırlı Standartlaştırılmış Varyans Cezası:
        pred_std = torch.std(y_pred_2d, dim=-1, correction=0)
        var_penalty = 0.50 * torch.mean((pred_std - 1.0) ** 2)

        # 8. TEKEL KİLİTLENMESİ CEZASI (Anti-Monopoly Cross-Asset Diversity Penalty):
        # Tahmin logitlerinin en yüksek olduğu varlığın tüm varlıklar içindeki baskınlığını cezalandırır.
        top_asset_probs = F.softmax(y_pred_2d, dim=-1)
        max_prob = torch.max(top_asset_probs, dim=-1)[0]
        diversity_penalty = 2.0 * torch.mean(torch.clamp(max_prob - 0.15, min=0.0))

        total_loss = ndcg_loss + delist_loss + var_penalty + diversity_penalty
        
        # Telemetry Hook
        if telemetry_logger is not None:
            telemetry_logger.log_soft_ndcg(
                loss_value=total_loss.item(),
                iteration=iteration,
                stage=stage
            )
            
        return total_loss


