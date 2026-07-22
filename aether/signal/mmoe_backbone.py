"""
AetherForecaster-X v35.2 Multi-Asset Master
Katman 2: Clamped Scheduled Soft-NDCG & Detach-Coupled MMoE

Modül: MMoE Trunk & Detach-Coupled Prediction Heads
Faz 3.3 / Adım 3.3.1: Head A logitinin Autograd grafiğinden koparılarak (.detach()) Head B
girdisine eklenmesini ve gradyan sızıntısını (gradient leakage) engelleyen MMoE mimarisi.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MMoEOutput:
    """
    MMoE Model İleri İletim (Forward Pass) Sonuç Raporu.
    """
    logit_head_a: torch.Tensor          # Head A (Ana Alpha / Beklenen Getiri Tahmini)
    logit_head_b: torch.Tensor          # Head B (Sıralama / Göreli Ayarlama Tahmini)
    trunk_representation: torch.Tensor  # MMoE Trunk temsili h_i
    gate_a_weights: torch.Tensor        # Expert kapılama ağırlıkları (Head A)
    gate_b_weights: torch.Tensor        # Expert kapılama ağırlıkları (Head B)


class DetachCoupledPredictionHeads(nn.Module):
    """
    Adım 3.3.1: Head A ve Head B Detach-Coupling Katmanı.
    Head B girdisi: input_HeadB = torch.cat([h_i, logit_HeadA.detach(), V_t_hybrid], dim=-1)
    """

    def __init__(
        self,
        trunk_dim: int,
        hybrid_vol_dim: int,
        head_a_hidden_dim: int = 64,
        head_b_hidden_dim: int = 64
    ):
        """
        :param trunk_dim: MMoE Trunk temsil boyutu (dim(h_i)).
        :param hybrid_vol_dim: Hibrit volatilite vektör boyutu (dim(V_t_hybrid)).
        :param head_a_hidden_dim: Head A gizli katman boyutu.
        :param head_b_hidden_dim: Head B gizli katman boyutu.
        """
        super().__init__()
        self.trunk_dim = trunk_dim
        self.hybrid_vol_dim = hybrid_vol_dim

        # Head A: Primary Return Prediction Network (h_i -> logit_HeadA)
        self.head_a_net = nn.Sequential(
            nn.Linear(trunk_dim, head_a_hidden_dim),
            nn.GELU(),
            nn.Linear(head_a_hidden_dim, 1)
        )

        # Head B Input Dimension = trunk_dim (h_i) + 1 (logit_HeadA.detach()) + hybrid_vol_dim (V_t_hybrid)
        head_b_input_dim = trunk_dim + 1 + hybrid_vol_dim

        # Head B: Ranking / Relative Adjustment Network
        self.head_b_net = nn.Sequential(
            nn.Linear(head_b_input_dim, head_b_hidden_dim),
            nn.GELU(),
            nn.Linear(head_b_hidden_dim, 1)
        )

    def forward(
        self,
        h_i: torch.Tensor,
        v_hybrid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param h_i: MMoE Trunk temsili, shape (batch_size, n_assets, trunk_dim) veya (batch_size, trunk_dim)
        :param v_hybrid: Hibrit volatilite vektörü, shape (batch_size, n_assets, hybrid_vol_dim) veya (batch_size, hybrid_vol_dim)
        :return: (logit_head_a, logit_head_b) tuple
        """
        # 1. Head A Logiti Hesabı: logit_A = HeadA(h_i)
        logit_head_a = self.head_a_net(h_i)

        # 2. ADIM 3.3.1: Autograd Grafiğinden Koparma (DETACH)
        # Gradient Leakage engellemek için Head A logitinin gradyan bağlantısı kesilir.
        logit_head_a_detached = logit_head_a.detach()

        # 3. Head B Girdisi İnşası: input_HeadB = concat(h_i, logit_HeadA.detach(), V_t_hybrid, dim=-1)
        head_b_input = torch.cat([h_i, logit_head_a_detached, v_hybrid], dim=-1)

        # 4. Head B Logiti Hesabı: logit_B = HeadB(input_HeadB)
        logit_head_b = self.head_b_net(head_b_input)

        # Logit Şişmesini / Patlamasını Engellemek İçin Kırpma Zırhı ([-10.0, +10.0])
        logit_head_a_clamped = torch.clamp(logit_head_a, min=-10.0, max=10.0)
        logit_head_b_clamped = torch.clamp(logit_head_b, min=-10.0, max=10.0)

        return logit_head_a_clamped, logit_head_b_clamped


class MMoEBackboneModule(nn.Module):
    """
    Adım 3.5.1: PyTorch ile Çoklu Expert ve Kapılama (Gating) MMoE Mimarisinin Kodlanması.
    - N adet paylaşımlı Shared Expert ağı (MLP).
    - Head A için özel Gate A (Softmax weighting).
    - Head B için özel Gate B (Softmax weighting).
    - DetachCoupledPredictionHeads ile entegre kriz zırhlı omurga.
    """

    def __init__(
        self,
        input_dim: int,
        hybrid_vol_dim: int,
        num_experts: int = 4,
        expert_hidden_dim: int = 64,
        expert_output_dim: int = 32,
        head_a_hidden_dim: int = 64,
        head_b_hidden_dim: int = 64
    ):
        """
        :param input_dim: Girdi özellik boyutu (dim(x)).
        :param hybrid_vol_dim: Hibrit volatilite vektör boyutu.
        :param num_experts: Paylaşımlı expert sayısı (Varsayılan 4).
        :param expert_hidden_dim: Expert gizli katman genişliği.
        :param expert_output_dim: Expert çıktı temsil genişliği (trunk_dim).
        :param head_a_hidden_dim: Head A gizli katman genişliği.
        :param head_b_hidden_dim: Head B gizli katman genişliği.
        """
        super().__init__()
        self.num_experts = num_experts
        self.expert_output_dim = expert_output_dim

        # 0. Cross-Asset Self-Attention Katmanı (Çapraz Varlık Göreli Güç Dikkat Katmanı)
        self.cross_asset_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=2, batch_first=True)
        self.attn_norm = nn.LayerNorm(input_dim)

        # 1. Shared Experts (Paylaşımlı Uzman Ağlar)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden_dim),
                nn.GELU(),
                nn.Linear(expert_hidden_dim, expert_output_dim)
            ) for _ in range(num_experts)
        ])

        # 2. Gate A (Head A için kapılama ağı)
        self.gate_a = nn.Linear(input_dim, num_experts)

        # 3. Gate B (Head B için kapılama ağı)
        self.gate_b = nn.Linear(input_dim, num_experts)

        # 4. Detach-Coupled Prediction Heads (Faz 3.3)
        self.prediction_heads = DetachCoupledPredictionHeads(
            trunk_dim=expert_output_dim,
            hybrid_vol_dim=hybrid_vol_dim,
            head_a_hidden_dim=head_a_hidden_dim,
            head_b_hidden_dim=head_b_hidden_dim
        )

    def forward(
        self,
        x: torch.Tensor,
        v_hybrid: torch.Tensor
    ) -> MMoEOutput:
        """
        MMoE İleri İletim Operasyonu.

        :param x: Girdi özellik tensorü, shape (batch_size, n_assets, input_dim) veya (batch_size, input_dim)
        :param v_hybrid: Hibrit volatilite tensorü, shape (batch_size, n_assets, hybrid_vol_dim) veya (batch_size, hybrid_vol_dim)
        :return: MMoEOutput nesnesi
        """
        # 1. Expert Çıktılarının Hesaplanması
        # Her expert E_m(x) -> shape (..., expert_output_dim)
        expert_outputs = [expert(x) for expert in self.experts]
        
        # Expert çıktılarını istifle (stack) -> shape (..., num_experts, expert_output_dim)
        stacked_experts = torch.stack(expert_outputs, dim=-2)

        # 2. Kapılama Ağıllıklarının Hesaplanması (Softmax Weighting)
        # Gate A weights -> shape (..., num_experts)
        gate_a_weights = F.softmax(self.gate_a(x), dim=-1)
        # Gate B weights -> shape (..., num_experts)
        gate_b_weights = F.softmax(self.gate_b(x), dim=-1)

        # 3. Ağırlaştırılmış Toplam (Weighted Sum of Experts)
        # h_A = Sum_m ( gate_a_m * E_m(x) ) -> shape (..., expert_output_dim)
        h_a = torch.sum(gate_a_weights.unsqueeze(-1) * stacked_experts, dim=-2)
        # h_B = Sum_m ( gate_b_m * E_m(x) ) -> shape (..., expert_output_dim)
        h_b = torch.sum(gate_b_weights.unsqueeze(-1) * stacked_experts, dim=-2)

        # 4. Prediction Heads (Head A uses h_a, Head B uses h_b + logit_A.detach() + v_hybrid)
        logit_a, _ = self.prediction_heads(h_a, v_hybrid)
        _, logit_b = self.prediction_heads(h_b, v_hybrid)

        return MMoEOutput(
            logit_head_a=logit_a,
            logit_head_b=logit_b,
            trunk_representation=h_a,
            gate_a_weights=gate_a_weights,
            gate_b_weights=gate_b_weights
        )

    def calculate_gating_entropy_loss(self, gate_weights: torch.Tensor, lambda_entropy: float = 0.05) -> torch.Tensor:
        """
        MMoE Gating Entropy Regularization:
        Kapılama ağının tek bir uzmana (expert) kilitlenmesini engeller ve uzmanların
        tamamının dinamik şekilde kullanılmasını sağlar.
        Loss_entropy = - lambda_entropy * Mean(Sum(p_k * log(p_k + 1e-8)))
        """
        p = torch.clamp(gate_weights, min=1e-8, max=1.0)
        entropy = -torch.sum(p * torch.log(p), dim=-1)
        # Entropiyi maksimize etmek için eksi entropi cezası uygulanır
        return -lambda_entropy * torch.mean(entropy)


