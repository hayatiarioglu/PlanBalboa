"""
AetherForecaster-X v35.2 Multi-Asset Master
Katman 2: Clamped Scheduled Soft-NDCG & Detach-Coupled MMoE

Modül: Signal Neural Pipeline Orchestrator (Katman 2 Bütünsel Entegratör)
Faz 3.6 / Adım 3.6.1: Katman 2 nöral ağ bileşenlerinin (Winsorized Scaler, Delist Decoupling,
MMoE Backbone, Temperature Decay, Clamped Soft-NDCG Loss) uçtan uca tek noktadan orkestrasyonu.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from aether.signal.target_generator import DelistDecouplingTargetGenerator, TargetGeneratorResult
from aether.signal.mmoe_backbone import MMoEBackboneModule, MMoEOutput
from aether.signal.temperature_decay import ScheduledTemperatureDecayEngine
from aether.signal.soft_ndcg_loss import ClampedScheduledSoftNDCGLoss


@dataclass
class SignalPipelineResult:
    """
    Katman 2 Nöral Pipeline Eğitim ve Tahmin Sonuç Raporu.
    """
    total_loss: torch.Tensor
    logit_head_a: torch.Tensor
    logit_head_b: torch.Tensor
    targets: torch.Tensor
    current_tau: float
    gate_a_weights: torch.Tensor
    gate_b_weights: torch.Tensor


class SignalPipelineOrchestrator(nn.Module):
    """
    Adım 3.6.1: Katman 2 Nöral Ağ Bileşenlerinin Bütünsel Entegrasyon Orkestratörü.
    """

    def __init__(
        self,
        input_dim: int,
        hybrid_vol_dim: int,
        num_experts: int = 4,
        expert_hidden_dim: int = 64,
        expert_output_dim: int = 32,
        head_a_hidden_dim: int = 64,
        head_b_hidden_dim: int = 64,
        tau_0: float = 1.0,
        tau_min: float = 0.10,
        alpha_decay: float = 0.05,
        eta_delist: float = 1.0,
        top_k: Optional[int] = 5
    ):
        super().__init__()
        # 1. Target Generator & Delist Decoupler (Faz 3.1)
        self.target_generator = DelistDecouplingTargetGenerator(delist_target_penalty=-2.0)

        # 2. MMoE Backbone & Detach-Coupled Heads (Faz 3.3 & 3.5)
        self.mmoe_model = MMoEBackboneModule(
            input_dim=input_dim,
            hybrid_vol_dim=hybrid_vol_dim,
            num_experts=num_experts,
            expert_hidden_dim=expert_hidden_dim,
            expert_output_dim=expert_output_dim,
            head_a_hidden_dim=head_a_hidden_dim,
            head_b_hidden_dim=head_b_hidden_dim
        )

        # 3. Scheduled Temperature Decay Engine (Faz 3.4)
        self.decay_engine = ScheduledTemperatureDecayEngine(
            tau_0=tau_0,
            tau_min=tau_min,
            alpha=alpha_decay
        )

        # 4. Clamped Scheduled Soft-NDCG Loss Function (Faz 3.2)
        self.loss_fn = ClampedScheduledSoftNDCGLoss(
            top_k=top_k,
            tau_min=tau_min,
            clamp_val=10.0,
            eta_delist=eta_delist
        )

    def forward(
        self,
        x: torch.Tensor,
        returns: Union[pd.Series, np.ndarray, torch.Tensor],
        v_hybrid: torch.Tensor,
        epoch: int = 0,
        delisted_flags: Optional[Union[pd.Series, np.ndarray, torch.Tensor]] = None
    ) -> SignalPipelineResult:
        """
        Uçtan Uca Forward Pass ve Loss Hesaplama Operasyonu.

        :param x: Girdi özellik tensorü, shape (batch_size, n_assets, input_dim) veya (n_assets, input_dim)
        :param returns: Ham haftalık getiriler
        :param v_hybrid: Hibrit volatilite tensorü, shape (batch_size, n_assets, hybrid_vol_dim)
        :param epoch: Güncel eğitim epoch adımı
        :param delisted_flags: Delist hisse bayrakları
        :return: SignalPipelineResult nesnesi
        """
        device = x.device
        dtype = x.dtype

        # 1. Target Preparation (Faz 3.1: Decoupling + Winsorizing)
        if isinstance(returns, torch.Tensor):
            ret_np = returns.detach().cpu().numpy()
        else:
            ret_np = returns

        if delisted_flags is not None and isinstance(delisted_flags, torch.Tensor):
            delist_np = delisted_flags.detach().cpu().numpy()
        else:
            delist_np = delisted_flags

        target_res = self.target_generator.generate_targets(ret_np, delisted_flags=delist_np)

        targets_tensor = torch.tensor(target_res.combined_targets, device=device, dtype=dtype)
        delist_mask_tensor = torch.tensor(target_res.delisted_mask, device=device, dtype=torch.bool)

        # 2. Temperature Decay Schedule (Faz 3.4)
        tau_res = self.decay_engine.get_tau(epoch=epoch, device=device, dtype=dtype)

        # 3. MMoE Backbone Forward Pass (Faz 3.3 & 3.5)
        mmoe_out = self.mmoe_model(x, v_hybrid)

        # 4. Soft-NDCG Loss Calculation (Faz 3.2)
        loss_val = self.loss_fn(
            y_pred=mmoe_out.logit_head_b.squeeze(-1),
            y_true=targets_tensor,
            delisted_mask=delist_mask_tensor,
            tau=tau_res.tau_tensor
        )

        return SignalPipelineResult(
            total_loss=loss_val,
            logit_head_a=mmoe_out.logit_head_a,
            logit_head_b=mmoe_out.logit_head_b,
            targets=targets_tensor,
            current_tau=tau_res.tau,
            gate_a_weights=mmoe_out.gate_a_weights,
            gate_b_weights=mmoe_out.gate_b_weights
        )
