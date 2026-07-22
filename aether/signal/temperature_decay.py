"""
AetherForecaster-X v35.2 Multi-Asset Master
Katman 2: Clamped Scheduled Soft-NDCG & Detach-Coupled MMoE

Modül: Scheduled Temperature Decay Engine (Dinamik Sıcaklık Sönümleme Motoru)
Faz 3.4 / Adım 3.4.1: Soft-NDCG sıralama yumuşatmasını epoch ilerledikçe dinamik olarak
sertleştiren tau(e) = tau_min + (tau_0 - tau_min) * exp(-alpha * e) sönümleme katmanı.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union
import math
import numpy as np
import torch


@dataclass
class TemperatureDecayResult:
    """
    Sıcaklık sönümleme adım çıktısı.
    """
    epoch: int
    tau: float
    tau_tensor: torch.Tensor


class ScheduledTemperatureDecayEngine:
    """
    Adım 3.4.1: Sıcaklık parametresini Epoch adımına bağlayan decay modülü.
    tau(e) = tau_min + (tau_0 - tau_min) * exp(-alpha * e)
    """

    def __init__(
        self,
        tau_0: float = 1.0,
        tau_min: float = 0.10,
        alpha: float = 0.05,
        total_epochs: Optional[int] = None
    ):
        """
        :param tau_0: Başlangıç sıcaklığı (Epoch 0'da pürüzsüz gradyan akışı için 1.0).
        :param tau_min: Taban sıcaklık (Float overflow zırhı için tau_min >= 0.10).
        :param alpha: Üstel sönümleme hızı (katsayısı alpha > 0).
        :param total_epochs: Eğer verilirse alpha otomatik hesaplanır: tau(E) ~ tau_min + 0.01.
        """
        if tau_min < 0.10:
            raise ValueError(f"Geçersiz tau_min! Float overflow zırhı için tau_min >= 0.10 olmalıdır: {tau_min}")
        if tau_0 <= tau_min:
            raise ValueError(f"Geçersiz tau_0! tau_0 ({tau_0}) > tau_min ({tau_min}) olmalıdır.")

        self.tau_0 = tau_0
        self.tau_min = tau_min

        if total_epochs is not None and total_epochs > 0:
            # tau(E) = tau_min + (tau_0 - tau_min) * exp(-alpha * E) = tau_min + 0.01
            # => exp(-alpha * E) = 0.01 / (tau_0 - tau_min)
            # => alpha = -ln(0.01 / (tau_0 - tau_min)) / E
            target_delta = 0.01
            init_delta = tau_0 - tau_min
            if init_delta > target_delta:
                self.alpha = -math.log(target_delta / init_delta) / float(total_epochs)
            else:
                self.alpha = alpha
        else:
            self.alpha = alpha

    def get_tau(
        self,
        epoch: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ) -> TemperatureDecayResult:
        """
        Verilen epoch adımı için tau(e) değerini hesaplar.

        :param epoch: Güncel epoch adımı (e >= 0).
        :param device: PyTorch cihazı (CPU/CUDA).
        :param dtype: Tensor veri tipi.
        :return: TemperatureDecayResult nesnesi.
        """
        if epoch < 0:
            raise ValueError(f"Epoch adımı negatif olamaz: {epoch}")

        # tau(e) = tau_min + (tau_0 - tau_min) * exp(-alpha * e)
        decay_factor = math.exp(-self.alpha * float(epoch))
        tau_val = self.tau_min + (self.tau_0 - self.tau_min) * decay_factor

        # Güvenlik kırpması (tau >= tau_min)
        tau_val = max(float(tau_val), self.tau_min)

        tau_tensor = torch.tensor(tau_val, device=device, dtype=dtype)

        return TemperatureDecayResult(
            epoch=epoch,
            tau=tau_val,
            tau_tensor=tau_tensor
        )
