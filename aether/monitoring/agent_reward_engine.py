import numpy as np
import torch
import torch.nn as nn

class AgentProximityRewardEngine:
    """
    Bireysel Ajan Tahmin Yakınlık Ödül Motoru (Proximity Reward & Penalty Engine).
    
    Her bir ajanın tahmini ile gerçekleşen getiri arasındaki farkı (Delta) hesaplar:
    - Delta <= %1.0  => +100 Puan (Zirve Ödül / Tam İsabet)
    - Delta <= %3.0  => +50 Puan  (Çok Yakın)
    - Delta <= %5.0  => +20 Puan  (Makul)
    - Delta > %10.0  => -50 Puan  (Ağır Ceza)
    """

    @staticmethod
    def calculate_proximity_score(pred_return: float, actual_return: float) -> float:
        delta = abs(pred_return - actual_return) * 100.0  # Yüzde cinsinden fark
        
        if delta <= 1.0:
            return 100.0
        elif delta <= 3.0:
            return 50.0
        elif delta <= 5.0:
            return 20.0
        elif delta <= 10.0:
            return 0.0
        else:
            return -50.0

    @staticmethod
    def compute_proximity_loss(pred_tensor: torch.Tensor, actual_tensor: torch.Tensor) -> torch.Tensor:
        """
        PyTorch gradient uyumlu MSE + Proximity Smooth L1 Loss.
        Ajanın sapmasını (Delta) sıfıra yaklaştırmayı hedefler.
        """
        diff = torch.abs(pred_tensor - actual_tensor)
        # Huber / Smooth L1 loss: Küçük sapmalarda karesel, büyüklerde doğrusal
        loss = torch.where(diff < 0.03, 0.5 * (diff / 0.03) ** 2, diff / 0.03 - 0.5)
        return torch.mean(loss)
