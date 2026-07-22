import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ExpertNetwork(nn.Module):
    """
    Belirli bir piyasa rejimine (Boğa, Ayı, Yatay) özelleşmiş Uzman Nöral Ağ.
    """
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64):
        super(ExpertNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class GatingNetwork(nn.Module):
    """
    Piyasa Rejimini (Boğa, Ayı, Yatay) tespit edip uzmanlar arası ağırlık dağıtan Hakem Ağ.
    """
    def __init__(self, input_dim: int = 5, num_experts: int = 3):
        super(GatingNetwork, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Linear(32, num_experts)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Softmax rasyoları: sum(weights) = 1.0
        return F.softmax(self.gate(x), dim=-1)

class MoESpecialistNetwork(nn.Module):
    """
    3/3 Extreme Tail Engine Mixture of Experts (MoE) & Expert Collapse Kalkanı.
    
    1. Boğa Uzmanı (Expert 1)
    2. Ayı / Defans Uzmanı (Expert 2)
    3. Yatay Testere Uzmanı (Expert 3)
    4. Gating Hakem Ağı (Dynamic Router)
    5. Auxiliary Load-Balancing Loss (Uzman Çökmesi Kalkanı)
    """
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, num_experts: int = 3):
        super(MoESpecialistNetwork, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            ExpertNetwork(input_dim=input_dim, hidden_dim=hidden_dim)
            for _ in range(num_experts)
        ])
        self.gating = GatingNetwork(input_dim=input_dim, num_experts=num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (N, input_dim) - O günkü N varlığın faktör matrisi
        returns: (moe_output: (N, 1), aux_loss: scalar)
        """
        # 1. Hakem Ağı Ağırlıklarını Hesapla: (N, num_experts)
        gate_weights = self.gating(x)
        
        # 2. Load-Balancing Auxiliary Loss (Uzman Çökmesini Önleyen Ceza)
        P_k = gate_weights.mean(dim=0) # Ortalam gating olasılığı
        top_expert = torch.argmax(gate_weights, dim=1)
        f_k = torch.bincount(top_expert, minlength=self.num_experts).float() / x.shape[0]
        aux_loss = self.num_experts * torch.sum(f_k * P_k)
        
        # 3. Her bir uzmanın tahminlerini al: (num_experts, N, 1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=0) # (3, N, 1)
        
        # 4. Uzman Tahminlerini Hakem Ağırlıkları ile Harmanla
        gate_weights_expanded = gate_weights.unsqueeze(-1) # (N, 3, 1)
        expert_outputs_permuted = expert_outputs.permute(1, 0, 2) # (N, 3, 1)
        
        moe_output = torch.sum(gate_weights_expanded * expert_outputs_permuted, dim=1) # (N, 1)
        return moe_output, aux_loss
