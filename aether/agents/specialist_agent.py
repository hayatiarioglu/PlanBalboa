try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    optim = None

import numpy as np
from typing import Dict, Any

class SpecialistAgentNetwork:
    """
    Tek bir varlığa (ör: THYAO, MGROS) özel Derin Nöral Ağ Modeli.
    Varlığın 20-faktörlü mikro dinamiklerini ve davranışını öğrenir.
    """
    def __init__(self, input_dim: int = 20, hidden_dim: int = 64):
        if HAS_TORCH:
            import torch.nn as nn
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        else:
            # Numpy Fallback weights
            np.random.seed(42)
            self.w1 = np.random.randn(input_dim, hidden_dim) * 0.1
            self.w2 = np.random.randn(hidden_dim, 1) * 0.1

    def forward(self, x) -> Any:
        if HAS_TORCH:
            return self.feature_extractor(x)
        else:
            h = np.maximum(0, np.dot(x, self.w1)) # ReLU
            return np.dot(h, self.w2)

class SpecialistAgent:
    """
    34 Varlığın Her Biri İçin Olağanüstü Profesyonel Uzman Varlık Ajanı.
    """
    def __init__(self, symbol: str, input_dim: int = 20, lr: float = 1e-3):
        self.symbol = symbol
        self.model = SpecialistAgentNetwork(input_dim=input_dim)
        if HAS_TORCH:
            self.optimizer = optim.AdamW(self.model.feature_extractor.parameters(), lr=lr, weight_decay=1e-4)
            self.criterion = nn.SmoothL1Loss()
        self.total_reward_score = 0.0

    def predict(self, feature_vector: np.ndarray) -> float:
        """
        Varlığın 20-faktörlü öznitelik vektörünü alıp tahmini getirisini döner.
        """
        if HAS_TORCH:
            self.model.feature_extractor.eval()
            with torch.no_grad():
                x_t = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
                pred = self.model(x_t).item()
            return pred
        else:
            x_arr = np.array(feature_vector, dtype=np.float32).flatten()
            res = self.model.forward(x_arr)
            return float(res[0] if isinstance(res, np.ndarray) else res)

    def update_agent(self, feature_vector: np.ndarray, actual_return: float, proximity_score: float) -> float:
        """
        Gerçekleşen getiri açıldığında ajanın kendi nöral ağırlıklarını günceller.
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        x_t = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
        y_t = torch.tensor([[actual_return]], dtype=torch.float32)
        
        pred = self.model(x_t)
        loss = self.criterion(pred, y_t)
        loss.backward()
        self.optimizer.step()
        
        self.total_reward_score += proximity_score
        return loss.item()
