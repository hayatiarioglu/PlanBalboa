import torch
import random
from typing import Tuple, List, Dict

class ExperienceReplayBuffer:
    """
    Experience Replay Buffer (Deneyim Hafızası Motoru)
    
    Modelin son 104 haftalık (2 yıllık) haftalık cross-sectional snapshot'larını saklar.
    Fine-tuning adımlarında yeni haftanın verisi (%20) ile geçmiş hafızadan rastgele
    örneklenen verileri (%80) harmanlayarak Catastrophic Forgetting (Yıkıcı Unutma)
    ve tek haftalık gürültüye ezber yapma (Overfitting to Noise) riskini sıfırlar.
    """
    def __init__(self, capacity_weeks: int = 104):
        self.capacity = capacity_weeks
        self.buffer: List[Dict[str, torch.Tensor]] = []
        
    def add_snapshot(self, features: torch.Tensor, targets: torch.Tensor, metadata: Dict = None):
        """
        1 haftalık cross-sectional snapshot'ı hafızaya ekler.
        Features: [Assets, Input_Dim]
        Targets: [Assets] (Relevance grades / returns)
        """
        snapshot = {
            "features": features.detach().clone(),
            "targets": targets.detach().clone(),
            "metadata": metadata or {}
        }
        
        self.buffer.append(snapshot)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0) # En eski haftayı çıkar (Circular Buffer)
            
    def __len__(self):
        return len(self.buffer)
        
    def sample_blended_batch(self, new_features: torch.Tensor, new_targets: torch.Tensor, replay_ratio: float = 0.80) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Yeni haftanın verisi ile hafızadaki eski haftalık snapshot'ları harmanlar.
        
        Parametreler:
        - new_features: [1, Assets, Input_Dim] (T_{t+1} verisi)
        - new_targets: [1, Assets]
        - replay_ratio: 0.80 (%80 Eski Hafıza + %20 Yeni Hafta)
        
        Döndürür:
        - blended_X: [Batch_Size, Assets, Input_Dim]
        - blended_Y: [Batch_Size, Assets]
        """
        if len(self.buffer) == 0:
            # Hafıza henüz boşsa sadece yeni haftayı dön
            return new_features, new_targets
            
        num_replay_samples = int(1.0 / (1.0 - replay_ratio)) - 1
        num_replay_samples = max(1, min(num_replay_samples, len(self.buffer)))
        
        sampled_snapshots = random.sample(self.buffer, num_replay_samples)
        
        all_features = [new_features.squeeze(0) if new_features.dim() == 3 else new_features]
        all_targets = [new_targets.squeeze(0) if new_targets.dim() == 2 else new_targets]
        
        for snap in sampled_snapshots:
            all_features.append(snap["features"])
            all_targets.append(snap["targets"])
            
        blended_X = torch.stack(all_features, dim=0) # [Batch, Assets, Input_Dim]
        blended_Y = torch.stack(all_targets, dim=0)  # [Batch, Assets]
        
        return blended_X, blended_Y
