import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Set

class MasterRanker:
    """
    3/3 Extreme Tail Engine Master Ranker & Epistemic Uncertainty Gate.
    
    Zırhlar:
    1. Portföy Kota Kalkanı: En fazla 1 TEFAS Fonu, En az 2 BIST Hissesi.
    2. Epistemic Uncertainty Gate: Monte Carlo Dropout ile 10 alt simülasyon koşturulur.
    3. Konsensüs Güven Şartı: 10 geçişin en az 8'i (%80+) aynı varlık üzerinde uzlaşıyorsa İŞLEM AÇILIR; aksi takdirde "PAS GEÇ / BEKLE" kararı verilir.
    """
    def __init__(self, hysteresis_threshold: int = 5, max_funds: int = 1, min_stocks: int = 2, confidence_threshold: float = 0.70):
        self.hysteresis_threshold = hysteresis_threshold
        self.max_funds = max_funds
        self.min_stocks = min_stocks
        self.confidence_threshold = confidence_threshold

    def select_top3_with_quota(self, sorted_assets: List[str]) -> List[str]:
        """
        Sıralı varlık listesinden en fazla 1 TEFAS fonu ve en az 2 BIST hissesi seçer.
        """
        selected = []
        fund_count = 0
        
        for asset in sorted_assets:
            if len(selected) >= 3:
                break
                
            is_fund = not asset.endswith(".IS")
            
            if is_fund:
                if fund_count < self.max_funds:
                    selected.append(asset)
                    fund_count += 1
            else:
                selected.append(asset)
                
        return selected

    def evaluate_uncertainty_gate(self, model: nn.Module, x_day: torch.Tensor, assets: List[str], num_passes: int = 10) -> Tuple[List[str], List[str], float, bool]:
        """
        Monte Carlo Dropout Ensemble ile Belirsizlik Kalkanını (Epistemic Uncertainty Gate) çalıştırır.
        """
        model.train() # Monte Carlo Dropout için train modunu aç
        
        up_counts = {a: 0 for a in assets}
        down_counts = {a: 0 for a in assets}
        
        for _ in range(num_passes):
            with torch.no_grad():
                preds_pass, _ = model(x_day)
                preds_np = preds_pass.squeeze(-1).cpu().numpy()
                
            s_preds = pd.Series(preds_np, index=assets).sort_values(ascending=False)
            top3_up_pass = self.select_top3_with_quota(list(s_preds.index))
            top3_down_pass = self.select_top3_with_quota(list(s_preds.index[::-1]))
            
            for a in top3_up_pass:
                up_counts[a] += 1
            for a in top3_down_pass:
                down_counts[a] += 1
                
        top3_up_sorted = sorted(up_counts.keys(), key=lambda k: up_counts[k], reverse=True)
        top3_down_sorted = sorted(down_counts.keys(), key=lambda k: down_counts[k], reverse=True)
        
        top3_up = self.select_top3_with_quota(top3_up_sorted)
        top3_down = self.select_top3_with_quota(top3_down_sorted)
        
        avg_up_consensus = sum(up_counts[a] for a in top3_up) / (3.0 * num_passes)
        is_confident = (avg_up_consensus >= self.confidence_threshold)
        
        return top3_up, top3_down, avg_up_consensus, is_confident
