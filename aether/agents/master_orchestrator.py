import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

class MasterOrchestrator:
    """
    34 Uzman Varlık Ajanının Raporlarını Toplayan ve Havuz Sıralamasını Yapan Master Ajan.
    """
    def __init__(self):
        self.total_pool_score = 0.0

    def rank_predictions(self, agent_predictions: Dict[str, float]) -> Tuple[pd.Series, List[str], List[str]]:
        """
        34 ajandan gelen tahminleri alır, 1'den 34'e sıralar ve Top 3 Yükselen ile Düşen isimleri çıkarır.
        """
        s_preds = pd.Series(agent_predictions).sort_values(ascending=False)
        top3_up = list(s_preds.index[:3])
        top3_down = list(s_preds.index[-3:])
        return s_preds, top3_up, top3_down

    def evaluate_top3_hits(self, pred_top3: List[str], actual_top3: List[str]) -> Tuple[int, float]:
        """
        Kullanıcının Belirttiği Üstel İsabet Skalası:
        - 1 İsabet: +10 Puan
        - 2 İsabet: +50 Puan
        - 3 İsabet (3'te 3 Tam İsabet): +100 Puan!
        """
        hits = len(set(pred_top3).intersection(set(actual_top3)))
        if hits == 1:
            score = 10.0
        elif hits == 2:
            score = 50.0
        elif hits == 3:
            score = 100.0
        else:
            score = 0.0
            
        self.total_pool_score += score
        return hits, score
