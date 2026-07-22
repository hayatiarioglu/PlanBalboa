from typing import List, Dict
import scipy.stats as stats

class CrossSectionalRankEngine:
    """
    FAZ 21 DÜZELTME 5: Kesit Alan Keskinleştirme (Cross-Sectional Point-in-Time Percentile Ranking).
    Tüm sürekli metrikleri her t anında evrendeki diğer hisselere kıyasla [0, 1] dilimine dönüştürür.
    Gelecek/Geçmiş veri sızıntısını (Global Scaling Leakage) %100 engeller.
    """

    @staticmethod
    def transform_percentile_rank(values: List[float]) -> List[float]:
        n = len(values)
        if n <= 1:
            return [0.5] * n

        ranks = stats.rankdata(values, method="average")
        percentile_ranks = [(r - 1.0) / (n - 1.0) for r in ranks]
        return [round(x, 4) for x in percentile_ranks]
