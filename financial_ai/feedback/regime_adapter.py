from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class FeedbackRecord:
    ticker: str
    predicted_score: float
    predicted_pe: float
    actual_return_t3: float
    actual_return_t6: float
    actual_return_t12: float
    earnings_surprise_pct: float

class MarketRegimeAdapter:
    """
    Öğrenme, Geri Besleme ve Piyasa Rejimi Adaptörü
    (Feedback Loop & Concept Drift Detection)
    """

    def __init__(self, initial_pe_weight: float = 0.35):
        self.pe_weight = initial_pe_weight
        self.ev_ebitda_weight = 0.35
        self.fcf_weight = 0.30
        self.history: List[FeedbackRecord] = []

    def adapt_regime(self, risk_free_rate: float) -> Dict[str, float]:
        """
        Piyasa Rejimi Adaptasyonu:
        - Yüksek Faiz Rejimi (Rf > 0.20): Yüksek F/K cezası artar, F/K katsayısı düşürülür, NCF/EV katsayısı öne çıkarılır.
        - Sıfır Faiz Rejimi (Rf ~ 0.00): Yüksek F/K tolere edilir, büyüme (G) katsayı ağırlığı artar.
        """
        if risk_free_rate > 0.20:
            regime = "HIGH_INTEREST_RATE"
            high_pe_penalty_multiplier = 1.8
            growth_tolerance = 0.6
        elif risk_free_rate < 0.03:
            regime = "ZERO_INTEREST_RATE"
            high_pe_penalty_multiplier = 0.5
            growth_tolerance = 1.5
        else:
            regime = "NORMAL_RATE"
            high_pe_penalty_multiplier = 1.0
            growth_tolerance = 1.0

        return {
            "regime": regime,
            "high_pe_penalty_multiplier": high_pe_penalty_multiplier,
            "growth_tolerance": growth_tolerance,
            "current_pe_weight": self.pe_weight
        }

    def record_feedback(self, record: FeedbackRecord):
        """Tahmin ve gerçekleşen getiri (T+6) arasındaki sapmayı izleme ve katsayı güncelleme"""
        self.history.append(record)
        if len(self.history) >= 10:
            self._update_weights()

    def _update_weights(self):
        """Root Mean Squared Error (RMSE) bazlı dinamik ağırlık güncelleme"""
        total_error = 0.0
        for rec in self.history[-10:]:
            # Getiri beklentisi ile T+6 gerçekleşen getiri sapması
            expected_return = (rec.predicted_score - 0.5) * 0.4  # % cinsinden getiri tahmini
            error = (rec.actual_return_t6 - expected_return) ** 2
            total_error += error
        
        rmse = (total_error / 10.0) ** 0.5
        
        # Eğer RMSE yüksekse F/K ağırlığını düşür, EV/EBITDA ve FCF ağırlığını artır
        if rmse > 0.15:
            self.pe_weight = max(0.15, self.pe_weight - 0.05)
            self.ev_ebitda_weight += 0.025
            self.fcf_weight += 0.025
