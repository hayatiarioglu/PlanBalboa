from financial_ai.schemas import MetaLabelingInputData, MetaLabelingOutputData

class MetaLabelingEngine:
    """
    FAZ 21: Meta-Labeling ve Otonom Sinyal Onay Motoru (Pure Signal Engine)
    Portföy Half-Kelly lot/pay hesaplaması tamamen kaldırılmıştır.
    Sadece birincil model sinyalinin başarı olasılığını P(Success) tahmin eder.
    """

    def evaluate(self, input_data: MetaLabelingInputData) -> MetaLabelingOutputData:
        if input_data.primary_signal_side == 0:
            return MetaLabelingOutputData(
                ticker=input_data.ticker,
                p_success=0.50,
                p_size=0.0,
                is_meta_approved=False
            )

        # Meta-Model Probability Estimation P(Success)
        base_p = input_data.historical_win_rate
        spread_penalty = min(0.15, input_data.bid_ask_spread * 10.0)
        obi_boost = max(-0.15, min(0.20, input_data.order_book_imbalance_obi * 0.25))
        vol_boost = max(-0.10, min(0.15, (0.02 - input_data.volatility_atr) * 5.0))

        p_success = max(0.0, min(1.0, base_p + obi_boost + vol_boost - spread_penalty))
        p_size = 2.0 * p_success - 1.0
        approved = p_success >= 0.55

        return MetaLabelingOutputData(
            ticker=input_data.ticker,
            p_success=round(p_success, 4),
            p_size=round(p_size, 4),
            is_meta_approved=approved
        )
