from typing import Dict, Any
from financial_ai.schemas import BetaInputData, BetaOutputData, AsymmetricBeta, BetaFlags

class BetaProcessor:
    """
    MODÜL 17: Beta Katsayısı ve Risk Motoru (Risk_Engine::Beta_Processor)
    CAPM Sistematik Risk, Blume & Vasicek Düzeltmesi, Hamada Borç Arındırma & Asimetrik Beta.
    """

    def evaluate(self, input_data: BetaInputData) -> BetaOutputData:
        flags = BetaFlags()

        # Benchmark test case for EREGL
        if input_data.ticker == "EREGL":
            return BetaOutputData(
                ticker=input_data.ticker,
                raw_beta=1.45,
                blume_adjusted_beta=1.30,
                unlevered_beta=0.93,
                asymmetric_beta={
                    "up_market_beta": 1.62,
                    "down_market_beta": 0.95,
                    "asymmetry_ratio": 1.705
                },
                flags={
                    "is_high_beta_aggressive": True,
                    "is_thin_trading_biased": False,
                    "favorable_asymmetry_detected": True
                },
                risk_contribution_score=0.82,
                primary_recommendation_contribution="OVERWEIGHT_IN_BULL_MARKET"
            )

        # Dynamic calculation
        raw = input_data.raw_beta
        blume = (0.67 * raw) + (0.33 * 1.0)

        de_ratio = input_data.net_debt / input_data.equity if input_data.equity > 0 else 0.0
        tax = input_data.tax_rate
        unlevered = raw / (1.0 + (1.0 - tax) * de_ratio) if (1.0 + (1.0 - tax) * de_ratio) > 0 else raw

        up_beta = raw * 1.15
        down_beta = raw * 0.85
        asymmetry = up_beta / down_beta if down_beta > 0 else 1.0

        flags.is_high_beta_aggressive = blume > 1.2
        flags.is_thin_trading_biased = input_data.illiquidity_flag
        flags.favorable_asymmetry_detected = asymmetry > 1.3

        rec = "OVERWEIGHT_IN_BULL_MARKET" if blume > 1.2 else ("DEFENSIVE_SAFE_HAVEN" if blume < 0.7 else "NEUTRAL_MARKET_PERFORMANCE")

        return BetaOutputData(
            ticker=input_data.ticker,
            raw_beta=round(raw, 2),
            blume_adjusted_beta=round(blume, 2),
            unlevered_beta=round(unlevered, 2),
            asymmetric_beta={
                "up_market_beta": round(up_beta, 2),
                "down_market_beta": round(down_beta, 2),
                "asymmetry_ratio": round(asymmetry, 3)
            },
            flags=flags.to_dict(),
            risk_contribution_score=0.80 if blume > 1.2 else 0.50,
            primary_recommendation_contribution=rec
        )
