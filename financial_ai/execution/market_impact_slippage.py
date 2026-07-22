import math
from typing import Dict, Any
from financial_ai.schemas import MarketImpactInputData, MarketImpactOutputData

class MarketImpactSlippageEngine:
    """
    FAZ 21 DÜZELTME 4: Non-linear Market Impact, Slippage & Taban/Tavan Kilitlenme Modeli.
    Gerçek hayattaki likidite yokluğu ve piyasa darbe etkisini simüle ederek stop-loss kaymalarını ve
    taban serilerindeki imkansız çıkışları (Limit-Down Lock) hesaba katar.
    """

    def simulate_execution(self, input_data: MarketImpactInputData) -> MarketImpactOutputData:
        # 1. Taban Kilitlenme Kontrolü (BIST Taban Serisi / Zero Bid Volume)
        if input_data.is_limit_down_locked:
            return MarketImpactOutputData(
                ticker=input_data.ticker,
                executed_price=input_data.lower_barrier_price * 0.80, # %20 derinleşen zarar
                slippage_pct=0.20,
                execution_feasible=False,
                rejection_reason="LIMIT_DOWN_LOCK_NO_BID_VOLUME"
            )

        # 2. Non-linear Market Impact Calculation
        adv = input_data.adv_20_shares if input_data.adv_20_shares > 0 else 1.0
        vol_ratio = input_data.order_volume_shares / adv

        impact_factor = input_data.gamma * ((vol_ratio) ** input_data.alpha) * input_data.daily_volatility
        slippage_pct = max(0.001, impact_factor) # Minimum %0.1 slippage

        executed_price = input_data.lower_barrier_price * (1.0 - slippage_pct)

        return MarketImpactOutputData(
            ticker=input_data.ticker,
            executed_price=round(executed_price, 2),
            slippage_pct=round(slippage_pct, 4),
            execution_feasible=True,
            rejection_reason=None
        )
