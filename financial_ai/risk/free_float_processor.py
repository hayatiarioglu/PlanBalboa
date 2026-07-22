from financial_ai.schemas import FreeFloatInputData, FreeFloatOutputData

class FreeFloatProcessor:
    """
    MODÜL 18: Serbest Dolaşım Oranı ve Likidite Risk Motoru (Pure Signal Engine)
    Portföy lot hesabı kaldırılmıştır. Sadece Sığ Tahta Kitleme Riski (Cornering Risk)
    ve Sert Sistem İnfaz Engeli (Hard System Block) sinyali üretir.
    """

    def evaluate(self, input_data: FreeFloatInputData) -> FreeFloatOutputData:
        free_float_pct = (input_data.free_float_shares / input_data.total_shares_outstanding) * 100.0
        free_float_mc_tl = input_data.free_float_shares * input_data.current_price

        # Dolaşım Rotasyon Hızı
        rotation_speed = input_data.daily_volume_shares / input_data.free_float_shares if input_data.free_float_shares > 0 else 0.0

        # Cornering Risk Index (CRI)
        import math
        mc_log = math.log10(free_float_mc_tl) if free_float_mc_tl > 1.0 else 1.0
        cri = 1.0 / (mc_log * (free_float_pct / 100.0)) if free_float_pct > 0 else 99.0

        is_micro_float = free_float_pct < 10.0 or cri > 8.0
        block_trading = free_float_pct < 10.0 or cri > 8.0

        flags = {
            "is_micro_float_danger": is_micro_float,
            "lockup_cliff_risk_active": False,
            "trading_execution_blocked": block_trading
        }

        risk_score = max(0.0, min(1.0, cri / 10.0))

        contrib = "HARD_SYSTEM_BLOCK_LOW_FLOAT" if block_trading else "NORMAL_FLOAT_LIQUIDITY"

        return FreeFloatOutputData(
            ticker=input_data.ticker,
            free_float_pct=round(free_float_pct, 2),
            free_float_market_cap_tl=round(free_float_mc_tl, 2),
            cornering_risk_index=round(cri, 2),
            float_rotation_speed=round(rotation_speed, 2),
            flags=flags,
            liquidity_risk_score=round(risk_score, 2),
            primary_recommendation_contribution=contrib
        )
