from financial_ai.schemas import MacroOverlayInputData, MacroOverlayOutputData

class AbsoluteMacroOverlayEngine:
    """
    KUSURSUZ MİMARİ DÜZELTME 2: Mutlak Makro Rejim Katmanı (Absolute Macro Overlay Engine).
    Cross-Sectional Percentile Rank'in makro piyasa çöküşlerindeki göreli körlüğünü engeller.
    BIST100 < EMA_200 VE Crash_Flag == True ise FORCE_CASH_SYSTEMIC_CRASH_PROTECTION tetikler.
    """

    def evaluate(self, input_data: MacroOverlayInputData) -> MacroOverlayOutputData:
        index_below_ema200 = input_data.bist100_index_price < input_data.bist100_ema_200
        high_volatility = input_data.volatility_index_vix > input_data.volatility_threshold

        is_systemic_crash = input_data.systemic_crash_flag or (index_below_ema200 and high_volatility)

        if is_systemic_crash:
            return MacroOverlayOutputData(
                macro_state=0,
                is_cash_protection_active=True,
                macro_regime_label="FORCE_CASH_SYSTEMIC_CRASH_PROTECTION",
                risk_multiplier=0.0
            )

        return MacroOverlayOutputData(
            macro_state=1,
            is_cash_protection_active=False,
            macro_regime_label="NORMAL_MARKET_REGIME",
            risk_multiplier=1.0
        )
