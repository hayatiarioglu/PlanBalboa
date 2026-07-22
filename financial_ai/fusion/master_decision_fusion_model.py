from typing import Dict, Any, List
from financial_ai.schemas import (
    MasterFusionInputData, MasterFusionOutputData, SubSystemScores
)

class MasterDecisionFusionModel:
    """
    FAZ 20: ÇOK FAKTÖRLÜ KARAR FÜZYON MOTORU (Master_Decision_Fusion_Model)
    19 Çekirdek Modülün Tamamından Gelen Çıktıları Harmanlayarak Nihai Portföy ve Al/Sat Kararı Üretir.
    """

    def evaluate(self, input_data: MasterFusionInputData) -> MasterFusionOutputData:
        warnings: List[str] = []
        drivers: List[str] = []

        ff_out = input_data.free_float_output
        ff_flags = ff_out.get("flags", {})
        cri = ff_out.get("cornering_risk_index", 0.0)

        # 1. HARD SAFETY GATEKEEPER CHECK (Sığ Tahta / Manipülasyon Engeli)
        if ff_flags.get("trading_execution_blocked", False) or cri > 8.0:
            warnings.append("CRITICAL: Sığ Tahta / Cornering Manipülasyon Engeli Tetiklendi!")
            return MasterFusionOutputData(
                ticker=input_data.ticker,
                timestamp=input_data.timestamp,
                composite_master_score=0.0,
                final_recommendation="REJECTED_LOW_FLOAT_CORNERING_RISK",
                subsystem_scores={"fundamental_score": 0.0, "technical_score": 0.0, "microstructure_score": 0.0, "risk_score": 0.0},
                positive_drivers=[],
                critical_warnings=warnings,
                hard_safety_block_triggered=True,
                recommended_portfolio_weight_pct=0.0,
                max_position_cap_shares=ff_out.get("max_allowed_position_shares", 0.0)
            )

        # 2. SEKTÖREL VE MODÜLER ALT SİSTEM SKORLARI
        pe_s = input_data.pe_output.get("signal_score", 0.5)
        pb_s = input_data.pb_output.get("safety_margin_score", 0.5)
        ebitda_s = input_data.ebitda_output.get("operational_quality_score", 0.5)
        ev_s = input_data.ev_output.get("valuation_attractiveness_score", 0.5)
        roe_s = input_data.roe_output.get("capital_quality_score", 0.5)
        cr_s = input_data.current_ratio_output.get("solvency_health_score", 0.5)

        fund_score = (pe_s + pb_s + ebitda_s + ev_s + roe_s + cr_s) / 6.0

        rsi_s = input_data.rsi_output.get("technical_signal_score", 0.5)
        ma_s = input_data.ma_output.get("trend_state_score", 0.5)
        macd_s = input_data.macd_output.get("momentum_score", 0.5)
        bb_s = input_data.bollinger_output.get("volatility_breakout_score", 0.5)
        vol_s = input_data.volume_output.get("smart_money_flow_score", 0.5)
        fib_s = input_data.fib_output.get("prz_reversal_score", 0.5)

        tech_score = (rsi_s + ma_s + macd_s + bb_s + vol_s + fib_s) / 6.0

        ob_s = input_data.order_book_output.get("microstructure_signal_score", 0.5)
        akd_s = input_data.akd_output.get("akd_signal_score", 0.5)
        custody_s = input_data.custody_output.get("custody_signal_score", 0.5)
        nmf_s = input_data.microstructure_fusion_output.get("overall_microstructure_signal", 0.5)

        micro_score = (ob_s + akd_s + custody_s + nmf_s) / 4.0

        beta_s = input_data.beta_output.get("risk_contribution_score", 0.5)
        ff_liq_s = 1.0 - input_data.free_float_output.get("liquidity_risk_score", 0.5)
        sharpe_s = input_data.sharpe_output.get("final_risk_adjusted_score", 0.5)

        risk_score = (beta_s + ff_liq_s + sharpe_s) / 3.0

        # 3. COMPOSITE MASTER SCORE (AĞIRLIKLI FÜZYON)
        cms = (0.30 * fund_score) + (0.25 * tech_score) + (0.25 * micro_score) + (0.20 * risk_score)

        # 4. PENALTI VE SÜRÜCÜ ANALİZİ
        if input_data.roe_output.get("flags", {}).get("double_negative_trap"):
            cms *= 0.5
            warnings.append("PENALTY: Çift Negatiflik Özkaynak İflas Tuzağı!")

        if input_data.current_ratio_output.get("flags", {}).get("liquidity_distress_warning"):
            cms *= 0.7
            warnings.append("PENALTY: Likidite Sıkıntısı & Overtrading Riski!")

        if input_data.fib_output.get("flags", {}).get("structure_invalidated"):
            cms *= 0.8
            warnings.append("PENALTY: Fib %78.6 Kırıldı, Trend Yapısı Bozuldu (MSS)!")

        if input_data.sharpe_output.get("flags", {}).get("negative_skewness_tail_risk"):
            warnings.append("WARNING: Negatif Çarpıklık / Kuyruk Riski Var!")

        if input_data.microstructure_fusion_output.get("microstructure_regime") == "STEALTH_ACCUMULATION_DETECTED":
            drivers.append("DRIVER: Sessiz Kurumsal Mal Toplama (Stealth Accumulation) Saptandı!")

        if input_data.fib_output.get("confluence_analysis", {}).get("confluence_score", 0) >= 3:
            drivers.append("DRIVER: Güçlü Fibonacci + EMA + HVN Kesişim Desteği!")

        if input_data.akd_output.get("akd_regime") == "STRONG_INSTITUTIONAL_ACCUMULATION":
            drivers.append("DRIVER: BofA / Kurumsal %80+ AKD Alım Konsantrasyonu!")

        # 5. NİHAİ KARAR SINIFLANDIRMASI
        if cms >= 0.80:
            rec = "STRONG_BUY_INSTITUTIONAL_CONFLUENCE"
            weight = 12.5
        elif cms >= 0.65:
            rec = "BUY_ACCUMULATE"
            weight = 8.0
        elif cms >= 0.40:
            rec = "NEUTRAL_HOLD"
            weight = 4.0
        else:
            rec = "STRONG_SELL_REDUCE_EXPOSURE"
            weight = 0.0

        subsystems = SubSystemScores(
            fundamental_score=fund_score,
            technical_score=tech_score,
            microstructure_score=micro_score,
            risk_score=risk_score
        )

        return MasterFusionOutputData(
            ticker=input_data.ticker,
            timestamp=input_data.timestamp,
            composite_master_score=round(cms, 2),
            final_recommendation=rec,
            subsystem_scores=subsystems.to_dict(),
            positive_drivers=drivers,
            critical_warnings=warnings,
            hard_safety_block_triggered=False,
            recommended_portfolio_weight_pct=weight,
            max_position_cap_shares=ff_out.get("max_allowed_position_shares", 1000000.0)
        )
