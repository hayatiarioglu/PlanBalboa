from typing import Dict, Any, Tuple, Optional
from financial_ai.schemas import EBITDAInputData, EBITDAFlags, EBITDAOutputData

class EBITDAValuationEngine:
    """
    MODÜL: Valuation_Engine::EBITDA_Proc
    FAVÖK / EBITDA (Faiz, Amortisman, Vergi Öncesi Kâr) İşleme ve Operasyonel Kalite Motoru.
    """

    def evaluate(self, input_data: EBITDAInputData) -> EBITDAOutputData:
        flags = EBITDAFlags()

        # ---------------------------------------------------------
        # ADIM 1: ÖN İŞLEM VE DÜZELTMELER
        # ---------------------------------------------------------
        # Ham EBITDA = EBIT + D&A
        raw_ebitda = input_data.ebit + input_data.depreciation_amortization

        # A. IFRS 16 Kira Düzeltmesi: EBITDA_Real = Raw_EBITDA - Nakit Kira Ödemeleri
        lease_payment = input_data.ifrs16_lease_payments
        if raw_ebitda > 0 and (lease_payment / raw_ebitda) > 0.10:
            flags.ifrs16_distortion_high = True

        # B. Hisse Bazlı Ödeme (SBC) İzolasyonu
        sbc = input_data.stock_based_compensation
        if raw_ebitda > 0 and (sbc / raw_ebitda) > 0.05:
            flags.sbc_dilution_risk = True

        # C. Aktifleştirilen Harcamalar (Capitalized CapEx) Düzeltmesi
        cap_capex = input_data.capitalized_capex
        if raw_ebitda > 0 and (cap_capex / raw_ebitda) > 0.03:
            flags.capitalized_capex_warning = True

        # Düzeltilmiş EBITDA (Adjusted EBITDA)
        adjusted_ebitda = raw_ebitda - lease_payment - sbc - cap_capex
        adjusted_ebitda = max(0.01, adjusted_ebitda)

        # EBITDA Marjı = Adjusted_EBITDA / Revenue
        revenue = max(1.0, input_data.revenue)
        ebitda_margin = adjusted_ebitda / revenue

        # ---------------------------------------------------------
        # ADIM 2: ÖZNİTELİK MÜHENDİSLİĞİ VE NAKİT KALİTESİ
        # ---------------------------------------------------------
        free_cash_flow = input_data.operating_cash_flow - input_data.capex
        cash_conversion_rate = free_cash_flow / adjusted_ebitda

        if cash_conversion_rate < 0.30:
            flags.cash_flow_decoupling_risk = True

        # Kaldıraç Kapsama Süresi (Net Debt / Adjusted EBITDA)
        net_debt_to_ebitda = input_data.net_debt / adjusted_ebitda
        if net_debt_to_ebitda > 4.0:
            flags.high_leverage_warning = True

        # Turnaround Kontrolü: Net Kâr zararda fakat operasyonel olarak nakit basıyor
        if input_data.net_income_gaap < 0 and adjusted_ebitda > 0 and net_debt_to_ebitda < 4.0:
            flags.turnaround_candidate = True

        # ---------------------------------------------------------
        # ADIM 3: OPERASYONEL KALİTE SKORU VE KARAR MATRİSİ
        # ---------------------------------------------------------
        quality_score, recommendation = self._compute_quality_score(
            adjusted_ebitda=adjusted_ebitda,
            ebitda_margin=ebitda_margin,
            cash_conversion_rate=cash_conversion_rate,
            net_debt_to_ebitda=net_debt_to_ebitda,
            flags=flags,
            input_data=input_data
        )

        return EBITDAOutputData(
            ticker=input_data.ticker,
            raw_ebitda=raw_ebitda,
            adjusted_ebitda=adjusted_ebitda,
            ebitda_margin=ebitda_margin,
            cash_conversion_rate=cash_conversion_rate,
            net_debt_to_ebitda=net_debt_to_ebitda,
            flags=flags.to_dict(),
            operational_quality_score=quality_score,
            primary_recommendation_contribution=recommendation
        )

    def _compute_quality_score(
        self,
        adjusted_ebitda: float,
        ebitda_margin: float,
        cash_conversion_rate: float,
        net_debt_to_ebitda: float,
        flags: EBITDAFlags,
        input_data: EBITDAInputData
    ) -> Tuple[float, str]:
        """
        Operasyonel Kalite Skoru ve Karar Matrisi
        """
        # Nakit Akışı Kopuş Tuzağı
        if flags.cash_flow_decoupling_risk:
            return 0.18, "VALUE_TRAP_NO_CASH_CONVERSION"

        # Turnaround Fırsatı
        if flags.turnaround_candidate:
            return 0.78, "TURNAROUND_BUY"

        base_score = 0.60

        # Nakit Dönüşüm Katkısı
        if cash_conversion_rate >= 0.40:
            base_score += 0.15
        elif cash_conversion_rate < 0.30:
            base_score -= 0.20

        # EBITDA Marj Katkısı
        if ebitda_margin > 0.10:
            base_score += 0.10

        # Kaldıraç Katkısı
        if net_debt_to_ebitda < 3.0:
            base_score += 0.05
        elif net_debt_to_ebitda > 4.0:
            base_score -= 0.15

        # Bayrak Cezaları
        if flags.capitalized_capex_warning:
            base_score -= 0.05
        if flags.ifrs16_distortion_high:
            base_score -= 0.04
        if flags.sbc_dilution_risk:
            base_score -= 0.10

        final_score = max(0.0, min(1.0, base_score))

        if final_score >= 0.75:
            recommendation = "BULLISH"
        elif final_score >= 0.60:
            recommendation = "MODERATE_BULLISH"
        elif final_score >= 0.40:
            recommendation = "NEUTRAL"
        else:
            recommendation = "BEARISH"

        return final_score, recommendation
