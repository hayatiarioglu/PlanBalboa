import math
from typing import Dict, Any, Tuple, Optional
from financial_ai.schemas import EVInputData, EVFlags, EVOutputData

class EVValuationEngine:
    """
    MODÜL: Valuation_Engine::EV_EBITDA
    Firma Değeri / FAVÖK (EV / EBITDA) Değerleme Motoru.
    
    Tüm aşamaları kapsar:
    1. Ön İşlem (Bloke Nakit Temizliği, İştirak Çift Sayım İzolasyonu, EV Hesabı)
    2. Mantık Ağacı (CapEx Yoğunluk Tuzağı, Likidasyon Net-Net Fırsatı, M&A Satın Alma Hedefi Tespiti)
    3. Öznitelik Mühendisliği (EV/(EBITDA-CapEx), Leverage to EV Ratio, EV FCF Yield)
    4. Değerleme Çekicilik Skoru (Valuation Attractiveness Score) & Sinyal Üretimi
    """

    def evaluate(self, input_data: EVInputData) -> EVOutputData:
        flags = EVFlags()

        mc = input_data.market_cap
        gross_debt = input_data.gross_debt
        leases = input_data.lease_liabilities
        minority = input_data.minority_interest
        associates = input_data.associates_value
        total_cash = input_data.total_cash
        restricted_cash = input_data.restricted_cash

        # ---------------------------------------------------------
        # ADIM 1: ÖN İŞLEM VE FİRMA DEĞERİ (EV) HESABI
        # ---------------------------------------------------------
        # A. Kısıtlı Nakit İzolasyonu
        adjusted_cash = total_cash - restricted_cash
        if restricted_cash > 0:
            flags.restricted_cash_adjusted = True

        # B. Düzeltilmiş Firma Değeri (Enterprise Value)
        # EV = MC + Gross Debt + Lease Liabilities - Adjusted Cash - Associates + (Minority Interest * 0.20)
        enterprise_value = mc + gross_debt + leases - adjusted_cash - associates + (minority * 0.20)

        adjusted_ebitda = input_data.adjusted_ebitda
        capex = input_data.capex

        # C. Negatif EBITDA İzolasyonu
        if adjusted_ebitda <= 0:
            flags.is_negative_ebitda = True
            return EVOutputData(
                ticker=input_data.ticker,
                enterprise_value=enterprise_value,
                ev_to_adjusted_ebitda=None,
                ev_to_ebitda_minus_capex=None,
                leverage_to_ev_ratio=gross_debt / max(1.0, enterprise_value),
                flags=flags.to_dict(),
                valuation_attractiveness_score=0.15,
                primary_recommendation_contribution="CASH_BURN_RED_FLAG"
            )

        # D. Negatif EV (Net-Net / Likidasyon Değeri Fırsatı)
        if enterprise_value < 0:
            flags.is_net_cash_company = True

        # ---------------------------------------------------------
        # ADIM 2: ÇARPAN HESAPLAMALARI VE CAPEX TUZAĞI TESTİ
        # ---------------------------------------------------------
        ev_to_adjusted_ebitda = enterprise_value / adjusted_ebitda

        ebitda_minus_capex = adjusted_ebitda - capex
        if ebitda_minus_capex > 0:
            ev_to_ebitda_minus_capex = enterprise_value / ebitda_minus_capex
        else:
            ev_to_ebitda_minus_capex = 999.0

        # CapEx Yoğunluk Tuzağı (EBITDA görünüşte ucuz fakat CapEx sonrası aşırı pahalı)
        if (ev_to_ebitda_minus_capex / max(0.1, ev_to_adjusted_ebitda)) > 2.5:
            flags.capex_trap_risk = True

        # Leverage to EV Ratio (Gross Debt / EV)
        leverage_to_ev_ratio = gross_debt / max(1.0, abs(enterprise_value))

        # M&A Satınalma Hedefi Tespiti
        sec_median = input_data.sector_median_ev_ebitda
        if ev_to_adjusted_ebitda < (sec_median * 0.75) and not flags.capex_trap_risk:
            flags.ma_target_potential = True

        # ---------------------------------------------------------
        # ADIM 3: DEĞERLEME ÇEKİCİLİK SKORU VE KARAR MATRİSİ
        # ---------------------------------------------------------
        attractiveness_score, recommendation = self._compute_attractiveness_score(
            enterprise_value=enterprise_value,
            ev_to_adjusted_ebitda=ev_to_adjusted_ebitda,
            ev_to_ebitda_minus_capex=ev_to_ebitda_minus_capex,
            leverage_to_ev_ratio=leverage_to_ev_ratio,
            flags=flags,
            input_data=input_data
        )

        return EVOutputData(
            ticker=input_data.ticker,
            enterprise_value=enterprise_value,
            ev_to_adjusted_ebitda=ev_to_adjusted_ebitda,
            ev_to_ebitda_minus_capex=ev_to_ebitda_minus_capex,
            leverage_to_ev_ratio=leverage_to_ev_ratio,
            flags=flags.to_dict(),
            valuation_attractiveness_score=attractiveness_score,
            primary_recommendation_contribution=recommendation
        )

    def _compute_attractiveness_score(
        self,
        enterprise_value: float,
        ev_to_adjusted_ebitda: float,
        ev_to_ebitda_minus_capex: float,
        leverage_to_ev_ratio: float,
        flags: EVFlags,
        input_data: EVInputData
    ) -> Tuple[float, str]:
        """
        Değerleme Çekicilik Skoru ve Karar Matrisi
        """
        # 1. Likidasyon Fırsatı (Net-Net Şirket)
        if flags.is_net_cash_company:
            return 0.95, "STRONG_BUY_NET_NET_LIQUIDATION"

        # 2. CapEx Tuzağı (Sermaye Emici Sektörler)
        if flags.capex_trap_risk:
            return 0.22, "BEARISH_CAPEX_TRAP"

        base_score = 0.50

        # Rule 1: Sektörüne göre ucuz VE M&A Hedef Potansiyeli Yüksek
        sec_median = input_data.sector_median_ev_ebitda
        if ev_to_adjusted_ebitda < (sec_median * 0.70) and leverage_to_ev_ratio < 0.50:
            base_score = 0.86
            return base_score, "BULLISH"

        # EV/EBITDA Sektör Göreli Skorlaması
        if ev_to_adjusted_ebitda < (sec_median * 0.85):
            base_score += 0.15
        elif ev_to_adjusted_ebitda > (sec_median * 1.40):
            base_score -= 0.20

        # M&A Potansiyel Katkısı
        if flags.ma_target_potential:
            base_score += 0.10

        # Kaldıraç Riski Düzeltmesi
        if leverage_to_ev_ratio > 0.60:
            base_score -= 0.15

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
