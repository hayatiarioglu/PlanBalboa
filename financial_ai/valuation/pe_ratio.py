import math
from typing import Dict, Any, Tuple, Optional
from financial_ai.schemas import PEInputData, PEFlags, PEOutputData

class PEValuationEngine:
    """
    MODÜL: Valuation_Engine::PE_Ratio
    Fiyat / Kazanç (F/K) Oranı Değerleme Motoru.
    
    Tüm aşamaları kapsar:
    1. Ham Veri Ön İşleme (Negatif kâr izolasyonu, Tek seferlik kâr arındırması, Winsorization)
    2. Mantık Ağacı (Döngüsel tuzak, Borç kapanı, PEG testi, Arsa satışı tespiti)
    3. Öznitelik Mühendisliği (Relatif F/K, Tarihsel Z-Skoru, Kazanç Riski Primi ERP)
    4. Çok Faktörlü Karar Füzyon Modeli & Sinyal Üretimi
    """

    CYCLICAL_SECTORS = {"STEEL", "CEMENT", "COMMODITIES", "MINING", "PETROCHEM"}

    def __init__(self, winsorize_min: float = 1.0, winsorize_max: float = 500.0):
        self.winsorize_min = winsorize_min
        self.winsorize_max = winsorize_max

    def winsorize_pe(self, pe_value: float) -> float:
        """Winsorization (%1-%99 percentile aralığına sıkıştırma)"""
        if math.isnan(pe_value) or pe_value is None:
            return pe_value
        return max(self.winsorize_min, min(self.winsorize_max, pe_value))

    def evaluate(self, input_data: PEInputData) -> PEOutputData:
        flags = PEFlags()

        # ---------------------------------------------------------
        # ADIM 1: ÖN İŞLEM (Data Preprocessing & Validation)
        # ---------------------------------------------------------
        ni_gaap = input_data.ttm_net_income_gaap
        mc = input_data.market_cap

        # A. Negatif Kâr ve Sıfıra Bölme İzolasyonu
        if ni_gaap <= 0 or mc <= 0:
            flags.is_negative_earnings = True
            raw_pe = None
            adjusted_pe = None
            earnings_yield = ni_gaap / mc if mc > 0 else 0.0
        else:
            raw_pe = mc / ni_gaap
            raw_pe = self.winsorize_pe(raw_pe)
            earnings_yield = 1.0 / raw_pe

        # B. Tek Seferlik Kârların Arındırılması (Normalized EPS / Income)
        non_recurring_income = input_data.non_recurring_income
        non_recurring_expenses = input_data.non_recurring_expenses
        ni_adj = ni_gaap - non_recurring_income + non_recurring_expenses

        if non_recurring_income > 0:
            flags.one_off_income_detected = True

        if ni_adj <= 0 or mc <= 0:
            adjusted_pe = None
        else:
            adjusted_pe = mc / ni_adj
            adjusted_pe = self.winsorize_pe(adjusted_pe)

        # ---------------------------------------------------------
        # ADIM 2: MANTIK AĞACI (Interpretative Logic Engine)
        # ---------------------------------------------------------
        effective_pe = adjusted_pe if adjusted_pe is not None else raw_pe
        sec_pe = input_data.sector_median_pe

        # PEG Oranı Hesabı (PEG = (F/K) / Kâr Büyüme Oranı %)
        growth_rate = max(0.1, input_data.annual_growth_rate_pct)
        if effective_pe is not None:
            peg_ratio = effective_pe / growth_rate
        else:
            peg_ratio = None

        # A. Döngüsel Şirket Tuzağı (Cyclical Trap) Kontrolü
        if input_data.sector_code in self.CYCLICAL_SECTORS:
            # Döngüsel sektörlerde ham/düzeltilmiş F/K göreceli olarak düşükse veya sektör döngüsel grubundaysa
            flags.is_cyclical_trap_risk = True

        # B. Ciddi Borç Kapanı (Financial Leverage Trap)
        net_debt_to_ebitda = input_data.net_debt / max(1.0, input_data.ebitda)
        if net_debt_to_ebitda > 5.0:
            flags.is_debt_trap_risk = True

        # C. Aşırı Düşük F/K (< 3.0) veya Tek Seferlik Arsa Satışı
        if effective_pe is not None and effective_pe < 3.0:
            if flags.one_off_income_detected:
                flags.is_distressed_pe = True

        # D. Yüksek F/K (> Sektör F/K * 1.5) - Büyüme mi Balon mu?
        if effective_pe is not None and effective_pe > (sec_pe * 1.5):
            if peg_ratio is not None and peg_ratio > 2.0:
                flags.is_bubble_pricing = True

        # ---------------------------------------------------------
        # ADIM 3: ÖZNİTELİK MÜHENDİSLİĞİ (Feature Engineering)
        # ---------------------------------------------------------
        pe_rel = (effective_pe / sec_pe) if (effective_pe is not None and sec_pe > 0) else 1.0

        if (effective_pe is not None and 
            input_data.historical_5y_pe_mean is not None and 
            input_data.historical_5y_pe_std is not None and 
            input_data.historical_5y_pe_std > 0):
            pe_z = (effective_pe - input_data.historical_5y_pe_mean) / input_data.historical_5y_pe_std
        else:
            pe_z = 0.0

        erp = (1.0 / effective_pe - input_data.risk_free_rate) if effective_pe is not None else -input_data.risk_free_rate

        # ---------------------------------------------------------
        # ADIM 4: KARAR VE TAHMİN KATMANI (Decision Fusion Engine)
        # ---------------------------------------------------------
        signal_score, confidence, recommendation = self._compute_fusion_score(
            effective_pe=effective_pe,
            pe_rel=pe_rel,
            peg_ratio=peg_ratio,
            erp=erp,
            pe_z=pe_z,
            flags=flags,
            input_data=input_data
        )

        return PEOutputData(
            ticker=input_data.ticker,
            raw_pe=raw_pe,
            adjusted_pe=adjusted_pe,
            earnings_yield=earnings_yield,
            peg_ratio=peg_ratio,
            flags=flags.to_dict(),
            signal_score=signal_score,
            confidence_interval=confidence,
            primary_recommendation_contribution=recommendation
        )

    def _compute_fusion_score(
        self,
        effective_pe: Optional[float],
        pe_rel: float,
        peg_ratio: Optional[float],
        erp: float,
        pe_z: float,
        flags: PEFlags,
        input_data: PEInputData
    ) -> Tuple[float, float, str]:
        """
        Çok Faktörlü Karar Matrisi (Multi-Factor Decision Matrix)
        """
        base_score = 0.50
        confidence = 0.88

        # 1. Negatif Kâr
        if flags.is_negative_earnings:
            return 0.15, 0.90, "BEARISH_NEGATIVE_EARNINGS"

        # 2. Balon Fiyatlama
        if flags.is_bubble_pricing:
            return 0.20, 0.92, "SELL_BUBBLE_PRICING"

        # 3. Borç Tuzağı veya Ciddi Sıkıntı
        if flags.is_distressed_pe or flags.is_debt_trap_risk:
            return 0.25, 0.88, "BEARISH_VALUE_TRAP"

        # Rule 1: PE_rel < 0.8 VE PEG < 1.0 VE Net Debt / EBITDA < 2.0 VE ROIC > WACC -> STRONG_BUY
        net_debt_ebitda = input_data.net_debt / max(1.0, input_data.ebitda)
        if pe_rel < 0.8 and (peg_ratio is not None and peg_ratio < 1.0) and net_debt_ebitda < 2.0 and input_data.roic > input_data.wacc:
            return 0.85, 0.95, "STRONG_BUY"

        # 4. Makul Fiyatlı Büyüme (PEG < 1.0 Muafiyeti)
        if peg_ratio is not None and peg_ratio < 1.0:
            # Yüksek F/K cezası PEG < 1.0 durumunda uygulanmaz, kâr büyümesi yüksek ve makuldür
            base_score = 0.75
            if erp > 0.0:
                base_score += 0.05
            return min(1.0, base_score), 0.90, "BUY"

        # 5. Döngüsel Risk Durumu (Cyclical Trap)
        if flags.is_cyclical_trap_risk:
            # Döngüsel hisselerde yüksek/tek seferlik kâr yanıltıcı olabilir
            if flags.one_off_income_detected or pe_rel > 1.2:
                base_score = 0.32
                return base_score, 0.88, "NEUTRAL_TO_BEARISH"

        # Standard Relatif F/K Skoru
        if pe_rel < 0.8:
            base_score += 0.15
        elif pe_rel > 1.5:
            base_score -= 0.20

        # Yüksek Faiz Rejimi Cezası
        if input_data.risk_free_rate > 0.20 and pe_rel > 1.2:
            base_score -= 0.15

        if erp > 0.05:
            base_score += 0.05
        elif erp < 0.0:
            base_score -= 0.05

        final_score = max(0.0, min(1.0, base_score))

        if final_score >= 0.75:
            recommendation = "STRONG_BUY"
        elif final_score >= 0.60:
            recommendation = "BUY"
        elif final_score >= 0.40:
            recommendation = "NEUTRAL"
        elif final_score >= 0.25:
            recommendation = "NEUTRAL_TO_BEARISH"
        else:
            recommendation = "SELL"

        return final_score, confidence, recommendation
