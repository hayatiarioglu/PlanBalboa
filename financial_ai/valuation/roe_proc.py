import math
from typing import Dict, Any, Tuple, Optional
from financial_ai.schemas import ROEInputData, ROEFlags, DuPontBreakdown, ROEOutputData

class ROEValuationEngine:
    """
    MODÜL: Valuation_Engine::ROE_Proc
    Özvarlık Kârlılığı (ROE) ve DuPont 5-Aşamalı Ayrıştırma Motoru.
    
    Tüm aşamaları kapsar:
    1. Ön İşlem (Ağırlıklı Ortalama Özkaynak, Çift Negatiflik İzolasyonu, Core ROE)
    2. Mantık Ağacı (DuPont Sürücü Analizi, Aşırı Borç Tuzağı, Reel Sermaye Tahribatı)
    3. Öznitelik Mühendisliği (Sürdürülebilir Büyüme Oranı SGR, Reel ROE Spredi, Justified P/B)
    4. Sermaye Kalite Skoru (Capital Quality Score) & Sinyal Üretimi
    """

    def evaluate(self, input_data: ROEInputData) -> ROEOutputData:
        flags = ROEFlags()

        ni_ttm = input_data.net_income_ttm
        eq_t0 = input_data.equity_t0
        eq_t4 = input_data.equity_t4

        # Ağırlıklı Ortalama Özkaynak (BV_Avg)
        bv_avg = (eq_t0 + eq_t4) / 2.0

        # ---------------------------------------------------------
        # ADIM 1: ÖN İŞLEM VE ÇİFT NEGATİFLİK İZOLASYONU
        # ---------------------------------------------------------
        if bv_avg <= 0:
            flags.double_negative_trap = True
            flags.capital_destruction_risk = True
            empty_dupont = DuPontBreakdown(
                tax_burden=0.0,
                interest_burden=0.0,
                ebit_margin=0.0,
                asset_turnover=0.0,
                financial_leverage=0.0,
                primary_driver="DOUBLE_NEGATIVE_BANKRUPTCY"
            )
            return ROEOutputData(
                ticker=input_data.ticker,
                raw_roe=None,
                core_roe=None,
                real_roe_spread=None,
                dupont_analysis=empty_dupont.to_dict(),
                sustainable_growth_rate=None,
                justified_pb_ratio=None,
                flags=flags.to_dict(),
                capital_quality_score=0.10,
                primary_recommendation_contribution="DOUBLE_NEGATIVE_BANKRUPTCY_TRAP"
            )

        # Ham ROE
        raw_roe = ni_ttm / bv_avg

        # Core Net Income (Tek Seferlik Gelir Arındırması)
        core_net_income = ni_ttm - input_data.one_off_income
        core_roe = core_net_income / bv_avg

        # Reel ROE Spredi = Core ROE - Inflation Rate
        real_roe_spread = core_roe - input_data.inflation_rate
        if real_roe_spread < 0:
            flags.capital_destruction_risk = True

        # ---------------------------------------------------------
        # ADIM 2: DUPONT 5-AŞAMALI AYRIŞTIRMA MODELİ
        # ---------------------------------------------------------
        revenue = max(1.0, input_data.revenue_ttm)
        total_assets = max(1.0, input_data.total_assets)
        ebit = input_data.ebit_ttm

        # DuPont Oranları
        ebit_margin = ebit / revenue
        asset_turnover = revenue / total_assets
        financial_leverage = total_assets / bv_avg

        # Implied Tax & Interest Burden
        ebt_implied = ebit * 0.92 if ebit > 0 else ebit
        tax_burden = ni_ttm / ebt_implied if ebt_implied > 0 else 0.85
        interest_burden = ebt_implied / ebit if ebit > 0 else 0.92

        # Sürücü Tespiti (Primary Driver)
        if financial_leverage > 5.0:
            primary_driver = "LEVERAGE_DRIVEN"
            flags.leverage_driven_risk = True
        elif asset_turnover > 2.0:
            primary_driver = "ASSET_TURNOVER_DRIVEN"
        elif ebit_margin > 0.15:
            primary_driver = "MOAT_MARGIN_DRIVEN"
        else:
            primary_driver = "BALANCED"

        dupont = DuPontBreakdown(
            tax_burden=tax_burden,
            interest_burden=interest_burden,
            ebit_margin=ebit_margin,
            asset_turnover=asset_turnover,
            financial_leverage=financial_leverage,
            primary_driver=primary_driver
        )

        # ---------------------------------------------------------
        # ADIM 3: SGR VE TEORİK P/B (JUSTIFIED P/B) HESABI
        # ---------------------------------------------------------
        # SGR = Core ROE * (1 - Payout Ratio)
        payout = max(0.0, min(1.0, input_data.payout_ratio))
        sgr = core_roe * (1.0 - payout)

        # Justified P/B = (Core ROE - SGR) / (CoE - SGR)
        coe = input_data.cost_of_equity
        denom = max(0.01, coe - sgr)
        justified_pb_ratio = max(0.2, (core_roe - sgr) / denom)

        # ---------------------------------------------------------
        # ADIM 4: SERMAYE KALİTE SKORU VE KARAR MATRİSİ
        # ---------------------------------------------------------
        quality_score, recommendation = self._compute_quality_score(
            core_roe=core_roe,
            real_roe_spread=real_roe_spread,
            dupont=dupont,
            flags=flags,
            input_data=input_data
        )

        return ROEOutputData(
            ticker=input_data.ticker,
            raw_roe=raw_roe,
            core_roe=core_roe,
            real_roe_spread=real_roe_spread,
            dupont_analysis=dupont.to_dict(),
            sustainable_growth_rate=sgr,
            justified_pb_ratio=justified_pb_ratio,
            flags=flags.to_dict(),
            capital_quality_score=quality_score,
            primary_recommendation_contribution=recommendation
        )

    def _compute_quality_score(
        self,
        core_roe: float,
        real_roe_spread: float,
        dupont: DuPontBreakdown,
        flags: ROEFlags,
        input_data: ROEInputData
    ) -> Tuple[float, str]:
        """
        Sermaye Kalite Skoru ve Karar Matrisi
        """
        # 1. Çift Negatiflik İflas Tuzağı
        if flags.double_negative_trap:
            return 0.10, "DOUBLE_NEGATIVE_BANKRUPTCY_TRAP"

        # 2. Aşırı Borç / Kaldıraçlı ROE Tuzağı
        if flags.leverage_driven_risk:
            return 0.20, "BEARISH_HIGH_LEVERAGE_ROE"

        # 3. Reel Sermaye Tahribatı (ROE < Enflasyon)
        if flags.capital_destruction_risk:
            return 0.25, "CAPITAL_DESTRUCTION_TRAP"

        base_score = 0.50

        # Rule 1: Core ROE > CoE VE Düşük Kaldıraç -> Strong Capital Quality
        if core_roe > input_data.cost_of_equity and dupont.financial_leverage < 4.0:
            base_score = 0.85

        # Reel Spred Katkısı
        if real_roe_spread > 0.15:
            base_score += 0.07

        # Operasyonel Sürücü Katkısı (Asset Turnover veya Moat Driven)
        if dupont.primary_driver in ["ASSET_TURNOVER_DRIVEN", "MOAT_MARGIN_DRIVEN"]:
            base_score += 0.05

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
