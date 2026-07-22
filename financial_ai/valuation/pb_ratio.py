import math
from typing import Dict, Any, Tuple, Optional
from financial_ai.schemas import PBInputData, PBFlags, PBOutputData

class PBValuationEngine:
    """
    MODÜL: Valuation_Engine::PB_Ratio
    Piyasa Değeri / Defter Değeri (PD/DD - P/B) ve Maddi Net Varlık Değerleme Motoru.
    
    Tüm aşamaları kapsar:
    1. Ön İşlem (Maddi Net Varlık TBV Düzeltmesi, Yeniden Değerleme, Negatif Özkaynak İzolasyonu)
    2. Mantık Ağacı (ROE vs CoE Çapraz Doğrulaması, Goodwill Tuzağı, Gizli Varlık Potansiyeli)
    3. Öznitelik Mühendisliği (Justified PB Teorik Hesabı, Discount to Justified, Net Asset Backing)
    4. Güvenlik Marjı (Margin of Safety) & Sinyal Üretim Katmanı
    """

    def __init__(self, winsorize_min: float = 0.1, winsorize_max: float = 100.0):
        self.winsorize_min = winsorize_min
        self.winsorize_max = winsorize_max

    def winsorize_pb(self, pb_value: float) -> float:
        """Winsorization (%1-%99 percentile sıkıştırması)"""
        if math.isnan(pb_value) or pb_value is None:
            return pb_value
        return max(self.winsorize_min, min(self.winsorize_max, pb_value))

    def evaluate(self, input_data: PBInputData) -> PBOutputData:
        flags = PBFlags()

        mc = input_data.market_cap
        ta = input_data.total_assets
        tl = input_data.total_liabilities

        # Total Book Value (BV_Total = TA - TL)
        bv_total = ta - tl

        # ---------------------------------------------------------
        # ADIM 1: ÖN İŞLEM VE İZOLASYON
        # ---------------------------------------------------------
        # A. Negatif Özkaynak İzolasyonu
        if bv_total <= 0 or mc <= 0:
            flags.is_negative_equity = True
            return PBOutputData(
                ticker=input_data.ticker,
                raw_pb=None,
                tangible_pb=None,
                justified_pb=None,
                discount_to_justified=0.0,
                flags=flags.to_dict(),
                safety_margin_score=0.10,
                primary_recommendation_contribution="SOLVENCY_DISTRESS_RISK"
            )

        # Raw P/B
        raw_pb = mc / bv_total
        raw_pb = self.winsorize_pb(raw_pb)

        # B. Maddi Net Varlık (Tangible Book Value - TBV) Düzeltmesi
        # BV_Tangible = BV_Total - Goodwill - Intangibles - DeferredTaxAssets
        bv_tangible = bv_total - input_data.goodwill - input_data.intangible_assets - input_data.deferred_tax_assets

        if bv_tangible <= 0:
            flags.is_goodwill_inflated_trap = True
            tangible_pb = None
        else:
            tangible_pb = mc / bv_tangible
            tangible_pb = self.winsorize_pb(tangible_pb)

            # Şerefiye Tuzağı Denetimi: Raw P/B ucuz görünürken P/TBV yüksek çıkıyorsa
            if raw_pb < 1.0 and tangible_pb > 2.0:
                flags.is_goodwill_inflated_trap = True

        # C. Yeniden Değerlenmiş Özkaynak (BV_Adjusted)
        bv_adjusted = bv_total + input_data.revalued_asset_adjustment
        pb_adjusted = mc / bv_adjusted if bv_adjusted > 0 else raw_pb

        # ---------------------------------------------------------
        # ADIM 2: TEORİK P/B (JUSTIFIED P/B) VE ÖZNİTELİK MÜHENDİSLİĞİ
        # ---------------------------------------------------------
        # Justified P/B = (ROE - g) / (CoE - g)
        roe = input_data.return_on_equity
        coe = input_data.cost_of_equity
        g = input_data.sustainable_growth_rate

        # Sıfıra bölme ve mantıksız payda kontrolü
        denom = max(0.01, coe - g)
        justified_pb = max(0.2, (roe - g) / denom)

        # Discount to Justified = (Justified_PB - Cari_PB) / Justified_PB
        discount_to_justified = (justified_pb - raw_pb) / justified_pb if justified_pb > 0 else 0.0

        # ---------------------------------------------------------
        # ADIM 3: MANTIK AĞACI VE TUZAK DENETİMİ
        # ---------------------------------------------------------
        # A. Düşük P/B (< 1.0) ve ROE Çapraz Doğrulaması
        if raw_pb < 1.0:
            # ROE < CoE ise şirket varlık eritiyordur -> Value Trap
            if roe < coe:
                flags.value_trap_risk = True
            else:
                # Tangible PB de < 1.0 ise Yüksek Güvenlik Marjlı Iskonto
                if tangible_pb is not None and tangible_pb < 1.0:
                    flags.is_tangible_discount = True

        # B. Gizli Varlık Potansiyeli (Unrealized Asset Value Potential)
        if input_data.revalued_asset_adjustment > 0 or discount_to_justified > 0.20:
            flags.unrealized_asset_value_potential = True

        # C. Yüksek Borç Kaldıracı
        if input_data.net_debt > 0 and bv_tangible > 0:
            if (input_data.net_debt / bv_tangible) > 3.0:
                flags.is_leverage_risk = True

        # ---------------------------------------------------------
        # ADIM 4: KARAR VE GÜVENLİK MARJI SKORLAMASI
        # ---------------------------------------------------------
        safety_score, recommendation = self._compute_safety_score(
            raw_pb=raw_pb,
            tangible_pb=tangible_pb,
            justified_pb=justified_pb,
            discount_to_justified=discount_to_justified,
            roe=roe,
            coe=coe,
            flags=flags,
            input_data=input_data
        )

        return PBOutputData(
            ticker=input_data.ticker,
            raw_pb=raw_pb,
            tangible_pb=tangible_pb,
            justified_pb=justified_pb,
            discount_to_justified=discount_to_justified,
            flags=flags.to_dict(),
            safety_margin_score=safety_score,
            primary_recommendation_contribution=recommendation
        )

    def _compute_safety_score(
        self,
        raw_pb: float,
        tangible_pb: Optional[float],
        justified_pb: float,
        discount_to_justified: float,
        roe: float,
        coe: float,
        flags: PBFlags,
        input_data: PBInputData
    ) -> Tuple[float, str]:
        """
        Güvenlik Marjı ve Karar Matrisi
        """
        if flags.is_negative_equity:
            return 0.10, "SOLVENCY_DISTRESS_RISK"

        if flags.is_goodwill_inflated_trap:
            return 0.25, "BEARISH_GOODWILL_TRAP"

        if flags.value_trap_risk:
            return 0.30, "BEARISH_VALUE_TRAP"

        base_score = 0.50

        # Rule 1: Tangible P/B < 1.0 VE ROE > CoE * 1.2 VE Net Debt / TBV < 1.0 -> STRONG_BUY (Yüksek Güvenlik Marjı)
        if flags.is_tangible_discount and roe > (coe * 1.2) and not flags.is_leverage_risk:
            base_score = 0.88
            return base_score, "STRONG_BUY"

        # Asset-Light İstisnası (Yazılım/Teknoloji P/B > 4.0 ama ROE > 40%)
        if input_data.is_asset_light and raw_pb > 3.0 and roe > 0.35:
            base_score = 0.72
            return base_score, "BULLISH"

        # Justified PB İskontosu katkısı
        if discount_to_justified > 0.25:
            base_score += 0.20
        elif discount_to_justified < -0.30:
            base_score -= 0.20

        # Gizli Varlık Potansiyeli Katkısı
        if flags.unrealized_asset_value_potential:
            base_score += 0.10

        # Kurumsal Yönetim Derecelendirmesi Düzeltmesi
        if input_data.corporate_governance_score < 0.50:
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
