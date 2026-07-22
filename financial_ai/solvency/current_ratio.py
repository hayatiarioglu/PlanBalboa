import math
from typing import Dict, Any, Tuple, Optional
from financial_ai.schemas import CurrentRatioInputData, CurrentRatioFlags, CurrentRatioOutputData

class CurrentRatioSolvencyEngine:
    """
    MODÜL: Solvency_Engine::Current_Ratio
    Cari Oran, Likidite Sağlığı ve Borç Ödeme Riski Analiz Motoru.
    """

    RETAIL_SECTORS = {"RETAIL", "FMCG", "SUPERMARKET", "GROCERY", "PERAKENDE"}

    def evaluate(self, input_data: CurrentRatioInputData) -> CurrentRatioOutputData:
        flags = CurrentRatioFlags()

        ca = input_data.current_assets
        cl = max(1.0, input_data.current_liabilities)
        inv = input_data.inventories
        rel_ar = input_data.related_party_receivables
        dou_ar = input_data.doubtful_receivables
        cash = input_data.cash_and_equivalents
        cfo = input_data.operating_cash_flow
        sector = input_data.sector_code.upper()

        # Ham Cari Oran
        raw_cr = ca / cl

        # Alacak Kalitesi ve İlişkili Taraf Temizliği
        ca_adjusted = ca - rel_ar - dou_ar
        adjusted_cr = ca_adjusted / cl

        # Asit-Test (Quick Ratio) ve Nakit Oranı (Cash Ratio)
        quick_ratio = (ca_adjusted - inv) / cl
        cash_ratio = cash / cl

        # Net Çalışma Sermayesi (NWC)
        net_working_capital = ca - cl

        # ---------------------------------------------------------
        # FLAK VE SEKTÖREL İSTİSNA KONTROLLERİ
        # ---------------------------------------------------------

        # 1. İlişkili Taraf Alacak Riski
        if rel_ar / max(1.0, ca) > 0.15:
            flags.related_party_receivable_risk = True

        # 2. Stok Yoğun Likidite Tuzağı (Cari Oran iyi görünür fakat stoklar çıkınca Asit-Test düşük kalır)
        if adjusted_cr > 1.1 and quick_ratio < 0.85:
            flags.inventory_heavy_liquidity = True

        # 3. Perakende / FMCG İş Modeli İstisnası (Negatif Çalışma Sermayesi)
        if sector in self.RETAIL_SECTORS and adjusted_cr >= 0.75:
            flags.retail_model_exception = True

        # 4. Likidite Kriz / Sıkıntı Uyarısı
        if not flags.retail_model_exception and adjusted_cr < 1.0:
            flags.liquidity_distress_warning = True

        # ---------------------------------------------------------
        # SOLVENCY SAĞLIK SKORU VE KARAR MATRİSİ
        # ---------------------------------------------------------
        health_score, recommendation = self._compute_health_score(
            adjusted_cr=adjusted_cr,
            quick_ratio=quick_ratio,
            cash_ratio=cash_ratio,
            cfo_to_cl=cfo / cl,
            flags=flags
        )

        return CurrentRatioOutputData(
            ticker=input_data.ticker,
            raw_current_ratio=raw_cr,
            adjusted_current_ratio=adjusted_cr,
            quick_ratio=quick_ratio,
            cash_ratio=cash_ratio,
            net_working_capital=net_working_capital,
            flags=flags.to_dict(),
            solvency_health_score=health_score,
            primary_recommendation_contribution=recommendation
        )

    def _compute_health_score(
        self,
        adjusted_cr: float,
        quick_ratio: float,
        cash_ratio: float,
        cfo_to_cl: float,
        flags: CurrentRatioFlags
    ) -> Tuple[float, str]:
        """
        Likidite Sağlık Skoru ve Karar Matrisi
        """
        # 1. Kritik İflas / Temerrüt Tehlikesi
        if not flags.retail_model_exception and adjusted_cr < 1.0 and cash_ratio < 0.15 and cfo_to_cl < 0.10:
            return 0.08, "CRITICAL_LIQUIDITY_RISK"

        # 2. Perakende İstisnası
        if flags.retail_model_exception:
            return 0.75, "NEUTRAL"

        # 3. Genel Sanayi / İmalat Değerlendirmesi
        base_score = 0.50

        if adjusted_cr >= 1.5 and quick_ratio >= 1.0:
            base_score = 0.89
            recommendation = "BULLISH"
        elif adjusted_cr >= 1.2:
            base_score = 0.68
            recommendation = "NEUTRAL"
        elif adjusted_cr >= 1.0:
            base_score = 0.45
            recommendation = "BEARISH_LIQUIDITY_WARNING"
        else:
            base_score = 0.20
            recommendation = "CRITICAL_LIQUIDITY_RISK"

        # Stok Yoğunluğu Cezası
        if flags.inventory_heavy_liquidity and recommendation == "BULLISH":
            recommendation = "NEUTRAL"
            base_score -= 0.10

        return base_score, recommendation
