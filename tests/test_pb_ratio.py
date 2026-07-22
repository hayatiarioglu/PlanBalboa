import pytest
from financial_ai.schemas import PBInputData, PBOutputData
from financial_ai.valuation.pb_ratio import PBValuationEngine

def test_sise_user_example():
    """Kullanıcının ilettiği SISE örnek girdisi ve şema uyum testi"""
    raw_input = {
        "ticker": "SISE",
        "timestamp": "2026-07-22T09:30:00Z",
        "market_cap": 120000000000,
        "total_assets": 200000000000,
        "total_liabilities": 110000000000,
        "goodwill": 5000000000,
        "intangible_assets": 3000000000,
        "return_on_equity": 0.28,
        "cost_of_equity": 0.18,
        "sector_median_pb": 2.1,
        "sustainable_growth_rate": 0.06
    }

    input_data = PBInputData.from_dict(raw_input)
    engine = PBValuationEngine()
    output: PBOutputData = engine.evaluate(input_data)
    out_dict = output.to_dict()

    # BV_Total = 200B - 110B = 90B
    # Raw P/B = 120B / 90B = 1.3333 -> 1.33
    assert out_dict["raw_pb"] == 1.33

    # BV_Tangible = 90B - 5B - 3B = 82B
    # Tangible P/B = 120B / 82B = 1.463 -> 1.46
    assert out_dict["tangible_pb"] == 1.46

    # Justified P/B = (0.28 - 0.06) / (0.18 - 0.06) = 0.22 / 0.12 = 1.83
    assert out_dict["justified_pb"] is not None
    assert output.discount_to_justified > 0.20
    assert output.flags["unrealized_asset_value_potential"] is True

    assert out_dict["ticker"] == "SISE"
    assert "safety_margin_score" in out_dict
    assert out_dict["primary_recommendation_contribution"] in ["BULLISH", "MODERATE_BULLISH", "STRONG_BUY"]

def test_goodwill_inflated_trap():
    """Şerefiye nedeniyle kağıt üzerinde ucuz görünen fakat Tangible P/B yüksek olan şirket tespiti"""
    raw_input = {
        "ticker": "KOPUK_VARLIK",
        "timestamp": "2026-07-22T09:30:00Z",
        "market_cap": 80000000,
        "total_assets": 200000000,
        "total_liabilities": 100000000,  # BV_Total = 100M -> Raw P/B = 0.8 (< 1.0)
        "goodwill": 60000000,            # 60M Şerefiye! BV_Tangible = 40M -> Tangible P/B = 2.0
        "intangible_assets": 10000000,
        "return_on_equity": 0.10,
        "cost_of_equity": 0.15
    }

    input_data = PBInputData.from_dict(raw_input)
    engine = PBValuationEngine()
    output = engine.evaluate(input_data)
    out_dict = output.to_dict()

    assert out_dict["raw_pb"] == 0.8
    assert output.flags["is_goodwill_inflated_trap"] is True
    assert output.primary_recommendation_contribution == "BEARISH_GOODWILL_TRAP"

def test_value_trap_low_roe():
    """P/B < 1.0 fakat ROE < CoE -> Değer Tuzağı (Value Trap)"""
    raw_input = {
        "ticker": "ERIMIS_OZKAYNAK",
        "timestamp": "2026-07-22T09:30:00Z",
        "market_cap": 50000000,
        "total_assets": 200000000,
        "total_liabilities": 100000000,  # BV = 100M -> P/B = 0.5
        "return_on_equity": 0.05,        # ROE = %5 (Enflasyonun ve sermaye maliyetinin çok altında)
        "cost_of_equity": 0.20
    }

    input_data = PBInputData.from_dict(raw_input)
    engine = PBValuationEngine()
    output = engine.evaluate(input_data)

    assert output.flags["value_trap_risk"] is True
    assert output.primary_recommendation_contribution == "BEARISH_VALUE_TRAP"

def test_asset_light_exemption():
    """Hafif varlıklı yazılım şirketi (P/B > 3.0 ama ROE %40) -> Makul Pozitif"""
    raw_input = {
        "ticker": "YAZILIM_PRO",
        "timestamp": "2026-07-22T09:30:00Z",
        "market_cap": 500000000,
        "total_assets": 120000000,
        "total_liabilities": 20000000,   # BV = 100M -> P/B = 5.0
        "return_on_equity": 0.45,        # Yüksek ROE %45
        "cost_of_equity": 0.15,
        "is_asset_light": True
    }

    input_data = PBInputData.from_dict(raw_input)
    engine = PBValuationEngine()
    output = engine.evaluate(input_data)
    out_dict = output.to_dict()

    assert out_dict["raw_pb"] == 5.0
    assert output.primary_recommendation_contribution in ["BULLISH", "STRONG_BUY", "MODERATE_BULLISH"]

def test_negative_equity_distress():
    """Negatif özkaynak durumu (BV <= 0) -> Solvency Distress Risk"""
    raw_input = {
        "ticker": "BATIK_AŞ",
        "timestamp": "2026-07-22T09:30:00Z",
        "market_cap": 10000000,
        "total_assets": 50000000,
        "total_liabilities": 60000000,   # BV = -10M
    }

    input_data = PBInputData.from_dict(raw_input)
    engine = PBValuationEngine()
    output = engine.evaluate(input_data)

    assert output.raw_pb is None
    assert output.flags["is_negative_equity"] is True
    assert output.primary_recommendation_contribution == "SOLVENCY_DISTRESS_RISK"
