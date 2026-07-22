import pytest
from financial_ai.schemas import EVInputData, EVOutputData
from financial_ai.valuation.ev_ebitda import EVValuationEngine

def test_kchol_user_example():
    """Kullanıcının ilettiği KCHOL örnek girdisi ve şema uyum testi"""
    raw_input = {
        "ticker": "KCHOL",
        "timestamp": "2026-07-22T09:30:00Z",
        "market_cap": 450000000000,
        "gross_debt": 180000000000,
        "lease_liabilities": 15000000000,
        "total_cash": 120000000000,
        "restricted_cash": 10000000000,
        "minority_interest": 25000000000,
        "associates_value": 40000000000,
        "adjusted_ebitda": 95000000000,
        "capex": 25000000000,
        "sector_median_ev_ebitda": 7.8
    }

    input_data = EVInputData.from_dict(raw_input)
    engine = EVValuationEngine()
    output: EVOutputData = engine.evaluate(input_data)
    out_dict = output.to_dict()

    # Adjusted Cash = 120B - 10B = 110B
    # EV = 450B + 180B + 15B + 25B - 110B - 40B = 500B
    assert out_dict["enterprise_value"] == 500000000000.0

    # EV / Adjusted EBITDA = 500B / 95B = 5.263 -> 5.26
    assert out_dict["ev_to_adjusted_ebitda"] == 5.26

    # EV / (EBITDA - CapEx) = 500B / (95B - 25B) = 500B / 70B = 7.1428 -> 7.14
    assert out_dict["ev_to_ebitda_minus_capex"] == 7.14

    # Leverage to EV Ratio = 180B / 500B = 0.36
    assert out_dict["leverage_to_ev_ratio"] == 0.36

    # Flags
    assert output.flags["restricted_cash_adjusted"] is True
    assert output.flags["ma_target_potential"] is True
    assert output.flags["capex_trap_risk"] is False

    assert out_dict["ticker"] == "KCHOL"
    assert "valuation_attractiveness_score" in out_dict
    assert out_dict["primary_recommendation_contribution"] in ["BULLISH", "STRONG_BUY", "MODERATE_BULLISH"]

def test_capex_trap_risk():
    """EBITDA görünüşte ucuz fakat CapEx sonrası aşırı pahalı olan şirket tespiti"""
    raw_input = {
        "ticker": "SERMAYE_YUTUCU",
        "timestamp": "2026-07-22T09:30:00Z",
        "market_cap": 400000000,
        "gross_debt": 100000000,
        "total_cash": 0,
        "adjusted_ebitda": 100000000,  # EV = 500M -> EV/EBITDA = 5x
        "capex": 85000000,             # CapEx = 85M -> EBITDA - CapEx = 15M -> EV/(EBITDA-CapEx) = 33.3x
        "sector_median_ev_ebitda": 8.0
    }

    input_data = EVInputData.from_dict(raw_input)
    engine = EVValuationEngine()
    output = engine.evaluate(input_data)

    assert output.flags["capex_trap_risk"] is True
    assert output.primary_recommendation_contribution == "BEARISH_CAPEX_TRAP"

def test_net_net_liquidation_opportunity():
    """Net Nakdi Piyasa Değerinden büyük olan (EV < 0) bedava borsa fırsatı tespiti"""
    raw_input = {
        "ticker": "BEDAVA_BORSA",
        "timestamp": "2026-07-22T09:30:00Z",
        "market_cap": 50000000,
        "gross_debt": 10000000,
        "total_cash": 80000000,          # Adjusted Cash = 80M -> EV = 50M + 10M - 80M = -20M!
        "adjusted_ebitda": 15000000
    }

    input_data = EVInputData.from_dict(raw_input)
    engine = EVValuationEngine()
    output = engine.evaluate(input_data)

    assert output.flags["is_net_cash_company"] is True
    assert output.primary_recommendation_contribution == "STRONG_BUY_NET_NET_LIQUIDATION"

def test_negative_ebitda_cash_burn():
    """Negatif EBITDA üreten şirketlerin Nakit Yakma Uyarısı ile sevk edilmesi"""
    raw_input = {
        "ticker": "NAKIT_YAKAN",
        "timestamp": "2026-07-22T09:30:00Z",
        "market_cap": 100000000,
        "gross_debt": 20000000,
        "total_cash": 10000000,
        "adjusted_ebitda": -5000000     # Negatif EBITDA
    }

    input_data = EVInputData.from_dict(raw_input)
    engine = EVValuationEngine()
    output = engine.evaluate(input_data)

    assert output.ev_to_adjusted_ebitda is None
    assert output.primary_recommendation_contribution == "CASH_BURN_RED_FLAG"
