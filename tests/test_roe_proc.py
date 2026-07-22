import pytest
from financial_ai.schemas import ROEInputData, ROEOutputData
from financial_ai.valuation.roe_proc import ROEValuationEngine

def test_bimas_user_example():
    """Kullanıcının ilettiği BIMAS örnek girdisi ve şema uyum testi"""
    raw_input = {
        "ticker": "BIMAS",
        "timestamp": "2026-07-22T09:30:00Z",
        "net_income_ttm": 18000000000,
        "equity_t0": 35000000000,
        "equity_t4": 25000000000,
        "revenue_ttm": 220000000000,
        "total_assets": 85000000000,
        "ebit_ttm": 24000000000,
        "one_off_income": 1200000000,
        "inflation_rate": 0.38,
        "risk_free_rate": 0.30,
        "cost_of_equity": 0.35,
        "payout_ratio": 0.50
    }

    input_data = ROEInputData.from_dict(raw_input)
    engine = ROEValuationEngine()
    output: ROEOutputData = engine.evaluate(input_data)
    out_dict = output.to_dict()

    # Raw ROE = 18B / 30B = 0.60
    assert out_dict["raw_roe"] == 0.60

    # Core Net Income = 18B - 1.2B = 16.8B -> Core ROE = 16.8B / 30B = 0.56
    assert out_dict["core_roe"] == 0.56

    # Real ROE Spread = 0.56 - 0.38 = 0.18
    assert out_dict["real_roe_spread"] == 0.18

    # DuPont Breakdown
    dupont = out_dict["dupont_analysis"]
    assert dupont["asset_turnover"] in [2.58, 2.59]
    assert dupont["financial_leverage"] == 2.83
    assert dupont["primary_driver"] == "ASSET_TURNOVER_DRIVEN"

    # Sustainable Growth Rate (SGR) = 0.56 * (1 - 0.50) = 0.28
    assert out_dict["sustainable_growth_rate"] == 0.28

    # Flags
    assert output.flags["double_negative_trap"] is False
    assert output.flags["leverage_driven_risk"] is False
    assert output.flags["capital_destruction_risk"] is False

    assert out_dict["ticker"] == "BIMAS"
    assert "capital_quality_score" in out_dict
    assert out_dict["primary_recommendation_contribution"] in ["BULLISH", "STRONG_BUY", "MODERATE_BULLISH"]

def test_double_negative_bankruptcy_trap():
    """Çift Negatiflik Tuzağı (Net Kâr ve Özkaynakların ikisi de negatif)"""
    raw_input = {
        "ticker": "BATIK_TEKSTİL",
        "timestamp": "2026-07-22T09:30:00Z",
        "net_income_ttm": -10000000,
        "equity_t0": -20000000,
        "equity_t4": -10000000,          # BV_Avg = -15M (Eksi)
        "revenue_ttm": 50000000,
        "total_assets": 40000000,
        "ebit_ttm": -5000000
    }

    input_data = ROEInputData.from_dict(raw_input)
    engine = ROEValuationEngine()
    output = engine.evaluate(input_data)

    assert output.raw_roe is None
    assert output.flags["double_negative_trap"] is True
    assert output.primary_recommendation_contribution == "DOUBLE_NEGATIVE_BANKRUPTCY_TRAP"

def test_leverage_driven_roe_trap():
    """Aşırı Borçlanma kaynaklı sahte yüksek ROE tespiti (TA / BV > 5.0)"""
    raw_input = {
        "ticker": "BORC_ILUZYONU",
        "timestamp": "2026-07-22T09:30:00Z",
        "net_income_ttm": 60000000,
        "equity_t0": 100000000,
        "equity_t4": 100000000,          # BV_Avg = 100M -> ROE = 60%
        "revenue_ttm": 200000000,        # Margin = 60M / 200M = 30%
        "total_assets": 700000000,       # Total Assets = 700M -> Leverage = 7.0x (> 5.0)
        "ebit_ttm": 70000000
    }

    input_data = ROEInputData.from_dict(raw_input)
    engine = ROEValuationEngine()
    output = engine.evaluate(input_data)

    assert output.flags["leverage_driven_risk"] is True
    assert output.primary_recommendation_contribution == "BEARISH_HIGH_LEVERAGE_ROE"

def test_capital_destruction_trap():
    """Enflasyonun altında kalan ROE (Reel Sermaye Tahribatı) tespiti"""
    raw_input = {
        "ticker": "ENFLASYON_ERİĞİ",
        "timestamp": "2026-07-22T09:30:00Z",
        "net_income_ttm": 20000000,
        "equity_t0": 100000000,
        "equity_t4": 100000000,          # Core ROE = 20%
        "revenue_ttm": 300000000,
        "total_assets": 200000000,
        "ebit_ttm": 25000000,
        "inflation_rate": 0.45            # Enflasyon %45 -> Real Spread = -25%
    }

    input_data = ROEInputData.from_dict(raw_input)
    engine = ROEValuationEngine()
    output = engine.evaluate(input_data)

    assert output.flags["capital_destruction_risk"] is True
    assert output.primary_recommendation_contribution == "CAPITAL_DESTRUCTION_TRAP"
