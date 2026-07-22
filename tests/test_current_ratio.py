import pytest
from financial_ai.schemas import CurrentRatioInputData, CurrentRatioOutputData
from financial_ai.solvency.current_ratio import CurrentRatioSolvencyEngine

def test_arclk_user_example():
    """Kullanıcının ilettiği ARCLK örnek girdisi ve şema uyum testi"""
    raw_input = {
        "ticker": "ARCLK",
        "timestamp": "2026-07-22T09:30:00Z",
        "sector_code": "MANUFACTURING",
        "current_assets": 85000000000,
        "current_liabilities": 65000000000,
        "inventories": 30000000000,
        "accounts_receivable": 28000000000,
        "related_party_receivables": 4000000000,
        "doubtful_receivables": 1000000000,
        "cash_and_equivalents": 15000000000,
        "operating_cash_flow": 8000000000
    }

    input_data = CurrentRatioInputData.from_dict(raw_input)
    engine = CurrentRatioSolvencyEngine()
    output: CurrentRatioOutputData = engine.evaluate(input_data)
    out_dict = output.to_dict()

    # Raw CR = 85B / 65B = 1.30769 -> 1.308 or 1.31
    assert out_dict["raw_current_ratio"] in [1.307, 1.308, 1.31]

    # Adjusted CA = 85B - 4B - 1B = 80B -> Adjusted CR = 80B / 65B = 1.23
    assert out_dict["adjusted_current_ratio"] == 1.23

    # Quick Ratio = (80B - 30B) / 65B = 50B / 65B = 0.769
    assert out_dict["quick_ratio"] == 0.769

    # Cash Ratio = 15B / 65B = 0.23
    assert out_dict["cash_ratio"] == 0.23

    # NWC = 85B - 65B = 20B
    assert out_dict["net_working_capital"] == 20000000000.0

    # Flags
    assert output.flags["inventory_heavy_liquidity"] is True
    assert output.flags["related_party_receivable_risk"] is False
    assert output.flags["liquidity_distress_warning"] is False
    assert output.flags["retail_model_exception"] is False

    assert out_dict["solvency_health_score"] == 0.68
    assert out_dict["primary_recommendation_contribution"] == "NEUTRAL"

def test_retail_model_exception():
    """Perakende Sektörü Negatif Çalışma Sermayesi İstisna Testi"""
    raw_input = {
        "ticker": "BIMAS",
        "timestamp": "2026-07-22T09:30:00Z",
        "sector_code": "RETAIL",
        "current_assets": 17000000000,
        "current_liabilities": 20000000000,   # Adjusted CR = 0.85 (< 1.0)
        "inventories": 8000000000,
        "cash_and_equivalents": 5000000000
    }

    input_data = CurrentRatioInputData.from_dict(raw_input)
    engine = CurrentRatioSolvencyEngine()
    output = engine.evaluate(input_data)

    assert output.flags["retail_model_exception"] is True
    assert output.flags["liquidity_distress_warning"] is False
    assert output.primary_recommendation_contribution == "NEUTRAL"

def test_critical_liquidity_distress():
    """Sanayi Şirketinde Kritik İflas / Temerrüt Riski Testi"""
    raw_input = {
        "ticker": "BATIK_SANAYI",
        "timestamp": "2026-07-22T09:30:00Z",
        "sector_code": "MANUFACTURING",
        "current_assets": 40000000,
        "current_liabilities": 50000000,     # Adjusted CR = 0.80 (< 1.0)
        "cash_and_equivalents": 2000000,     # Cash Ratio = 0.04 (< 0.15)
        "operating_cash_flow": 1000000       # CFO / CL = 0.02 (< 0.10)
    }

    input_data = CurrentRatioInputData.from_dict(raw_input)
    engine = CurrentRatioSolvencyEngine()
    output = engine.evaluate(input_data)

    assert output.flags["liquidity_distress_warning"] is True
    assert output.primary_recommendation_contribution == "CRITICAL_LIQUIDITY_RISK"
