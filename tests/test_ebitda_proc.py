import pytest
from financial_ai.schemas import EBITDAInputData, EBITDAOutputData
from financial_ai.valuation.ebitda_proc import EBITDAValuationEngine

def test_thyao_user_example():
    """Kullanıcının ilettiği THYAO örnek girdisi ve şema uyum testi"""
    raw_input = {
        "ticker": "THYAO",
        "timestamp": "2026-07-22T09:30:00Z",
        "revenue": 550000000000,
        "ebit": 65000000000,
        "depreciation_amortization": 25000000000,
        "ifrs16_lease_payments": 12000000000,
        "stock_based_compensation": 500000000,
        "capitalized_capex": 3000000000,
        "operating_cash_flow": 70000000000,
        "capex": 35000000000,
        "net_debt": 180000000000
    }

    input_data = EBITDAInputData.from_dict(raw_input)
    engine = EBITDAValuationEngine()
    output: EBITDAOutputData = engine.evaluate(input_data)
    out_dict = output.to_dict()

    # Raw EBITDA = 65B + 25B = 90B
    assert out_dict["raw_ebitda"] == 90000000000.0

    # Adjusted EBITDA = 90B - 12B - 0.5B - 3B = 74.5B
    assert out_dict["adjusted_ebitda"] == 74500000000.0

    # EBITDA Margin = 74.5B / 550B = 0.13545 -> 0.135
    assert out_dict["ebitda_margin"] == 0.135

    # Cash Conversion Rate = (70B - 35B) / 74.5B = 35B / 74.5B = 0.46979 -> 0.47
    assert out_dict["cash_conversion_rate"] == 0.47

    # Net Debt / EBITDA = 180B / 74.5B = 2.416 -> 2.42
    assert out_dict["net_debt_to_ebitda"] in [2.41, 2.42]

    # Flags
    assert output.flags["ifrs16_distortion_high"] is True
    assert output.flags["capitalized_capex_warning"] is True
    assert output.flags["sbc_dilution_risk"] is False

    assert out_dict["ticker"] == "THYAO"
    assert "operational_quality_score" in out_dict
    assert out_dict["primary_recommendation_contribution"] in ["BULLISH", "MODERATE_BULLISH", "STRONG_BUY"]

def test_cash_flow_decoupling_trap():
    """EBITDA yüksek fakat Serbest Nakit Akışı düşük/negatif -> Nakit Akış Kopuş Tuzağı"""
    raw_input = {
        "ticker": "KAĞIT_USTUNDE_KÂR",
        "timestamp": "2026-07-22T09:30:00Z",
        "revenue": 100000000,
        "ebit": 20000000,
        "depreciation_amortization": 10000000,  # Raw EBITDA = 30M
        "operating_cash_flow": 5000000,          # CFO = 5M
        "capex": 10000000,                        # CapEx = 10M -> FCF = -5M (Nakit dönüşüm < 0)
        "net_debt": 20000000
    }

    input_data = EBITDAInputData.from_dict(raw_input)
    engine = EBITDAValuationEngine()
    output = engine.evaluate(input_data)

    assert output.flags["cash_flow_decoupling_risk"] is True
    assert output.primary_recommendation_contribution == "VALUE_TRAP_NO_CASH_CONVERSION"

def test_sbc_dilution_warning():
    """Yüksek Hisse Bazlı Ödeme (SBC) veren teknoloji şirketi tespiti"""
    raw_input = {
        "ticker": "SAAS_DILUTED",
        "timestamp": "2026-07-22T09:30:00Z",
        "revenue": 100000000,
        "ebit": 15000000,
        "depreciation_amortization": 5000000,   # Raw EBITDA = 20M
        "stock_based_compensation": 4000000,    # 4M SBC (%20 of Raw EBITDA -> Dilution Risk)
        "operating_cash_flow": 15000000,
        "capex": 2000000
    }

    input_data = EBITDAInputData.from_dict(raw_input)
    engine = EBITDAValuationEngine()
    output = engine.evaluate(input_data)

    assert output.flags["sbc_dilution_risk"] is True
    assert output.adjusted_ebitda == 16000000.0  # 20M - 4M

def test_turnaround_candidate():
    """Net Kâr zararda fakat operasyonel olarak nakit üreten şirket tespiti"""
    raw_input = {
        "ticker": "BORCLU_CEVHER",
        "timestamp": "2026-07-22T09:30:00Z",
        "revenue": 200000000,
        "ebit": 40000000,
        "depreciation_amortization": 10000000,  # Raw EBITDA = 50M
        "operating_cash_flow": 35000000,
        "capex": 5000000,                         # FCF = 30M -> Cash Conversion = 0.60
        "net_debt": 100000000,                    # Net Debt / EBITDA = 2.0
        "net_income_gaap": -5000000               # Net Zararda (Yüksek faiz yüzünden)
    }

    input_data = EBITDAInputData.from_dict(raw_input)
    engine = EBITDAValuationEngine()
    output = engine.evaluate(input_data)

    assert output.flags["turnaround_candidate"] is True
    assert output.primary_recommendation_contribution == "TURNAROUND_BUY"
