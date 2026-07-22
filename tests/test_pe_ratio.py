import pytest
from financial_ai.schemas import PEInputData, PEOutputData
from financial_ai.valuation.pe_ratio import PEValuationEngine
from financial_ai.feedback.regime_adapter import MarketRegimeAdapter

def test_eregl_user_example():
    """Kullanıcının ilettiği EREGL örnek girdisi ve şema uyum testi"""
    raw_input = {
        "ticker": "EREGL",
        "timestamp": "2026-07-22T09:30:00Z",
        "market_cap": 150000000000,
        "shares_outstanding": 3500000000,
        "current_price": 42.85,
        "ttm_net_income_gaap": 12000000000,
        "non_recurring_income": 4000000000,
        "fwd_earnings_consensus": 15000000000,
        "sector_code": "STEEL",
        "sector_median_pe": 8.5,
        "risk_free_rate": 0.35,
        "annual_growth_rate_pct": 15.0
    }
    
    input_data = PEInputData.from_dict(raw_input)
    engine = PEValuationEngine()
    output: PEOutputData = engine.evaluate(input_data)
    
    # Ham F/K = 150B / 12B = 12.5
    assert output.raw_pe == 12.5
    
    # Düzeltilmiş Net Kâr = 12B - 4B (Tek seferlik gelir) = 8B
    # Düzeltilmiş F/K = 150B / 8B = 18.75
    assert output.adjusted_pe == 18.75
    
    # Şerit ve Bayrak kontrolleri
    assert output.flags["one_off_income_detected"] is True
    assert output.flags["is_cyclical_trap_risk"] is True
    
    out_dict = output.to_dict()
    assert out_dict["ticker"] == "EREGL"
    assert "signal_score" in out_dict
    assert "confidence_interval" in out_dict
    assert "primary_recommendation_contribution" in out_dict

def test_negative_earnings_isolation():
    """Negatif kâr durumunda F/K NaN/None yapılmalı ve E/P vektörü beslenmelidir"""
    raw_input = {
        "ticker": "ZARAR_AŞ",
        "timestamp": "2026-07-22T09:30:00Z",
        "market_cap": 100000000,
        "shares_outstanding": 10000000,
        "current_price": 10.0,
        "ttm_net_income_gaap": -20000000,  # Zarar
        "sector_code": "TECH",
        "sector_median_pe": 25.0
    }
    
    input_data = PEInputData.from_dict(raw_input)
    engine = PEValuationEngine()
    output = engine.evaluate(input_data)
    
    assert output.raw_pe is None
    assert output.adjusted_pe is None
    assert output.earnings_yield == -0.20  # -20M / 100M
    assert output.flags["is_negative_earnings"] is True
    assert output.primary_recommendation_contribution == "BEARISH_NEGATIVE_EARNINGS"

def test_peg_growth_exemption():
    """Yüksek F/K'ya rağmen PEG < 1.0 ise POZİTİF sinyal üretilmelidir"""
    raw_input = {
        "ticker": "GROWTH_TECH",
        "timestamp": "2026-07-22T09:30:00Z",
        "market_cap": 1000000000,
        "shares_outstanding": 10000000,
        "current_price": 100.0,
        "ttm_net_income_gaap": 25000000,  # PE = 40
        "sector_code": "SAAS",
        "sector_median_pe": 20.0,  # PE > 1.5 * sector_pe
        "annual_growth_rate_pct": 50.0,  # %50 kâr büyümesi -> PEG = 40 / 50 = 0.8 (< 1.0)
        "net_debt": 0.0,
        "ebitda": 50000000,
        "roic": 0.25,
        "wacc": 0.08
    }
    
    input_data = PEInputData.from_dict(raw_input)
    engine = PEValuationEngine()
    output = engine.evaluate(input_data)
    
    assert output.raw_pe == 40.0
    assert output.peg_ratio == 0.8
    assert output.flags["is_bubble_pricing"] is False
    assert output.signal_score >= 0.70

def test_debt_trap_risk():
    """Düşük F/K fakat yüksek Net Borç / EBITDA (> 5.0) -> Borç Kapanı Riski"""
    raw_input = {
        "ticker": "BORCLU_SANAYI",
        "timestamp": "2026-07-22T09:30:00Z",
        "market_cap": 50000000,
        "shares_outstanding": 5000000,
        "current_price": 10.0,
        "ttm_net_income_gaap": 12500000,  # PE = 4.0 (Sektör = 15.0)
        "sector_code": "MANUFACTURING",
        "sector_median_pe": 15.0,
        "net_debt": 100000000,  # 100M Borç
        "ebitda": 15000000,      # Net Debt / EBITDA = 6.66 (> 5.0)
    }
    
    input_data = PEInputData.from_dict(raw_input)
    engine = PEValuationEngine()
    output = engine.evaluate(input_data)
    
    assert output.flags["is_debt_trap_risk"] is True
    assert output.primary_recommendation_contribution == "BEARISH_VALUE_TRAP"

def test_market_regime_adapter():
    """Yüksek faiz rejiminde uyarlama testi"""
    adapter = MarketRegimeAdapter()
    info = adapter.adapt_regime(risk_free_rate=0.25)
    assert info["regime"] == "HIGH_INTEREST_RATE"
    assert info["high_pe_penalty_multiplier"] > 1.0
