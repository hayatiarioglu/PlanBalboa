import pytest
from financial_ai.schemas import BetaInputData
from financial_ai.risk.beta_processor import BetaProcessor

def test_eregl_user_example():
    """Modül 17: EREGL Beta Katsayısı ve Risk Motoru Testi"""
    raw_input = {
        "ticker": "EREGL",
        "benchmark_ticker": "XU100",
        "timestamp": "2026-07-22T09:30:00Z",
        "lookback_window_days": 252,
        "raw_beta": 1.45,
        "stock_returns_std": 0.028,
        "benchmark_returns_std": 0.016,
        "correlation": 0.82,
        "net_debt": 45000000000.0,
        "equity": 85000000000.0,
        "tax_rate": 0.25,
        "illiquidity_flag": False
    }

    input_data = BetaInputData.from_dict(raw_input)
    engine = BetaProcessor()
    output = engine.evaluate(input_data)
    out_dict = output.to_dict()

    assert out_dict["ticker"] == "EREGL"
    assert out_dict["raw_beta"] == 1.45
    assert out_dict["blume_adjusted_beta"] == 1.30
    assert out_dict["unlevered_beta"] == 0.93

    asym = out_dict["asymmetric_beta"]
    assert asym["up_market_beta"] == 1.62
    assert asym["down_market_beta"] == 0.95
    assert asym["asymmetry_ratio"] == 1.705

    assert output.flags["is_high_beta_aggressive"] is True
    assert output.flags["is_thin_trading_biased"] is False
    assert output.flags["favorable_asymmetry_detected"] is True

    assert out_dict["risk_contribution_score"] == 0.82
    assert out_dict["primary_recommendation_contribution"] == "OVERWEIGHT_IN_BULL_MARKET"

def test_thin_trading_and_unlevered_beta():
    """Modül 17: Blume Düzeltmesi ve Hamada Borç Arındırması Formül Doğrulama Testi"""
    raw_input = {
        "ticker": "DEFENSIVE_STOCK",
        "benchmark_ticker": "XU100",
        "timestamp": "2026-07-22T09:30:00Z",
        "lookback_window_days": 252,
        "raw_beta": 0.50,
        "stock_returns_std": 0.012,
        "benchmark_returns_std": 0.015,
        "correlation": 0.50,
        "net_debt": 100000000.0,
        "equity": 200000000.0,
        "tax_rate": 0.20,
        "illiquidity_flag": True
    }

    input_data = BetaInputData.from_dict(raw_input)
    engine = BetaProcessor()
    output = engine.evaluate(input_data)

    # Blume: 0.67 * 0.50 + 0.33 * 1.0 = 0.665 -> 0.67
    assert output.blume_adjusted_beta == 0.67
    # D/E = 0.5, Hamada: 0.5 / (1 + (1 - 0.2) * 0.5) = 0.5 / 1.4 = 0.357 -> 0.36
    assert output.unlevered_beta == 0.36
    assert output.flags["is_thin_trading_biased"] is True
