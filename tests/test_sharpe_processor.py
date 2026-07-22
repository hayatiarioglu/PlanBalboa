import pytest
from financial_ai.schemas import SharpeInputData
from financial_ai.risk.sharpe_processor import SharpeProcessor

def test_ti1_capital_fund_user_example():
    """Modül 19: TI1_CAPITAL_FUND Sharpe Oranı ve Performans Motoru Testi"""
    raw_input = {
        "asset_id": "TI1_CAPITAL_FUND",
        "timestamp": "2026-07-22T09:30:00Z",
        "frequency": "DAILY",
        "lookback_periods": 252,
        "asset_returns": [0.0012, -0.0005, 0.0021, 0.0018, -0.0010],
        "risk_free_rate_annual": 0.48,
        "skewness": -1.25,
        "kurtosis": 5.80,
        "backtest_trials_count": 45
    }

    input_data = SharpeInputData.from_dict(raw_input)
    engine = SharpeProcessor()
    output = engine.evaluate(input_data)
    out_dict = output.to_dict()

    assert out_dict["asset_id"] == "TI1_CAPITAL_FUND"
    assert out_dict["raw_annualized_sharpe"] == 1.42
    assert out_dict["lo_autocorr_adjusted_sharpe"] == 1.15
    assert out_dict["favre_galeano_adjusted_sharpe"] == 0.82
    assert out_dict["deflated_sharpe_ratio_dsr"] == 0.78

    assert output.flags["is_gamed_or_smoothed"] is False
    assert output.flags["negative_skewness_tail_risk"] is True
    assert output.flags["exceeds_risk_free_significantly"] is True
    assert output.flags["overfitting_rejected"] is True

    assert out_dict["final_risk_adjusted_score"] == 0.64
    assert out_dict["primary_recommendation_contribution"] == "UNDERWEIGHT_DUE_TO_TAIL_RISK"

def test_high_quality_fund():
    """Modül 19: Yüksek Kaliteli Fon Sharpe Doğrulama Testi"""
    raw_input = {
        "asset_id": "EXCELLENT_FUND",
        "timestamp": "2026-07-22T09:30:00Z",
        "frequency": "DAILY",
        "lookback_periods": 252,
        "asset_returns": [0.005, 0.004, 0.003, 0.006, 0.002],
        "risk_free_rate_annual": 0.10,
        "skewness": 0.20,
        "kurtosis": 3.0,
        "backtest_trials_count": 1
    }

    input_data = SharpeInputData.from_dict(raw_input)
    engine = SharpeProcessor()
    output = engine.evaluate(input_data)

    assert output.flags["negative_skewness_tail_risk"] is False
    assert output.flags["overfitting_rejected"] is False
    assert output.primary_recommendation_contribution == "STRONG_RISK_ADJUSTED_PERFORMANCE"
