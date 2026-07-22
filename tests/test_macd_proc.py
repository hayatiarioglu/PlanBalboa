import pytest
from financial_ai.schemas import MACDInputData, MACDOutputData
from financial_ai.technical.macd_proc import MACDTechnicalEngine

def test_eupwr_user_example():
    """Kullanıcının ilettiği EUPWR örnek girdisi ve şema uyum testi"""
    raw_input = {
        "ticker": "EUPWR",
        "timeframe": "4h",
        "timestamp": "2026-07-22T09:30:00Z",
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9,
        "ohlcv_data": [
            {"timestamp": "2026-07-21T12:00:00Z", "close": 82.5, "volume": 3200000},
            {"timestamp": "2026-07-22T09:00:00Z", "close": 86.0, "volume": 5800000}
        ],
        "adx_val": 26.1,
        "atr_val": 2.1
    }

    input_data = MACDInputData.from_dict(raw_input)
    engine = MACDTechnicalEngine()
    output: MACDOutputData = engine.evaluate(input_data)
    out_dict = output.to_dict()

    assert out_dict["ppo_line"] == 1.85
    assert out_dict["signal_line"] == 1.12
    assert out_dict["histogram_value"] == 0.73
    assert out_dict["delta_histogram"] == 0.18
    assert out_dict["histogram_acceleration"] == 0.05
    assert out_dict["zero_line_status"] == "ABOVE_ZERO_BULLISH"

    cross = out_dict["crossover_events"]
    assert cross["bullish_cross_active"] is True
    assert cross["candles_since_cross"] == 2
    assert cross["cross_quality"] == "HIGH_CONFIRMATION"

    assert output.flags["is_decelerating"] is False
    assert output.flags["whipsaw_risk"] is False

    assert out_dict["momentum_score"] == 0.91
    assert out_dict["primary_recommendation_contribution"] == "STRONG_BULLISH_ACCELERATION"

def test_macd_whipsaw_risk_low_adx():
    """Yatay Piyasada MACD Sahte Kesişim Testi (ADX < 18)"""
    raw_input = {
        "ticker": "TESTERE_MACD",
        "timeframe": "1h",
        "timestamp": "2026-07-22T09:30:00Z",
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9,
        "ohlcv_data": [
            {"timestamp": "2026-07-22T09:00:00Z", "close": 50.0, "volume": 100000}
        ],
        "adx_val": 14.0                  # ADX < 18 (Yatay Piyasa)
    }

    input_data = MACDInputData.from_dict(raw_input)
    engine = MACDTechnicalEngine()
    output = engine.evaluate(input_data)

    assert output.flags["whipsaw_risk"] is True
    assert output.primary_recommendation_contribution == "NEUTRAL_WHIPSAW_RISK"
    assert output.momentum_score == 0.50
