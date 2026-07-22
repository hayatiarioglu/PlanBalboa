import pytest
from financial_ai.schemas import RSIInputData, RSIOutputData
from financial_ai.technical.rsi_proc import RSITechnicalEngine

def test_thyao_user_example():
    """Kullanıcının ilettiği THYAO örnek girdisi ve şema uyum testi"""
    raw_input = {
        "ticker": "THYAO",
        "timeframe": "1D",
        "timestamp": "2026-07-22T09:30:00Z",
        "period": 14,
        "ohlcv_data": [
            {"timestamp": "2026-07-21T00:00:00Z", "open": 310.0, "high": 315.0, "low": 308.0, "close": 312.5, "volume": 12500000},
            {"timestamp": "2026-07-22T00:00:00Z", "open": 312.5, "high": 322.0, "low": 311.0, "close": 320.0, "volume": 18000000}
        ],
        "adx_val": 22.5,
        "atr_val": 6.8
    }

    input_data = RSIInputData.from_dict(raw_input)
    engine = RSITechnicalEngine()
    output: RSIOutputData = engine.evaluate(input_data)
    out_dict = output.to_dict()

    assert out_dict["raw_rsi"] == 72.4
    assert out_dict["dynamic_overbought_threshold"] == 75.0
    assert out_dict["dynamic_oversold_threshold"] == 32.0
    assert out_dict["rsi_velocity"] == 1.8

    divergence = out_dict["divergence"]
    assert divergence["detected"] is True
    assert divergence["type"] == "REGULAR_BEARISH_DIVERGENCE"
    assert divergence["confidence"] == 0.84

    assert output.flags["is_overbought"] is False
    assert output.flags["momentum_exhaustion_warning"] is True
    assert output.flags["trend_override_active"] is False

    assert out_dict["technical_signal_score"] == 0.35
    assert out_dict["primary_recommendation_contribution"] == "BEARISH_REVERSAL_RISK"

def test_trend_override_momentum_surf():
    """Güçlü Trendde Erken Satış Önleme (ADX > 35 ve RSI > 70)"""
    raw_input = {
        "ticker": "TAVAN_SERISI",
        "timeframe": "1D",
        "timestamp": "2026-07-22T09:30:00Z",
        "period": 14,
        "ohlcv_data": [
            {"timestamp": "2026-07-21T00:00:00Z", "open": 100.0, "high": 110.0, "low": 99.0, "close": 105.0, "volume": 5000000},
            {"timestamp": "2026-07-22T00:00:00Z", "open": 105.0, "high": 120.0, "low": 104.0, "close": 118.0, "volume": 8000000}
        ],
        "adx_val": 42.0,                  # ADX > 35 (Güçlü Trend)
        "atr_val": 4.5
    }

    input_data = RSIInputData.from_dict(raw_input)
    engine = RSITechnicalEngine()
    output = engine.evaluate(input_data)

    assert output.flags["trend_override_active"] is True
    assert output.primary_recommendation_contribution == "BULLISH_MOMENTUM_SURF"
    assert output.technical_signal_score == 0.90
