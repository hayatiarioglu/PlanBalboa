import pytest
from financial_ai.schemas import FibInputData, FibOutputData
from financial_ai.technical.fib_proc import FibTechnicalEngine

def test_thyao_user_example():
    """Kullanıcının ilettiği THYAO Fibonacci örnek girdisi ve şema uyum testi"""
    raw_input = {
        "ticker": "THYAO",
        "timeframe": "4h",
        "timestamp": "2026-07-22T09:30:00Z",
        "pivot_left": 10,
        "pivot_right": 10,
        "ohlcv_data": [
            {"timestamp": "2026-07-21T12:00:00Z", "high": 330.0, "low": 310.0, "close": 315.0, "volume": 14000000},
            {"timestamp": "2026-07-22T09:00:00Z", "high": 318.0, "low": 302.0, "close": 305.0, "volume": 19000000}
        ],
        "atr_val": 5.4,
        "confluence_inputs": {
            "ema_200": 303.5,
            "hvn_price": 304.0
        }
    }

    input_data = FibInputData.from_dict(raw_input)
    engine = FibTechnicalEngine()
    output: FibOutputData = engine.evaluate(input_data)
    out_dict = output.to_dict()

    assert out_dict["trend_direction"] == "BULLISH_RETRACEMENT_DOWN"

    anchors = out_dict["anchors"]
    assert anchors["swing_high"] == 330.0
    assert anchors["swing_low"] == 260.0

    fibs = out_dict["fib_levels"]
    assert fibs["level_0236"] == 313.48
    assert fibs["level_0382"] == 303.26
    assert fibs["level_0500"] == 295.00
    assert fibs["golden_pocket_0618_0650"] == [286.74, 284.50]
    assert fibs["level_0786"] == 274.98

    conf = out_dict["confluence_analysis"]
    assert conf["active_zone"] == "LEVEL_0382"
    assert conf["confluence_score"] == 3
    assert conf["matched_elements"] == ["FIB_0382", "EMA_200", "VOLUME_PROFILE_HVN"]

    assert output.flags["price_in_prz_zone"] is True
    assert output.flags["structure_invalidated"] is False

    assert out_dict["prz_reversal_score"] == 0.91
    assert out_dict["primary_recommendation_contribution"] == "STRONG_CONFLUENCE_SUPPORT_BUY"

def test_market_structure_invalidation_below_0786():
    """Fibonacci %78.6 Seviyesinin Kırılmasıyla Trend Yapısının Bozulması (MSS) Testi"""
    bars = [
        {"timestamp": "2026-07-20T00:00:00Z", "high": 200.0, "low": 100.0, "close": 150.0, "volume": 1000000},
        {"timestamp": "2026-07-22T00:00:00Z", "high": 120.0, "low": 115.0, "close": 118.0, "volume": 2000000}    # 118.0 < Fib 0.786 (121.4)
    ]

    raw_input = {
        "ticker": "BROKEN_STRUCTURE_STOCK",
        "timeframe": "1D",
        "timestamp": "2026-07-22T09:30:00Z",
        "pivot_left": 5,
        "pivot_right": 5,
        "ohlcv_data": bars
    }

    input_data = FibInputData.from_dict(raw_input)
    engine = FibTechnicalEngine()
    output = engine.evaluate(input_data)

    assert output.flags["structure_invalidated"] is True
    assert output.primary_recommendation_contribution == "MARKET_STRUCTURE_INVALIDATION"
    assert output.prz_reversal_score == 0.12
