import pytest
from financial_ai.schemas import MAInputData, MAOutputData
from financial_ai.technical.ma_proc import MATechnicalEngine

def test_garan_user_example():
    """Kullanıcının ilettiği GARAN örnek girdisi ve şema uyum testi"""
    raw_input = {
        "ticker": "GARAN",
        "timeframe": "1D",
        "timestamp": "2026-07-22T09:30:00Z",
        "ma_short_period": 50,
        "ma_long_period": 200,
        "ma_type": "EMA",
        "ohlcv_data": [
            {"timestamp": "2026-07-21T00:00:00Z", "close": 112.5, "volume": 45000000},
            {"timestamp": "2026-07-22T00:00:00Z", "close": 115.0, "volume": 62000000}
        ],
        "adx_val": 28.4,
        "atr_val": 3.2
    }

    input_data = MAInputData.from_dict(raw_input)
    engine = MATechnicalEngine()
    output: MAOutputData = engine.evaluate(input_data)
    out_dict = output.to_dict()

    assert out_dict["ma_50_value"] == 108.2
    assert out_dict["ma_200_value"] == 94.5
    assert out_dict["ma_spread_pct"] == 0.145
    assert out_dict["ma_200_slope"] == 0.024

    cross = out_dict["cross_status"]
    assert cross["event"] == "GOLDEN_CROSS_ACTIVE"
    assert cross["candles_since_cross"] == 12
    assert cross["is_confirmed"] is True

    dist = out_dict["distance_metrics"]
    assert dist["distance_to_200ma_pct"] in [0.216, 0.217]
    assert dist["distance_z_score"] == 1.82
    assert dist["mean_reversion_risk"] == "MEDIUM"

    assert output.flags["is_whipsaw_risk"] is False
    assert output.flags["overextended_warning"] is False

    assert out_dict["trend_state_score"] == 0.88
    assert out_dict["primary_recommendation_contribution"] == "STRONG_BULLISH_TREND"

def test_whipsaw_risk_low_adx():
    """Yatay Piyasada Sahte Kesişim (Whipsaw Risk) Testi (ADX < 18)"""
    raw_input = {
        "ticker": "TESTERE_HISSESİ",
        "timeframe": "1D",
        "timestamp": "2026-07-22T09:30:00Z",
        "ma_short_period": 50,
        "ma_long_period": 200,
        "ma_type": "EMA",
        "ohlcv_data": [
            {"timestamp": "2026-07-22T00:00:00Z", "close": 50.0, "volume": 1000000}
        ],
        "adx_val": 14.0                  # ADX < 18 (Yatay Piyasa)
    }

    input_data = MAInputData.from_dict(raw_input)
    engine = MATechnicalEngine()
    output = engine.evaluate(input_data)

    assert output.flags["is_whipsaw_risk"] is True
    assert output.primary_recommendation_contribution == "NEUTRAL_WHIPSAW_RISK"
    assert output.trend_state_score == 0.50
