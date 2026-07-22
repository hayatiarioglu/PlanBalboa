import pytest
from financial_ai.schemas import BollingerInputData, BollingerOutputData
from financial_ai.technical.bollinger_proc import BollingerTechnicalEngine

def test_asels_user_example():
    """Kullanıcının ilettiği ASELS örnek girdisi ve şema uyum testi"""
    raw_input = {
        "ticker": "ASELS",
        "timeframe": "1D",
        "timestamp": "2026-07-22T09:30:00Z",
        "period": 20,
        "k_std_dev": 2.0,
        "ohlcv_data": [
            {"timestamp": "2026-07-21T00:00:00Z", "close": 61.5, "volume": 18000000},
            {"timestamp": "2026-07-22T00:00:00Z", "close": 65.0, "volume": 42000000}
        ],
        "adx_val": 34.2,
        "atr_val": 1.85
    }

    input_data = BollingerInputData.from_dict(raw_input)
    engine = BollingerTechnicalEngine()
    output: BollingerOutputData = engine.evaluate(input_data)
    out_dict = output.to_dict()

    assert out_dict["upper_band"] == 63.80
    assert out_dict["middle_band"] == 58.50
    assert out_dict["lower_band"] == 53.20
    assert out_dict["bandwidth"] == 0.181
    assert out_dict["percent_b"] == 1.113
    assert out_dict["bw_z_score"] == -1.85

    sq = out_dict["squeeze_status"]
    assert sq["is_squeezing"] is True
    assert sq["squeeze_duration_candles"] == 14
    assert sq["breakout_direction"] == "UPWARD_EXPANSION"

    assert output.flags["band_walking_active"] is True
    assert output.flags["mean_reversion_risk"] is False
    assert output.flags["head_fake_warning"] is False
    assert output.flags["volume_confirmed"] is True

    assert out_dict["volatility_breakout_score"] == 0.93
    assert out_dict["primary_recommendation_contribution"] == "STRONG_BULLISH_BREAKOUT"

def test_bollinger_mean_reversion_low_adx():
    """Yatay Piyasada Üst Bant İhlali ve Ortalamaya Dönüş Riski (ADX < 20)"""
    bars = [{"timestamp": f"2026-07-0{i}T00:00:00Z", "close": 50.0 + (i * 0.5), "volume": 100000} for i in range(1, 22)]
    bars[-1]["close"] = 65.0            # Üst bandı kırdı (%B > 1.0)

    raw_input = {
        "ticker": "MEAN_REV_TEST",
        "timeframe": "1D",
        "timestamp": "2026-07-22T09:30:00Z",
        "period": 20,
        "k_std_dev": 2.0,
        "ohlcv_data": bars,
        "adx_val": 15.0                  # ADX < 20 (Yatay / Zayıf Trend)
    }

    input_data = BollingerInputData.from_dict(raw_input)
    engine = BollingerTechnicalEngine()
    output = engine.evaluate(input_data)

    assert output.flags["mean_reversion_risk"] is True
    assert output.flags["band_walking_active"] is False
    assert output.primary_recommendation_contribution == "MEAN_REVERSION_OVERBOUGHT"
    assert output.volatility_breakout_score == 0.15
