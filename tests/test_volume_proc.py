import pytest
from financial_ai.schemas import VolumeInputData, VolumeOutputData
from financial_ai.technical.volume_proc import VolumeTechnicalEngine

def test_kchol_user_example():
    """Kullanıcının ilettiği KCHOL örnek girdisi ve şema uyum testi"""
    raw_input = {
        "ticker": "KCHOL",
        "timeframe": "1D",
        "timestamp": "2026-07-22T09:30:00Z",
        "ohlcv_data": [
            {"timestamp": "2026-07-21T00:00:00Z", "close": 230.0, "volume": 12000000},
            {"timestamp": "2026-07-22T00:00:00Z", "close": 238.5, "volume": 35000000}
        ],
        "bid_volume": 19000000,
        "ask_volume": 16000000,
        "market_cap_tier": "LARGE_CAP"
    }

    input_data = VolumeInputData.from_dict(raw_input)
    engine = VolumeTechnicalEngine()
    output: VolumeOutputData = engine.evaluate(input_data)
    out_dict = output.to_dict()

    assert out_dict["obv_value"] == 452000000
    assert out_dict["obv_ema_20"] == 410000000
    assert out_dict["rvol_normalized"] == 2.45
    assert out_dict["bid_ask_delta"] == 3000000

    div = out_dict["divergence"]
    assert div["detected"] is True
    assert div["type"] == "BULLISH_ACCUMULATION_DIVERGENCE"
    assert div["confidence"] == 0.89

    assert output.flags["is_volume_spike"] is True
    assert output.flags["wash_trading_risk"] is False
    assert output.flags["illiquid_stock_trap"] is False
    assert output.flags["institutional_buying_detected"] is True

    assert out_dict["smart_money_flow_score"] == 0.92
    assert out_dict["primary_recommendation_contribution"] == "STRONG_INSTITUTIONAL_ACCUMULATION"

def test_unsupported_volume_trap():
    """Hacimsiz Sahte Yükseliş Tuzağı Testi (Fiyat +%3, RVOL < 0.6)"""
    bars = [
        {"timestamp": "2026-07-20T00:00:00Z", "close": 100.0, "volume": 10000000},
        {"timestamp": "2026-07-21T00:00:00Z", "close": 101.0, "volume": 10000000},
        {"timestamp": "2026-07-22T00:00:00Z", "close": 104.0, "volume": 2000000}     # Fiyat +%3, Hacim çok düşük (RVOL = 0.2)
    ]

    raw_input = {
        "ticker": "FAKE_RALLY_STOCK",
        "timeframe": "1D",
        "timestamp": "2026-07-22T09:30:00Z",
        "ohlcv_data": bars
    }

    input_data = VolumeInputData.from_dict(raw_input)
    engine = VolumeTechnicalEngine()
    output = engine.evaluate(input_data)

    assert output.primary_recommendation_contribution == "UNSUPPORTED_VOLUME_TRAP"
    assert output.smart_money_flow_score == 0.18
