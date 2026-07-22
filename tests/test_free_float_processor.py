import pytest
from financial_ai.schemas import FreeFloatInputData
from financial_ai.risk.free_float_processor import FreeFloatProcessor

def test_kboru_user_example():
    """Modül 18: KBORU Serbest Dolaşım Oranı ve Sıkışma Riski Testi (Pure Signal)"""
    raw_input = {
        "ticker": "KBORU",
        "timestamp": "2026-07-22T09:30:00Z",
        "total_shares_outstanding": 100000000,
        "free_float_shares": 8500000,
        "current_price": 145.50,
        "daily_volume_shares": 12500000,
        "lockup_expiration_date": "2026-07-28T00:00:00Z",
        "lockup_shares_count": 25000000
    }

    input_data = FreeFloatInputData.from_dict(raw_input)
    engine = FreeFloatProcessor()
    output = engine.evaluate(input_data)
    out_dict = output.to_dict()

    assert out_dict["ticker"] == "KBORU"
    assert out_dict["free_float_pct"] == 8.50
    assert out_dict["free_float_market_cap_tl"] == 1236750000.0
    assert out_dict["cornering_risk_index"] == 1.29
    assert out_dict["flags"]["is_micro_float_danger"] is True
    assert out_dict["flags"]["trading_execution_blocked"] is True

def test_healthy_float_stock():
    """Modül 18: Sağlıklı Dolaşım Oranı ve Sinyal Onayı Testi"""
    raw_input = {
        "ticker": "HEALTHY_FLOAT_STOCK",
        "timestamp": "2026-07-22T09:30:00Z",
        "total_shares_outstanding": 500000000,
        "free_float_shares": 150000000,   # 30% Float
        "current_price": 50.0,
        "daily_volume_shares": 10000000
    }

    input_data = FreeFloatInputData.from_dict(raw_input)
    engine = FreeFloatProcessor()
    output = engine.evaluate(input_data)

    assert output.free_float_pct == 30.0
    assert output.flags["trading_execution_blocked"] is False
