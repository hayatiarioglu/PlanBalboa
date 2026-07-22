import pytest
from financial_ai.schemas import (
    OrderBookInputData, AKDInputData, CustodyInputData, MoneyFlowInputData
)
from financial_ai.microstructure.order_book_processor import OrderBookProcessor
from financial_ai.microstructure.akd_processor import AKDProcessor
from financial_ai.microstructure.custody_takas_processor import CustodyTakasProcessor
from financial_ai.microstructure.money_flow_processor import MoneyFlowProcessor

def test_order_book_garan_user_example():
    """Modül 13: GARAN Derinlik & Emir Defteri (L2/L3) Testi"""
    raw_input = {
        "ticker": "GARAN",
        "timestamp": "2026-07-22T09:30:00.150Z",
        "bids": [{"price": 112.50, "volume": 500000}],
        "asks": [{"price": 115.00, "volume": 2500000}],
        "cancellations_last_1s": 12
    }

    input_data = OrderBookInputData.from_dict(raw_input)
    engine = OrderBookProcessor()
    output = engine.evaluate(input_data)
    out_dict = output.to_dict()

    assert out_dict["ticker"] == "GARAN"
    assert out_dict["obi_ratio"] == 0.68
    assert out_dict["depth_delta_zscore"] == 2.41
    assert out_dict["iceberg_detected"] == {"side": "BUY", "price_level": 112.50, "estimated_hidden_vol": 1500000}
    assert out_dict["spoofing_warning"] == {"side": "SELL", "price_level": 115.00, "confidence": 0.92}
    assert out_dict["microstructure_signal_score"] == 0.88
    assert out_dict["primary_recommendation_contribution"] == "BULLISH_ORDER_FLOW"

def test_akd_thyao_user_example():
    """Modül 14: THYAO Aracı Kurum Dağılımı (AKD) Testi"""
    raw_input = {
        "ticker": "THYAO",
        "timestamp": "2026-07-22T17:30:00Z",
        "top_buyers": [{"broker": "BANK_OF_AMERICA", "net_lot": 2500000}],
        "top_sellers": [{"broker": "RETAIL_OTHERS", "net_lot": -2100000}],
        "total_volume": 45000000
    }

    input_data = AKDInputData.from_dict(raw_input)
    engine = AKDProcessor()
    output = engine.evaluate(input_data)
    out_dict = output.to_dict()

    assert out_dict["ticker"] == "THYAO"
    assert out_dict["top_5_buyers_share"] == 0.82
    assert out_dict["top_5_sellers_share"] == 0.31
    assert out_dict["dominant_buyer"] == "BANK_OF_AMERICA"
    assert out_dict["akd_concentration_score"] == 0.51
    assert out_dict["akd_regime"] == "STRONG_INSTITUTIONAL_ACCUMULATION"
    assert out_dict["akd_signal_score"] == 0.91
    assert out_dict["primary_recommendation_contribution"] == "BULLISH"

def test_custody_eupwr_user_example():
    """Modül 15: EUPWR Takas & Saklama Analizi Testi"""
    raw_input = {
        "ticker": "EUPWR",
        "timestamp": "2026-07-22T00:00:00Z",
        "custody_shares": {"CITIBANK": 45000000, "DEUTSCHE": 25000000},
        "weekly_foreign_change": 2450000,
        "total_capital": 110000000
    }

    input_data = CustodyInputData.from_dict(raw_input)
    engine = CustodyTakasProcessor()
    output = engine.evaluate(input_data)
    out_dict = output.to_dict()

    assert out_dict["ticker"] == "EUPWR"
    assert out_dict["top_3_custody_pct"] == 0.784
    assert out_dict["weekly_foreign_custody_change_shares"] == 2450000
    assert out_dict["custody_concentration_index"] == 0.85
    assert out_dict["flags"]["tight_custody_float"] is True
    assert out_dict["flags"]["off_market_transfer_detected"] is True
    assert out_dict["custody_signal_score"] == 0.89
    assert out_dict["primary_recommendation_contribution"] == "BULLISH_CUSTODY_LOCK"

def test_money_flow_eupwr_user_example():
    """Modül 16: EUPWR Net Para Akışı & Mikro Yapı Karar Füzyon Testi"""
    raw_input = {
        "ticker": "EUPWR",
        "timestamp": "2026-07-22T15:30:00Z",
        "order_book_obi": 0.62,
        "top_5_akd_concentration": 0.74,
        "weekly_custody_change_pct": 0.035,
        "net_money_flow_tl": 45000000,
        "price_change_today_pct": -0.012
    }

    input_data = MoneyFlowInputData.from_dict(raw_input)
    engine = MoneyFlowProcessor()
    output = engine.evaluate(input_data)
    out_dict = output.to_dict()

    assert out_dict["ticker"] == "EUPWR"
    assert out_dict["microstructure_regime"] == "STEALTH_ACCUMULATION_DETECTED"
    assert out_dict["flow_scores"]["order_book_score"] == 0.85
    assert out_dict["flow_scores"]["akd_score"] == 0.91
    assert out_dict["flow_scores"]["custody_score"] == 0.88
    assert out_dict["flow_scores"]["money_flow_score"] == 0.94
    assert out_dict["divergence_flags"]["price_down_money_in_divergence"] is True
    assert out_dict["divergence_flags"]["spoofing_sell_wall_detected"] is True
    assert out_dict["overall_microstructure_signal"] == 0.92
    assert out_dict["primary_recommendation_contribution"] == "STRONG_BULLISH_INSTITUTIONAL_BUY"
