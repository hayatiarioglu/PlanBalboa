import pytest
from financial_ai.schemas import MasterFusionInputData
from financial_ai.fusion.master_decision_fusion_model import MasterDecisionFusionModel

def test_full_19_module_confluence_strong_buy():
    """Faz 20: 19 Modülün Tamamından Gelen Verilerle Güçlü Alım (Strong Buy Confluence) Testi"""
    raw_input = {
        "ticker": "EUPWR",
        "timestamp": "2026-07-22T17:30:00Z",
        "pe_output": {"signal_score": 0.85},
        "pb_output": {"safety_margin_score": 0.82},
        "ebitda_output": {"operational_quality_score": 0.88},
        "ev_output": {"valuation_attractiveness_score": 0.90},
        "roe_output": {"capital_quality_score": 0.94},
        "current_ratio_output": {"solvency_health_score": 0.80},
        "rsi_output": {"technical_signal_score": 0.85},
        "ma_output": {"trend_state_score": 0.88},
        "macd_output": {"momentum_score": 0.91},
        "bollinger_output": {"volatility_breakout_score": 0.93},
        "volume_output": {"smart_money_flow_score": 0.92},
        "fib_output": {
            "prz_reversal_score": 0.91,
            "confluence_analysis": {"confluence_score": 3}
        },
        "order_book_output": {"microstructure_signal_score": 0.88},
        "akd_output": {
            "akd_signal_score": 0.91,
            "akd_regime": "STRONG_INSTITUTIONAL_ACCUMULATION"
        },
        "custody_output": {"custody_signal_score": 0.89},
        "microstructure_fusion_output": {
            "overall_microstructure_signal": 0.92,
            "microstructure_regime": "STEALTH_ACCUMULATION_DETECTED"
        },
        "beta_output": {"risk_contribution_score": 0.82},
        "free_float_output": {
            "liquidity_risk_score": 0.10,
            "max_allowed_position_shares": 500000.0,
            "flags": {"trading_execution_blocked": False}
        },
        "sharpe_output": {"final_risk_adjusted_score": 0.85}
    }

    input_data = MasterFusionInputData.from_dict(raw_input)
    engine = MasterDecisionFusionModel()
    output = engine.evaluate(input_data)
    out_dict = output.to_dict()

    assert out_dict["ticker"] == "EUPWR"
    assert out_dict["composite_master_score"] >= 0.85
    assert out_dict["final_recommendation"] == "STRONG_BUY_INSTITUTIONAL_CONFLUENCE"
    assert out_dict["hard_safety_block_triggered"] is False
    assert out_dict["recommended_portfolio_weight_pct"] == 12.5
    assert len(out_dict["positive_drivers"]) >= 2

def test_hard_system_block_low_float():
    """Faz 20: Serbest Dolaşım Oranı %8.5 ve CRI > 8.0 Nedeniyle Sert İnfaz Engeli (Hard Block) Testi"""
    raw_input = {
        "ticker": "KBORU",
        "timestamp": "2026-07-22T17:30:00Z",
        "free_float_output": {
            "cornering_risk_index": 9.24,
            "max_allowed_position_shares": 85000.0,
            "flags": {"trading_execution_blocked": True}
        }
    }

    input_data = MasterFusionInputData.from_dict(raw_input)
    engine = MasterDecisionFusionModel()
    output = engine.evaluate(input_data)
    out_dict = output.to_dict()

    assert out_dict["ticker"] == "KBORU"
    assert out_dict["composite_master_score"] == 0.0
    assert out_dict["final_recommendation"] == "REJECTED_LOW_FLOAT_CORNERING_RISK"
    assert out_dict["hard_safety_block_triggered"] is True
    assert out_dict["recommended_portfolio_weight_pct"] == 0.0
    assert out_dict["max_position_cap_shares"] == 85000.0

def test_double_negative_bankruptcy_penalty():
    """Faz 20: Çift Negatiflik İflas Tuzağında Otomatik Skorsal Penaltı Uygulama Testi"""
    raw_input = {
        "ticker": "DISTRESSED_STOCK",
        "timestamp": "2026-07-22T17:30:00Z",
        "roe_output": {
            "capital_quality_score": 0.20,
            "flags": {"double_negative_trap": True}
        },
        "free_float_output": {
            "flags": {"trading_execution_blocked": False}
        }
    }

    input_data = MasterFusionInputData.from_dict(raw_input)
    engine = MasterDecisionFusionModel()
    output = engine.evaluate(input_data)

    assert "PENALTY: Çift Negatiflik Özkaynak İflas Tuzağı!" in output.critical_warnings
    assert output.composite_master_score < 0.40
    assert output.final_recommendation == "STRONG_SELL_REDUCE_EXPOSURE"
