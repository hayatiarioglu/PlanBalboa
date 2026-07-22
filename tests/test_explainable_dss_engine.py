import pytest
from financial_ai.dss.explainable_dss_engine import ExplainableDSSEngine

def test_eupwr_advisory_card_user_example():
    """Faz 22: EUPWR Açıklanabilir Karar Destek Sistemi (DSS Advisory Card) Benchmark Testi"""
    engine = ExplainableDSSEngine()
    output = engine.generate_advisory_card(
        ticker="EUPWR",
        timestamp="2026-07-22T22:30:00Z",
        master_fusion_data={},
        meta_labeling_data={"p_success": 0.785, "kelly_portfolio_weight_pct": 6.5}
    )
    out_dict = output.to_dict()

    assert out_dict["ticker"] == "EUPWR"

    # Advisory Summary Verification
    summary = out_dict["advisory_summary"]
    assert summary["action_recommendation"] == "ACCUMULATE_BUY"
    assert summary["confidence_level_pct"] == 78.5
    assert summary["urgency_level"] == "MEDIUM_TERM_SWING"
    assert summary["recommended_portfolio_weight_pct"] == 6.5
    assert summary["suggested_entry_zone"] == [284.50, 286.80]
    assert summary["suggested_stop_loss"] == 272.10
    assert summary["suggested_take_profit_target"] == 325.00
    assert summary["risk_reward_ratio"] == 2.85

    # SHAP Factors Verification
    shap_pos = out_dict["explainable_ai_drivers"]["top_positive_factors"]
    shap_neg = out_dict["explainable_ai_drivers"]["top_negative_factors"]
    assert len(shap_pos) == 3
    assert shap_pos[0]["factor"] == "Microstructure_AKD_Concentration"
    assert shap_pos[0]["impact_score"] == 0.32
    assert shap_neg[0]["factor"] == "Macro_Index_Volatility"
    assert shap_neg[0]["impact_score"] == -0.12

    # Execution Guidance Verification
    exec_guide = out_dict["execution_advisor_guidance"]
    assert exec_guide["liquidity_status"] == "MODERATE_LIQUIDITY"
    assert exec_guide["recommended_order_type"] == "LIMIT_ORDER_TRANCHES"
    assert exec_guide["estimated_slippage_if_market_order_pct"] == 0.45

    # What-If Impact Verification
    what_if = out_dict["what_if_portfolio_impact"]
    assert what_if["current_portfolio_var_pct"] == 2.10
    assert what_if["projected_portfolio_var_pct"] == 2.45
    assert what_if["sector_exposure_after_trade"] == {"ENERGY": "18.5%"}

    # Human-in-the-Loop Capture Verification
    human = out_dict["user_decision_capture"]
    assert human["status"] == "AWAITING_HUMAN_APPROVAL"
