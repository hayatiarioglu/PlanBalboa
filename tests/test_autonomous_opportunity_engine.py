import pytest
from financial_ai.dss.autonomous_opportunity_engine import AutonomousOpportunityEngine

def test_thyao_1_month_opportunity_card_user_example():
    """Faz 23: THYAO 1-Aylık Trend Otonom Sinyal ve Fırsat Kartı Benchmark Testi"""
    engine = AutonomousOpportunityEngine()
    output = engine.generate_opportunity_card(
        ticker="THYAO",
        timestamp="2026-07-22T22:30:00Z",
        time_horizon="1_MONTH",
        current_price=300.00
    )
    out_dict = output.to_dict()

    assert out_dict["recommendation_id"] == "REC_20260722_THYAO"
    assert out_dict["ticker"] == "THYAO"
    assert out_dict["time_horizon"] == "1_MONTH"
    assert out_dict["recommendation_type"] == "STRONGLY_RECOMMEND_BUY"

    # Pricing Targets Verification
    targets = out_dict["pricing_targets"]
    assert targets["current_price"] == 300.00
    assert targets["target_price_low"] == 337.50
    assert targets["target_price_high"] == 348.00
    assert targets["stop_loss_price"] == 288.00
    assert targets["expected_return_pct_range"] == [12.5, 16.0]
    assert targets["risk_reward_ratio"] == 3.0

    # AI Confidence Verification
    confidence = out_dict["ai_confidence"]
    assert confidence["win_probability_pct"] == 82.0
    assert confidence["model_verdict"] == "HIGH_CONFLUENCE"

    # Key Reasons SHAP Verification
    assert len(out_dict["key_reasons_shap"]) == 3
    assert "AKD: BofA" in out_dict["key_reasons_shap"][0]

    # Autonomous Learning Metadata Verification
    meta = out_dict["autonomous_learning_metadata"]
    assert meta["evaluation_due_date"] == "2026-08-22T22:30:00Z"
    assert meta["status"] == "PENDING_REALIZATION"
    assert meta["realized_outcome"] is None

def test_1_week_scalp_opportunity_card():
    """Faz 23: 1-Haftalık Kısa Vadeli Scalp Fırsat Kartı Testi"""
    engine = AutonomousOpportunityEngine()
    output = engine.generate_opportunity_card(
        ticker="X_HISSE",
        timestamp="2026-07-22T22:30:00Z",
        time_horizon="1_WEEK",
        current_price=50.00
    )
    out_dict = output.to_dict()

    assert out_dict["ticker"] == "X_HISSE"
    assert out_dict["time_horizon"] == "1_WEEK"
    assert out_dict["recommendation_type"] == "WEEKLY_SCALP_BUY"
    assert out_dict["pricing_targets"]["expected_return_pct_range"] == [6.0, 9.0]
    assert out_dict["ai_confidence"]["win_probability_pct"] == 74.0

def test_autonomous_self_labeling_loop():
    """Faz 23: İnsansız Otonom Kendi Kendine Öğrenme ve Etiketleme Döngüsü (Self-Labeling Loop) Testi"""
    engine = AutonomousOpportunityEngine()
    card = engine.generate_opportunity_card(
        ticker="THYAO",
        timestamp="2026-07-22T22:30:00Z",
        time_horizon="1_MONTH",
        current_price=300.00
    )

    # Simulation 1: Target Achieved (Realized Price >= 337.50)
    evaluated_success = engine.evaluate_realized_outcome(
        opportunity_card=card,
        max_realized_price=345.00,
        min_realized_price=295.00
    )
    assert evaluated_success.autonomous_learning_metadata.status == "REALIZED_SUCCESS"
    assert evaluated_success.autonomous_learning_metadata.realized_outcome == "TARGET_ACHIEVED_SUCCESS_LABEL_PLUS_ONE"

    # Simulation 2: Stop Loss Hit (Realized Price <= 288.00)
    evaluated_failure = engine.evaluate_realized_outcome(
        opportunity_card=card,
        max_realized_price=310.00,
        min_realized_price=280.00
    )
    assert evaluated_failure.autonomous_learning_metadata.status == "REALIZED_FAILURE"
    assert evaluated_failure.autonomous_learning_metadata.realized_outcome == "STOP_LOSS_HIT_FAILURE_LABEL_MINUS_ONE"
