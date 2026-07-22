import pytest
from financial_ai.schemas import (
    MetaLabelingInputData, MarketImpactInputData, SampleUniquenessInputData, BiTemporalTimestamp
)
from financial_ai.fusion.meta_labeling_engine import MetaLabelingEngine
from financial_ai.preprocessing.sample_uniqueness import SampleUniquenessEngine
from financial_ai.execution.market_impact_slippage import MarketImpactSlippageEngine
from financial_ai.preprocessing.cross_sectional_rank import CrossSectionalRankEngine

def test_meta_labeling_and_signal_confidence():
    """Faz 21: Meta-Labeling Sinyal Başarı Olasılığı P(Success) Testi (Pure Signal)"""
    raw_input = {
        "ticker": "EUPWR",
        "timestamp": "2026-07-22T17:30:00Z",
        "primary_signal_side": 1,
        "volatility_atr": 0.015,
        "bid_ask_spread": 0.002,
        "order_book_imbalance_obi": 0.65,
        "volume_intensity": 2.5,
        "historical_win_rate": 0.58
    }

    input_data = MetaLabelingInputData(**raw_input)
    engine = MetaLabelingEngine()
    output = engine.evaluate(input_data)
    out_dict = output.to_dict()

    assert out_dict["ticker"] == "EUPWR"
    assert out_dict["p_success"] > 0.60
    assert out_dict["is_meta_approved"] is True

def test_absolute_macro_overlay_crash_protection():
    """Kusursuz Mimari Düzeltme 2: Mutlak Makro Rejim Katmanı (Systemic Crash Protection) Testi"""
    from financial_ai.schemas import MacroOverlayInputData
    from financial_ai.macro.macro_overlay_engine import AbsoluteMacroOverlayEngine

    # Case 1: Systemic Crash (BIST100 < EMA200 & High Volatility)
    crash_input = MacroOverlayInputData(
        timestamp="2026-07-22T17:30:00Z",
        bist100_index_price=8500.0,
        bist100_ema_200=9200.0,
        volatility_index_vix=35.0,
        systemic_crash_flag=True
    )
    macro_engine = AbsoluteMacroOverlayEngine()
    output_crash = macro_engine.evaluate(crash_input)

    assert output_crash.macro_state == 0
    assert output_crash.is_cash_protection_active is True
    assert output_crash.macro_regime_label == "FORCE_CASH_SYSTEMIC_CRASH_PROTECTION"
    assert output_crash.risk_multiplier == 0.0

    # Case 2: Normal Market
    normal_macro = MacroOverlayInputData(
        timestamp="2026-07-22T17:30:00Z",
        bist100_index_price=9800.0,
        bist100_ema_200=9200.0,
        volatility_index_vix=18.0
    )
    output_normal = macro_engine.evaluate(normal_macro)

    assert output_normal.macro_state == 1
    assert output_normal.is_cash_protection_active is False
    assert output_normal.macro_regime_label == "NORMAL_MARKET_REGIME"
    assert output_normal.risk_multiplier == 1.0

def test_sample_uniqueness_concurrency():
    """Faz 21 Düzeltme 2: Marcos López de Prado Sample Uniqueness (u_i) Testi"""
    # 3 overlapping Triple-Barrier labels
    raw_input = {
        "label_start_times": [1, 2, 5],
        "label_end_times": [5, 6, 8],
        "returns": [0.03, 0.02, -0.01]
    }

    input_data = SampleUniquenessInputData(**raw_input)
    engine = SampleUniquenessEngine()
    output = engine.compute_uniqueness(input_data)

    assert len(output.mean_uniqueness_scores) == 3
    assert len(output.sample_weights) == 3
    # Overlapping samples should have uniqueness score < 1.0
    assert output.mean_uniqueness_scores[0] < 1.0

def test_market_impact_and_limit_down_lock():
    """Faz 21 Düzeltme 4: Non-linear Market Impact Slippage & Taban Kilitlenme Testi"""
    # Case 1: Normal execution with slippage
    normal_input = MarketImpactInputData(
        ticker="THYAO",
        order_volume_shares=50000.0,
        adv_20_shares=1000000.0,
        daily_volatility=0.025,
        lower_barrier_price=300.0,
        is_limit_down_locked=False
    )
    engine = MarketImpactSlippageEngine()
    output_normal = engine.simulate_execution(normal_input)

    assert output_normal.execution_feasible is True
    assert output_normal.executed_price < 300.0  # Slippage lowers execution price
    assert output_normal.slippage_pct > 0.0

    # Case 2: Limit-Down Lock execution failure
    locked_input = MarketImpactInputData(
        ticker="DISTRESSED",
        order_volume_shares=100000.0,
        adv_20_shares=500000.0,
        daily_volatility=0.05,
        lower_barrier_price=50.0,
        is_limit_down_locked=True
    )
    output_locked = engine.simulate_execution(locked_input)

    assert output_locked.execution_feasible is False
    assert output_locked.rejection_reason == "LIMIT_DOWN_LOCK_NO_BID_VOLUME"

def test_cross_sectional_percentile_ranking():
    """Faz 21 Düzeltme 5: Cross-Sectional Percentile Ranking Testi"""
    pe_ratios = [12.5, 45.0, 8.2, 18.0, 100.0]
    scaled = CrossSectionalRankEngine.transform_percentile_rank(pe_ratios)

    assert len(scaled) == 5
    assert min(scaled) == 0.0   # Lowest value gets 0.0
    assert max(scaled) == 1.0   # Highest value gets 1.0
    assert scaled[2] == 0.0     # 8.2 is lowest -> 0.0
    assert scaled[4] == 1.0     # 100.0 is highest -> 1.0

def test_bi_temporal_timestamp_validation():
    """Faz 21 Düzeltme 3: Bi-Temporal Timestamp (T+2 Lag) Testi"""
    # Event happened on 15 March, settled/known on 17 March
    ts = BiTemporalTimestamp(
        event_timestamp="2026-03-15T17:00:00Z",
        knowledge_timestamp="2026-03-17T09:00:00Z"
    )

    # On 15 March (simulated time), data should NOT be visible (Lookahead Leakage Prevention)
    assert ts.is_valid_at("2026-03-15T18:00:00Z") is False

    # On 17 March, data becomes visible
    assert ts.is_valid_at("2026-03-17T10:00:00Z") is True
