import torch
import pandas as pd
import numpy as np
from pathlib import Path
from aether.monitoring.telemetry import TelemetryLogger
from aether.signal.soft_ndcg_loss import ClampedScheduledSoftNDCGLoss
from aether.master_pipeline import AetherMasterPipeline, PortfolioQPConfig
import json

def test_telemetry():
    print("Testing Telemetry Integration...")
    logger = TelemetryLogger(log_dir="test_logs", log_filename="test_metrics.jsonl")
    
    # 1. Test Soft-NDCG Loss Telemetry
    y_pred = torch.randn(10)
    y_true = torch.rand(10)
    
    loss_fn = ClampedScheduledSoftNDCGLoss(tau_min=0.1)
    loss = loss_fn(y_pred, y_true, telemetry_logger=logger, iteration=42, stage="test_stage")
    print(f"Loss computed: {loss.item()}")
    
    # 2. Test Master Pipeline Telemetry
    pipeline = AetherMasterPipeline(checkpoint_dir="test_checkpoints")
    
    # Prepare dummy snapshot
    assets = [f"ASSET_{i}" for i in range(10)]
    snapshot = {
        "assets": assets,
        "x": torch.randn(1, 10, 5),
        "v_hybrid": torch.randn(1, 10, 3),
        "returns": torch.zeros(10),
        "raw_alpha": np.random.randn(10) * 0.05,
        "cov_matrix": pd.DataFrame(np.eye(10) * 0.002, index=assets, columns=assets),
        "volatilities": pd.Series(np.ones(10)*0.01, index=assets),
        "adv_series": pd.Series(np.ones(10)*1e6, index=assets),
        "settlement_days": pd.Series(np.ones(10)*2, index=assets)
    }
    
    config = PortfolioQPConfig(w_prev={assets[i]: 0.0 for i in range(10)})
    # Overwrite pipeline's telemetry to use our test logger
    pipeline.telemetry = logger
    
    res = pipeline.execute_friday_pipeline(
        snapshot=snapshot,
        total_capital=10_000_000.0,
        config=config
    )
    print(f"Pipeline executed. Success: {res.optimization_result.is_success}")
    
    # Verify file contents
    log_file = Path("test_logs/test_metrics.jsonl")
    assert log_file.exists(), "Log file was not created!"
    
    with open(log_file, "r") as f:
        lines = f.readlines()
        assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"
        
        loss_log = json.loads(lines[0])
        assert loss_log["event"] == "soft_ndcg_loss"
        assert loss_log["metrics"]["iteration"] == 42
        
        opt_log = json.loads(lines[1])
        assert opt_log["event"] == "portfolio_optimization_completed"
        assert "solver_status" in opt_log["metrics"]
        assert "active_positions_count" in opt_log["metrics"]
        
    print("ALL TELEMETRY TESTS PASSED!")

if __name__ == "__main__":
    test_telemetry()
