import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd


class TelemetryLogger:
    """
    Faz 4.4.1: Gerçek Zamanlı Loglama ve Metrik Takibi
    (Real-time Logging & Telemetry)
    """
    def __init__(self, log_dir: str = "logs", log_filename: str = "aether_metrics.jsonl"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / log_filename

        # Setup standard Python logger as well (for console/debug)
        self.logger = logging.getLogger("AetherTelemetry")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def log_metric(self, event_name: str, metrics: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Herhangi bir metrik paketini JSONL formatında kaydeder.
        """
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event": event_name,
            "metrics": metrics,
            "metadata": metadata or {}
        }
        
        # Write to JSONL file
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            self.logger.error(f"Telemetri dosyasına yazılamadı: {e}")

        # Also log to console
        self.logger.info(f"[{event_name}] {metrics}")

    def log_optimization_result(self, result: Any, pipeline_duration_ms: float, loaded_checkpoint: str) -> None:
        """
        PortfolioOptimizationResult objesinden ve master pipeline verilerinden log oluşturur.
        """
        metrics = {
            "is_success": result.is_success,
            "solver_status": result.status,
            "condition_number": result.condition_number,
            "objective_value": result.objective_value,
            "total_turnover": result.total_turnover,
            "estimated_impact_cost_tl": result.estimated_impact_cost_tl,
            "estimated_settlement_cost_tl": result.estimated_settlement_cost_tl,
            "opt_run_time_ms": result.run_time_ms,
            "pipeline_run_time_ms": pipeline_duration_ms
        }

        metadata = {
            "loaded_checkpoint": loaded_checkpoint,
            "solver_info": result.solver_info
        }
        
        # Portföy ağırlıkları istatistiklerini de ekleyelim
        if hasattr(result, 'weights') and isinstance(result.weights, pd.Series):
            weights = result.weights
            metrics["active_positions_count"] = int((weights.abs() > 1e-6).sum())
            metrics["max_weight"] = float(weights.max())
            metrics["min_weight"] = float(weights.min())
            metrics["sum_weights"] = float(weights.sum())

        self.log_metric("portfolio_optimization_completed", metrics, metadata)

    def log_soft_ndcg(self, loss_value: float, iteration: int, stage: str = "train") -> None:
        """
        Soft-NDCG loss takibi
        """
        metrics = {
            "loss_value": float(loss_value),
            "iteration": int(iteration),
            "stage": stage
        }
        self.log_metric("soft_ndcg_loss", metrics)
