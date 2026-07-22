import unittest
import tempfile
import shutil
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from aether.master_pipeline import AetherMasterPipeline, MasterPipelineExecutionResult
from aether.signal.signal_orchestrator import SignalPipelineOrchestrator
from aether.trainer.checkpoint_manager import CheckpointManager
from aether.trainer.data_pipeline import InsufficientAssetCoverageError

class TestAetherMasterPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.assets = [f"GARAN_{i:02d}" for i in range(10)]
        self.expected_universe = self.assets
        
        # Instantiate signal model
        self.signal_model = SignalPipelineOrchestrator(
            input_dim=16,
            hybrid_vol_dim=4
        )
        
        # Save a dummy checkpoint W_{t-1}
        self.cp_manager = CheckpointManager(self.test_dir)
        self.cp_path = self.cp_manager.save_checkpoint_atomically(
            model=self.signal_model,
            optimizer=None,
            version_tag="2026_W28"
        )
        
        # Create pipeline
        self.pipeline = AetherMasterPipeline(
            checkpoint_dir=self.test_dir,
            expected_universe=self.expected_universe,
            signal_orchestrator=self.signal_model
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_friday_pipeline_end_to_end_execution(self):
        """
        Tests Friday 17:52 live pipeline execution from PIT Snapshot -> Signal -> Covariance -> OSQP Weights.
        """
        n_assets = len(self.assets)
        x = torch.randn(n_assets, 16)
        v_hybrid = torch.randn(n_assets, 4)
        
        # Covariance matrix
        raw_cov = np.eye(n_assets) * 0.005 + 0.001
        cov_df = pd.DataFrame(raw_cov, index=self.assets, columns=self.assets)
        
        snapshot = {
            "assets": self.assets,
            "x": x,
            "v_hybrid": v_hybrid,
            "cov_matrix": cov_df,
            "volatilities": pd.Series(0.02, index=self.assets),
            "adv_series": pd.Series(50_000_000.0, index=self.assets),
            "settlement_days": pd.Series(2, index=self.assets)
        }
        
        result = self.pipeline.execute_friday_pipeline(
            snapshot=snapshot,
            total_capital=100_000_000.0
        )
        
        self.assertIsInstance(result, MasterPipelineExecutionResult)
        self.assertTrue(result.optimization_result.is_success)
        self.assertEqual(len(result.target_weights), n_assets)
        self.assertAlmostEqual(result.target_weights.sum(), 1.0, places=4)
        self.assertEqual(result.loaded_checkpoint_version, "2026_W28")

    def test_pipeline_aborts_on_insufficient_coverage(self):
        """
        Verifies pipeline aborts if asset coverage is below 95%.
        """
        # Only 5 out of 10 assets provided (50% coverage < 95%)
        snapshot = {
            "assets": self.assets[:5],
            "x": torch.randn(5, 16),
            "v_hybrid": torch.randn(5, 4)
        }
        
        with self.assertRaises(InsufficientAssetCoverageError):
            self.pipeline.execute_friday_pipeline(
                snapshot=snapshot,
                total_capital=10_000_000.0
            )

    def test_realtime_sla(self):
        """
        SLA Quality Test (Step 4.3.2): Verifies end-to-end execution completes within 3.0 seconds (3000 ms).
        """
        import time
        
        # Realistic universe of 50 assets
        large_universe = [f"ASSET_{i:02d}" for i in range(50)]
        large_pipeline = AetherMasterPipeline(
            checkpoint_dir=self.test_dir,
            expected_universe=large_universe,
            signal_orchestrator=self.signal_model
        )
        
        n_assets = len(large_universe)
        raw_cov = np.eye(n_assets) * 0.005 + 0.001
        cov_df = pd.DataFrame(raw_cov, index=large_universe, columns=large_universe)
        
        snapshot = {
            "assets": large_universe,
            "x": torch.randn(n_assets, 16),
            "v_hybrid": torch.randn(n_assets, 4),
            "cov_matrix": cov_df,
            "volatilities": pd.Series(0.02, index=large_universe),
            "adv_series": pd.Series(50_000_000.0, index=large_universe),
            "settlement_days": pd.Series(2, index=large_universe)
        }
        
        start_time = time.perf_counter()
        result = large_pipeline.execute_friday_pipeline(
            snapshot=snapshot,
            total_capital=500_000_000.0
        )
        elapsed_seconds = time.perf_counter() - start_time
        
        self.assertTrue(result.optimization_result.is_success)
        # SLA Requirement: Must be < 3.0 seconds
        self.assertLess(elapsed_seconds, 3.0, f"Pipeline execution took {elapsed_seconds:.4f}s, exceeding 3.0s SLA target.")
        self.assertLess(result.optimization_result.run_time_ms, 3000.0)

if __name__ == "__main__":
    unittest.main()
