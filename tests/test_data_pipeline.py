import unittest
from aether.trainer.data_pipeline import NightlyDataPipeline, InsufficientAssetCoverageError

class TestNightlyDataPipeline(unittest.TestCase):
    def setUp(self):
        # 100 sample assets
        self.universe = [f"ASSET_{i:03d}" for i in range(100)]
        self.pipeline = NightlyDataPipeline(expected_universe=self.universe, min_threshold_ratio=0.95)

    def test_empty_universe_error(self):
        """
        Verifies initialization fails if expected universe is empty.
        """
        with self.assertRaises(ValueError):
            NightlyDataPipeline(expected_universe=[])

    def test_valid_coverage_above_threshold(self):
        """
        Verifies 96% coverage passes successfully.
        """
        available_96 = self.universe[:96]
        ratio = self.pipeline.validate_coverage(available_96)
        self.assertAlmostEqual(ratio, 0.96)

    def test_exact_threshold_boundary(self):
        """
        Verifies exactly 95% coverage passes at boundary.
        """
        available_95 = self.universe[:95]
        ratio = self.pipeline.validate_coverage(available_95)
        self.assertAlmostEqual(ratio, 0.95)

    def test_insufficient_coverage_below_threshold(self):
        """
        Verifies 94% coverage raises InsufficientAssetCoverageError and aborts.
        """
        available_94 = self.universe[:94]
        with self.assertRaises(InsufficientAssetCoverageError) as ctx:
            self.pipeline.validate_coverage(available_94)
            
        self.assertIn("94.00%", str(ctx.exception))
        self.assertIn("95.00%", str(ctx.exception))

    def test_prepare_batch_success(self):
        """
        Verifies snapshot dict validation and payload enrichment.
        """
        snapshot = {
            "assets": self.universe[:98],
            "features": [1, 2, 3]
        }
        prepared = self.pipeline.prepare_batch(snapshot)
        self.assertIn("coverage_ratio", prepared)
        self.assertAlmostEqual(prepared["coverage_ratio"], 0.98)

if __name__ == "__main__":
    unittest.main()
