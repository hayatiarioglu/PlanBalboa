from typing import List, Dict, Any, Sequence, Set

class InsufficientAssetCoverageError(Exception):
    """Raised when available asset snapshot coverage falls below the required threshold (95%)."""
    pass

class NightlyDataPipeline:
    """
    Nightly Data Pipeline & Min-Asset Threshold Guard (Armor 2).
    Verifies that incoming PIT Data Lake snapshots contain at least 95% valid asset coverage
    relative to the expected asset universe before allowing model training to proceed.
    """
    
    def __init__(self, expected_universe: Sequence[str], min_threshold_ratio: float = 0.95):
        """
        Args:
            expected_universe: The list/sequence of expected asset symbols (e.g. BIST100 + TEFAS funds).
            min_threshold_ratio: Minimum required active coverage ratio (default 0.95 = 95%).
        """
        if not expected_universe:
            raise ValueError("Expected universe cannot be empty.")
            
        self.expected_universe: Set[str] = set(expected_universe)
        self.total_expected_count: int = len(self.expected_universe)
        self.min_threshold_ratio: float = min_threshold_ratio
        
    def validate_coverage(self, available_assets: Sequence[str]) -> float:
        """
        Validates coverage ratio of available_assets against expected_universe.
        Raises InsufficientAssetCoverageError if ratio < min_threshold_ratio.
        """
        available_set = set(available_assets)
        valid_intersection = self.expected_universe.intersection(available_set)
        valid_count = len(valid_intersection)
        
        coverage_ratio = valid_count / float(self.total_expected_count)
        
        if coverage_ratio < self.min_threshold_ratio:
            missing_count = self.total_expected_count - valid_count
            raise InsufficientAssetCoverageError(
                f"Asset coverage ratio {coverage_ratio * 100:.2f}% is below required threshold "
                f"{self.min_threshold_ratio * 100:.2f}%. Valid assets: {valid_count}/{self.total_expected_count} "
                f"(Missing: {missing_count}). Training aborted."
            )
            
        return coverage_ratio
        
    def prepare_batch(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates and prepares a data snapshot batch for model training.
        """
        if "assets" not in snapshot:
            raise KeyError("Snapshot dictionary must contain an 'assets' key.")
            
        available_assets = snapshot["assets"]
        coverage_ratio = self.validate_coverage(available_assets)
        
        snapshot_prepared = dict(snapshot)
        snapshot_prepared["coverage_ratio"] = coverage_ratio
        
        return snapshot_prepared
