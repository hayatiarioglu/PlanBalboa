"""
AetherForecaster-X v35.2 Multi-Asset Master
Katman 2: Clamped Scheduled Soft-NDCG & Detach-Coupled MMoE
"""

from aether.signal.target_generator import (
    RobustWinsorizedScaler,
    WinsorizedScalerResult,
    DelistDecouplingTargetGenerator,
    TargetGeneratorResult
)
from aether.signal.soft_ndcg_loss import (
    EpsilonProtectedPositiveIDCG,
    DelistSigmoidPenalty,
    ClampedLogitDifference,
    ClampedScheduledSoftNDCGLoss
)
from aether.signal.mmoe_backbone import DetachCoupledPredictionHeads, MMoEBackboneModule, MMoEOutput
from aether.signal.temperature_decay import ScheduledTemperatureDecayEngine, TemperatureDecayResult
from aether.signal.signal_orchestrator import SignalPipelineOrchestrator, SignalPipelineResult

__all__ = [
    "RobustWinsorizedScaler",
    "WinsorizedScalerResult",
    "DelistDecouplingTargetGenerator",
    "TargetGeneratorResult",
    "EpsilonProtectedPositiveIDCG",
    "DelistSigmoidPenalty",
    "ClampedLogitDifference",
    "ClampedScheduledSoftNDCGLoss",
    "DetachCoupledPredictionHeads",
    "MMoEBackboneModule",
    "MMoEOutput",
    "ScheduledTemperatureDecayEngine",
    "TemperatureDecayResult",
    "SignalPipelineOrchestrator",
    "SignalPipelineResult"
]








