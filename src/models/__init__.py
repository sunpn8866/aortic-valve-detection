"""Model architecture modules."""

from .two_stage_detector import (
    TwoStageDetector,
    TwoStageTrainer,
    AttentionPooling,
    AuxiliarySegmentationLoss,
)

__all__ = [
    "TwoStageDetector",
    "TwoStageTrainer",
    "AttentionPooling",
    "AuxiliarySegmentationLoss",
]
