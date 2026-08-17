"""Presentation-neutral application services."""

from .training import TrainingRequest, TrainingRun, train_xor

__all__ = ["TrainingRequest", "TrainingRun", "train_xor"]
