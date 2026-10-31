"""Version 1 algorithmic flood-risk engine (no machine learning)."""
from .config import LEVELS, EngineConfig, default_config, load_config
from .core import evaluate_segment, level_at_least
from .types import EvalContext, RainWindow, SegmentRisk, SegmentStatic

__all__ = [
    "LEVELS", "EngineConfig", "default_config", "load_config",
    "evaluate_segment", "level_at_least",
    "EvalContext", "RainWindow", "SegmentRisk", "SegmentStatic",
]
