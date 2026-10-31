"""Load and validate the engine configuration (config/engine.yaml)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "engine.yaml"

LEVELS = ("low", "moderate", "high", "very_high")
LEVEL_RANK = {name: i for i, name in enumerate(LEVELS)}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class EngineConfig:
    raw: dict[str, Any]

    # convenience accessors -------------------------------------------------
    @property
    def version(self) -> str:
        return str(self.raw["version"])

    def section(self, name: str) -> dict[str, Any]:
        return self.raw[name]

    @property
    def s_weights(self) -> dict[str, float]:
        return self.raw["susceptibility"]["weights"]

    @property
    def s_min(self) -> float:
        return float(self.raw["susceptibility"]["s_min"])

    @property
    def dv_weights(self) -> dict[str, float]:
        return self.raw["drainage"]["weights"]

    @property
    def t0(self) -> float:
        return float(self.raw["threshold"]["t0_mm_h"])

    @property
    def beta(self) -> float:
        th = self.raw["threshold"]
        return min(float(th["beta"]), float(th["beta_max"]))

    @property
    def level_cuts(self) -> dict[str, float]:
        return self.raw["levels"]


def _check_sum(name: str, weights: dict[str, float]) -> None:
    total = sum(float(v) for v in weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ConfigError(f"{name} weights must sum to 1.0 (got {total:.4f})")
    if any(float(v) < 0 for v in weights.values()):
        raise ConfigError(f"{name} weights must be non-negative")


def validate(raw: dict[str, Any]) -> EngineConfig:
    _check_sum("susceptibility", raw["susceptibility"]["weights"])
    _check_sum("drainage", raw["drainage"]["weights"])
    lv = raw["levels"]
    if not (0 < lv["moderate"] < lv["high"] < lv["very_high"] <= 100):
        raise ConfigError("levels must satisfy 0 < moderate < high < very_high <= 100")
    th = raw["threshold"]
    if th["t0_mm_h"] <= 0 or not (0 <= th["beta"] < 1):
        raise ConfigError("threshold: t0 must be > 0 and 0 <= beta < 1")
    if not (0 <= raw["susceptibility"]["s_min"] < 1):
        raise ConfigError("s_min must be in [0, 1)")
    return EngineConfig(raw=raw)


def load_config(path: str | Path | None = None) -> EngineConfig:
    path = Path(path or os.environ.get("FLOOD_ENGINE_CONFIG", DEFAULT_PATH))
    with open(path, encoding="utf-8") as fh:
        return validate(yaml.safe_load(fh))


@lru_cache(maxsize=1)
def default_config() -> EngineConfig:
    return load_config()
