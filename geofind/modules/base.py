"""Abstract base class for detection modules."""

from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import Any

from geofind.core.candidate import ModuleHit
from geofind.core.config import PipelineConfig

logger = logging.getLogger(__name__)


class BaseModule(abc.ABC):
    """Base class all detection modules must implement."""

    name: str = "base"

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._ready = False

    @abc.abstractmethod
    def detect(
        self,
        media_path: Path,
        *,
        frames: list[Any] | None = None,
        audio_path: Path | None = None,
    ) -> list[ModuleHit]:
        """Run detection on media. Return list of geographic hits."""
        ...

    def is_available(self) -> bool:
        """Check if this module's dependencies are installed."""
        return True

    def prepare(self) -> None:
        """Optional one-time preparation (download models, etc.)."""
        self._ready = True

    def _make_hit(
        self,
        lat: float,
        lon: float,
        confidence: float,
        **metadata: Any,
    ) -> ModuleHit:
        return ModuleHit(
            module=self.name,
            lat=lat,
            lon=lon,
            confidence=confidence,
            metadata=metadata,
        )

    def _log(self, msg: str, level: int = logging.INFO) -> None:
        logger.log(level, f"[{self.name}] {msg}")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
