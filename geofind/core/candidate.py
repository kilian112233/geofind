"""Candidate location data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModuleHit:
    """A single hit from a detection module."""
    module: str
    lat: float
    lon: float
    confidence: float  # 0.0–1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateLocation:
    """A ranked candidate location with probability."""
    lat: float
    lon: float
    probability: float = 0.0
    log_posterior: float = 0.0
    hits: list[ModuleHit] = field(default_factory=list)
    buff_multiplier: float = 1.0

    @property
    def country_hint(self) -> str:
        return self.hits[0].metadata.get("country", "") if self.hits else ""

    def add_hit(self, hit: ModuleHit) -> None:
        self.hits.append(hit)


@dataclass
class GeoResult:
    """Final result bundle from the pipeline."""
    candidates: list[CandidateLocation]
    consensus_lat: float
    consensus_lon: float
    agreement_strength: float  # 0.0–1.0
    modules_run: list[str] = field(default_factory=list)
    modules_failed: list[str] = field(default_factory=list)
    processing_time_s: float = 0.0

    @property
    def top_candidate(self) -> CandidateLocation | None:
        return self.candidates[0] if self.candidates else None

    @property
    def top_n(self) -> list[CandidateLocation]:
        return self.candidates[:20]
