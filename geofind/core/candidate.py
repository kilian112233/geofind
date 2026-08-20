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
    sigma_km: float | None = None  # Per-hit Gaussian spread; None = use module/grid default
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
    is_exact: bool = False  # True when coords are from a precision source (e.g. EXIF GPS)
    is_fine_refined: bool = False  # True when coords come from hierarchical fine-grid pass
    distance_to_truth_km: float | None = None  # Set during self-test for diagnostics

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
    all_module_hits: dict[str, list[ModuleHit]] = field(default_factory=dict)
    hierarchical_pass: bool = False  # True if hierarchical refinement ran

    @property
    def top_candidate(self) -> CandidateLocation | None:
        return self.candidates[0] if self.candidates else None

    @property
    def top_n(self) -> list[CandidateLocation]:
        return self.candidates[:20]
