"""Module configuration and weights."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModuleConfig:
    """Configuration for a single detection module."""
    name: str
    weight: float = 1.0
    enabled: bool = True
    optional: bool = False  # If True, failure doesn't block pipeline
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Global pipeline configuration."""
    # Paths
    models_dir: Path = Path("W:/geofind/models")
    output_dir: Path = Path("W:/geofind/output")

    # Grid — coarse pass
    grid_resolution_deg: float = 1.0
    gaussian_sigma_km: float = 100.0

    # Hierarchical refinement — fine pass around top coarse candidates
    hierarchical_enabled: bool = True
    fine_resolution_deg: float = 0.005        # ~555m per cell (balance precision vs speed)
    fine_sigma_km: float = 1.0                # 1km Gaussian spread (forgiving enough for CLIP-level precision)
    fine_region_radius_deg: float = 1.5       # ±1.5° around each coarse candidate
    fine_top_n: int = 3                       # refine top-3 coarse candidates
    fine_candidates_per_region: int = 5000    # max cells per fine region

    # Reranking
    top_candidates: int = 20_000
    consensus_top_n: int = 5
    consensus_top_half_buff: float = 1.5
    consensus_top_n_buff: float = 3.0
    min_module_weight: float = 0.01

    # Video
    video_max_frames: int = 60
    video_frame_interval_s: float = 2.0

    # Output
    output_json: bool = True
    output_html: bool = True
    top_display: int = 20

    # Module configs
    modules: dict[str, ModuleConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.modules:
            self.modules = _default_module_configs()


def _default_module_configs() -> dict[str, ModuleConfig]:
    """Default module configurations.

    Precise modules (GeoCLIP, landmark, clip_visual) get higher weights.
    Area-level modules (OCR script, driving_side, vegetation, etc.) get
    lower weights — their per-hit sigma handles the precision difference.
    """
    return {
        "exif": ModuleConfig(name="exif", weight=5.0, optional=False),
        "geoclip": ModuleConfig(name="geoclip", weight=6.0, optional=True),
        "clip_visual": ModuleConfig(name="clip_visual", weight=4.0, optional=True),
        "landmark": ModuleConfig(name="landmark", weight=5.0, optional=True),
        "ocr_text": ModuleConfig(name="ocr_text", weight=0.8, optional=True),
        "audio_power": ModuleConfig(name="audio_power", weight=0.5, optional=True),
        "birdnet": ModuleConfig(name="birdnet", weight=2.0, optional=True),
        "sun_clock": ModuleConfig(name="sun_clock", weight=2.0, optional=True),
        "shadow_angle": ModuleConfig(name="shadow_angle", weight=1.0, optional=True),
        "vision_llm": ModuleConfig(name="vision_llm", weight=3.0, optional=True),
        "driving_side": ModuleConfig(name="driving_side", weight=0.5, optional=True),
        "vegetation": ModuleConfig(name="vegetation", weight=0.5, optional=True),
        "license_plate": ModuleConfig(name="license_plate", weight=0.5, optional=True),
        "currency": ModuleConfig(name="currency", weight=0.3, optional=True),
        "audio_scene": ModuleConfig(name="audio_scene", weight=0.5, optional=True),
    }
