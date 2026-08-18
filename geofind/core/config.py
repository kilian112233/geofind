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

    # Grid
    grid_resolution_deg: float = 1.0
    gaussian_sigma_km: float = 100.0

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
    """Default module configurations."""
    return {
        "exif": ModuleConfig(name="exif", weight=5.0, optional=False),
        "clip_visual": ModuleConfig(name="clip_visual", weight=2.0, optional=True),
        "landmark": ModuleConfig(name="landmark", weight=3.0, optional=True),
        "ocr_text": ModuleConfig(name="ocr_text", weight=2.5, optional=True),
        "audio_power": ModuleConfig(name="audio_power", weight=1.5, optional=True),
        "birdnet": ModuleConfig(name="birdnet", weight=2.0, optional=True),
        "sun_clock": ModuleConfig(name="sun_clock", weight=2.0, optional=True),
        "shadow_angle": ModuleConfig(name="shadow_angle", weight=1.0, optional=True),
        "vision_llm": ModuleConfig(name="vision_llm", weight=3.0, optional=True),
        "driving_side": ModuleConfig(name="driving_side", weight=1.5, optional=True),
        "vegetation": ModuleConfig(name="vegetation", weight=1.5, optional=True),
        "license_plate": ModuleConfig(name="license_plate", weight=2.5, optional=True),
        "currency": ModuleConfig(name="currency", weight=2.0, optional=True),
        "audio_scene": ModuleConfig(name="audio_scene", weight=1.5, optional=True),
    }
