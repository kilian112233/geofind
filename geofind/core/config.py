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
    fine_top_n: int = 8                       # refine top-8 coarse candidates
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

    # Feature flags — toggle capability groups on/off
    online_geocoding: bool = True       # Nominatim + Overpass text geocoding
    terrain_matching: bool = False      # DEM-based terrain/horizon matching (requires DEM data)
    shadow_analysis: bool = True        # Solar geometry + shadow cross-reference

    # Online geocoding settings
    nominatim_timeout_s: float = 5.0    # Per-query timeout for Nominatim
    overpass_timeout_s: float = 10.0    # Per-query timeout for Overpass API
    geocoding_rate_limit_s: float = 1.1 # Min seconds between API calls

    # Terrain matching settings
    dem_cache_dir: Path = Path("W:/geofind/models/dem")
    dem_resolution_m: int = 90          # SRTM resolution (30 or 90 meters)

    # Module configs
    modules: dict[str, ModuleConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dem_cache_dir.mkdir(parents=True, exist_ok=True)
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
        "geoclip": ModuleConfig(name="geoclip", weight=5.0, optional=True),
        "clip_visual": ModuleConfig(name="clip_visual", weight=1.5, optional=True),
        "clip_retrieval": ModuleConfig(name="clip_retrieval", weight=4.0, optional=True),
        "places365": ModuleConfig(name="places365", weight=1.5, optional=True),
        "landmark": ModuleConfig(name="landmark", weight=3.0, optional=True),
        "ocr_text": ModuleConfig(name="ocr_text", weight=2.5, optional=True),
        "text_geocoder": ModuleConfig(name="text_geocoder", weight=6.0, optional=True),
        "streetclip": ModuleConfig(name="streetclip", weight=3.0, optional=True),
        "solar_geolocate": ModuleConfig(name="solar_geolocate", weight=2.0, optional=True),
        "terrain_match": ModuleConfig(name="terrain_match", weight=3.0, optional=True),
        "audio_power": ModuleConfig(name="audio_power", weight=0.5, optional=True),
        "birdnet": ModuleConfig(name="birdnet", weight=1.5, optional=True),
        "sun_clock": ModuleConfig(name="sun_clock", weight=2.0, optional=True),
        "shadow_angle": ModuleConfig(name="shadow_angle", weight=1.0, optional=True),

        "driving_side": ModuleConfig(name="driving_side", weight=0.5, optional=True),
        "vegetation": ModuleConfig(name="vegetation", weight=0.5, optional=True),
        "license_plate": ModuleConfig(name="license_plate", weight=0.5, optional=True),
        "currency": ModuleConfig(name="currency", weight=0.3, optional=True),
        "audio_scene": ModuleConfig(name="audio_scene", weight=0.5, optional=True),
        "region_voter": ModuleConfig(name="region_voter", weight=1.0, optional=True),
    }
