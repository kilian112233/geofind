"""Terrain/Horizon Matching module - DEM-based terrain comparison.

Extracts horizon profile from image (sky/ground boundary silhouette)
and compares it against SRTM DEM-computed horizon profiles at
candidate locations.

Requires: DEM data (on-demand download), numpy, scipy
"""

from __future__ import annotations

import logging
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np

from geofind.core.candidate import ModuleHit
from geofind.core.config import PipelineConfig
from geofind.modules.base import BaseModule

logger = logging.getLogger(__name__)

# --- DEM utilities ---

DEG_PER_TILE = 5  # SRTM tiles are 5x5 degrees
SRTM_SAMPLES = 3601  # 3601x3601 for SRTM 1-arcsec


def _dem_tile_path(cache_dir: Path, lat: int, lon: int) -> Path:
    """Get path for a cached SRTM HGT tile file."""
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    return cache_dir / f"{lat_dir}{abs(lat):02d}{lon_dir}{abs(lon):03d}.hgt"


def _download_srtm_tile(cache_dir: Path, lat: int, lon: int) -> Path | None:
    """Download a single SRTM HGT tile on demand from AWS Open Data."""
    tile_lat = (lat // DEG_PER_TILE) * DEG_PER_TILE
    tile_lon = (lon // DEG_PER_TILE) * DEG_PER_TILE
    path = _dem_tile_path(cache_dir, tile_lat, tile_lon)

    if path.exists():
        return path

    try:
        import requests

        lat_dir = "N" if tile_lat >= 0 else "S"
        lon_dir = "E" if tile_lon >= 0 else "W"
        fname = f"{lat_dir}{abs(tile_lat):02d}{lon_dir}{abs(tile_lon):03d}.hgt"
        url = f"https://s3.amazonaws.com/elevation-tiles-prod/Tilesets/1-arc-sec/{fname}"

        path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200 and len(resp.content) > 0:
            path.write_bytes(resp.content)
            return path
        else:
            logger.warning("SRTM download failed: %s -> %d", url, resp.status_code)
            return None
    except Exception as e:
        logger.warning("SRTM download error: %s", e)
        return None


def _read_elevation(hgt_path: Path, lat: float, lon: float) -> float | None:
    """Read elevation from an HGT file at given lat/lon."""
    if not hgt_path.exists():
        return None

    # Determine tile base lat/lon from filename
    name = hgt_path.stem
    ns = name[0]
    ew = name[3]
    tile_lat = int(name[1:3]) * (1 if ns == "N" else -1)
    tile_lon = int(name[4:7]) * (1 if ew == "E" else -1)

    # Compute pixel indices
    lat_offset = lat - tile_lat
    lon_offset = lon - tile_lon

    row = int(lat_offset * (SRTM_SAMPLES - 1))
    col = int(lon_offset * (SRTM_SAMPLES - 1))

    if not (0 <= row < SRTM_SAMPLES and 0 <= col < SRTM_SAMPLES):
        return None

    idx = row * SRTM_SAMPLES + col
    offset = idx * 2  # 2 bytes per sample (big-endian int16)

    with open(hgt_path, "rb") as f:
        f.seek(offset)
        data = f.read(2)
        if len(data) < 2:
            return None
        elev = struct.unpack(">h", data)[0]
        if elev == -32768:
            return 0.0  # Void value, treat as sea level
        return float(elev)


def _compute_dem_horizon(
    cache_dir: Path,
    lat: float,
    lon: float,
    elevation: float,
    num_azimuths: int = 72,
    max_dist_km: float = 30.0,
) -> list[float] | None:
    """Compute horizon profile from DEM for a given viewpoint.

    For each azimuth, finds the maximum elevation angle of terrain
    along that direction out to max_dist_km.

    Returns list of elevation angles in degrees, or None if DEM data
    is unavailable.
    """
    # Get elevation at viewpoint
    tile_lat = (lat // DEG_PER_TILE) * DEG_PER_TILE
    tile_lon = (lon // DEG_PER_TILE) * DEG_PER_TILE
    hgt_path = _dem_tile_path(cache_dir, tile_lat, tile_lon)

    if not hgt_path.exists():
        return None

    view_elev = _read_elevation(hgt_path, lat, lon)
    if view_elev is None:
        view_elev = elevation

    horizon_angles = []

    for i in range(num_azimuths):
        azimuth = i * 360.0 / num_azimuths
        az_rad = math.radians(azimuth)
        cos_az = math.cos(az_rad)
        sin_az = math.sin(az_rad)

        max_angle = 0.0

        # Sample terrain along this azimuth
        step_m = 500.0  # 500m steps
        num_steps = int(max_dist_km * 1000 / step_m)

        for step in range(1, num_steps + 1):
            dist_m = step * step_m
            d_lat = (dist_m / 111320.0) * cos_az
            d_lon = (dist_m / (111320.0 * math.cos(math.radians(lat)))) * sin_az

            sample_lat = lat + d_lat
            sample_lon = lon + d_lon

            # Get elevation from DEM
            s_tile_lat = (sample_lat // DEG_PER_TILE) * DEG_PER_TILE
            s_tile_lon = (sample_lon // DEG_PER_TILE) * DEG_PER_TILE
            s_hgt = _dem_tile_path(cache_dir, s_tile_lat, s_tile_lon)

            if not s_hgt.exists():
                continue

            s_elev = _read_elevation(s_hgt, sample_lat, sample_lon)
            if s_elev is None:
                continue

            # Compute elevation angle
            height_diff = s_elev - view_elev
            if dist_m > 0:
                angle = math.degrees(math.atan2(height_diff, dist_m))
                if angle > max_angle:
                    max_angle = angle

        horizon_angles.append(max_angle)

    return horizon_angles


def _extract_image_horizon(
    image_array: np.ndarray, num_azimuths: int = 72
) -> list[float] | None:
    """Extract approximate horizon profile from image.

    Uses edge detection to find the sky/ground boundary,
    then maps it to azimuth-based elevation angles.
    """
    try:
        import cv2

        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array

        h, w = gray.shape

        # Detect edges
        edges = cv2.Canny(gray, 30, 100)

        # Focus on middle horizontal band (likely horizon area)
        horizon_band = edges[h // 3 : 2 * h // 3, :]

        if horizon_band.size == 0:
            return None

        # Find the strongest edge in each column (horizon line)
        horizon_line = []
        for col in range(w):
            col_data = horizon_band[:, col % horizon_band.shape[1]]
            max_row = np.argmax(col_data)
            if col_data[max_row] > 0:
                # Map row to elevation angle
                # Top of image = higher angle, bottom = lower/negative
                # Center = 0 degrees
                rel_row = (max_row - horizon_band.shape[0] / 2) / (
                    horizon_band.shape[0] / 2
                )
                horizon_line.append(-rel_row * 30)  # Approx +/- 30 degrees
            else:
                horizon_line.append(0.0)

        if len(horizon_line) < num_azimuths:
            return None

        # Resample to num_azimuths bins
        indices = np.linspace(0, len(horizon_line) - 1, num_azimuths).astype(int)
        profile = [horizon_line[i] for i in indices]

        return profile

    except Exception as e:
        logger.warning("Horizon extraction failed: %s", e)
        return None


def _horizon_similarity(
    profile1: list[float], profile2: list[float]
) -> float:
    """Compare two horizon profiles using normalized correlation.

    Returns similarity score in [0, 1].
    """
    n = min(len(profile1), len(profile2))
    if n < 5:
        return 0.0

    a = np.array(profile1[:n], dtype=np.float64)
    b = np.array(profile2[:n], dtype=np.float64)

    # Normalize
    a_std = np.std(a)
    b_std = np.std(b)

    if a_std < 0.01 and b_std < 0.01:
        return 0.5  # Both flat - no signal

    if a_std < 0.01 or b_std < 0.01:
        return 0.1  # One flat, one not

    a_norm = (a - np.mean(a)) / a_std
    b_norm = (b - np.mean(b)) / b_std

    corr = np.mean(a_norm * b_norm)

    # Map from [-1, 1] to [0, 1]
    return max(0.0, (corr + 1.0) / 2.0)


class TerrainMatchModule(BaseModule):
    """Terrain/Horizon matching module.

    Compares image horizon profile against DEM-computed horizon profiles
    at candidate locations. Works best in mountainous/hilly terrain.

    Requires: on-demand SRTM DEM tile downloads, numpy
    """

    name = "terrain_match"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)

    def is_available(self) -> bool:
        if not self.config.terrain_matching:
            return False
        try:
            import requests  # noqa: F401
            return True
        except ImportError:
            return False

    def detect(
        self,
        media_path: Path,
        *,
        frames: list[Any] | None = None,
        audio_path: Path | None = None,
    ) -> list[ModuleHit]:
        if not self._ready:
            return []

        # Extract horizon profile from image
        img_horizon = self._extract_image_horizon(media_path, frames)
        if img_horizon is None:
            self._log("Could not extract horizon from image", logging.WARNING)
            return []

        self._log(f"Extracted horizon profile ({len(img_horizon)} azimuths)")

        # The module doesn't search globally - it validates candidates
        # from other modules. We emit a signal that the reranker can use
        # to validate top candidates.
        # For now, we emit the profile as metadata for downstream use.

        # Emit a single "horizon profile available" hit at (0, 0) with
        # very low confidence - it's a signal, not a location claim.
        hits = [self._make_hit(
            0.0, 0.0, 0.1,
            sigma_km=1000.0,
            horizon_profile=img_horizon,
            source="terrain_match",
            method="image_horizon",
        )]

        return hits

    def validate_candidate(
        self,
        lat: float,
        lon: float,
        image_horizon: list[float],
    ) -> float:
        """Validate a specific candidate against image horizon.

        This is called by the pipeline/reranker for top candidates.
        Returns similarity score [0, 1].
        """
        # Download DEM tile if needed
        tile_lat = (lat // DEG_PER_TILE) * DEG_PER_TILE
        tile_lon = (lon // DEG_PER_TILE) * DEG_PER_TILE

        hgt = _download_srtm_tile(
            self.config.dem_cache_dir, tile_lat, tile_lon
        )
        if hgt is None:
            return 0.5  # Can't validate - return neutral

        view_elev = _read_elevation(hgt, lat, lon)
        if view_elev is None:
            view_elev = 0.0

        # Compute DEM horizon
        dem_horizon = _compute_dem_horizon(
            self.config.dem_cache_dir,
            lat, lon, view_elev,
            num_azimuths=len(image_horizon),
        )

        if dem_horizon is None:
            return 0.5  # Can't compute

        return _horizon_similarity(image_horizon, dem_horizon)

    def _extract_image_horizon(
        self, media_path: Path, frames: list[Any] | None
    ) -> list[float] | None:
        """Extract horizon profile from image."""
        try:
            from PIL import Image

            if frames:
                f = frames[0]
                if isinstance(f, Image.Image):
                    arr = np.array(f.convert("RGB"))
                else:
                    arr = np.array(f)
            else:
                from geofind.utils.media import load_image
                pil = load_image(media_path)
                if pil is None:
                    return None
                arr = np.array(pil.convert("RGB"))

            return _extract_image_horizon(arr)
        except Exception as e:
            self._log(f"Image horizon extraction failed: {e}", logging.WARNING)
            return None
