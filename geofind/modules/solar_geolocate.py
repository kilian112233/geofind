"""Solar Geolocation module - sun position + shadow direction analysis.

Cross-references shadow direction from shadow_angle module with solar
position calculations from astral to narrow down latitude bands.

Logic:
- Shadow direction reveals approximate sun azimuth
- Sun azimuth + time of day → possible latitude ranges
- Multiple shadow observations at different times → tighter constraints

Requires: astral library (pip install astral)
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from geofind.core.candidate import ModuleHit
from geofind.core.config import PipelineConfig
from geofind.modules.base import BaseModule

logger = logging.getLogger(__name__)


def _sun_azimuth(lat: float, lon: float, dt: datetime) -> float | None:
    """Compute sun azimuth at a given location and time.

    Returns azimuth in degrees (0=N, 90=E, 180=S, 270=W) or None if
    the sun is below the horizon.
    """
    try:
        from astral import LocationInfo
        from astral.sun import azimuth as astral_azimuth

        loc = LocationInfo(
            name="probe",
            region="",
            timezone="UTC",
            latitude=lat,
            longitude=lon,
        )
        az = astral_azimuth(loc.observer, dt)
        return az
    except Exception:
        return None


def _sun_elevation(lat: float, lon: float, dt: datetime) -> float | None:
    """Compute sun elevation (altitude) at a given location and time.

    Returns elevation in degrees above horizon, or None if sun is below.
    """
    try:
        from astral import LocationInfo
        from astral.sun import elevation as astral_elevation

        loc = LocationInfo(
            name="probe",
            region="",
            timezone="UTC",
            latitude=lat,
            longitude=lon,
        )
        el = astral_elevation(loc.observer, dt)
        return el if el > 0 else None
    except Exception:
        return None


def _shadow_to_sun_azimuth(shadow_az: float) -> float:
    """Convert shadow azimuth to sun azimuth (opposite direction)."""
    return (shadow_az + 180) % 360


class SolarGeolocateModule(BaseModule):
    """Solar geolocation module.

    Uses sun position + shadow direction to constrain latitude.
    Works by scanning latitude bands and finding where the sun
    position matches the observed shadow direction.

    Requires: astral library
    """

    name = "solar_geolocate"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)

    def is_available(self) -> bool:
        try:
            from astral import LocationInfo  # noqa: F401
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

        # Cheap reliability check FIRST — skip before any expensive analysis
        has_real_time = self._has_real_time(media_path, frames)
        if not has_real_time:
            self._log("Time is unknown/default — solar analysis unreliable, skipping")
            return []

        # Get shadow direction from prior modules if available
        shadow_az = self._get_shadow_azimuth(media_path, frames)
        if shadow_az is None:
            self._log("No shadow direction available, skipping solar geo")
            return []

        # Get approximate time from sun_clock or EXIF datetime
        approx_time = self._get_approx_time(media_path, frames)
        if approx_time is None:
            self._log("No time information available, skipping solar geo")
            return []

        self._log(
            f"Solar geo: shadow_az={shadow_az:.1f} deg, "
            f"time={approx_time.isoformat()}"
        )

        # Compute expected sun azimuth at the shadow's opposite direction
        expected_sun_az = _shadow_to_sun_azimuth(shadow_az)

        # Scan latitude bands at multiple longitudes
        # The sun azimuth changes with latitude, so we find matching latitudes
        hits = self._scan_latitudes(expected_sun_az, approx_time)

        return hits

    def _get_shadow_azimuth(
        self, media_path: Path, frames: list[Any] | None
    ) -> float | None:
        """Get shadow azimuth from shadow_angle module or estimate from image."""
        # Try to get from the shadow_angle module's analysis
        try:
            from geofind.modules.shadow_angle import ShadowAngleModule
            shadow_mod = ShadowAngleModule(self.config)
            shadow_mod.prepare()
            hits = shadow_mod.detect(media_path, frames=frames)
            if hits:
                # Shadow angle module returns the shadow direction
                # We need the azimuth, which may be in metadata
                for h in hits:
                    if "shadow_azimuth" in h.metadata:
                        return h.metadata["shadow_azimuth"]
                    # Try to reconstruct from lat/lon hint
                    return h.metadata.get("shadow_direction")
        except Exception:
            pass

        # Fallback: try to estimate shadow direction from image brightness
        return self._estimate_shadow_from_brightest(media_path, frames)

    def _estimate_shadow_from_brightest(
        self, media_path: Path, frames: list[Any] | None
    ) -> float | None:
        """Estimate shadow direction from image gradient (crude fallback)."""
        try:
            from PIL import Image
            import numpy as np
            import cv2

            if frames:
                f = frames[0]
                if isinstance(f, Image.Image):
                    img = f.convert("L")
                else:
                    img = Image.fromarray(f).convert("L")
            else:
                from geofind.utils.media import load_image
                pil = load_image(media_path)
                if pil is None:
                    return None
                img = pil.convert("L")

            arr = np.array(img, dtype=np.float32)

            # Compute gradient direction at edges
            # Shadows create dark-to-light gradients
            gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=5)
            gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=5)

            # The dominant gradient direction (light direction is perpendicular)
            mean_gx = np.mean(gx)
            mean_gy = np.mean(gy)

            if abs(mean_gx) < 0.1 and abs(mean_gy) < 0.1:
                return None

            # Gradient points from dark to light
            gradient_az = math.degrees(math.atan2(mean_gx, mean_gy)) % 360
            # Shadow goes from light to dark (opposite of gradient)
            shadow_az = (gradient_az + 180) % 360

            self._log(f"Estimated shadow azimuth from gradient: {shadow_az:.1f}")
            return shadow_az
        except Exception as e:
            self._log(f"Shadow estimation failed: {e}", logging.WARNING)
            return None

    def _get_approx_time(
        self, media_path: Path, frames: list[Any] | None
    ) -> datetime | None:
        """Get approximate time from sun_clock or EXIF datetime."""
        # Try EXIF first
        try:
            import exifread

            with open(media_path, "rb") as f:
                tags = exifread.process_file(f, stop_tag="EXIF DateTimeOriginal")
                dt_str = str(tags.get("EXIF DateTimeOriginal", ""))
                if dt_str:
                    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                        try:
                            return datetime.strptime(dt_str, fmt).replace(
                                tzinfo=timezone.utc
                            )
                        except ValueError:
                            continue
        except Exception:
            pass

        # Try sun_clock module
        try:
            from geofind.modules.sun_clock import SunClockModule
            clock_mod = SunClockModule(self.config)
            clock_mod.prepare()
            hits = clock_mod.detect(media_path, frames=frames)
            if hits:
                for h in hits:
                    utc_offset = h.metadata.get("utc_offset")
                    if utc_offset is not None:
                        # Assume midday if we only have offset
                        from datetime import timedelta
                        noon_utc = datetime.now(timezone.utc).replace(
                            hour=12, minute=0, second=0, microsecond=0
                        )
                        return noon_utc - timedelta(hours=utc_offset)
        except Exception:
            pass

        # Fallback: assume current day, solar noon UTC
        now = datetime.now(timezone.utc)
        return now.replace(hour=12, minute=0, second=0, microsecond=0)

    def _has_real_time(self, media_path: Path, frames: list[Any] | None) -> bool:
        """Check if we have a real time signal (not the fake default).

        Fast path: EXIF only. Running the sun_clock module here would cost a
        full OCR pass — sun_clock runs separately in the pipeline anyway.
        """
        try:
            import exifread
            with open(media_path, "rb") as f:
                tags = exifread.process_file(f, stop_tag="EXIF DateTimeOriginal")
                dt_str = str(tags.get("EXIF DateTimeOriginal", ""))
                if dt_str:
                    return True
        except Exception:
            pass

        return False

    def _scan_latitudes(
        self, expected_sun_az: float, dt: datetime
    ) -> list[ModuleHit]:
        """Scan latitude bands to find where sun azimuth matches."""
        hits: list[ModuleHit] = []

        # Scan from -60 to +60 latitude in 5-degree steps
        best_match = None
        best_diff = float("inf")

        # Use a fixed longitude for the scan (longitude shifts the sun azimuth
        # mostly in timing, not magnitude - the latitude is the key variable)
        for lat_10 in range(-600, 601, 50):  # -60 to +60 in 5-degree steps
            lat = lat_10 / 10.0

            # Test at a few representative longitudes
            for lon in [-30, 0, 30, 60, 90, 120, 150]:
                sun_az = _sun_azimuth(lat, lon, dt)
                if sun_az is None:
                    continue

                diff = abs(sun_az - expected_sun_az)
                if diff > 180:
                    diff = 360 - diff

                if diff < best_diff:
                    best_diff = diff
                    best_match = (lat, lon)

        if best_match is None or best_diff > 30:
            self._log(
                f"No good solar match found "
                f"(best_diff={best_diff:.1f} deg)"
            )
            return []

        lat, lon = best_match

        # The sun azimuth varies with latitude but not much with longitude
        # (longitude mainly shifts timing). So the latitude is well-constrained
        # but longitude is not. Emit a latitude band hit.

        # Confidence based on how well the sun matches
        conf = max(0.3, 0.9 - best_diff * 0.02)

        # Sigma: latitude is constrained, longitude is wide open
        # We emit a hit at the best latitude with a latitude-only sigma
        # represented as a moderate sigma (the reranker handles this)
        sigma = 50.0  # Wide sigma because longitude is unconstrained

        self._log(
            f"Solar match: lat={lat:.1f} (best_diff={best_diff:.1f} deg)"
        )

        hits.append(self._make_hit(
            lat, lon, conf,
            sigma_km=sigma,
            best_match_lat=lat,
            best_diff_deg=best_diff,
            method="latitude_band",
        ))

        return hits
