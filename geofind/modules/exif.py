"""EXIF GPS Extraction module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from geofind.core.candidate import ModuleHit
from geofind.core.config import PipelineConfig
from geofind.modules.base import BaseModule

logger = logging.getLogger(__name__)


def _dms_to_decimal(dms: tuple, ref: str) -> float:
    """Convert EXIF DMS (degrees, minutes, seconds) to decimal degrees."""
    try:
        d = float(dms[0].numerator) / float(dms[0].denominator)
        m = float(dms[1].numerator) / float(dms[1].denominator)
        s = float(dms[2].numerator) / float(dms[2].denominator)
    except (AttributeError, IndexError, ZeroDivisionError):
        d = float(dms[0])
        m = float(dms[1])
        s = float(dms[2])

    decimal = d + m / 60.0 + s / 3600.0
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


class ExifModule(BaseModule):
    """Extract GPS coordinates from image EXIF data."""

    name = "exif"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)

    def is_available(self) -> bool:
        try:
            import exifread  # noqa: F401
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
        hits: list[ModuleHit] = []

        images = frames or []
        if not images and media_path.suffix.lower() in {".jpg", ".jpeg", ".tiff", ".tif"}:
            try:
                from geofind.utils.media import load_image
                img = load_image(media_path)
                images = [img]
            except Exception:
                pass

        if not images:
            return hits

        try:
            import exifread
        except ImportError:
            self._log("exifread not installed", logging.WARNING)
            return hits

        img_path = media_path if media_path.suffix.lower() in {".jpg", ".jpeg", ".tiff", ".tif"} else None
        if img_path is None:
            return hits

        try:
            with open(img_path, "rb") as f:
                tags = exifread.process_file(f, details=False)
        except Exception as e:
            self._log(f"Failed to read EXIF: {e}", logging.WARNING)
            return hits

        lat = self._get_gps_coord(tags, "GPS GPSLatitude", "GPS GPSLatitudeRef")
        lon = self._get_gps_coord(tags, "GPS GPSLongitude", "GPS GPSLongitudeRef")

        if lat is not None and lon is not None:
            confidence = 0.95
            if abs(lat) < 0.01 and abs(lon) < 0.01:
                confidence = 0.3
                self._log("GPS coordinates near (0,0), likely invalid", logging.WARNING)
            else:
                self._log(f"GPS found: {lat:.6f}, {lon:.6f}")
            hits.append(self._make_hit(
                lat, lon, confidence,
                source="exif_gps",
                country="",
            ))

        return hits

    @staticmethod
    def _get_gps_coord(tags: dict, tag_name: str, ref_name: str) -> float | None:
        """Extract a GPS coordinate from EXIF tags."""
        try:
            dms = tags[tag_name].values
            ref_tag = tags.get(ref_name)
            ref = str(ref_tag.values[0]) if ref_tag else ""
            return _dms_to_decimal(dms, ref)
        except (KeyError, AttributeError, IndexError, TypeError, ZeroDivisionError):
            return None
