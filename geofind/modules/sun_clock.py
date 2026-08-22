"""Sun Position + Clock Reading module."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from geofind.core.candidate import ModuleHit
from geofind.core.config import PipelineConfig
from geofind.modules.base import BaseModule

logger = logging.getLogger(__name__)


class SunClockModule(BaseModule):
    """Estimate location from sun position and clock time in image."""

    name = "sun_clock"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)
        self._ocr_reader = None

    def is_available(self) -> bool:
        try:
            import astral  # noqa: F401
            return True
        except ImportError:
            return False

    def prepare(self) -> None:
        try:
            import easyocr
            from geofind.utils.models import get_cached_model

            def _load():
                return easyocr.Reader(["en"], gpu=False)

            self._ocr_reader = get_cached_model("easyocr", _load)
        except ImportError:
            self._log("easyocr not available for clock reading", logging.INFO)

        super().prepare()

    def detect(
        self,
        media_path: Path,
        *,
        frames: list[Any] | None = None,
        audio_path: Path | None = None,
    ) -> list[ModuleHit]:
        from PIL import Image

        image = self._get_image(media_path, frames)
        if image is None:
            return []

        time_str = self._read_clock(image)
        if time_str is None:
            self._log("No clock time detected")
            return []

        hours, minutes = time_str
        self._log(f"Clock reads: {hours:02d}:{minutes:02d}")

        file_mtime = self._get_file_time(media_path)
        if file_mtime is None:
            self._log("No timestamp available for sun calculation")
            return []

        hits = self._estimate_from_sun(hours, minutes, file_mtime)
        return hits

    def _read_clock(self, image: Any) -> tuple[int, int] | None:
        """Find a clock time in the image's OCR text.

        Uses the shared cached OCR text (no extra EasyOCR pass). Falls back
        to a raw read only if the shared pipeline is unavailable.
        """
        import re

        try:
            from geofind.utils.models import extract_ocr_text_cached
            full_text = extract_ocr_text_cached(image)
        except Exception:
            full_text = ""

        if not full_text and self._ocr_reader is not None:
            try:
                import numpy as np
                results = self._ocr_reader.readtext(np.array(image))
                full_text = " ".join(r[1] for r in results if len(r) > 1)
            except Exception as e:
                self._log(f"Clock OCR failed: {e}", logging.WARNING)
                return None

        if not full_text:
            return None

        time_pattern = re.compile(r"(\d{1,2})[:\s.](\d{2})")

        for match in time_pattern.finditer(full_text):
            h = int(match.group(1))
            m = int(match.group(2))
            if 0 <= h <= 23 and 0 <= m <= 59:
                return (h, m)

        return None

    def _get_file_time(self, media_path: Path) -> datetime | None:
        try:
            import exifread
            with open(media_path, "rb") as f:
                tags = exifread.process_file(f, details=False)
            date_str = str(tags.get("EXIF DateTimeOriginal", ""))
            if date_str:
                return datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S").replace(
                    tzinfo=timezone.utc
                )
        except Exception:
            pass

        try:
            mtime = media_path.stat().st_mtime
            return datetime.fromtimestamp(mtime, tz=timezone.utc)
        except Exception:
            return None

    def _estimate_from_sun(
        self, clock_h: int, clock_m: int, file_time: datetime
    ) -> list[ModuleHit]:
        from geofind.utils.constants import REGION_UTC_OFFSETS

        hits: list[ModuleHit] = []
        target_utc = file_time.replace(
            hour=clock_h, minute=clock_m, second=0, microsecond=0
        )

        for region, (offset_min, offset_max) in REGION_UTC_OFFSETS.items():
            offset = (offset_min + offset_max) / 2.0
            local_time = target_utc.replace(tzinfo=timezone.utc)
            estimated_utc = local_time.timestamp() - offset * 3600

            lat, lon = self._region_to_coords(region)
            if lat is None:
                continue

            try:
                from astral import LocationInfo
                from astral.sun import sun

                loc = LocationInfo(region, "", "", lat, lon)
                s = sun(loc.observer, date=datetime.fromtimestamp(estimated_utc, tz=timezone.utc))
                sunrise = s["sunrise"]
                sunset = s["sunset"]
                daylight_hours = (sunset - sunrise).total_seconds() / 3600

                from geofind.utils.geo import LatLon, haversine_km
                equator = LatLon(0.0, lon)
                actual = LatLon(lat, lon)
                dist = haversine_km(equator, actual)
                expected_daylight = 12.0
                deviation = abs(daylight_hours - expected_daylight)
                lat_penalty = abs(lat) / 90.0

                confidence = max(0.05, 0.4 - deviation * 0.05 - lat_penalty * 0.1)

                hits.append(self._make_hit(
                    lat, lon, confidence,
                    region=region,
                    utc_offset=offset,
                    daylight_hours=daylight_hours,
                ))
            except Exception:
                hits.append(self._make_hit(
                    lat, lon, 0.15,
                    region=region,
                    utc_offset=offset,
                ))

        return hits

    def _region_to_coords(self, region: str) -> tuple[float, float] | None:
        mapping = {
            "US_eastern": (40.0, -75.0), "US_central": (38.0, -95.0),
            "US_mountain": (39.0, -105.0), "US_pacific": (37.0, -120.0),
            "UK": (52.0, -1.0), "central_europe": (48.0, 10.0),
            "eastern_europe": (47.0, 25.0), "russia_moscow": (55.0, 38.0),
            "india": (22.0, 78.0), "china": (35.0, 105.0),
            "japan": (36.0, 138.0), "korea": (36.0, 128.0),
            "australia_east": (-30.0, 145.0), "australia_west": (-28.0, 118.0),
        }
        return mapping.get(region)

    def _get_image(self, media_path: Path, frames: list[Any] | None) -> Any | None:
        from PIL import Image

        if frames:
            f = frames[0]
            if isinstance(f, Image.Image):
                return f.convert("RGB")
            try:
                import numpy as np
                return Image.fromarray(f).convert("RGB")
            except Exception:
                pass

        try:
            from geofind.utils.media import load_image
            return load_image(media_path)
        except Exception:
            return None
