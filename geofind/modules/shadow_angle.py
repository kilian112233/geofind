"""Shadow Angle Analysis module."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from geofind.core.candidate import ModuleHit
from geofind.core.config import PipelineConfig
from geofind.modules.base import BaseModule

logger = logging.getLogger(__name__)


class ShadowAngleModule(BaseModule):
    """Estimate latitude from shadow direction and time-of-day."""

    name = "shadow_angle"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)

    def is_available(self) -> bool:
        try:
            import cv2  # noqa: F401
            import numpy  # noqa: F401
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
        image = self._get_image(media_path, frames)
        if image is None:
            return []

        import cv2
        import numpy as np

        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        shadow_angle = self._estimate_shadow_angle(gray)
        if shadow_angle is None:
            self._log("Could not estimate shadow angle")
            return []

        self._log(f"Estimated shadow azimuth: {shadow_angle:.1f} degrees")

        file_time = self._get_file_time(media_path)
        hour = None
        if file_time is not None:
            hour = file_time.hour + file_time.minute / 60.0

        hits = self._estimate_latitude(shadow_angle, hour)
        return hits

    def _estimate_shadow_angle(self, gray: Any) -> float | None:
        import cv2
        import numpy as np

        try:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            h, w = edges.shape
            center_x, center_y = w // 2, h // 2

            y_coords, x_coords = np.nonzero(edges)
            if len(x_coords) < 50:
                return None

            dx = x_coords - center_x
            dy = y_coords - center_y
            angles = np.arctan2(dy, dx)

            hist, bin_edges = np.histogram(angles, bins=72, range=(-np.pi, np.pi))
            dominant_bin = np.argmax(hist)
            dominant_angle = (bin_edges[dominant_bin] + bin_edges[dominant_bin + 1]) / 2

            brightness_gradient = self._compute_brightness_gradient(gray)

            if brightness_gradient is not None:
                sun_azimuth = (np.degrees(dominant_angle) + 180) % 360
            else:
                sun_azimuth = np.degrees(dominant_angle)

            return sun_azimuth

        except Exception as e:
            self._log(f"Shadow analysis failed: {e}", logging.WARNING)
            return None

    def _compute_brightness_gradient(self, gray: Any) -> float | None:
        import numpy as np

        try:
            h, w = gray.shape
            left_half = gray[:, :w // 2].mean()
            right_half = gray[:, w // 2:].mean()
            top_half = gray[:h // 2, :].mean()
            bottom_half = gray[h // 2:, :].mean()

            dx = right_half - left_half
            dy = bottom_half - top_half

            if abs(dx) < 5 and abs(dy) < 5:
                return None

            return float(np.degrees(np.arctan2(dy, dx)))
        except Exception:
            return None

    def _estimate_latitude(
        self, shadow_azimuth: float, hour: float | None
    ) -> list[ModuleHit]:
        import math

        hits: list[ModuleHit] = []

        if hour is not None:
            sun_azimuth_offset = (hour - 12.0) / 12.0 * 180.0
            adjusted_azimuth = (shadow_azimuth + 180 + sun_azimuth_offset) % 360

            if 150 <= adjusted_azimuth <= 210:
                lat_estimate = 0.0
                confidence = 0.35
            elif adjusted_azimuth < 180:
                lat_estimate = 30.0 + (adjusted_azimuth - 150) * 2.0
                confidence = 0.2
            else:
                lat_estimate = 30.0 + (210 - adjusted_azimuth) * 2.0
                confidence = 0.2

            lat_estimate = max(-60, min(60, lat_estimate))

            hits.append(self._make_hit(
                lat_estimate, 0.0, confidence,
                shadow_azimuth=shadow_azimuth,
                estimated_hour=hour,
                method="shadow_time_combined",
            ))
        else:
            for lat_est in [-30, -15, 0, 15, 30, 45]:
                confidence = 0.08
                hits.append(self._make_hit(
                    lat_est, 0.0, confidence,
                    shadow_azimuth=shadow_azimuth,
                    method="shadow_only",
                ))

        return hits

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
            return datetime.fromtimestamp(media_path.stat().st_mtime, tz=timezone.utc)
        except Exception:
            return None

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
