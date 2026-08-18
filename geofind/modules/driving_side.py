"""Driving Side Detection module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from geofind.core.candidate import ModuleHit
from geofind.core.config import PipelineConfig
from geofind.modules.base import BaseModule

logger = logging.getLogger(__name__)

_COUNTRY_CENTROIDS: dict[str, tuple[float, float]] = {
    "AU": (-25.27, 133.78), "NZ": (-40.90, 174.89), "GB": (55.38, -3.44),
    "IE": (53.14, -7.69), "IN": (20.59, 78.96), "PK": (30.38, 69.35),
    "TH": (15.87, 100.99), "ID": (-0.79, 113.92), "JP": (36.20, 138.25),
    "MY": (4.21, 101.98), "SG": (1.35, 103.82), "HK": (22.40, 114.11),
    "ZA": (-30.56, 22.94), "KE": (-0.02, 37.91), "TZ": (-6.37, 34.89),
    "NG": (9.08, 8.68), "US": (37.09, -95.71), "CA": (56.13, -106.35),
    "MX": (23.63, -102.55), "BR": (-14.24, -51.93), "DE": (51.17, 10.45),
    "FR": (46.23, 2.21), "IT": (41.88, 12.57), "ES": (40.46, -3.75),
    "RU": (61.52, 105.32), "CN": (35.86, 104.20), "KR": (35.91, 127.77),
    "SA": (23.89, 45.08),
}


class DrivingSideModule(BaseModule):
    """Detect which side of the road vehicles drive on from image."""

    name = "driving_side"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)
        self._yolo_model = None

    def is_available(self) -> bool:
        try:
            from ultralytics import YOLO  # noqa: F401
            import cv2  # noqa: F401
            return True
        except ImportError:
            return False

    def prepare(self) -> None:
        from geofind.utils.models import get_cached_model

        def _load():
            from ultralytics import YOLO
            return YOLO("yolov8n.pt")

        self._yolo_model = get_cached_model("yolo_driving", _load)
        super().prepare()

    def detect(
        self,
        media_path: Path,
        *,
        frames: list[Any] | None = None,
        audio_path: Path | None = None,
    ) -> list[ModuleHit]:
        if self._yolo_model is None:
            return []

        image = self._get_image(media_path, frames)
        if image is None:
            return []

        import cv2
        import numpy as np

        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array

        try:
            results = self._yolo_model(img_bgr, conf=0.3, verbose=False)
        except Exception as e:
            self._log(f"YOLO detection failed: {e}", logging.WARNING)
            return []

        vehicles = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in {2, 3, 5, 7}:
                    vehicles.append(box.xyxy[0].cpu().numpy())

        if len(vehicles) < 2:
            self._log(f"Only {len(vehicles)} vehicles detected, need at least 2")
            return []

        h, w = img_bgr.shape[:2]
        driving_side = self._analyze_vehicles(vehicles, w, h)

        if driving_side is None:
            return []

        self._log(f"Detected driving side: {driving_side}")

        from geofind.utils.constants import DRIVING_SIDE_COUNTRIES

        matching = [
            cc for cc, side in DRIVING_SIDE_COUNTRIES.items()
            if side == driving_side
        ]

        if not matching:
            return []

        hits: list[ModuleHit] = []
        n = len(matching)
        confidence = min(0.3 + len(vehicles) * 0.05, 0.7)
        for cc in matching:
            lat, lon = _COUNTRY_CENTROIDS.get(cc, (0.0, 0.0))
            hits.append(self._make_hit(
                lat, lon, confidence / n,
                country=cc,
                driving_side=driving_side,
                vehicle_count=len(vehicles),
            ))

        return hits

    def _analyze_vehicles(
        self, vehicles: list[Any], img_width: int, img_height: int
    ) -> str | None:
        import numpy as np

        centers_x = []
        for v in vehicles:
            x_min, y_min, x_max, y_max = v[:4]
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            centers_x.append((cx, cy))

        if not centers_x:
            return None

        lower_third = [(cx, cy) for cx, cy in centers_x if cy > img_height * 0.5]
        if len(lower_third) < 2:
            lower_third = centers_x

        left_count = sum(1 for cx, _ in lower_third if cx < img_width * 0.4)
        right_count = sum(1 for cx, _ in lower_third if cx > img_width * 0.6)

        total = left_count + right_count
        if total == 0:
            return None

        if left_count > right_count * 1.5:
            return "left"
        elif right_count > left_count * 1.5:
            return "right"

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
