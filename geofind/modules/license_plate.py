"""License Plate Detection module."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from geofind.core.candidate import ModuleHit
from geofind.core.config import PipelineConfig
from geofind.modules.base import BaseModule

logger = logging.getLogger(__name__)

_COUNTRY_CENTROIDS: dict[str, tuple[float, float]] = {
    "US": (37.09, -95.71), "GB": (55.38, -3.44), "DE": (51.17, 10.45),
    "FR": (46.23, 2.21), "JP": (36.20, 138.25), "IN": (20.59, 78.96),
    "BR": (-14.24, -51.93), "AU": (-25.27, 133.78), "IT": (41.88, 12.57),
    "ES": (40.46, -3.75), "RU": (61.52, 105.32), "KR": (35.91, 127.77),
    "CN": (35.86, 104.20),
}


class LicensePlateModule(BaseModule):
    """Detect license plates and match format patterns to countries."""

    name = "license_plate"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)
        self._yolo_model = None
        self._ocr_reader = None

    def is_available(self) -> bool:
        try:
            from ultralytics import YOLO  # noqa: F401
            import easyocr  # noqa: F401
            return True
        except ImportError:
            return False

    def prepare(self) -> None:
        from geofind.utils.models import get_cached_model

        def _load_yolo():
            from ultralytics import YOLO
            return YOLO("yolov8n.pt")

        def _load_ocr():
            import easyocr
            return easyocr.Reader(["en"], gpu=False)

        self._yolo_model = get_cached_model("yolo_plate", _load_yolo)
        self._ocr_reader = get_cached_model("easyocr", _load_ocr)
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
            results = self._yolo_model(img_bgr, conf=0.25, verbose=False)
        except Exception as e:
            self._log(f"YOLO detection failed: {e}", logging.WARNING)
            return []

        plate_boxes = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 2:
                    plate_boxes.append(box.xyxy[0].cpu().numpy())

        if not plate_boxes:
            self._log("No license plates detected")
            return []

        hits: list[ModuleHit] = []
        for box in plate_boxes:
            x_min, y_min, x_max, y_max = box[:4]
            pad = 5
            x_min = max(0, int(x_min) - pad)
            y_min = max(0, int(y_min) - pad)
            x_max = min(img_bgr.shape[1], int(x_max) + pad)
            y_max = min(img_bgr.shape[0], int(y_max) + pad)

            plate_crop = img_bgr[y_min:y_max, x_min:x_max]
            if plate_crop.size == 0:
                continue

            plate_text = self._read_plate_text(plate_crop)
            if not plate_text:
                continue

            self._log(f"Plate text: {plate_text}")

            matched = self._match_plate_pattern(plate_text)
            for cc, confidence in matched:
                lat, lon = _COUNTRY_CENTROIDS.get(cc, (0.0, 0.0))
                hits.append(self._make_hit(
                    lat, lon, confidence,
                    sigma_km=600.0,  # Country-level — plate pattern matches a country
                    country=cc,
                    plate_text=plate_text,
                ))

        return hits

    def _read_plate_text(self, plate_crop: Any) -> str | None:
        if self._ocr_reader is None:
            return None

        import numpy as np

        try:
            results = self._ocr_reader.readtext(plate_crop)
            if results:
                texts = [r[1] for r in results if len(r) > 1]
                combined = " ".join(texts).strip()
                combined = re.sub(r'[^A-Za-z0-9가-힣\u4e00-\u9fff\u3040-\u30ff-]', '', combined)
                if len(combined) >= 3:
                    return combined
        except Exception as e:
            self._log(f"Plate OCR failed: {e}", logging.WARNING)

        return None

    def _match_plate_pattern(self, text: str) -> list[tuple[str, float]]:
        from geofind.utils.constants import PLATE_PATTERNS

        matches: list[tuple[str, float]] = []
        text_upper = text.upper()

        for cc, pattern in PLATE_PATTERNS.items():
            try:
                if re.search(pattern.format_regex, text, re.IGNORECASE):
                    confidence = 0.7
                    if cc in _COUNTRY_CENTROIDS:
                        confidence = 0.75
                    matches.append((cc, confidence))
            except re.error:
                continue

        if not matches:
            if re.match(r'^[A-Z]{1,3}[\s-]?[A-Z]{1,2}\s?\d{1,4}$', text_upper):
                matches.append(("DE", 0.4))
            elif re.match(r'^\d{3,4}\s?[A-Z]{3}\s?\d{2}$', text_upper):
                matches.append(("FR", 0.5))
            elif re.match(r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$', text_upper):
                matches.append(("IN", 0.6))

        return matches

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
