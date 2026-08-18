"""Local Vision LLM (LLaVA) module."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from geofind.core.candidate import ModuleHit
from geofind.core.config import PipelineConfig
from geofind.modules.base import BaseModule

logger = logging.getLogger(__name__)

_LOCATION_COORDS: dict[str, tuple[float, float]] = {
    "united states": (37.09, -95.71), "usa": (37.09, -95.71),
    "canada": (56.13, -106.35), "mexico": (23.63, -102.55),
    "brazil": (-14.24, -51.93), "argentina": (-38.42, -63.62),
    "united kingdom": (55.38, -3.44), "england": (52.36, -1.17),
    "scotland": (56.49, -4.20), "wales": (52.13, -3.78),
    "france": (46.23, 2.21), "germany": (51.17, 10.45),
    "italy": (41.88, 12.57), "spain": (40.46, -3.75),
    "portugal": (39.40, -8.22), "netherlands": (52.13, 5.29),
    "belgium": (50.50, 4.47), "switzerland": (46.82, 8.23),
    "austria": (47.52, 14.55), "sweden": (60.13, 18.64),
    "norway": (60.47, 8.47), "denmark": (56.26, 9.50),
    "finland": (61.92, 25.75), "poland": (51.92, 19.15),
    "czech republic": (49.82, 15.47), "czechia": (49.82, 15.47),
    "hungary": (47.16, 19.50), "romania": (45.94, 24.97),
    "greece": (39.07, 21.82), "turkey": (38.96, 35.24),
    "russia": (61.52, 105.32), "china": (35.86, 104.20),
    "japan": (36.20, 138.25), "south korea": (35.91, 127.77),
    "korea": (35.91, 127.77), "india": (20.59, 78.96),
    "thailand": (15.87, 100.99), "vietnam": (14.06, 108.28),
    "indonesia": (-0.79, 113.92), "philippines": (12.88, 121.77),
    "malaysia": (4.21, 101.98), "singapore": (1.35, 103.82),
    "australia": (-25.27, 133.78), "new zealand": (-40.90, 174.89),
    "south africa": (-30.56, 22.94), "egypt": (26.82, 30.80),
    "nigeria": (9.08, 8.68), "kenya": (-0.02, 37.91),
    "ethiopia": (9.15, 40.49), "morocco": (31.79, -7.09),
    "saudi arabia": (23.89, 45.08), "uae": (23.42, 53.85),
    "united arab emirates": (23.42, 53.85), "israel": (31.05, 34.85),
    "pakistan": (30.38, 69.35), "bangladesh": (23.68, 90.36),
    "nepal": (28.39, 84.12), "sri lanka": (7.87, 80.77),
    "colombia": (4.57, -74.30), "chile": (-35.68, -71.54),
    "peru": (-9.19, -75.02), "venezuela": (6.42, -66.59),
    "cuba": (21.52, -77.78), "jamaica": (18.11, -77.30),
    "ireland": (53.14, -7.69), "iceland": (64.96, -19.02),
    "ukraine": (48.38, 31.17), "croatia": (45.10, 15.20),
    "serbia": (44.02, 21.01), "bulgaria": (42.73, 25.49),
    "taiwan": (23.70, 120.96), "hong kong": (22.40, 114.11),
    "new york": (40.71, -74.01), "los angeles": (34.05, -118.24),
    "london": (51.51, -0.13), "paris": (48.86, 2.35),
    "tokyo": (35.68, 139.69), "beijing": (39.90, 116.40),
    "mumbai": (19.08, 72.88), "delhi": (28.61, 77.21),
    "sydney": (-33.87, 151.21), "melbourne": (-37.81, 144.96),
    "dubai": (25.20, 55.27), "bangkok": (13.76, 100.50),
    "berlin": (52.52, 13.41), "rome": (41.90, 12.50),
    "madrid": (40.42, -3.70), "amsterdam": (52.37, 4.90),
    "moscow": (55.76, 37.62), "seoul": (37.57, 126.98),
    "toronto": (43.65, -79.38), "vancouver": (49.28, -123.12),
    "cairo": (30.04, 31.24), "cape town": (-33.93, 18.42),
    "nairobi": (-1.29, 36.82), "buenos aires": (-34.60, -58.38),
    "santiago": (-33.45, -70.67), "lima": (-12.05, -77.04),
    "bogota": (4.71, -74.07), "rio de janeiro": (-22.91, -43.17),
    "sao paulo": (-23.55, -46.63),
}


class VisionLlmModule(BaseModule):
    """Use a local LLaVA vision model to identify locations in images."""

    name = "vision_llm"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)
        self._model = None
        self._model_path = None

    def is_available(self) -> bool:
        try:
            from llama_cpp import Llama  # noqa: F401
            return True
        except ImportError:
            return False

    def prepare(self) -> None:
        from pathlib import Path as P
        import glob

        models_dir = self.config.models_dir
        candidates = list(models_dir.glob("*.gguf"))
        llava_candidates = [c for c in candidates if "llava" in c.stem.lower()]

        if llava_candidates:
            self._model_path = llava_candidates[0]
            self._log(f"Found LLaVA model: {self._model_path.name}")
        elif candidates:
            self._model_path = candidates[0]
            self._log(f"Using model: {self._model_path.name}")
        else:
            self._log("No GGUF model found in models directory", logging.WARNING)

        super().prepare()

    def prepare_model(self) -> None:
        if self._model_path is None or self._model is not None:
            return

        try:
            from llama_cpp import Llama
            self._model = Llama(
                model_path=str(self._model_path),
                n_ctx=2048,
                n_gpu_layers=0,
                verbose=False,
            )
            self._log("LLaVA model loaded")
        except Exception as e:
            self._log(f"Failed to load LLaVA: {e}", logging.WARNING)

    def detect(
        self,
        media_path: Path,
        *,
        frames: list[Any] | None = None,
        audio_path: Path | None = None,
    ) -> list[ModuleHit]:
        if self._model_path is None:
            self._log("No vision LLM model available, skipping")
            return []

        self.prepare_model()
        if self._model is None:
            return []

        image = self._get_image(media_path, frames)
        if image is None:
            return []

        hits: list[ModuleHit] = []
        prompt = (
            "What country, city, or region is this photo taken in? "
            "Be specific. Answer with just the location name."
        )

        try:
            import base64
            import io

            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            response = self._model.create_chat_completion(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                max_tokens=100,
                temperature=0.1,
            )
            text = response["choices"][0]["message"]["content"].strip()
            self._log(f"LLM response: {text}")

            hits = self._parse_location(text)

        except Exception as e:
            self._log(f"LLaVA inference failed: {e}", logging.WARNING)

        return hits

    def _parse_location(self, text: str) -> list[ModuleHit]:
        hits: list[ModuleHit] = []
        text_lower = text.lower()

        sorted_locations = sorted(_LOCATION_COORDS.keys(), key=len, reverse=True)
        matched: list[str] = []

        for loc in sorted_locations:
            if loc in text_lower:
                if not any(loc in m or m in loc for m in matched):
                    matched.append(loc)

        if not matched:
            words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            for w in words:
                wl = w.lower()
                if wl in _LOCATION_COORDS:
                    matched.append(wl)

        for loc in matched:
            lat, lon = _LOCATION_COORDS[loc]
            confidence = min(0.6 + len(loc) * 0.02, 0.9)
            hits.append(self._make_hit(
                lat, lon, confidence,
                location_mention=loc,
                llm_response=text[:500],
            ))

        return hits

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
