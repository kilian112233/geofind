"""OCR + Script/Language Detection module."""

from __future__ import annotations

import logging
import unicodedata
from pathlib import Path
from typing import Any

from geofind.core.candidate import ModuleHit
from geofind.core.config import PipelineConfig
from geofind.modules.base import BaseModule

logger = logging.getLogger(__name__)

_COUNTRY_CENTROIDS: dict[str, tuple[float, float]] = {
    "TH": (13.75, 100.50), "JP": (36.20, 138.25), "KR": (35.91, 127.77),
    "CN": (35.86, 104.20), "TW": (23.70, 120.96), "HK": (22.40, 114.11),
    "MO": (22.20, 113.55), "SA": (23.89, 45.08), "AE": (23.42, 53.85),
    "EG": (26.82, 30.80), "MA": (31.79, -7.09), "TN": (33.89, 9.54),
    "IQ": (33.22, 43.68), "IR": (32.43, 53.69), "JO": (30.59, 36.24),
    "LB": (33.85, 35.86), "RU": (61.52, 105.32), "UA": (48.38, 31.17),
    "BG": (42.73, 25.49), "RS": (44.02, 21.01), "BY": (53.71, 27.95),
    "KZ": (48.02, 66.92), "KG": (41.20, 74.77), "IN": (20.59, 78.96),
    "NP": (28.39, 84.12), "BD": (23.68, 90.36), "LK": (7.87, 80.77),
    "TR": (38.96, 35.24), "VN": (14.06, 108.28), "ID": (-0.79, 113.92),
    "PH": (12.88, 121.77), "MY": (4.21, 101.98), "GR": (39.07, 21.82),
    "IL": (31.05, 34.85), "GE": (42.32, 43.36), "AM": (40.07, 45.04),
    "ET": (9.15, 40.49), "MM": (21.91, 95.96), "KH": (12.57, 104.99),
    "LA": (19.86, 102.50),
}

_SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    "thai": [(0x0E00, 0x0E7F)],
    "japanese_hiragana": [(0x3040, 0x309F)],
    "japanese_katakana": [(0x30A0, 0x30FF)],
    "korean": [(0xAC00, 0xD7AF), (0x1100, 0x11FF)],
    "chinese_cjk": [(0x4E00, 0x9FFF)],
    "arabic": [(0x0600, 0x06FF), (0x0750, 0x077F)],
    "cyrillic": [(0x0400, 0x04FF)],
    "devanagari": [(0x0900, 0x097F)],
    "bengali": [(0x0980, 0x09FF)],
    "tamil": [(0x0B80, 0x0BFF)],
    "telugu": [(0x0C00, 0x0C7F)],
    "kannada": [(0x0C80, 0x0CFF)],
    "malayalam": [(0x0D00, 0x0D7F)],
    "gurmukhi": [(0x0A00, 0x0A7F)],
    "greek": [(0x0370, 0x03FF)],
    "hebrew": [(0x0590, 0x05FF)],
    "georgian": [(0x10A0, 0x10FF)],
    "armenian": [(0x0530, 0x058F)],
    "ethiopic": [(0x1200, 0x137F)],
    "myanmar": [(0x1000, 0x109F)],
    "khmer": [(0x1780, 0x17FF)],
    "lao": [(0x0E80, 0x0EFF)],
    "tibetan": [(0x0F00, 0x0FFF)],
}

_SCRIPT_TO_HINT_KEY: dict[str, str] = {
    "thai": "thai", "japanese_hiragana": "japanese", "japanese_katakana": "japanese",
    "korean": "korean", "chinese_cjk": "chinese_similar", "arabic": "arabic",
    "cyrillic": "cyrillic", "devanagari": "devanagari", "bengali": "bengali",
    "tamil": "tamil", "telugu": "telugu", "kannada": "kannada",
    "malayalam": "malayalam", "gurmukhi": "gurmukhi", "greek": "greek",
    "hebrew": "hebrew", "georgian": "georgian", "armenian": "armenian",
    "ethiopic": "ethiopic", "myanmar": "myanmar", "khmer": "khmer",
    "lao": "lao", "tibetan": "tibetan",
}

_LATIN_EXTENDED_RANGES: list[tuple[int, int]] = [
    (0x00C0, 0x024F), (0x1E00, 0x1EFF), (0x0100, 0x017F),
    (0x0180, 0x024F), (0x0250, 0x02AF),
]


class OcrTextModule(BaseModule):
    """Extract text via OCR, detect script/language, map to country hints."""

    name = "ocr_text"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)

    def is_available(self) -> bool:
        try:
            import easyocr  # noqa: F401
            return True
        except ImportError:
            return False

    def prepare(self) -> None:
        from geofind.utils.models import get_cached_model, ensure_easyocr_langs

        langs = ensure_easyocr_langs()

        def _load():
            import easyocr
            return easyocr.Reader(langs, gpu=False)

        self._reader = get_cached_model("easyocr", _load)
        super().prepare()

    def detect(
        self,
        media_path: Path,
        *,
        frames: list[Any] | None = None,
        audio_path: Path | None = None,
    ) -> list[ModuleHit]:
        if not self._ready:
            return []

        from PIL import Image

        image = self._get_image(media_path, frames)
        if image is None:
            return []

        try:
            import numpy as np
            img_array = np.array(image)
            results = self._reader.readtext(img_array)
        except Exception as e:
            self._log(f"OCR failed: {e}", logging.WARNING)
            return []

        if not results:
            return []

        full_text = " ".join(r[1] for r in results if len(r) > 1)
        if not full_text.strip():
            return []

        self._log(f"OCR text: {full_text[:100]}")

        scripts = self._detect_scripts(full_text)
        country_votes: dict[str, float] = {}

        from geofind.utils.constants import SCRIPT_COUNTRY_HINTS

        for script, count in scripts.items():
            hint_key = _SCRIPT_TO_HINT_KEY.get(script, script)
            countries = SCRIPT_COUNTRY_HINTS.get(hint_key, [])
            weight = count / max(len(full_text), 1)
            for cc in countries:
                country_votes[cc] = country_votes.get(cc, 0.0) + weight

        hits: list[ModuleHit] = []
        if country_votes:
            total = max(sum(country_votes.values()), 1e-9)
            for cc, score in sorted(country_votes.items(), key=lambda x: -x[1]):
                if score / total < 0.01:
                    continue
                lat, lon = _COUNTRY_CENTROIDS.get(cc, (0.0, 0.0))
                hits.append(self._make_hit(
                    lat, lon, min(score / total, 1.0),
                    country=cc,
                    ocr_text=full_text[:500],
                    scripts=list(scripts.keys()),
                ))
        else:
            latin_score = self._count_latin_extended(full_text) / max(len(full_text), 1)
            if latin_score > 0.1:
                hits.append(self._make_hit(
                    37.0, 15.0, 0.3,
                    scripts=["latin_extended"],
                    ocr_text=full_text[:500],
                ))

        return hits

    def _detect_scripts(self, text: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for ch in text:
            cp = ord(ch)
            for script_name, ranges in _SCRIPT_RANGES.items():
                for start, end in ranges:
                    if start <= cp <= end:
                        counts[script_name] = counts.get(script_name, 0) + 1
                        break
        return counts

    def _count_latin_extended(self, text: str) -> int:
        count = 0
        for ch in text:
            cp = ord(ch)
            for start, end in _LATIN_EXTENDED_RANGES:
                if start <= cp <= end:
                    count += 1
                    break
        return count

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
