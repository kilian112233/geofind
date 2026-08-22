"""Vegetation/Biome Classification module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from geofind.core.candidate import ModuleHit
from geofind.core.config import PipelineConfig
from geofind.modules.base import BaseModule

logger = logging.getLogger(__name__)


class VegetationModule(BaseModule):
    """Classify vegetation type via CLIP and map to geographic biome regions."""

    name = "vegetation"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)

    def is_available(self) -> bool:
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    def prepare(self) -> None:
        from geofind.utils.models import get_clip_shared
        from geofind.utils.constants import BIOME_PROMPTS

        self._model, self._processor = get_clip_shared()
        self._biome_prompts = BIOME_PROMPTS
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

        from geofind.utils.models import clip_softmax_scores

        image = self._get_image(media_path, frames)
        if image is None:
            return []

        all_prompts: list[str] = []
        prompt_to_biome: dict[int, str] = {}
        for biome, prompts in self._biome_prompts.items():
            for p in prompts:
                idx = len(all_prompts)
                all_prompts.append(p)
                prompt_to_biome[idx] = biome

        probs = clip_softmax_scores(image, "vegetation", all_prompts)

        biome_scores: dict[str, float] = {}
        for idx, prob in enumerate(probs):
            biome = prompt_to_biome[idx]
            biome_scores[biome] = biome_scores.get(biome, 0.0) + prob

        from geofind.utils.constants import BIOME_ZONES

        hits: list[ModuleHit] = []
        total = max(sum(biome_scores.values()), 1e-9)

        for biome, score in sorted(biome_scores.items(), key=lambda x: -x[1]):
            norm_score = score / total
            if norm_score < 0.05:
                continue

            zones = [
                (name, min_lat, max_lat, min_lon, max_lon)
                for name, min_lat, max_lat, min_lon, max_lon in BIOME_ZONES
                if name == biome
            ]

            if zones:
                _, min_lat, max_lat, min_lon, max_lon = zones[0]
                lat = (min_lat + max_lat) / 2.0
                lon = (min_lon + max_lon) / 2.0
            else:
                lat, lon = 0.0, 0.0

            confidence = min(norm_score * 1.5, 0.85)
            hits.append(self._make_hit(
                lat, lon, confidence,
                sigma_km=500.0,  # Biome-level — wide spread
                biome=biome,
                raw_score=score,
                hint_level="biome",
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
