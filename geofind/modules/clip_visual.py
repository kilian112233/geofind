"""CLIP Visual Scene Classification module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from geofind.core.candidate import ModuleHit
from geofind.core.config import PipelineConfig
from geofind.modules.base import BaseModule

logger = logging.getLogger(__name__)

_BIOME_CENTROIDS: dict[str, tuple[float, float]] = {
    "tropical_rainforest": (0.0, 40.0),
    "desert": (25.0, 30.0),
    "temperate_forest": (42.0, 10.0),
    "boreal_forest": (60.0, 0.0),
    "tundra": (72.0, 0.0),
    "mediterranean": (37.0, 15.0),
    "grassland": (40.0, -40.0),
    "subtropical": (27.0, 60.0),
}


class ClipVisualModule(BaseModule):
    """Zero-shot CLIP scene classification mapped to geographic biome priors."""

    name = "clip_visual"

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
        from geofind.utils.models import get_cached_model, ensure_clip_model
        from geofind.utils.constants import BIOME_PROMPTS

        model_name = ensure_clip_model()

        def _load():
            from transformers import CLIPProcessor, CLIPModel
            model = CLIPModel.from_pretrained(model_name)
            processor = CLIPProcessor.from_pretrained(model_name)
            model.eval()
            return model, processor

        self._model, self._processor = get_cached_model("clip_visual", _load)
        self._biome_prompts = BIOME_PROMPTS
        self._biome_centroids = _BIOME_CENTROIDS
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

        import torch
        from PIL import Image

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

        inputs = self._processor(
            text=all_prompts, images=image, return_tensors="pt", padding=True
        )

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits_per_image[0]
            probs = logits.softmax(dim=0)

        biome_scores: dict[str, float] = {}
        for idx, prob in enumerate(probs.tolist()):
            biome = prompt_to_biome[idx]
            biome_scores[biome] = biome_scores.get(biome, 0.0) + prob

        hits: list[ModuleHit] = []
        for biome, score in sorted(biome_scores.items(), key=lambda x: -x[1]):
            if score < 0.05:
                continue
            lat, lon = self._biome_centroids.get(biome, (0.0, 0.0))
            hits.append(self._make_hit(
                lat, lon, min(score, 1.0),
                biome=biome,
                raw_score=score,
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
