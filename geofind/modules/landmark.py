"""Landmark Recognition module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from geofind.core.candidate import ModuleHit
from geofind.core.config import PipelineConfig
from geofind.modules.base import BaseModule

logger = logging.getLogger(__name__)

# CLIP ViT-B/32 temperature is 100.0
_CLIP_TEMPERATURE = 100.0
# Threshold on approximate cosine similarity to keep a landmark
_PASS1_THRESHOLD = 0.20
# Refined-pass threshold (slightly lower after re-scoring)
_PASS2_THRESHOLD = 0.22
# Candidate count to send through pass 2
_PASS1_TOP_K = 15


class LandmarkModule(BaseModule):
    """Recognize famous landmarks using CLIP cosine similarity."""

    name = "landmark"

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
        from geofind.utils.constants import LANDMARK_CATEGORIES

        self._model, self._processor = get_clip_shared()
        self._landmarks = LANDMARK_CATEGORIES

        self._landmark_prompts: list[str] = []
        self._prompt_to_key: dict[int, str] = {}
        self._refined_prompts: dict[int, str] = {}
        for key, info in self._landmarks.items():
            name = key.replace("_", " ")
            prompt = f"a photo of {name}, famous landmark, tourist attraction, travel photography"
            refined = f"a photo of {name}, famous landmark, tourist attraction, travel photography, street view"
            idx = len(self._landmark_prompts)
            self._landmark_prompts.append(prompt)
            self._prompt_to_key[idx] = key
            self._refined_prompts[idx] = refined
        super().prepare()

    def _cosine_scores(self, image: Any, prompts: list[str]) -> list[float]:
        """Run CLIP and return approximate cosine similarity scores."""
        import torch

        inputs = self._processor(
            text=prompts, images=image,
            return_tensors="pt", padding=True,
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits_per_image[0]
        return [logit / _CLIP_TEMPERATURE for logit in logits.tolist()]

    @staticmethod
    def _score_to_confidence(cosine_score: float) -> float:
        """Map cosine similarity to a 0-1 confidence value."""
        # Linear ramp: 0.20 → 0.05, 0.25 → 0.3, 0.30 → 0.6, 0.35 → 0.9
        raw = (cosine_score - 0.20) * 8.0
        return min(0.9, max(0.05, raw))

    def detect(
        self,
        media_path: Path,
        *,
        frames: list[Any] | None = None,
        audio_path: Path | None = None,
    ) -> list[ModuleHit]:
        if not self._ready:
            return []

        image = self._get_image(media_path, frames)
        if image is None:
            return []

        # Pass 1: score all landmarks with cosine similarity
        scores = self._cosine_scores(image, self._landmark_prompts)

        # Collect candidates above first-pass threshold
        pass1 = [
            (idx, score)
            for idx, score in enumerate(scores)
            if score >= _PASS1_THRESHOLD
        ]
        if not pass1:
            return []

        # Take top K for pass 2
        pass1.sort(key=lambda x: x[1], reverse=True)
        top_k = pass1[:_PASS1_TOP_K]

        # Pass 2: re-score top candidates with refined prompts
        refined = [self._refined_prompts[idx] for idx, _ in top_k]
        refined_scores = self._cosine_scores(image, refined)

        hits: list[ModuleHit] = []
        for (idx, _score), refined_score in zip(top_k, refined_scores):
            if refined_score < _PASS2_THRESHOLD:
                continue
            key = self._prompt_to_key[idx]
            info = self._landmarks[key]
            lat = info["lat"]
            lon = info["lon"]
            base_conf = info.get("confidence", 0.8)
            confidence = min(self._score_to_confidence(refined_score) * base_conf, 1.0)
            hits.append(self._make_hit(
                lat, lon, confidence,
                landmark=key,
                cosine_score=refined_score,
                pass1_score=_score,
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
