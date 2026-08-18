"""Landmark Recognition module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from geofind.core.candidate import ModuleHit
from geofind.core.config import PipelineConfig
from geofind.modules.base import BaseModule

logger = logging.getLogger(__name__)


class LandmarkModule(BaseModule):
    """Recognize famous landmarks using CLIP embeddings."""

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
        from geofind.utils.models import get_cached_model, ensure_clip_model
        from geofind.utils.constants import LANDMARK_CATEGORIES

        model_name = ensure_clip_model()

        def _load():
            from transformers import CLIPProcessor, CLIPModel
            model = CLIPModel.from_pretrained(model_name)
            processor = CLIPProcessor.from_pretrained(model_name)
            model.eval()
            return model, processor

        self._model, self._processor = get_cached_model("clip_landmark", _load)
        self._landmarks = LANDMARK_CATEGORIES

        self._landmark_prompts: list[str] = []
        self._prompt_to_key: dict[int, str] = {}
        for key, info in self._landmarks.items():
            name = key.replace("_", " ")
            prompt = f"a photo of {name}"
            idx = len(self._landmark_prompts)
            self._landmark_prompts.append(prompt)
            self._prompt_to_key[idx] = key
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

        inputs = self._processor(
            text=self._landmark_prompts, images=image,
            return_tensors="pt", padding=True,
        )

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits_per_image[0]
            probs = logits.softmax(dim=0)

        hits: list[ModuleHit] = []
        threshold = 0.01
        for idx, prob in enumerate(probs.tolist()):
            if prob < threshold:
                continue
            key = self._prompt_to_key[idx]
            info = self._landmarks[key]
            lat = info["lat"]
            lon = info["lon"]
            base_conf = info.get("confidence", 0.8)
            confidence = min(prob * base_conf, 1.0)
            hits.append(self._make_hit(
                lat, lon, confidence,
                landmark=key,
                raw_score=prob,
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
