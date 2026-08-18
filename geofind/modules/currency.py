"""Currency Identification module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from geofind.core.candidate import ModuleHit
from geofind.core.config import PipelineConfig
from geofind.modules.base import BaseModule

logger = logging.getLogger(__name__)

_COUNTRY_CENTROIDS: dict[str, tuple[float, float]] = {
    "US": (37.09, -95.71), "DE": (51.17, 10.45), "FR": (46.23, 2.21),
    "IT": (41.88, 12.57), "ES": (40.46, -3.75), "PT": (39.40, -8.22),
    "NL": (52.13, 5.29), "BE": (50.50, 4.47), "AT": (47.52, 14.55),
    "GR": (39.07, 21.82), "IE": (53.14, -7.69), "FI": (61.92, 25.75),
    "GB": (55.38, -3.44), "JP": (36.20, 138.25), "IN": (20.59, 78.96),
    "BR": (-14.24, -51.93), "AU": (-25.27, 133.78), "CA": (56.13, -106.35),
    "CN": (35.86, 104.20), "KR": (35.91, 127.77), "RU": (61.52, 105.32),
    "TR": (38.96, 35.24), "MX": (23.63, -102.55), "ZA": (-30.56, 22.94),
    "TH": (15.87, 100.99), "ID": (-0.79, 113.92), "MY": (4.21, 101.98),
    "PH": (12.88, 121.77), "SA": (23.89, 45.08), "AE": (23.42, 53.85),
}


class CurrencyModule(BaseModule):
    """Identify currency notes/bills in images using CLIP."""

    name = "currency"

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
        from geofind.utils.constants import CURRENCY_PROMPTS

        model_name = ensure_clip_model()

        def _load():
            from transformers import CLIPProcessor, CLIPModel
            model = CLIPModel.from_pretrained(model_name)
            processor = CLIPProcessor.from_pretrained(model_name)
            model.eval()
            return model, processor

        self._model, self._processor = get_cached_model("clip_currency", _load)
        self._currency_prompts = CURRENCY_PROMPTS

        self._all_prompts: list[str] = []
        self._prompt_to_currency: dict[int, str] = {}
        for currency, prompts in self._currency_prompts.items():
            for p in prompts:
                idx = len(self._all_prompts)
                self._all_prompts.append(p)
                self._prompt_to_currency[idx] = currency
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
            text=self._all_prompts, images=image,
            return_tensors="pt", padding=True,
        )

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits_per_image[0]
            probs = logits.softmax(dim=0)

        currency_scores: dict[str, float] = {}
        for idx, prob in enumerate(probs.tolist()):
            currency = self._prompt_to_currency[idx]
            currency_scores[currency] = currency_scores.get(currency, 0.0) + prob

        from geofind.utils.constants import CURRENCY_TO_COUNTRIES

        hits: list[ModuleHit] = []
        total = max(sum(currency_scores.values()), 1e-9)

        for currency, score in sorted(currency_scores.items(), key=lambda x: -x[1]):
            norm_score = score / total
            if norm_score < 0.05:
                continue

            countries = CURRENCY_TO_COUNTRIES.get(currency, [])
            if not countries:
                continue

            n = len(countries)
            for cc in countries:
                lat, lon = _COUNTRY_CENTROIDS.get(cc, (0.0, 0.0))
                confidence = min(norm_score / n, 0.8)
                hits.append(self._make_hit(
                    lat, lon, confidence,
                    currency=currency,
                    country=cc,
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
