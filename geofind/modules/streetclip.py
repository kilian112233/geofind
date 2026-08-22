"""StreetCLIP module — geolocation-specialized CLIP zero-shot classifier.

Uses `geolocal/StreetCLIP` (CLIP ViT-L/14 fine-tuned on Street View imagery,
336px input) for zero-shot country classification. Per arXiv 2302.00275,
StreetCLIP zero-shot beats supervised geolocation models trained on 4M+
images on Im2GPS/YFCC benchmarks.

Strategy: score every country with Street View-style prompts ("a Street
View photo from {country}"), average two prompt templates per country,
softmax, and emit country-level hits for the top candidates.
"""

from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path
from typing import Any

from geofind.core.candidate import ModuleHit
from geofind.core.config import PipelineConfig
from geofind.modules.base import BaseModule

logger = logging.getLogger(__name__)

# Prompt templates (averaged per country). StreetCLIP is trained on Street
# View imagery, so "Street View" phrasing matches its training distribution.
_PROMPT_TEMPLATES = [
    "a Street View photo from {name}",
    "a photo taken in {name}",
]

# Eastern-Europe / Balkan / Baltic extension — the base list in clip_visual
# skews Western Europe and misses countries where our test images cluster.
_EXTRA_COUNTRIES: dict[str, tuple[str, tuple[float, float]]] = {
    "MD": ("Moldova", (47.2, 28.5)),
    "BY": ("Belarus", (53.9, 27.6)),
    "SK": ("Slovakia", (48.7, 19.5)),
    "SI": ("Slovenia", (46.1, 14.8)),
    "HR": ("Croatia", (45.1, 15.2)),
    "RS": ("Serbia", (44.8, 20.5)),
    "BA": ("Bosnia and Herzegovina", (43.9, 17.7)),
    "MK": ("North Macedonia", (41.6, 21.7)),
    "AL": ("Albania", (41.2, 20.2)),
    "ME": ("Montenegro", (42.7, 19.4)),
    "BG": ("Bulgaria", (42.7, 25.5)),
    "LT": ("Lithuania", (55.2, 23.9)),
    "LV": ("Latvia", (56.9, 24.6)),
    "EE": ("Estonia", (58.6, 25.0)),
    "GE": ("Georgia", (41.7, 44.8)),
    "AM": ("Armenia", (40.1, 45.0)),
    "AZ": ("Azerbaijan", (40.4, 47.6)),
    "KZ": ("Kazakhstan", (51.2, 71.4)),
    "UZ": ("Uzbekistan", (41.3, 64.6)),
    "CY": ("Cyprus", (35.1, 33.4)),
    "MT": ("Malta", (35.9, 14.4)),
    "IS": ("Iceland", (64.1, -21.9)),
    "LU": ("Luxembourg", (49.6, 6.1)),
}


class StreetclipModule(BaseModule):
    """Zero-shot country classification with the StreetCLIP model."""

    name = "streetclip"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)
        self._img_emb_cache: dict[str, Any] = {}
        self._text_embs: Any | None = None

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def prepare(self) -> None:
        from geofind.utils.models import get_streetclip
        from geofind.modules.clip_visual import (
            _COUNTRY_NAMES as _BASE_NAMES,
            _COUNTRY_CENTROIDS as _BASE_CENTROIDS,
        )

        self._model, self._processor = get_streetclip()

        # Merge base country list with the Eastern-Europe extension
        merged: dict[str, tuple[str, tuple[float, float]]] = {}
        for cc, name in _BASE_NAMES.items():
            if cc in _BASE_CENTROIDS:
                merged[cc] = (name, _BASE_CENTROIDS[cc])
        for cc, (name, centroid) in _EXTRA_COUNTRIES.items():
            merged[cc] = (name, centroid)

        # Stable ordering
        self._country_list: list[tuple[str, str, tuple[float, float]]] = [
            (cc, name, centroid)
            for cc, (name, centroid) in sorted(merged.items())
        ]

        # Build prompts: 2 templates per country
        self._prompts: list[str] = []
        for _cc, name, _centroid in self._country_list:
            for tpl in _PROMPT_TEMPLATES:
                self._prompts.append(tpl.format(name=name))

        super().prepare()

    def _country_text_embeddings(self) -> Any:
        """Encode all country prompts once; returns [N_countries, D] matrix
        where each row is the normalized mean of that country's templates."""
        import torch

        if self._text_embs is not None:
            return self._text_embs

        n_tpl = len(_PROMPT_TEMPLATES)
        chunks: list[Any] = []
        with torch.no_grad():
            for i in range(0, len(self._prompts), 32):
                inputs = self._processor(
                    text=self._prompts[i : i + 32],
                    return_tensors="pt",
                    padding=True,
                )
                feats = self._model.get_text_features(**inputs)
                if hasattr(feats, "text_embeds") and getattr(
                    feats, "text_embeds", None
                ) is not None:
                    feats = feats.text_embeds
                elif getattr(feats, "pooler_output", None) is not None:
                    feats = feats.pooler_output
                feats = feats / feats.norm(dim=-1, keepdim=True)
                chunks.append(feats)

        all_embs = torch.cat(chunks)  # [N_countries * n_tpl, D]
        grouped = all_embs.view(len(self._country_list), n_tpl, -1)
        self._text_embs = grouped.mean(dim=1)
        self._text_embs = self._text_embs / self._text_embs.norm(
            dim=-1, keepdim=True
        )
        return self._text_embs

    def _image_embedding(self, image: Any) -> Any | None:
        """Encode image once per unique content (cached across images run)."""
        import torch

        try:
            key = hashlib.md5(image.tobytes()).hexdigest()
        except Exception:
            key = f"id:{id(image)}"

        emb = self._img_emb_cache.get(key)
        if emb is None:
            with torch.no_grad():
                inputs = self._processor(images=image, return_tensors="pt")
                feats = self._model.get_image_features(**inputs)
                if getattr(feats, "image_embeds", None) is not None:
                    feats = feats.image_embeds
                elif getattr(feats, "pooler_output", None) is not None:
                    feats = feats.pooler_output
                feats = feats / feats.norm(dim=-1, keepdim=True)
            emb = feats[0]
            self._img_emb_cache.clear()  # keep only most recent
            self._img_emb_cache[key] = emb
        return emb

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

        image = self._get_image(media_path, frames)
        if image is None:
            return []

        img_emb = self._image_embedding(image)
        if img_emb is None:
            return []
        txt_embs = self._country_text_embeddings()

        sims = (img_emb @ txt_embs.T).tolist()
        probs = torch.softmax(torch.tensor(sims) * 100.0, dim=0).tolist()

        hits: list[ModuleHit] = []
        ranked = sorted(
            zip(probs, self._country_list), key=lambda x: -x[0]
        )
        top = ranked[:5]
        for prob, (cc, name, (lat, lon)) in top:
            if prob < 0.03:
                continue
            hits.append(self._make_hit(
                lat, lon, min(prob, 1.0),
                sigma_km=800.0,
                country=cc,
                country_name=name,
                raw_score=prob,
                hint_level="country",
                model="streetclip",
            ))
            self._log(
                f"Country: {name} p={prob:.3f}"
            )

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
