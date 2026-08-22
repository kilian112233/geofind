"""CLIP Image Retrieval module — nearest-neighbour geolocation against an image database."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from geofind.core.candidate import ModuleHit
from geofind.core.config import PipelineConfig
from geofind.modules.base import BaseModule

logger = logging.getLogger(__name__)

# Retrieval database built by scripts/build_retrieval_db.py
_RETRIEVAL_DB_PATH = Path("W:/geofind/models/retrieval_db.npz")

# Cosine similarity below this is considered noise (ViT-L/14 range)
_SIMILARITY_THRESHOLD = 0.15
# Matches at/above this similarity are considered confident
_HIGH_SIM_THRESHOLD = 0.25
_SIGMA_HIGH_KM = 10.0
_SIGMA_MEDIUM_KM = 30.0
_TOP_K = 10
# Matches closer than this to each other are merged into one hit
_DEDUP_RADIUS_KM = 5.0


class ClipRetrievalModule(BaseModule):
    """Retrieve geotagged database images visually similar to the query via CLIP."""

    name = "clip_retrieval"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)
        self._db_features: Any | None = None
        self._db_lats: Any | None = None
        self._db_lons: Any | None = None
        self._db_names: Any | None = None

    def is_available(self) -> bool:
        try:
            import numpy  # noqa: F401
        except ImportError:
            return False
        return _RETRIEVAL_DB_PATH.exists()

    def prepare(self) -> None:
        try:
            import numpy as np

            if not _RETRIEVAL_DB_PATH.exists():
                self._log(
                    f"retrieval database not found: {_RETRIEVAL_DB_PATH}",
                    logging.WARNING,
                )
                return

            with np.load(_RETRIEVAL_DB_PATH) as db:
                self._db_features = np.asarray(db["features"], dtype=np.float32)
                self._db_lats = np.asarray(db["lats"], dtype=np.float64)
                self._db_lons = np.asarray(db["lons"], dtype=np.float64)
                self._db_names = np.asarray(db["names"])

            if len(self._db_features) == 0:
                self._log("retrieval database is empty", logging.WARNING)
                return

            self._log(
                f"loaded {len(self._db_features)} entries "
                f"({self._db_features.shape[1]}-d) from {_RETRIEVAL_DB_PATH.name}"
            )
            super().prepare()

        except Exception as e:
            self._log(f"failed to load retrieval database: {e}", logging.ERROR)

    def detect(
        self,
        media_path: Path,
        *,
        frames: list[Any] | None = None,
        audio_path: Path | None = None,
    ) -> list[ModuleHit]:
        if not self._ready:
            return []

        import numpy as np

        image = self._get_image(media_path, frames)
        if image is None:
            return []

        feat = self._extract_features(image)
        if feat is None:
            return []

        db_features = self._db_features
        if db_features is None or len(db_features) == 0:
            return []

        # ── Similarity search ────────────────────────────────────────────
        similarities = db_features @ feat  # (N,) cosine similarities
        k = min(_TOP_K, len(similarities))
        top_k_indices = np.argsort(similarities)[-k:][::-1]  # descending

        candidates: list[tuple[float, float, float, str]] = []
        for idx in top_k_indices:
            sim = float(similarities[idx])
            if sim <= _SIMILARITY_THRESHOLD:
                continue
            candidates.append((
                sim,
                float(self._db_lats[idx]),
                float(self._db_lons[idx]),
                str(self._db_names[idx]),
            ))

        if not candidates:
            return []

        # ── Merge nearby matches into single hits ────────────────────────
        hits: list[ModuleHit] = []
        for cluster in self._deduplicate(candidates):
            best_sim = cluster["best_sim"]
            sigma = (
                _SIGMA_HIGH_KM
                if best_sim >= _HIGH_SIM_THRESHOLD
                else _SIGMA_MEDIUM_KM
            )
            hits.append(self._make_hit(
                cluster["lat"],
                cluster["lon"],
                min(best_sim, 1.0),
                sigma_km=sigma,
                similarity=best_sim,
                match_count=cluster["count"],
                matches=cluster["names"],
            ))

        return hits

    def _extract_features(self, image: Any) -> Any | None:
        """Extract L2-normalized CLIP ViT-L/14 image embedding as (768,) vector.

        Uses the shared per-image embedding cache — if another module already
        encoded this exact frame, no forward pass is needed.
        """
        try:
            import numpy as np
            from geofind.utils.models import clip_image_embedding

            emb = clip_image_embedding(image)
            feat = emb.detach().cpu().numpy().flatten().astype(np.float32)

            norm = float(np.linalg.norm(feat))
            if norm == 0.0:
                return None
            return feat / norm

        except Exception as e:
            self._log(f"feature extraction failed: {e}", logging.ERROR)
            return None

    @staticmethod
    def _deduplicate(
        candidates: list[tuple[float, float, float, str]],
    ) -> list[dict[str, Any]]:
        """Greedy-cluster candidates within _DEDUP_RADIUS_KM.

        Candidates arrive sorted by similarity (descending). Each cluster keeps
        a similarity-weighted average coordinate; the strongest match anchors
        the cluster's confidence.
        """
        from geofind.utils.geo import LatLon, haversine_km

        clusters: list[dict[str, Any]] = []
        for sim, lat, lon, name in candidates:
            merged = False
            for cluster in clusters:
                anchor = LatLon(cluster["lat"], cluster["lon"])
                if haversine_km(anchor, LatLon(lat, lon)) <= _DEDUP_RADIUS_KM:
                    weight = cluster["weight"] + sim
                    cluster["lat"] = (cluster["lat"] * cluster["weight"] + lat * sim) / weight
                    cluster["lon"] = (cluster["lon"] * cluster["weight"] + lon * sim) / weight
                    cluster["weight"] = weight
                    cluster["count"] += 1
                    cluster["names"].append(name)
                    if sim > cluster["best_sim"]:
                        cluster["best_sim"] = sim
                    merged = True
                    break
            if not merged:
                clusters.append({
                    "lat": lat,
                    "lon": lon,
                    "weight": sim,
                    "best_sim": sim,
                    "count": 1,
                    "names": [name],
                })
        return clusters

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
