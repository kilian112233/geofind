"""GeoCLIP pre-trained geo-vision model module.

Uses the GeoCLIP model (VicenteVivan/geo-clip) trained on MP-16 dataset
(4.7M geotagged images worldwide) for direct GPS prediction from visual
features. This is the single most impactful module for non-EXIF accuracy,
providing ~1-10km accuracy on typical outdoor photos.

References:
    - github.com/VicenteVivan/geo-clip
    - MP-16 dataset: 4.7M geotagged images worldwide
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from geofind.core.candidate import ModuleHit
from geofind.core.config import PipelineConfig
from geofind.modules.base import BaseModule

logger = logging.getLogger(__name__)


class GeoclipModule(BaseModule):
    """Pre-trained GeoCLIP model for direct image geolocation.

    Predicts GPS coordinates directly from visual features using a
    CLIP-inspired architecture trained on 4.7M geotagged images.
    Typically provides ~1-10km accuracy for outdoor scenes.
    """

    name = "geoclip"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)

    def is_available(self) -> bool:
        try:
            import geoclip  # noqa: F401
            return True
        except ImportError:
            return False

    def prepare(self) -> None:
        from geofind.utils.models import get_cached_model

        def _load():
            from geoclip import GeoCLIP
            model = GeoCLIP()
            return model

        self._model = get_cached_model("geoclip", _load)
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

        # Get the first frame or original image
        image_path = self._get_image_path(media_path, frames)
        if image_path is None:
            return []

        try:
            # GeoCLIP predict returns (gps_predictions, probabilities)
            # gps_predictions: tensor of (lat, lon) pairs
            # probabilities: tensor of confidence scores
            gps_preds, probs = self._model.predict(
                str(image_path), top_k=10
            )

            hits: list[ModuleHit] = []

            # Convert predictions to module hits
            # gps_preds is typically a tensor of shape (top_k, 2)
            # probs is a tensor of shape (top_k,)
            import torch
            import math

            if isinstance(gps_preds, torch.Tensor):
                gps_np = gps_preds.detach().cpu().numpy()
            else:
                gps_np = gps_preds

            if isinstance(probs, torch.Tensor):
                probs_np = probs.detach().cpu().numpy()
            else:
                probs_np = probs

            # Filter valid predictions first
            valid_lats = []
            valid_lons = []
            valid_probs = []
            for i in range(min(len(gps_np), len(probs_np), 10)):
                lat = float(gps_np[i][0])
                lon = float(gps_np[i][1])
                prob = float(probs_np[i])
                if lat == 0.0 and lon == 0.0:
                    continue
                if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                    continue
                valid_lats.append(lat)
                valid_lons.append(lon)
                valid_probs.append(prob)

            if not valid_lats:
                return []

            # ── Clustering analysis ──────────────────────────────────────
            # Compute how tightly clustered the top predictions are.
            # If top-3 are within 200km, model is regionally confident.
            # If spread across >500km, model is guessing.
            def _hav_km(lat1, lon1, lat2, lon2):
                lat1r, lon1r = math.radians(lat1), math.radians(lon1)
                lat2r, lon2r = math.radians(lat2), math.radians(lon2)
                dl = lat2r - lat1r
                dn = lon2r - lon1r
                a = math.sin(dl/2)**2 + math.cos(lat1r)*math.cos(lat2r)*math.sin(dn/2)**2
                return 2 * 6371 * math.asin(math.sqrt(min(a, 1.0)))

            top_lat, top_lon = valid_lats[0], valid_lons[0]
            # Average distance of top-3 to top-1
            cluster_dists = []
            for j in range(1, min(3, len(valid_lats))):
                cluster_dists.append(_hav_km(top_lat, top_lon, valid_lats[j], valid_lons[j]))
            avg_cluster_km = sum(cluster_dists) / max(len(cluster_dists), 1)

            # Peak dominance: ratio of top prob to second prob
            if len(valid_probs) > 1 and valid_probs[1] > 0:
                peak_ratio = valid_probs[0] / valid_probs[1]
            else:
                peak_ratio = 10.0  # only one prediction → very dominant

            # Cluster spread → per-hit sigma
            # Tight cluster (<100km): precise, use 100km sigma
            # Moderate (100-300km): 250km sigma
            # Wide (>300km): 500km sigma (model is uncertain about region)
            if avg_cluster_km < 100:
                cluster_sigma = 100.0
                cluster_label = "tight"
            elif avg_cluster_km < 300:
                cluster_sigma = 250.0
                cluster_label = "moderate"
            else:
                cluster_sigma = 500.0
                cluster_label = "spread"

            self._log(
                f"Cluster analysis: avg_dist={avg_cluster_km:.0f}km "
                f"({cluster_label}), peak_ratio={peak_ratio:.1f}x, "
                f"sigma={cluster_sigma:.0f}km"
            )

            for i in range(len(valid_lats)):
                lat = valid_lats[i]
                lon = valid_lons[i]
                prob = valid_probs[i]

                if i == 0:
                    # Top prediction: confidence based on clustering + peak dominance
                    # Tight cluster + high dominance = confident
                    # Spread + low dominance = uncertain
                    base_conf = 0.3 + 0.4 * min(peak_ratio / 5.0, 1.0)
                    if cluster_label == "tight":
                        confidence = min(base_conf + 0.2, 0.9)
                    elif cluster_label == "moderate":
                        confidence = min(base_conf, 0.7)
                    else:
                        confidence = min(base_conf * 0.5, 0.4)
                else:
                    # Secondary predictions: much lower, scaled by cluster quality
                    rel_prob = prob / valid_probs[0] if valid_probs[0] > 0 else 0
                    if cluster_label == "tight":
                        confidence = max(0.05, min(0.35, rel_prob * 0.8))
                    else:
                        confidence = max(0.02, min(0.15, rel_prob * 0.3))

                hits.append(self._make_hit(
                    lat=lat,
                    lon=lon,
                    confidence=confidence,
                    sigma_km=cluster_sigma,  # Wide sigma based on cluster spread
                    rank=i,
                    raw_probability=prob,
                    cluster_label=cluster_label,
                    avg_cluster_km=avg_cluster_km,
                    model="geoclip",
                ))

                self._log(
                    f"Prediction #{i+1}: ({lat:.4f}, {lon:.4f}) "
                    f"conf={confidence:.3f} raw_prob={prob:.6f}"
                )

            return hits

        except Exception as e:
            self._log(f"GeoCLIP prediction failed: {e}", level=logging.WARNING)
            return []

    def _get_image_path(
        self, media_path: Path, frames: list[Any] | None
    ) -> Path | None:
        """Get image path for GeoCLIP.

        GeoCLIP works with file paths, not PIL images. If we have extracted
        frames (numpy arrays), save the first frame to a temp file.
        """
        import tempfile

        if frames:
            from PIL import Image
            import numpy as np

            f = frames[0]
            if isinstance(f, Image.Image):
                img = f
            elif isinstance(f, np.ndarray):
                img = Image.fromarray(f)
            else:
                return media_path

            # Save to temp file for GeoCLIP
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            img.convert("RGB").save(tmp.name, quality=95)
            return Path(tmp.name)

        return media_path
