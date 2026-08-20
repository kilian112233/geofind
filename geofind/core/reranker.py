"""3-pass Bayesian reranking with consensus clustering."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from geofind.core.candidate import CandidateLocation, ModuleHit
from geofind.core.config import PipelineConfig
from geofind.core.grid import GeoGrid
from geofind.utils.geo import LatLon, haversine_km

logger = logging.getLogger(__name__)


def _hav_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance between two points in km."""
    lat1r, lon1r = math.radians(lat1), math.radians(lon1)
    lat2r, lon2r = math.radians(lat2), math.radians(lon2)
    dl = lat2r - lat1r
    dn = lon2r - lon1r
    a = math.sin(dl / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dn / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(min(a, 1.0)))


class BayesianReranker:
    """Three-pass Bayesian reranking with consensus clustering boost."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def rerank(
        self,
        grid: GeoGrid,
        all_hits: dict[str, list[ModuleHit]],
        module_weights: dict[str, float],
    ) -> list[CandidateLocation]:
        """Full 3-pass reranking pipeline."""

        # ── Pass 1: Standard Bayesian posterior ──────────────────────────────
        log_post = grid.compute_posterior(module_weights)
        candidates = grid.get_top_candidates(log_post, n=self.config.top_candidates)
        grid.attach_hits_to_candidates(candidates, all_hits)

        if not candidates:
            return []

        # ── Pass 2: Consensus clustering buff ───────────────────────────────
        candidates, consensus = self._apply_consensus_buff(
            candidates, module_weights, all_hits,
        )

        # Apply buffs directly to probabilities
        total_prob = sum(c.probability for c in candidates)
        if total_prob > 0:
            for c in candidates:
                c.probability = (c.probability / total_prob) * c.buff_multiplier
            total_prob = sum(c.probability for c in candidates)
            if total_prob > 0:
                for c in candidates:
                    c.probability /= total_prob

        # ── Pass 3: Fuzzy module calibration ─────────────────────────────────
        candidates = self._fuzzy_calibration(candidates, consensus, module_weights)

        # Sort final
        candidates.sort(key=lambda c: c.probability, reverse=True)

        return candidates

    def _apply_consensus_buff(
        self,
        candidates: list[CandidateLocation],
        weights: dict[str, float],
        all_hits: dict[str, list[ModuleHit]],
    ) -> tuple[list[CandidateLocation], dict[str, Any]]:
        """Apply consensus clustering boost using MODULE-LEVEL agreement.

        Each module produces a weighted centroid from its hits.
        We find the densest cluster of module centroids within 500km,
        then boost candidates near that cluster.
        """
        if not candidates:
            return candidates, {"centroid_lat": None, "centroid_lon": None, "strength": 0.0}

        # ── Compute per-module centroids ─────────────────────────────────
        module_centroids: list[tuple[float, float, float, str]] = []
        for mod_name, hits in all_hits.items():
            if not hits:
                continue
            mod_weight = weights.get(mod_name, 1.0)
            w_sum = sum(h.confidence for h in hits)
            if w_sum == 0:
                continue
            m_lat = sum(h.confidence * h.lat for h in hits) / w_sum
            m_lon = sum(h.confidence * h.lon for h in hits) / w_sum
            module_centroids.append((m_lat, m_lon, mod_weight, mod_name))

        if not module_centroids:
            return candidates, {"centroid_lat": None, "centroid_lon": None, "strength": 0.0}

        # ── Find the densest cluster of module centroids ─────────────────
        best_cluster_size = 0
        best_lat, best_lon = 0.0, 0.0
        best_total_weight = 0.0

        for i in range(len(module_centroids)):
            lat_i, lon_i, _, _ = module_centroids[i]
            count = 0
            w_lat, w_lon, w_sum = 0.0, 0.0, 0.0
            for j in range(len(module_centroids)):
                lat_j, lon_j, w_j, _ = module_centroids[j]
                if _hav_km(lat_i, lon_i, lat_j, lon_j) < 300:
                    count += 1
                    w_lat += lat_j * w_j
                    w_lon += lon_j * w_j
                    w_sum += w_j
            if count > best_cluster_size or (count == best_cluster_size and w_sum > best_total_weight):
                best_cluster_size = count
                best_total_weight = w_sum
                best_lat = w_lat / max(w_sum, 1e-9)
                best_lon = w_lon / max(w_sum, 1e-9)

        # ── Compute agreement strength ──────────────────────────────────
        # Base: fraction of all modules in the best cluster
        base_strength = best_cluster_size / max(len(module_centroids), 1)

        # Boost when high-weight modules agree
        high_weight_agreeing = sum(
            1 for lat_i, lon_i, w_i, _ in module_centroids
            if w_i >= 2.0 and _hav_km(lat_i, lon_i, best_lat, best_lon) < 500
        )
        high_weight_total = sum(
            1 for _, _, w_i, _ in module_centroids if w_i >= 2.0
        )
        if high_weight_total > 0:
            hw_ratio = high_weight_agreeing / high_weight_total
            strength = 0.6 * base_strength + 0.4 * hw_ratio
        else:
            strength = base_strength

        # Pair agreement bonus for high-weight modules
        high_weight_names = {"geoclip", "landmark", "clip_visual", "vision_llm"}
        hw_centroids = [
            (lat, lon, name) for lat, lon, w, name in module_centroids
            if name in high_weight_names
        ]
        for i in range(len(hw_centroids)):
            for j in range(i + 1, len(hw_centroids)):
                if _hav_km(hw_centroids[i][0], hw_centroids[i][1],
                           hw_centroids[j][0], hw_centroids[j][1]) < 300:
                    strength = max(strength, 0.5)
                    break
            if strength >= 0.5:
                break

        logger.info(
            f"[reranker] Consensus: {best_cluster_size}/{len(module_centroids)} modules "
            f"agree within 300km, strength={strength:.2f}, "
            f"centroid=({best_lat:.2f}, {best_lon:.2f})"
        )

        # ── Apply proximity-based buffs to candidates ────────────────────
        c_lats = np.array([c.lat for c in candidates], dtype=np.float64)
        c_lons = np.array([c.lon for c in candidates], dtype=np.float64)
        c_lat_rad = np.radians(c_lats)
        c_lon_rad = np.radians(c_lons)
        clat_rad = math.radians(best_lat)
        clon_rad = math.radians(best_lon)

        dlat = clat_rad - c_lat_rad
        dlon = clon_rad - c_lon_rad
        h = (np.sin(dlat / 2) ** 2
             + np.cos(clat_rad) * np.cos(c_lat_rad) * np.sin(dlon / 2) ** 2)
        dists = 2 * 6371.0 * np.arcsin(np.sqrt(np.minimum(h, 1.0)))

        # Sort candidates by distance to centroid
        sorted_by_dist = np.argsort(dists)
        top_n = self.config.consensus_top_n
        top_half = len(candidates) // 2

        for i in range(len(candidates)):
            candidates[i].buff_multiplier = 1.0

        for rank, idx in enumerate(sorted_by_dist):
            if rank < top_n:
                candidates[idx].buff_multiplier = 5.0  # Strong boost for consensus leaders
            elif rank < top_half:
                candidates[idx].buff_multiplier = 2.0
            else:
                candidates[idx].buff_multiplier = 1.0
            # Scale buff by agreement strength
            candidates[idx].buff_multiplier = 1.0 + (candidates[idx].buff_multiplier - 1.0) * strength

        consensus = {
            "centroid_lat": best_lat,
            "centroid_lon": best_lon,
            "strength": strength,
            "definitive_count": best_cluster_size,
        }

        return candidates, consensus

    def _fuzzy_calibration(
        self,
        candidates: list[CandidateLocation],
        consensus: dict[str, Any],
        weights: dict[str, float],
    ) -> list[CandidateLocation]:
        """Pass 3: Calibrate based on module agreement with consensus.

        Boosts candidates that have high-confidence module hits agreeing with
        each other. Vectorized haversine for performance.
        """
        if not candidates or not consensus.get("centroid_lat"):
            return candidates

        centroid_lat = consensus["centroid_lat"]
        centroid_lon = consensus["centroid_lon"]

        # Vectorized distance from centroid for all candidates
        c_lats = np.array([c.lat for c in candidates], dtype=np.float64)
        c_lons = np.array([c.lon for c in candidates], dtype=np.float64)
        c_lat_rad = np.radians(c_lats)
        c_lon_rad = np.radians(c_lons)
        clat_rad = math.radians(centroid_lat)
        clon_rad = math.radians(centroid_lon)

        dlat = clat_rad - c_lat_rad
        dlon = clon_rad - c_lon_rad
        h = (np.sin(dlat / 2) ** 2
             + np.cos(clat_rad) * np.cos(c_lat_rad) * np.sin(dlon / 2) ** 2)
        dists_to_centroid = 2 * 6371.0 * np.arcsin(np.sqrt(np.minimum(h, 1.0)))

        for i, candidate in enumerate(candidates):
            dist = float(dists_to_centroid[i])

            # Proximity penalty for far-away candidates
            if dist > 3000:
                proximity_factor = math.exp(
                    -((dist - 3000) ** 2) / (2 * 5000 ** 2)
                )
                candidate.probability *= (0.8 + 0.2 * proximity_factor)

            # Bonus for multi-module agreement at this candidate
            if len(candidate.hits) >= 3:
                candidate.probability *= 1.2
            elif len(candidate.hits) >= 2:
                candidate.probability *= 1.1

        # Renormalize
        total = sum(c.probability for c in candidates)
        if total > 0:
            for c in candidates:
                c.probability /= total

        return candidates

    def compute_consensus(
        self,
        all_hits: dict[str, list[ModuleHit]],
        weights: dict[str, float],
    ) -> dict[str, Any]:
        """Compute consensus from per-module centroids."""
        module_centroids: list[tuple[float, LatLon]] = []
        for mod_name, hits in all_hits.items():
            if not hits:
                continue
            mod_weight = weights.get(mod_name, 1.0)
            w_sum = sum(h.confidence for h in hits)
            if w_sum == 0:
                continue
            m_lat = sum(h.confidence * h.lat for h in hits) / w_sum
            m_lon = sum(h.confidence * h.lon for h in hits) / w_sum
            module_centroids.append((mod_weight, LatLon(m_lat, m_lon)))

        if not module_centroids:
            return {
                "centroid_lat": 0.0,
                "centroid_lon": 0.0,
                "strength": 0.0,
                "modules_agreeing": 0,
            }

        total_w = sum(w for w, _ in module_centroids)
        centroid_lat = sum(w * mc.lat for w, mc in module_centroids) / total_w
        centroid_lon = sum(w * mc.lon for w, mc in module_centroids) / total_w
        centroid = LatLon(centroid_lat, centroid_lon)

        agreeing = sum(
            1 for _, mc in module_centroids
            if haversine_km(mc, centroid) < 500
        )

        strength = agreeing / max(len(module_centroids), 1)

        return {
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
            "strength": strength,
            "modules_agreeing": agreeing,
            "total_modules": len(module_centroids),
        }
