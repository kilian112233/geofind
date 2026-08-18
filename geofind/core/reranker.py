"""3-pass Bayesian reranking with consensus clustering."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from geofind.core.candidate import CandidateLocation, ModuleHit
from geofind.core.config import PipelineConfig
from geofind.core.grid import GeoGrid
from geofind.utils.geo import LatLon, haversine_km


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
        candidates, consensus = self._apply_consensus_buff(candidates, module_weights)

        # Apply buffs directly to probabilities (not via grid re-computation
        # which had index-mapping bugs). The buff multipliers from Pass 2
        # scale each candidate's posterior probability.
        total_prob = sum(c.probability for c in candidates)
        if total_prob > 0:
            for c in candidates:
                c.probability = (c.probability / total_prob) * c.buff_multiplier
            # Renormalize
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
    ) -> tuple[list[CandidateLocation], dict[str, Any]]:
        """Apply consensus clustering boost to candidates.

        Boosts candidates closest to the consensus centroid (proximity-based),
        not those highest in the ranked list. This ensures precise modules
        like EXIF are properly boosted when they agree with each other.

        Returns updated candidates and consensus metadata.
        """
        if not candidates:
            return candidates, {"centroid": None, "strength": 0.0}

        # Find the centroid of "definitive" candidates (those with multiple
        # module hits confirming the same area)
        definitive = [c for c in candidates[:100] if len(c.hits) >= 2]
        if not definitive:
            # Fall back to top candidates by probability
            definitive = candidates[:20]

        # Weighted centroid from definitive candidates
        total_weight = sum(
            max(h.confidence, 0.01) for c in definitive for h in c.hits
        )
        if total_weight == 0:
            return candidates, {"centroid": None, "strength": 0.0}

        centroid_lat = sum(
            c.lat * sum(max(h.confidence, 0.01) for h in c.hits)
            for c in definitive
        ) / total_weight
        centroid_lon = sum(
            c.lon * sum(max(h.confidence, 0.01) for h in c.hits)
            for c in definitive
        ) / total_weight
        centroid = LatLon(centroid_lat, centroid_lon)

        # Compute distances from centroid for all candidates
        dists = [
            haversine_km(centroid, LatLon(c.lat, c.lon))
            for c in candidates
        ]

        # Compute agreement strength: how tightly clustered are top hits?
        top_dists = dists[:min(20, len(dists))]
        if top_dists:
            avg_dist = sum(top_dists) / len(top_dists)
            # Strength: 1.0 when avg_dist < 100km, approaching 0 when > 2000km
            strength = max(0.0, min(1.0, 1.0 - (avg_dist - 100) / 1900))
        else:
            strength = 0.0

        # Sort candidates by distance to centroid (closest first)
        sorted_by_dist = sorted(range(len(candidates)), key=lambda i: dists[i])
        top_n = self.config.consensus_top_n
        top_half = len(candidates) // 2

        # Apply proximity-based buffs: closest to centroid get the biggest boost
        for i in range(len(candidates)):
            candidates[i].buff_multiplier = 1.0

        for rank, idx in enumerate(sorted_by_dist):
            if rank < top_n:
                candidates[idx].buff_multiplier = self.config.consensus_top_n_buff
            elif rank < top_half:
                candidates[idx].buff_multiplier = self.config.consensus_top_half_buff
            else:
                candidates[idx].buff_multiplier = 1.0

            # Scale buff by agreement strength
            candidates[idx].buff_multiplier = 1.0 + (candidates[idx].buff_multiplier - 1.0) * strength

        consensus = {
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
            "strength": strength,
            "definitive_count": len(definitive),
        }

        return candidates, consensus

    def _fuzzy_calibration(
        self,
        candidates: list[CandidateLocation],
        consensus: dict[str, Any],
        weights: dict[str, float],
    ) -> list[CandidateLocation]:
        """Pass 3: Calibrate based on module agreement with consensus.

        Instead of penalizing candidates far from consensus (which would
        hurt precise modules like EXIF when consensus is wrong), we boost
        candidates that have high-confidence module hits agreeing with
        each other. The key insight: a candidate with a high-weight module
        hit (e.g. EXIF GPS) should not be downweighted by fuzzy consensus.
        """
        if not candidates or not consensus.get("centroid_lat"):
            return candidates

        centroid = LatLon(consensus["centroid_lat"], consensus["centroid_lon"])

        # For each candidate, compute a confidence bonus based on:
        # 1. Number of modules that agree (hit count)
        # 2. Whether the highest-weight module (EXIF) hit this location
        for candidate in candidates:
            cand_point = LatLon(candidate.lat, candidate.lon)
            dist_to_centroid = haversine_km(cand_point, centroid)

            # Mild proximity factor — only penalize very far candidates
            # (e.g. >3000km from consensus), and only slightly
            if dist_to_centroid > 3000:
                proximity_factor = math.exp(
                    -((dist_to_centroid - 3000) ** 2) / (2 * 5000 ** 2)
                )
                candidate.probability *= (0.8 + 0.2 * proximity_factor)

            # Bonus for multi-module agreement
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
        """Compute consensus from per-module centroids.

        Each module produces a single centroid from its hits, weighted by
        module weight and hit confidence. The overall consensus is the
        weighted average of these module centroids. Agreement strength
        measures how many module centroids cluster within 500km.
        """
        # Compute per-module centroids (weighted by hit confidence)
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
            # Weight the module centroid by module weight
            module_centroids.append((mod_weight, LatLon(m_lat, m_lon)))

        if not module_centroids:
            return {
                "centroid_lat": 0.0,
                "centroid_lon": 0.0,
                "strength": 0.0,
                "modules_agreeing": 0,
            }

        # Overall centroid = weighted average of module centroids
        total_w = sum(w for w, _ in module_centroids)
        centroid_lat = sum(w * mc.lat for w, mc in module_centroids) / total_w
        centroid_lon = sum(w * mc.lon for w, mc in module_centroids) / total_w
        centroid = LatLon(centroid_lat, centroid_lon)

        # How many module centroids are within 500km of overall centroid
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
