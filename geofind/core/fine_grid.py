"""Fine-grained regional grid for hierarchical refinement."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from geofind.core.candidate import CandidateLocation, ModuleHit
from geofind.core.grid import GeoGrid
from geofind.utils.geo import (
    LatLon,
    gaussian_kernel,
    haversine_km,
)

EARTH_RADIUS_KM = 6371.0


class FineGrid:
    """High-resolution regional grid for hierarchical refinement.

    After the coarse 1° grid identifies top candidate regions, this grid
    builds a fine-grained grid (default 0.005° ≈ 555m) around each region
    and re-runs the Bayesian posterior at that resolution.
    """

    def __init__(
        self,
        resolution_deg: float = 0.005,
        sigma_km: float = 0.5,
    ) -> None:
        self.resolution = resolution_deg
        self.sigma_km = sigma_km

    def build_regional_grid_arrays(
        self,
        center_lat: float,
        center_lon: float,
        radius_deg: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build a fine grid as numpy arrays for vectorized computation.

        Returns:
            lats: 1D array of lat centers (degrees)
            lons: 1D array of lon centers (degrees)
            lats_rad: 1D array of lat centers (radians)
            lons_rad: 1D array of lon centers (radians)
            cos_lats: 1D array of cos(lat) for haversine
        """
        min_lat = center_lat - radius_deg
        max_lat = center_lat + radius_deg
        min_lon = center_lon - radius_deg
        max_lon = center_lon + radius_deg

        lat_vals = np.arange(min_lat + self.resolution / 2, max_lat, self.resolution)
        lon_vals = np.arange(min_lon + self.resolution / 2, max_lon, self.resolution)

        # Create 2D meshgrid then flatten
        lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
        lats = lat_grid.ravel()
        lons = lon_grid.ravel()

        lats_rad = np.radians(lats)
        lons_rad = np.radians(lons)
        cos_lats = np.cos(lats_rad)

        return lats, lons, lats_rad, lons_rad, cos_lats

    def compute_fine_posterior(
        self,
        center_lat: float,
        center_lon: float,
        radius_deg: float,
        all_hits: dict[str, list[ModuleHit]],
        module_weights: dict[str, float],
    ) -> list[CandidateLocation]:
        """Compute Bayesian posterior on a fine regional grid.

        Fully vectorized with numpy for performance.
        """
        lats, lons, lats_rad, lons_rad, cos_lats = self.build_regional_grid_arrays(
            center_lat, center_lon, radius_deg
        )
        n_cells = len(lats)
        if n_cells == 0:
            return []

        # Uniform prior over regional cells
        log_prior = np.full(n_cells, -math.log(n_cells))

        # Accumulate module likelihoods
        log_post = log_prior.copy()

        # Region bounds in km for hit prefiltering
        region_radius_km = radius_deg * 111.0

        for mod_name, hits in all_hits.items():
            weight = module_weights.get(mod_name, 0.0)
            if weight <= 0 or not hits:
                continue

            evidence = np.zeros(n_cells)

            for hit in hits:
                hit_lat_rad = math.radians(hit.lat)
                hit_lon_rad = math.radians(hit.lon)
                # Per-hit sigma for area-level modules
                hit_sigma = hit.sigma_km if hit.sigma_km is not None else self.sigma_km

                # Prefilter: skip hits too far from this region to matter.
                # Gaussian kernel is negligible beyond ~4.5 sigma.
                center_dlat = hit_lat_rad - center_lat * math.pi / 180
                center_dlon = hit_lon_rad - center_lon * math.pi / 180
                hc = (math.sin(center_dlat / 2) ** 2
                      + math.cos(hit_lat_rad) * math.cos(center_lat * math.pi / 180)
                      * math.sin(center_dlon / 2) ** 2)
                center_dist = 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(hc, 1.0)))
                if center_dist > region_radius_km * 1.2 + 4.5 * hit_sigma:
                    continue

                dlat = hit_lat_rad - lats_rad
                dlon = hit_lon_rad - lons_rad

                h = (np.sin(dlat / 2) ** 2
                     + np.cos(hit_lat_rad) * cos_lats * np.sin(dlon / 2) ** 2)
                dist = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.minimum(h, 1.0)))

                kernel = np.exp(-(dist ** 2) / (2 * hit_sigma ** 2))
                contrib = hit.confidence * kernel

                np.maximum(evidence, contrib, out=evidence)

            # Normalize
            max_ev = np.max(evidence)
            if max_ev > 0:
                evidence[evidence < max_ev * 1e-6] = 0.0
                total_ev = np.sum(evidence)
                if total_ev > 0:
                    evidence /= total_ev
                    log_lik = np.log(np.maximum(evidence, 1e-30))
                    log_post += weight * log_lik

        # Normalize posterior
        max_lp = np.max(log_post)
        log_post -= max_lp + math.log(np.sum(np.exp(log_post - max_lp)))

        # Extract top candidates sorted by probability
        top_n = min(200, n_cells)
        sorted_indices = np.argpartition(log_post, -top_n)[-top_n:]
        sorted_indices = sorted_indices[np.argsort(log_post[sorted_indices])[::-1]]

        candidates = []
        for idx in sorted_indices:
            candidates.append(CandidateLocation(
                lat=float(lats[idx]),
                lon=float(lons[idx]),
                probability=float(math.exp(log_post[idx])),
                log_posterior=float(log_post[idx]),
                is_fine_refined=True,
            ))

        return candidates

    def refine_candidates(
        self,
        coarse_candidates: list[CandidateLocation],
        all_hits: dict[str, list[ModuleHit]],
        module_weights: dict[str, float],
        radius_deg: float = 0.5,
        top_n: int = 5,
    ) -> list[CandidateLocation]:
        """Refine top coarse candidates on fine grids.

        For each of the top-N coarse candidates, builds a fine regional
        grid centered on that candidate and re-computes the posterior.
        """
        regions_to_refine = coarse_candidates[:top_n]

        all_fine_candidates: list[CandidateLocation] = []

        for coarse_cand in regions_to_refine:
            fine_cands = self.compute_fine_posterior(
                center_lat=coarse_cand.lat,
                center_lon=coarse_cand.lon,
                radius_deg=radius_deg,
                all_hits=all_hits,
                module_weights=module_weights,
            )

            for fc in fine_cands:
                fc.metadata = {"coarse_probability": coarse_cand.probability}

            all_fine_candidates.extend(fine_cands)

        # Deduplicate using KD-tree: O(n log n) instead of O(n²)
        all_fine_candidates.sort(key=lambda c: c.probability, reverse=True)
        if not all_fine_candidates:
            return []

        # Convert to radians for KD-tree (0.2km ≈ 0.0018° at equator)
        # Use approximate degree threshold: 0.2km / 111km_per_deg ≈ 0.0018
        threshold_deg = 0.2 / 111.0

        coords = np.array([[c.lat, c.lon] for c in all_fine_candidates])
        tree = cKDTree(coords)

        keep_mask = np.ones(len(all_fine_candidates), dtype=bool)
        for i in range(len(all_fine_candidates)):
            if not keep_mask[i]:
                continue
            # Mark all neighbors within threshold as duplicates
            neighbors = tree.query_ball_point(coords[i], threshold_deg)
            for j in neighbors:
                if j != i:
                    keep_mask[j] = False

        return [c for c, keep in zip(all_fine_candidates, keep_mask) if keep]
