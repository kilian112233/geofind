"""Earth grid and probability heatmap management."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from geofind.core.candidate import CandidateLocation, ModuleHit
from geofind.utils.geo import (
    LatLon,
    build_land_grid,
    gaussian_kernel,
    grid_idx_to_latlon,
    haversine_km,
    latlon_to_grid_idx,
)


class GeoGrid:
    """1°×1° Earth grid with log-probability accumulation."""

    def __init__(self, resolution: float = 1.0, sigma_km: float = 100.0) -> None:
        self.resolution = resolution
        self.sigma_km = sigma_km

        # Build land grid
        self.land_cells: list[LatLon] = build_land_grid(resolution)
        self.cell_count = len(self.land_cells)

        # Precompute cell coords as numpy arrays for vectorized distance calc
        self._cell_lats = np.array([c.lat for c in self.land_cells], dtype=np.float64)
        self._cell_lons = np.array([c.lon for c in self.land_cells], dtype=np.float64)
        self._cell_lats_rad = np.radians(self._cell_lats)
        self._cell_lons_rad = np.radians(self._cell_lons)
        self._cos_lats = np.cos(self._cell_lats_rad)

        # Index map: (row, col) → position in land_cells list
        self._idx_map: dict[tuple[int, int], int] = {}
        for i, cell in enumerate(self.land_cells):
            key = latlon_to_grid_idx(cell.lat, cell.lon, resolution)
            self._idx_map[key] = i

        # Log-prior: uniform over land cells
        self.log_prior = np.full(self.cell_count, -math.log(self.cell_count))

        # Accumulated log-likelihoods per module
        self._module_log_likelihoods: dict[str, np.ndarray] = {}

    @property
    def grid_shape(self) -> tuple[int, int]:
        lat_range = int(180 / self.resolution)
        lon_range = int(360 / self.resolution)
        return lat_range, lon_range

    def add_module_hits(self, module_name: str, hits: list[ModuleHit]) -> None:
        """Add hits from a module to the grid, spreading via Gaussian kernel.

        Each module's evidence is normalized to a proper probability
        distribution (sums to 1) before storage. This is critical: a module
        with 1 precise EXIF hit concentrates ~100% of its probability mass
        at the hit location, while a module with 16 scattered landmark hits
        spreads its probability thin. This prevents many imprecise modules
        from collectively overwhelming one precise module.

        Uses MAX (not logsumexp) to combine multiple hits from the same
        module: each cell gets the strongest evidence from whichever hit
        is closest.

        Vectorized with numpy for performance.
        """
        # Initialize with zero evidence (will be normalized later)
        evidence = np.zeros(self.cell_count)

        if not hits:
            self._module_log_likelihoods[module_name] = np.full(
                self.cell_count, -100.0
            )
            return

        # Vectorized haversine + gaussian for all hits at once
        for hit in hits:
            hit_lat_rad = math.radians(hit.lat)
            hit_lon_rad = math.radians(hit.lon)
            # Per-hit sigma: area-level modules use wider spread
            hit_sigma = hit.sigma_km if hit.sigma_km is not None else self.sigma_km

            dlat = hit_lat_rad - self._cell_lats_rad
            dlon = hit_lon_rad - self._cell_lons_rad

            h = (np.sin(dlat / 2) ** 2
                 + np.cos(hit_lat_rad) * self._cos_lats * np.sin(dlon / 2) ** 2)
            dist = 2 * 6371.0 * np.arcsin(np.sqrt(np.minimum(h, 1.0)))

            kernel = np.exp(-(dist ** 2) / (2 * hit_sigma ** 2))
            contrib = hit.confidence * kernel

            # MAX: take the strongest evidence at each cell
            np.maximum(evidence, contrib, out=evidence)

        # Hard cutoff: cells with negligible evidence get zero
        max_ev = np.max(evidence)
        if max_ev > 0:
            evidence[evidence < max_ev * 1e-6] = 0.0

        # Normalize to probability distribution: sum(evidence) = 1
        total_ev = np.sum(evidence)
        if total_ev > 0:
            evidence /= total_ev
            log_lik = np.log(np.maximum(evidence, 1e-30))
        else:
            log_lik = np.full(self.cell_count, -100.0)

        self._module_log_likelihoods[module_name] = log_lik

    def compute_posterior(
        self,
        weights: dict[str, float],
    ) -> np.ndarray:
        """Compute log-posterior given module weights.

        log P(c|all) = log P(c) + Σ_i w_i × log P_i(c)
        """
        log_post = self.log_prior.copy()

        for mod_name, weight in weights.items():
            if mod_name not in self._module_log_likelihoods:
                continue
            log_lik = self._module_log_likelihoods[mod_name]
            log_post += weight * log_lik

        # Normalize (log-sum-exp)
        max_lp = np.max(log_post)
        log_post -= max_lp + math.log(np.sum(np.exp(log_post - max_lp)))

        return log_post

    def get_top_candidates(
        self,
        log_posterior: np.ndarray,
        n: int = 20_000,
    ) -> list[CandidateLocation]:
        """Extract top-n candidates from log-posterior."""
        # Sort by log-posterior (descending)
        sorted_indices = np.argsort(log_posterior)[::-1][:n]

        candidates = []
        for idx in sorted_indices:
            cell = self.land_cells[idx]
            candidates.append(CandidateLocation(
                lat=cell.lat,
                lon=cell.lon,
                probability=math.exp(log_posterior[idx]),
                log_posterior=float(log_posterior[idx]),
            ))

        return candidates

    def attach_hits_to_candidates(
        self,
        candidates: list[CandidateLocation],
        all_hits: dict[str, list[ModuleHit]],
    ) -> None:
        """Attach module hits to nearby candidates. Vectorized."""
        if not candidates:
            return

        # Flatten all hits
        all_hits_flat: list[ModuleHit] = []
        for hits in all_hits.values():
            all_hits_flat.extend(hits)
        if not all_hits_flat:
            return

        # Build candidate arrays
        cand_lats = np.array([c.lat for c in candidates], dtype=np.float64)
        cand_lons = np.array([c.lon for c in candidates], dtype=np.float64)
        cand_lats_rad = np.radians(cand_lats)
        cand_lons_rad = np.radians(cand_lons)
        cos_cand = np.cos(cand_lats_rad)

        for hit in all_hits_flat:
            hit_lat_rad = math.radians(hit.lat)
            hit_lon_rad = math.radians(hit.lon)
            hit_sigma = hit.sigma_km if hit.sigma_km is not None else self.sigma_km

            dlat = hit_lat_rad - cand_lats_rad
            dlon = hit_lon_rad - cand_lons_rad

            h = (np.sin(dlat / 2) ** 2
                 + np.cos(hit_lat_rad) * cos_cand * np.sin(dlon / 2) ** 2)
            dist = 2 * 6371.0 * np.arcsin(np.sqrt(np.minimum(h, 1.0)))

            threshold_km = hit_sigma * 2
            nearby = np.where(dist < threshold_km)[0]
            for idx in nearby:
                candidates[idx].add_hit(hit)

    def get_module_names(self) -> list[str]:
        return list(self._module_log_likelihoods.keys())
