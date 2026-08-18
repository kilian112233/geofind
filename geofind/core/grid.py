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
        """
        # Initialize with zero evidence (will be normalized later)
        evidence = np.zeros(self.cell_count)

        if not hits:
            self._module_log_likelihoods[module_name] = np.full(
                self.cell_count, -100.0
            )
            return

        for hit in hits:
            hit_point = LatLon(hit.lat, hit.lon)
            for i, cell in enumerate(self.land_cells):
                dist = haversine_km(hit_point, cell)
                kernel = gaussian_kernel(dist, self.sigma_km)
                contrib = hit.confidence * kernel
                # MAX: take the strongest evidence at each cell
                if contrib > evidence[i]:
                    evidence[i] = contrib

        # Hard cutoff: cells with negligible evidence get zero
        # (prevents Gaussian tails from spreading evidence globally)
        max_ev = np.max(evidence)
        if max_ev > 0:
            evidence[evidence < max_ev * 1e-6] = 0.0

        # Normalize to probability distribution: sum(evidence) = 1
        # This is the KEY fix: a precise module concentrates probability,
        # while an imprecise module spreads it thin.
        total_ev = np.sum(evidence)
        if total_ev > 0:
            evidence /= total_ev
            # Convert to log-probabilities
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
        """Attach module hits to nearby candidates."""
        for candidate in candidates:
            cand_point = LatLon(candidate.lat, candidate.lon)
            for mod_name, hits in all_hits.items():
                for hit in hits:
                    dist = haversine_km(cand_point, LatLon(hit.lat, hit.lon))
                    if dist < self.sigma_km * 2:
                        candidate.add_hit(hit)

    def get_module_names(self) -> list[str]:
        return list(self._module_log_likelihoods.keys())
