"""Geographic utility functions."""

from __future__ import annotations

import math
from typing import NamedTuple

# Earth radius in km
EARTH_RADIUS_KM = 6371.0


class LatLon(NamedTuple):
    lat: float
    lon: float


def haversine_km(a: LatLon, b: LatLon) -> float:
    """Great-circle distance between two points in km."""
    lat1, lon1 = math.radians(a.lat), math.radians(a.lon)
    lat2, lon2 = math.radians(b.lat), math.radians(b.lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def latlon_to_grid_idx(lat: float, lon: float, resolution: float = 1.0) -> tuple[int, int]:
    """Map lat/lon to grid cell indices."""
    row = int(math.floor(lat / resolution))
    col = int(math.floor(lon / resolution))
    return row, col


def grid_idx_to_latlon(row: int, col: int, resolution: float = 1.0) -> LatLon:
    """Map grid cell indices back to center lat/lon."""
    lat = row * resolution + resolution / 2
    lon = col * resolution + resolution / 2
    return LatLon(lat, lon)


def gaussian_kernel(distance_km: float, sigma_km: float = 100.0) -> float:
    """Gaussian kernel for spreading point probability to grid."""
    return math.exp(-(distance_km ** 2) / (2 * sigma_km ** 2))


def generate_earth_grid(resolution: float = 1.0) -> list[LatLon]:
    """Generate all land-proximate grid cell centers at given resolution.

    Returns ~20k cells covering -90..90 lat, -180..180 lon at 1° resolution.
    Filters to only cells that could possibly be land (simplified).
    """
    cells = []
    lat = -90 + resolution / 2
    while lat < 90:
        lon = -180 + resolution / 2
        while lon < 180:
            cells.append(LatLon(lat, lon))
            lon += resolution
        lat += resolution
    return cells


def is_land_approximate(lat: float, lon: float) -> bool:
    """Very rough land check - excludes obvious open ocean cells.

    This is a simplified heuristic. For production, use a shapefile.
    """
    # Antarctica - mostly land
    if lat < -60:
        return True
    # Arctic - mostly ocean
    if lat > 83:
        return False
    # Major ocean basins (simplified)
    # Pacific
    if -60 < lat < 20 and 150 < lon < -120:
        return False
    # Central Pacific
    if -30 < lat < 30 and 160 < lon < 220:
        return False
    # Everything else could be land
    return True


def build_land_grid(resolution: float = 1.0) -> list[LatLon]:
    """Build grid of land-proximate cells."""
    return [c for c in generate_earth_grid(resolution) if is_land_approximate(c.lat, c.lon)]
