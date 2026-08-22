"""Region Voter meta-module: combines area-level module signals into regional GPS hits.

This module does NOT analyze images directly. Instead, it is called by the
pipeline after all other modules run, to combine their area-level hints
(OCR script, driving side, vegetation biome, CLIP country, etc.) into a
refined regional consensus hit.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from geofind.core.candidate import ModuleHit
from geofind.core.config import PipelineConfig
from geofind.modules.base import BaseModule

logger = logging.getLogger(__name__)

# Country centroids for voting — maps ISO-2 country code to (lat, lon, name)
_REGION_CENTROIDS: dict[str, tuple[float, float, str]] = {
    "US": (39.8, -98.5, "United States"),
    "CA": (56.1, -106.3, "Canada"),
    "MX": (23.6, -102.6, "Mexico"),
    "BR": (-14.2, -51.9, "Brazil"),
    "AR": (-38.4, -63.6, "Argentina"),
    "CL": (-35.7, -71.5, "Chile"),
    "CO": (4.6, -74.3, "Colombia"),
    "PE": (-9.2, -75.0, "Peru"),
    "GB": (54.0, -2.0, "United Kingdom"),
    "IE": (53.4, -8.2, "Ireland"),
    "FR": (46.6, 2.0, "France"),
    "DE": (51.2, 10.4, "Germany"),
    "IT": (42.5, 12.5, "Italy"),
    "ES": (40.0, -3.7, "Spain"),
    "PT": (39.4, -8.2, "Portugal"),
    "NL": (52.1, 5.3, "Netherlands"),
    "BE": (50.5, 4.5, "Belgium"),
    "CH": (46.8, 8.2, "Switzerland"),
    "AT": (47.3, 13.3, "Austria"),
    "PL": (51.9, 19.1, "Poland"),
    "CZ": (49.8, 15.5, "Czech Republic"),
    "HU": (47.2, 19.5, "Hungary"),
    "RO": (45.9, 24.9, "Romania"),
    "BG": (42.7, 25.5, "Bulgaria"),
    "HR": (45.1, 15.2, "Croatia"),
    "RS": (44.0, 21.0, "Serbia"),
    "BA": (43.9, 17.7, "Bosnia"),
    "SK": (48.7, 19.7, "Slovakia"),
    "SI": (46.2, 14.9, "Slovenia"),
    "UA": (49.0, 31.3, "Ukraine"),
    "BY": (53.7, 27.9, "Belarus"),
    "RU": (61.5, 105.3, "Russia"),
    "SE": (60.1, 18.6, "Sweden"),
    "NO": (60.5, 8.5, "Norway"),
    "FI": (61.9, 25.7, "Finland"),
    "DK": (56.3, 9.5, "Denmark"),
    "GR": (39.1, 21.8, "Greece"),
    "TR": (38.9, 35.2, "Turkey"),
    "JP": (36.2, 138.3, "Japan"),
    "CN": (35.9, 104.2, "China"),
    "KR": (35.9, 127.8, "South Korea"),
    "IN": (20.6, 78.9, "India"),
    "AU": (-25.3, 133.8, "Australia"),
    "NZ": (-40.9, 174.9, "New Zealand"),
    "ZA": (-30.6, 22.9, "South Africa"),
    "EG": (26.8, 30.8, "Egypt"),
    "NG": (9.1, 8.7, "Nigeria"),
    "KE": (-0.02, 37.9, "Kenya"),
    "MA": (31.8, -7.1, "Morocco"),
    "TH": (15.9, 100.9, "Thailand"),
    "VN": (14.1, 108.3, "Vietnam"),
    "ID": (-0.8, 113.9, "Indonesia"),
    "PH": (12.9, 121.8, "Philippines"),
    "MY": (4.2, 101.9, "Malaysia"),
    "SG": (1.4, 103.8, "Singapore"),
    "PK": (30.4, 69.3, "Pakistan"),
    "BD": (23.7, 90.4, "Bangladesh"),
    "LK": (7.9, 80.8, "Sri Lanka"),
    "NP": (28.4, 84.1, "Nepal"),
    "IS": (64.9, -19.0, "Iceland"),
    "LT": (55.2, 23.9, "Lithuania"),
    "LV": (56.9, 24.1, "Latvia"),
    "EE": (58.6, 25.0, "Estonia"),
    "MD": (47.0, 28.9, "Moldova"),
    "AL": (41.2, 20.2, "Albania"),
    "MK": (41.5, 21.7, "North Macedonia"),
    "ME": (42.7, 19.4, "Montenegro"),
}

# Modules whose hits are area-level (wide sigma → regional hint)
_AREA_LEVEL_MODULES = {
    "ocr_text", "driving_side", "vegetation", "currency",
    "license_plate", "clip_visual", "places365", "audio_scene",
    "audio_power",
}

# Modules whose hits are precise (narrow sigma → city/landmark level)
_PRECISE_MODULES = {
    "landmark", "clip_retrieval", "geoclip",
}

# Distance threshold for mapping a hit to a country (km)
_COUNTRY_ASSIGN_THRESHOLD_KM = 800.0


def _hav_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in km."""
    lat1r, lon1r = math.radians(lat1), math.radians(lon1)
    lat2r, lon2r = math.radians(lat2), math.radians(lon2)
    dl = lat2r - lat1r
    dn = lon2r - lon1r
    a = math.sin(dl / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dn / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(min(a, 1.0)))


def _assign_country(lat: float, lon: float) -> str | None:
    """Find the nearest country centroid within threshold."""
    best_code = None
    best_dist = _COUNTRY_ASSIGN_THRESHOLD_KM
    for code, (clat, clon, _) in _REGION_CENTROIDS.items():
        d = _hav_km(lat, lon, clat, clon)
        if d < best_dist:
            best_dist = d
            best_code = code
    return best_code


def _get_country_from_hit(hit: ModuleHit) -> str | None:
    """Try to extract country code from hit metadata, or assign by proximity."""
    # Check metadata first
    country = hit.metadata.get("country", "")
    if country and len(country) == 2:
        return country.upper()
    # Check country_hint
    country = hit.metadata.get("country_hint", "")
    if country and len(country) == 2:
        return country.upper()
    # Fall back to proximity assignment
    return _assign_country(hit.lat, hit.lon)


class RegionVoterModule(BaseModule):
    """Meta-module that combines area-level signals into regional GPS hits.

    This module is a BaseModule subclass for pipeline compatibility but
    never runs directly. The pipeline calls vote() as a post-processing step.
    """

    name = "region_voter"

    def is_available(self) -> bool:
        return True

    def prepare(self) -> None:
        self._ready = True

    def detect(self, media_path: Any, *, frames: Any = None, audio_path: Any = None) -> list[ModuleHit]:
        raise NotImplementedError("RegionVoterModule does not run directly; use vote()")

    @staticmethod
    def vote(
        all_hits: dict[str, list[ModuleHit]],
        config: PipelineConfig,
    ) -> list[ModuleHit]:
        """Combine area-level module signals into regional consensus hits.

        Returns top-3 country-level hits weighted by vote strength, plus
        optional agreement hit when precise modules converge.
        """
        # ── Collect country votes ────────────────────────────────────────
        country_votes: dict[str, float] = {}  # code → total confidence vote
        country_voters: dict[str, list[str]] = {}  # code → list of module names

        for mod_name, hits in all_hits.items():
            if not hits:
                continue
            mod_weight = config.modules.get(mod_name, None)
            w = mod_weight.weight if mod_weight else 1.0

            for hit in hits:
                hit_sigma = getattr(hit, 'sigma_km', None)
                is_area = hit_sigma is not None and hit_sigma >= 400
                # Also treat area-level modules as area hits even without explicit sigma
                if mod_name in _AREA_LEVEL_MODULES:
                    is_area = True
                if not is_area:
                    continue  # skip precise hits for country voting

                country = _get_country_from_hit(hit)
                if country is None:
                    continue

                # Vote weighted by hit confidence × module weight
                vote = hit.confidence * min(w, 5.0)  # cap module weight at 5 for voting
                country_votes[country] = country_votes.get(country, 0.0) + vote
                if country not in country_voters:
                    country_voters[country] = []
                if mod_name not in country_voters[country]:
                    country_voters[country].append(mod_name)

        if not country_votes:
            logger.debug("[region_voter] No area-level votes collected")
            return []

        # ── Rank countries by vote strength ──────────────────────────────
        total_votes = sum(country_votes.values())
        ranked = sorted(country_votes.items(), key=lambda x: -x[1])

        hits: list[ModuleHit] = []
        # Emit only TOP-1 country to concentrate evidence (not spread across 3)
        for code, votes in ranked[:1]:
            vote_share = votes / max(total_votes, 1e-9)
            if vote_share < 0.35:
                break  # require a decisive plurality — weak pluralities mislead
            clat, clon, name = _REGION_CENTROIDS.get(code, (0.0, 0.0, code))
            # Conservative confidence: even a unanimous vote stays a hint
            conf = min(0.5, vote_share)
            n_voters = len(country_voters.get(code, []))
            hits.append(ModuleHit(
                module="region_voter",
                lat=clat,
                lon=clon,
                confidence=conf,
                sigma_km=500.0,  # wide sigma — this is a country-level guess
                metadata={
                    "country": code,
                    "country_name": name,
                    "vote_share": vote_share,
                    "voter_count": n_voters,
                    "voters": country_voters.get(code, []),
                },
            ))

        if hits:
            logger.info(
                f"[region_voter] Top votes: "
                + ", ".join(
                    f"{code}({votes/total_votes:.0%})" for code, votes in ranked[:3]
                )
            )

        # ── Precise module agreement hit ─────────────────────────────────
        # If 2+ precise modules agree within 200km, emit an agreement hit
        precise_centroids: list[tuple[str, float, float, float]] = []
        for mod_name, hit_list in all_hits.items():
            if not hit_list or mod_name not in _PRECISE_MODULES:
                continue
            mod_weight = config.modules.get(mod_name, None)
            w = mod_weight.weight if mod_weight else 1.0
            w_sum = sum(h.confidence for h in hit_list)
            if w_sum == 0:
                continue
            avg_lat = sum(h.confidence * h.lat for h in hit_list) / w_sum
            avg_lon = sum(h.confidence * h.lon for h in hit_list) / w_sum
            precise_centroids.append((mod_name, avg_lat, avg_lon, w))

        # Check pairwise agreement
        if len(precise_centroids) >= 2:
            for i in range(len(precise_centroids)):
                for j in range(i + 1, len(precise_centroids)):
                    n1, lat1, lon1, w1 = precise_centroids[i]
                    n2, lat2, lon2, w2 = precise_centroids[j]
                    d = _hav_km(lat1, lon1, lat2, lon2)
                    if d < 200:
                        # Agreement — weighted centroid of these two modules
                        tw = w1 + w2
                        agree_lat = (lat1 * w1 + lat2 * w2) / tw
                        agree_lon = (lon1 * w1 + lon2 * w2) / tw
                        agree_conf = min(0.9, 0.5 + 0.2 * (w1 + w2) / 10.0)
                        hits.append(ModuleHit(
                            module="region_voter",
                            lat=agree_lat,
                            lon=agree_lon,
                            confidence=agree_conf,
                            sigma_km=100.0,
                            metadata={
                                "type": "precise_agreement",
                                "modules": [n1, n2],
                                "distance_km": d,
                            },
                        ))
                        break  # one agreement hit is enough
                else:
                    continue
                break

        return hits
