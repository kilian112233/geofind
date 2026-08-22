"""Text Geocoder module - Nominatim + Overpass API for OCR text intersection.

Takes text extracted by OCR and geocodes it via:
1. Nominatim (structured search) - for addresses, street names, cities
2. Overpass API (full-text) - for named features (buildings, monuments, etc.)

Requires internet access and user consent via config.online_geocoding flag.
"""

from __future__ import annotations

import logging
import math
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

from geofind.core.candidate import ModuleHit
from geofind.core.config import PipelineConfig
from geofind.modules.base import BaseModule

logger = logging.getLogger(__name__)

# --- Rate limiter ---

_last_request_time: float = 0.0


def _rate_limit(min_interval_s: float = 1.1) -> None:
    """Simple global rate limiter for API calls."""
    global _last_request_time
    now = time.monotonic()
    elapsed = now - _last_request_time
    if elapsed < min_interval_s:
        time.sleep(min_interval_s - elapsed)
    _last_request_time = time.monotonic()


# --- Text cleaning ---

_QUERY_BLACKLIST: set[str] = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "has", "his", "how", "its", "may",
    "new", "now", "old", "see", "way", "who", "did", "get", "let", "say",
    "she", "too", "use", "day", "got", "set", "run", "try", "ask",
    "own", "put", "end", "far", "hand", "high", "keep", "last",
    "long", "make", "many", "most", "name", "only", "over", "such",
    "take", "time", "very", "when", "come", "could", "been", "call",
    "first", "give", "them", "than", "back", "were", "that", "this",
    "with", "have", "from", "they", "said", "each", "which",
    "their", "will", "other", "about", "then", "would",
    "like", "so", "just", "know", "people", "into",
    "year", "your", "good", "some", "think", "also",
    "after", "two", "how", "work", "well", "even", "want",
    "because", "any", "these", "find", "here", "thing",
    "great", "between", "need", "large", "small",
    "stop", "exit", "entrance", "open", "close", "parking", "speed",
    "zone", "area", "info", "free", "wifi", "hotel",
    "restaurant", "cafe", "bar", "pub", "shop", "store", "market",
    "taxi", "bus", "train", "metro", "airport", "hospital", "police",
    "fuel", "gas", "petrol", "atm", "bank", "pharmacy", "school",
    "university", "museum", "church", "temple", "mosque", "castle",
    "tower", "bridge", "park", "garden", "square", "monument",
    "phone", "mobile", "call", "number", "email", "website",
    "street", "road", "avenue", "boulevard", "lane", "drive",
    "north", "south", "east", "west", "central", "downtown",
    "www", "com", "org", "net", "http", "https",
    "photo", "image", "hello", "welcome", "password",
    "video", "audio", "file", "folder", "logo", "icon",
    "app", "software", "code", "data", "system",
    "hot", "cold", "fast", "slow", "good", "bad", "best", "top",
    "low", "live", "love", "sun", "moon", "star", "rain",
}

_MIN_TEXT_LENGTH = 3


def _clean_for_query(text: str) -> str:
    """Clean OCR text into a geocodable query string."""
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"\b\d\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\w\s\-.,]", "", text)
    return text.strip()


def _extract_text_blocks(full_text: str) -> list[str]:
    """Extract meaningful text blocks from OCR output.

    Returns individual words/phrases that might be geocodable place names.
    Includes a garbage filter: if >60% of characters are non-alpha or the text
    looks random, return early to avoid wasting API calls.
    """
    # --- Garbage OCR detection ---
    # Count alphabetic vs total characters
    alpha_chars = sum(1 for c in full_text if c.isalpha())
    total_chars = len(full_text.replace(" ", ""))
    if total_chars > 0:
        alpha_ratio = alpha_chars / total_chars
    else:
        return []

    # If text is mostly non-alphabetic, it's garbled
    if alpha_ratio < 0.5:
        return []

    words = full_text.split()

    # Check for random/garbled patterns: excessive consonant clusters
    real_words = 0
    for w in words:
        w_clean = re.sub(r"[^a-zA-Z]", "", w).lower()
        if len(w_clean) < 3:
            continue
        # Simple heuristic: words with vowels and consonants mixed are more likely real
        has_vowel = any(c in "aeiou" for c in w_clean)
        consonants = sum(1 for c in w_clean if c not in "aeiou")
        consonant_ratio = consonants / len(w_clean) if len(w_clean) > 0 else 1.0
        # Real words rarely have >70% consonants (except abbreviations)
        if has_vowel and consonant_ratio < 0.75:
            real_words += 1

    # If very few words look real, text is garbled
    if len(words) > 3 and real_words < 2:
        return []

    meaningful = [
        w for w in words
        if len(w) >= _MIN_TEXT_LENGTH
        and w.lower() not in _QUERY_BLACKLIST
        and not w.isdigit()
    ]

    blocks: list[str] = []

    # Individual meaningful words (longer = more likely place names)
    for w in meaningful:
        if len(w) >= 4:
            blocks.append(w)

    # Consecutive pairs
    for i in range(len(meaningful) - 1):
        pair = f"{meaningful[i]} {meaningful[i+1]}"
        blocks.append(pair)

    # Full text (if not too long)
    cleaned = _clean_for_query(full_text)
    if 3 < len(cleaned) < 100:
        blocks.append(cleaned)

    # Deduplicate
    seen: set[str] = set()
    unique: list[str] = []
    for b in blocks:
        key = b.lower().strip()
        if key not in seen and len(key) >= _MIN_TEXT_LENGTH:
            seen.add(key)
            unique.append(b)

    return unique


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in km."""
    lat1r, lon1r = math.radians(lat1), math.radians(lon1)
    lat2r, lon2r = math.radians(lat2), math.radians(lon2)
    dl = lat2r - lat1r
    dn = lon2r - lon1r
    a = (math.sin(dl / 2) ** 2
         + math.cos(lat1r) * math.cos(lat2r) * math.sin(dn / 2) ** 2)
    return 2 * 6371 * math.asin(math.sqrt(min(a, 1.0)))


class TextGeocoderModule(BaseModule):
    """Geocode OCR text blocks via Nominatim and Overpass API.

    This module runs after ocr_text and uses the extracted text to find
    exact geographic locations. It emits high-confidence, low-sigma hits
    when multiple text blocks converge on the same location.
    """

    name = "text_geocoder"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)
        self._nominatim_url = "https://nominatim.openstreetmap.org"
        self._overpass_url = "https://overpass-api.de/api/interpreter"
        self._session = None

    def is_available(self) -> bool:
        if not self.config.online_geocoding:
            return False
        try:
            import requests  # noqa: F401
            return True
        except ImportError:
            return False

    def prepare(self) -> None:
        import requests as _requests
        self._session = _requests.Session()
        self._session.headers.update({
            "User-Agent": "geofind/1.0 (geolocation research)",
            "Accept": "application/json",
        })
        super().prepare()

    def detect(
        self,
        media_path: Path,
        *,
        frames: list[Any] | None = None,
        audio_path: Path | None = None,
    ) -> list[ModuleHit]:
        if not self._ready or self._session is None:
            return []

        # Extract text from image using shared OCR
        full_text = self._extract_ocr_text(media_path, frames)
        if not full_text:
            return []

        self._log(f"OCR text for geocoding: {full_text[:100]}...")

        blocks = _extract_text_blocks(full_text)
        if not blocks:
            self._log("No geocodable text blocks found")
            return []

        self._log(f"Geocoding {len(blocks)} text blocks")

        # Phase 1: Nominatim
        nominatim_hits = self._nominatim_batch(blocks)

        # Phase 2: Overpass
        overpass_hits = self._overpass_batch(blocks)

        # Combine and score
        all_hits = self._combine_hits(nominatim_hits, overpass_hits)

        return all_hits

    def _extract_ocr_text(
        self, media_path: Path, frames: list[Any] | None
    ) -> str:
        """Extract text from image using the shared cached OCR pipeline."""
        image = self._get_image(media_path, frames)
        if image is None:
            return ""

        try:
            from geofind.utils.models import extract_ocr_text_cached
            return extract_ocr_text_cached(image)
        except Exception as e:
            self._log(f"OCR extraction failed: {e}", logging.WARNING)
            return ""

    def _nominatim_batch(self, blocks: list[str]) -> list[dict[str, Any]]:
        """Query Nominatim for each text block."""
        results: list[dict[str, Any]] = []

        for block in blocks[:6]:  # Max 6 Nominatim queries to stay under 10s
            cleaned = _clean_for_query(block)
            if len(cleaned) < _MIN_TEXT_LENGTH:
                continue

            try:
                _rate_limit(self.config.geocoding_rate_limit_s)
                resp = self._session.get(
                    f"{self._nominatim_url}/search",
                    params={
                        "q": cleaned,
                        "format": "json",
                        "limit": 3,
                        "addressdetails": 1,
                        "accept-language": "en",
                    },
                    timeout=self.config.nominatim_timeout_s,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data:
                        lat = float(item.get("lat", 0))
                        lon = float(item.get("lon", 0))
                        importance = float(item.get("importance", 0.5))
                        place_type = item.get("type", "")
                        display = item.get("display_name", "")

                        if lat == 0 and lon == 0:
                            continue

                        conf = min(0.9, 0.4 + importance * 0.5)
                        if cleaned.lower() in display.lower():
                            conf = min(0.95, conf * 1.3)

                        results.append({
                            "lat": lat,
                            "lon": lon,
                            "confidence": conf,
                            "query": block,
                            "display": display[:200],
                            "type": place_type,
                            "source": "nominatim",
                        })
                        self._log(
                            f"Nominatim: '{cleaned}' -> "
                            f"({lat:.4f}, {lon:.4f}) "
                            f"[{place_type}, conf={conf:.2f}]"
                        )
                elif resp.status_code == 429:
                    self._log("Nominatim rate limited, backing off", logging.WARNING)
                    time.sleep(2.0)
            except Exception as e:
                self._log(
                    f"Nominatim failed for '{cleaned}': {e}", logging.WARNING
                )

        return results

    def _overpass_batch(self, blocks: list[str]) -> list[dict[str, Any]]:
        """Query Overpass API for named features matching text blocks."""
        results: list[dict[str, Any]] = []

        name_filters = []
        for block in blocks[:3]:  # Max 3 Overpass name filters
            cleaned = _clean_for_query(block)
            if len(cleaned) < 3:
                continue
            name_filters.append(
                f'["name"~"{re.escape(cleaned)}",i]'
            )

        if not name_filters:
            return results

        query_parts = []
        for filt in name_filters[:3]:
            query_parts.append(f"node{filt}(global);")
            query_parts.append(f"way{filt}(global);")

        query = (
            "[out:json][timeout:8];\n(\n  "
            + "\n  ".join(query_parts)
            + "\n);\nout center 20;"
        )

        try:
            _rate_limit(self.config.geocoding_rate_limit_s)
            resp = self._session.post(
                self._overpass_url,
                data={"data": query},
                timeout=self.config.overpass_timeout_s,
            )
            if resp.status_code == 200:
                data = resp.json()
                elements = data.get("elements", [])

                for elem in elements[:20]:
                    lat = elem.get("lat") or elem.get("center", {}).get("lat")
                    lon = elem.get("lon") or elem.get("center", {}).get("lon")
                    tags = elem.get("tags", {})
                    name = tags.get("name", "")

                    if lat is None or lon is None:
                        continue

                    conf = 0.6
                    if name:
                        conf = min(0.85, 0.5 + len(name) * 0.01)

                    results.append({
                        "lat": float(lat),
                        "lon": float(lon),
                        "confidence": conf,
                        "query": name,
                        "display": name,
                        "type": tags.get(
                            "amenity", tags.get("building", "unknown")
                        ),
                        "source": "overpass",
                    })
                    self._log(
                        f"Overpass: {name} -> ({lat:.4f}, {lon:.4f})"
                    )
            elif resp.status_code == 429:
                self._log("Overpass rate limited", logging.WARNING)
                time.sleep(3.0)
        except Exception as e:
            self._log(f"Overpass failed: {e}", logging.WARNING)

        return results

    def _combine_hits(
        self,
        nominatim_hits: list[dict[str, Any]],
        overpass_hits: list[dict[str, Any]],
    ) -> list[ModuleHit]:
        """Combine and deduplicate hits from both sources.

        When multiple text blocks converge on the same location,
        boost the confidence significantly.
        """
        all_raw = nominatim_hits + overpass_hits
        if not all_raw:
            return []

        # Cluster nearby hits (within 5km)
        clusters: list[list[dict[str, Any]]] = []
        used: set[int] = set()

        for i, h1 in enumerate(all_raw):
            if i in used:
                continue
            cluster = [h1]
            used.add(i)
            for j, h2 in enumerate(all_raw):
                if j in used:
                    continue
                dist = _haversine_km(h1["lat"], h1["lon"], h2["lat"], h2["lon"])
                if dist < 5.0:
                    cluster.append(h2)
                    used.add(j)
            clusters.append(cluster)

        # Emit ModuleHit for each cluster
        hits: list[ModuleHit] = []
        for cluster in clusters:
            total_conf = sum(h["confidence"] for h in cluster)
            if total_conf == 0:
                continue

            avg_lat = sum(
                h["lat"] * h["confidence"] for h in cluster
            ) / total_conf
            avg_lon = sum(
                h["lon"] * h["confidence"] for h in cluster
            ) / total_conf

            # Convergence boost
            n_sources = len(set(h["query"] for h in cluster))
            convergence_boost = min(1.5, 1.0 + (n_sources - 1) * 0.15)

            has_nominatim = any(h["source"] == "nominatim" for h in cluster)
            has_overpass = any(h["source"] == "overpass" for h in cluster)
            cross_boost = 1.15 if (has_nominatim and has_overpass) else 1.0

            base_conf = max(h["confidence"] for h in cluster)
            final_conf = min(0.95, base_conf * convergence_boost * cross_boost)

            # Precision sigma
            if has_nominatim and final_conf > 0.7:
                sigma = 0.5
            elif has_overpass and final_conf > 0.6:
                sigma = 2.0
            else:
                sigma = 5.0

            best_display = max(
                (h.get("display", "") for h in cluster), key=len
            )

            hits.append(self._make_hit(
                avg_lat, avg_lon, final_conf,
                sigma_km=sigma,
                source="+".join(sorted(set(h["source"] for h in cluster))),
                query_hits=n_sources,
                display_name=best_display[:200],
            ))
            self._log(
                f"Geocoder hit: ({avg_lat:.4f}, {avg_lon:.4f}) "
                f"conf={final_conf:.2f} sigma={sigma:.1f}km "
                f"sources={n_sources} size={len(cluster)}"
            )

        hits.sort(key=lambda h: h.confidence, reverse=True)
        return hits[:10]

    def _get_image(
        self, media_path: Path, frames: list[Any] | None
    ) -> Any | None:
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
