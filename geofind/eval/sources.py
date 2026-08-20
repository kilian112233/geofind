"""Image sources for the evaluation framework.

Provides geotagged images from multiple sources with ground-truth coordinates.
All images can be EXIF-stripped before analysis.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

USER_AGENT = "geofind-eval/1.0 (https://github.com/geofind; geofind-eval@example.com)"
BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
CACHE_DIR = Path("W:/geofind/dev")


@dataclass
class EvalImage:
    """A test image with ground-truth coordinates."""
    id: str
    name: str
    lat: float
    lon: float
    image_url: str
    category: str = "unknown"
    continent: str = "unknown"
    difficulty: str = "random"
    source: str = "wikimedia"

    # Filled after download
    local_path: Path | None = None
    stripped_path: Path | None = None


class ImageSource:
    """Base class for image sources."""

    def fetch(self, count: int) -> list[EvalImage]:
        raise NotImplementedError

    def download(
        self, images: list[EvalImage], dest_dir: Path, strip_exif: bool = True
    ) -> list[EvalImage]:
        """Download images and optionally strip EXIF GPS."""
        import requests

        dest_dir.mkdir(parents=True, exist_ok=True)
        results: list[EvalImage] = []

        for i, img in enumerate(images):
            dest = dest_dir / f"{img.id}.jpg"

            if not dest.exists() or dest.stat().st_size < 1024:
                try:
                    # Rate limit: 1 req/sec for Wikimedia
                    if i > 0:
                        time.sleep(1.0)

                    resp = requests.get(
                        img.image_url,
                        timeout=30,
                        headers={"User-Agent": BROWSER_UA},
                        stream=True,
                        allow_redirects=True,
                    )
                    # Retry on rate limit
                    if resp.status_code == 429 or (resp.status_code == 403 and "Too many" in resp.text):
                        time.sleep(3.0)
                        resp = requests.get(
                            img.image_url,
                            timeout=30,
                            headers={"User-Agent": BROWSER_UA},
                            stream=True,
                            allow_redirects=True,
                        )
                    resp.raise_for_status()
                    with open(dest, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                except Exception as e:
                    logger.warning(f"Failed to download {img.id}: {e}")
                    continue

            img.local_path = dest

            if strip_exif:
                stripped = dest_dir / f"{img.id}_stripped.jpg"
                if not stripped.exists():
                    try:
                        _strip_exif_gps(dest, stripped)
                    except Exception as e:
                        logger.warning(f"EXIF strip failed for {img.id}: {e}")
                        stripped = dest
                img.stripped_path = stripped
            else:
                img.stripped_path = dest

            results.append(img)

        return results


class WikimediaSource(ImageSource):
    """Fetch geotagged images from Wikimedia Commons via Wikidata SPARQL."""

    def __init__(self, cache_path: Path | None = None):
        self.cache_path = cache_path or CACHE_DIR / "cached_random_images.json"

    def fetch(self, count: int) -> list[EvalImage]:
        import requests

        # Try cache first
        if self.cache_path.exists():
            try:
                cached = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if len(cached) >= count:
                    logger.info(f"Using {count} cached Wikimedia images")
                    return [self._from_cache(c) for c in cached[:count]]
            except (json.JSONDecodeError, KeyError):
                pass

        # SPARQL query for geotagged photographs
        sparql = """
SELECT ?item ?itemLabel ?lat ?lon ?image WHERE {
  ?item wdt:P31/wdt:P279* wd:Q125191 .
  ?item p:P625 ?coordStatement .
  ?coordStatement psv:P625 ?coord .
  ?coord wikibase:geoLatitude ?lat .
  ?coord wikibase:geoLongitude ?lon .
  ?item wdt:P18 ?image .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
} LIMIT 200
"""
        results: list[EvalImage] = []
        try:
            resp = requests.get(
                "https://query.wikidata.org/sparql",
                params={"query": sparql, "format": "json"},
                headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"},
                timeout=10,
            )
            resp.raise_for_status()
            for row in resp.json().get("results", {}).get("bindings", []):
                if len(results) >= count:
                    break
                try:
                    item_id = row["item"]["value"].split("/")[-1]
                    title = row.get("itemLabel", {}).get("value", item_id)
                    lat = float(row["lat"]["value"])
                    lon = float(row["lon"]["value"])
                    image_raw = row["image"]["value"]
                    # SPARQL returns full URLs like http://commons.wikimedia.org/wiki/Special:FilePath/Foo.jpg
                    # Extract just the filename
                    if "Special:FilePath/" in image_raw:
                        image_title = image_raw.split("Special:FilePath/")[-1]
                    elif "/" in image_raw:
                        image_title = image_raw.split("/")[-1]
                    else:
                        image_title = image_raw
                    # Strip any query params
                    image_title = image_title.split("?")[0]
                    # Decode any existing URL encoding, then re-encode properly
                    import urllib.parse
                    decoded = urllib.parse.unquote(image_title)
                    encoded_title = urllib.parse.quote(decoded, safe="()!~'*-.")
                    url = (
                        f"https://commons.wikimedia.org/w/index.php"
                        f"?title=Special:Redirect/file/{encoded_title}&width=1200"
                    )
                    safe_id = item_id.replace("Q", "q")
                    results.append(EvalImage(
                        id=f"wiki_{safe_id}",
                        name=title,
                        lat=lat,
                        lon=lon,
                        image_url=url,
                        source="wikimedia",
                    ))
                except (KeyError, ValueError, TypeError):
                    continue
        except Exception as e:
            logger.warning(f"SPARQL query failed: {e}")

        # Geosearch fallback
        if len(results) < count:
            results.extend(self._geosearch_fallback(
                count - len(results),
                existing_ids={r.id for r in results},
            ))

        # Cache
        try:
            cache_data = [
                {"id": r.id, "name": r.name, "url": r.image_url,
                 "lat": r.lat, "lon": r.lon}
                for r in results
            ]
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(cache_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass

        return results[:count]

    def _geosearch_fallback(
        self, count: int, existing_ids: set[str] | None = None
    ) -> list[EvalImage]:
        import requests

        if existing_ids is None:
            existing_ids = set()

        results: list[EvalImage] = []
        search_points = [
            (40.0, -74.0), (48.85, 2.35), (-33.87, 151.21),
            (35.68, 139.69), (-22.91, -43.17), (51.51, -0.13),
            (41.90, 12.50), (52.52, 13.41), (19.07, 72.88),
            (55.75, 37.62), (59.33, 18.07), (37.98, 23.73),
            (49.28, -123.12), (-34.60, -58.38), (30.04, 31.24),
        ]
        rng = random.Random()
        rng.shuffle(search_points)

        for lat, lon in search_points:
            if len(results) >= count:
                break
            try:
                resp = requests.get(
                    "https://commons.wikimedia.org/w/api.php",
                    params={
                        "action": "query",
                        "list": "geosearch",
                        "gscoord": f"{lat}|{lon}",
                        "gsradius": 10000,
                        "gslimit": 10,
                        "gsnamespace": 6,
                        "format": "json",
                    },
                    headers={"User-Agent": USER_AGENT},
                    timeout=20,
                )
                resp.raise_for_status()
                data = resp.json()
                titles_to_fetch = []
                items_by_title = {}
                for item in data.get("query", {}).get("geosearch", []):
                    title = item.get("title", "")
                    if not title:
                        continue
                    img_lat = item.get("lat", lat)
                    img_lon = item.get("lon", lon)
                    titles_to_fetch.append(title)
                    items_by_title[title] = (img_lat, img_lon)

                # Fetch image URLs in a batch
                if titles_to_fetch:
                    try:
                        info_resp = requests.get(
                            "https://commons.wikimedia.org/w/api.php",
                            params={
                                "action": "query",
                                "titles": "|".join(titles_to_fetch[:10]),
                                "prop": "imageinfo",
                                "iiprop": "url",
                                "iiurlwidth": 1200,
                                "format": "json",
                            },
                            headers={"User-Agent": USER_AGENT},
                            timeout=20,
                        )
                        info_resp.raise_for_status()
                        for page in info_resp.json().get("query", {}).get("pages", {}).values():
                            if len(results) >= count:
                                break
                            title = page.get("title", "")
                            imageinfo = page.get("imageinfo", [{}])[0]
                            url = imageinfo.get("thumburl") or imageinfo.get("url", "")
                            if not url or title not in items_by_title:
                                continue
                            fid = f"commons_{hashlib.md5(title.encode()).hexdigest()[:12]}"
                            if fid in existing_ids:
                                continue
                            existing_ids.add(fid)
                            img_lat, img_lon = items_by_title[title]
                            results.append(EvalImage(
                                id=fid,
                                name=title.replace("File:", "").replace("_", " ")[:60],
                                lat=img_lat,
                                lon=img_lon,
                                image_url=url,
                                source="wikimedia-geosearch",
                            ))
                    except Exception as e:
                        logger.debug(f"Image info fetch failed: {e}")
                time.sleep(0.5)
            except Exception as e:
                logger.debug(f"Geosearch failed for ({lat},{lon}): {e}")
                continue

        return results

    def _from_cache(self, c: dict[str, Any]) -> EvalImage:
        return EvalImage(
            id=c["id"],
            name=c.get("name", c["id"]),
            lat=c["lat"],
            lon=c["lon"],
            image_url=c.get("url", ""),
            source="wikimedia-cached",
        )


class LocalDirectorySource(ImageSource):
    """Load images from a local directory with a JSON manifest."""

    def __init__(self, manifest_path: Path, images_dir: Path | None = None):
        self.manifest_path = manifest_path
        self.images_dir = images_dir or manifest_path.parent / "test_images"

    def fetch(self, count: int) -> list[EvalImage]:
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cases = data.get("test_cases", [])
        results: list[EvalImage] = []
        for tc in cases:
            if len(results) >= count:
                break
            img_path = self.images_dir / f"{tc['id']}.jpg"
            if not img_path.exists():
                continue
            results.append(EvalImage(
                id=tc["id"],
                name=tc.get("name", tc["id"]),
                lat=tc["expected_lat"],
                lon=tc["expected_lon"],
                image_url="",
                category=tc.get("category", "unknown"),
                continent=tc.get("continent", "unknown"),
                difficulty=tc.get("difficulty", "unknown"),
                source="local",
                local_path=img_path,
                stripped_path=img_path,
            ))
        return results


def get_source(name: str, **kwargs: Any) -> ImageSource:
    """Factory for image sources."""
    if name == "wikimedia":
        return WikimediaSource(**kwargs)
    elif name == "local":
        return LocalDirectorySource(**kwargs)
    else:
        raise ValueError(f"Unknown source: {name}")


def _strip_exif_gps(src: Path, dst: Path) -> None:
    """Remove all EXIF data from an image."""
    img = Image.open(src)
    data = list(img.getdata())
    clean = Image.new(img.mode, img.size)
    clean.putdata(data)
    clean.save(dst, "JPEG", quality=95)
