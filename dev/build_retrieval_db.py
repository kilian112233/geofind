#!/usr/bin/env python3
"""Build a CLIP image retrieval database from geotagged Wikimedia images.

Downloads geotagged images (cities, buildings, photographs) via Wikidata
SPARQL, extracts L2-normalised CLIP ViT-L/14 embeddings using the shared
project model (geofind.utils.models.get_clip_shared), and saves them as a
compressed numpy database at W:/geofind/models/retrieval_db.npz.

Usage:
    python dev/build_retrieval_db.py [--force] [--limit N] [--regions]
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_retrieval_db")
logging.getLogger("httpx").setLevel(logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "models" / "retrieval_db.npz"

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "geofind-retrieval-db/1.0 (https://github.com/geofind; geofind-dev)"

REQUEST_DELAY = 1.1
SPARQL_RETRIES = 5
DOWNLOAD_RETRIES = 3
MAX_IMAGE_DIM = 640
NAME_MAX_LEN = 50


def _sparql_query(cls: str, include_subclasses: bool, limit: int = 1000) -> str:
    """Build a geotagged-image SPARQL query for one Wikidata class."""
    path = "/wdt:P279*" if include_subclasses else ""
    return f"""
SELECT ?item ?itemLabel ?image ?lat ?lon WHERE {{
  ?item wdt:P31{path} wd:{cls} .
  ?item wdt:P18 ?image .
  ?item p:P625 ?coord .
  ?coord psv:P625 ?coordNode .
  ?coordNode wikibase:geoLatitude ?lat .
  ?coordNode wikibase:geoLongitude ?lon .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
}} LIMIT {limit}
"""


SPARQL_QUERIES: list[tuple[str, str]] = [
    ("cities", _sparql_query("Q515", include_subclasses=True, limit=5000)),
    ("buildings", _sparql_query("Q811979", include_subclasses=False, limit=5000)),
    ("photographs", _sparql_query("Q125191", include_subclasses=False, limit=2000)),
]

# Regional queries targeting under-represented areas.
# Note: entity IDs corrected from the draft to match intent:
#   Ukraine = Q212 (Q805 is Yemen), Slovakia = Q214 (Q140 is "lion"),
#   Scandinavia uses Norway Q20 / Finland Q33 (Q175 was Sao Paulo).
REGIONAL_QUERIES: list[str] = [
    # Eastern Europe - buildings in Ukraine
    """
    SELECT ?item ?itemLabel ?image ?lat ?lon WHERE {
      ?item wdt:P31/wdt:P279* wd:Q811979 .
      ?item wdt:P17 wd:Q212 .     # Ukraine
      ?item wdt:P18 ?image .
      ?item p:P625 ?coord .
      ?coord psv:P625 ?coord_node .
      ?coord_node wikibase:geoLatitude ?lat .
      ?coord_node wikibase:geoLongitude ?lon .
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    } LIMIT 500
    """,
    # Buildings in Germany
    """
    SELECT ?item ?itemLabel ?image ?lat ?lon WHERE {
      ?item wdt:P31/wdt:P279* wd:Q811979 .
      ?item wdt:P17 wd:Q183 .     # Germany
      ?item wdt:P18 ?image .
      ?item p:P625 ?coord .
      ?coord psv:P625 ?coord_node .
      ?coord_node wikibase:geoLatitude ?lat .
      ?coord_node wikibase:geoLongitude ?lon .
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    } LIMIT 500
    """,
    # Buildings in France
    """
    SELECT ?item ?itemLabel ?image ?lat ?lon WHERE {
      ?item wdt:P31/wdt:P279* wd:Q811979 .
      ?item wdt:P17 wd:Q142 .     # France
      ?item wdt:P18 ?image .
      ?item p:P625 ?coord .
      ?coord psv:P625 ?coord_node .
      ?coord_node wikibase:geoLatitude ?lat .
      ?coord_node wikibase:geoLongitude ?lon .
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    } LIMIT 500
    """,
    # Buildings in Hungary, Romania, Bulgaria, Slovakia, Poland
    """
    SELECT ?item ?itemLabel ?image ?lat ?lon WHERE {
      VALUES ?country { wd:Q28 wd:Q218 wd:Q219 wd:Q214 wd:Q36 }
      ?item wdt:P31/wdt:P279* wd:Q811979 .
      ?item wdt:P17 ?country .
      ?item wdt:P18 ?image .
      ?item p:P625 ?coord .
      ?coord psv:P625 ?coord_node .
      ?coord_node wikibase:geoLatitude ?lat .
      ?coord_node wikibase:geoLongitude ?lon .
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    } LIMIT 500
    """,
    # Photographs in Scandinavia
    """
    SELECT ?item ?itemLabel ?image ?lat ?lon WHERE {
      VALUES ?country { wd:Q20 wd:Q33 wd:Q34 wd:Q35 }
      ?item wdt:P31 wd:Q125191 .
      ?item wdt:P17 ?country .
      ?item wdt:P18 ?image .
      ?item p:P625 ?coord .
      ?coord psv:P625 ?coord_node .
      ?coord_node wikibase:geoLatitude ?lat .
      ?coord_node wikibase:geoLongitude ?lon .
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    } LIMIT 500
    """,
]


def run_sparql_query(sparql: str, label: str) -> list[dict]:
    """Execute one Wikidata SPARQL query and return parsed candidate rows."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/sparql-results+json",
    }
    for attempt in range(1, SPARQL_RETRIES + 1):
        try:
            resp = requests.get(
                SPARQL_ENDPOINT,
                params={"query": sparql, "format": "json"},
                headers=headers,
                timeout=180,
            )
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = int(retry_after) if retry_after else min(30 * attempt, 120)
                logger.warning(f"SPARQL [{label}] rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            rows: list[dict] = []
            for binding in data.get("results", {}).get("bindings", []):
                try:
                    rows.append({
                        "item": binding["item"]["value"],
                        "name": binding["itemLabel"]["value"],
                        "image": binding["image"]["value"],
                        "lat": float(binding["lat"]["value"]),
                        "lon": float(binding["lon"]["value"]),
                    })
                except (KeyError, ValueError, TypeError):
                    continue
            logger.info(f"SPARQL [{label}]: {len(rows)} usable rows")
            return rows
        except requests.exceptions.RequestException as e:
            logger.warning(f"SPARQL [{label}] attempt {attempt} failed: {e}")
            if attempt < SPARQL_RETRIES:
                time.sleep(min(20 * attempt, 90))
        except Exception as e:
            logger.error(f"SPARQL [{label}] unexpected error: {e}")
            break
    return []


def collect_candidates(
    queries: list[tuple[str, str] | str],
) -> list[dict]:
    """Gather candidates from all queries, deduped by item, name and grid cell."""
    candidates: list[dict] = []
    seen_items: set[str] = set()
    seen_cells: set[tuple[int, int]] = set()
    seen_names: set[str] = set()

    for idx, query in enumerate(queries):
        if isinstance(query, tuple):
            label, sparql = query
        else:
            label, sparql = f"regional-{idx}", query
        rows = run_sparql_query(sparql, label)
        added = 0
        for row in rows:
            if row["item"] in seen_items:
                continue
            seen_items.add(row["item"])
            cell = (int(round(row["lat"] * 100)), int(round(row["lon"] * 100)))
            if cell in seen_cells:
                continue
            seen_cells.add(cell)
            name_key = row["name"].strip().lower()
            if name_key:
                if name_key in seen_names:
                    continue
                seen_names.add(name_key)
            candidates.append(row)
            added += 1
        logger.info(f"[{label}] {added} new unique candidates")
        time.sleep(REQUEST_DELAY)

    random.shuffle(candidates)
    return candidates


def to_download_url(image_value: str) -> str | None:
    """Convert a SPARQL image value to a Commons thumbnail download URL."""
    value = image_value.strip()
    filename: str | None = None

    if "Special:FilePath/" in value:
        filename = value.split("Special:FilePath/", 1)[1]
    elif value.startswith("File:"):
        filename = value[len("File:"):]
    elif value.startswith("http"):
        url = value.replace("http://", "https://")
        return url if "?" in url else f"{url}?width={MAX_IMAGE_DIM}"
    else:
        filename = value

    if not filename:
        return None
    filename = unquote(filename.split("?")[0]).replace(" ", "_")
    return (
        f"https://commons.wikimedia.org/wiki/Special:Redirect/file/"
        f"{filename}?width={MAX_IMAGE_DIM}"
    )


def download_image(url: str, dest: Path) -> bool:
    """Download an image with retries, honouring 429 rate limits, skipping 404s."""
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=30, stream=True)

            if resp.status_code == 404:
                logger.debug(f"404, skipping: {url[:90]}")
                return False

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = int(retry_after) if retry_after else min(30 * attempt, 120)
                logger.warning(f"Rate limited (429), waiting {wait}s...")
                time.sleep(wait)
                continue

            resp.raise_for_status()

            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)

            if dest.stat().st_size < 1024:
                logger.debug(f"Suspiciously small file: {dest.name}")
                dest.unlink(missing_ok=True)
                continue

            return True

        except requests.exceptions.RequestException as e:
            logger.debug(f"Download attempt {attempt} failed: {e}")
            if attempt < DOWNLOAD_RETRIES:
                time.sleep(min(5 * attempt, 30))
        except Exception as e:
            logger.error(f"Unexpected download error: {e}")
            break

    return False


def embed_image(path: Path, model, processor) -> np.ndarray:
    """Load an image, resize to max 640px, and return its L2-normalised CLIP vector."""
    import torch
    from PIL import Image

    with Image.open(path) as img:
        img.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM))
        img = img.convert("RGB")

    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    features = outputs if torch.is_tensor(outputs) else outputs.pooler_output
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().flatten().astype(np.float32)


def build_database(
    limit: int,
    queries: list[tuple[str, str] | str],
) -> tuple[int, int]:
    """Download candidates and extract features. Returns (entries, examined)."""
    from geofind.utils.models import get_clip_shared

    candidates = collect_candidates(queries)
    logger.info(f"{len(candidates)} unique grid-cell candidates collected")

    model, processor = get_clip_shared()

    all_features: list[np.ndarray] = []
    all_lats: list[float] = []
    all_lons: list[float] = []
    all_names: list[str] = []

    tmp_dir = Path(tempfile.mkdtemp(prefix="geofind_retrieval_"))
    examined = 0
    try:
        for cand in candidates:
            if len(all_features) >= limit:
                break
            examined += 1

            url = to_download_url(cand["image"])
            if not url:
                continue

            ext = Path(urlparse(url).path).suffix.lower() or ".jpg"
            tmp_path = tmp_dir / f"img_{examined}{ext}"
            try:
                time.sleep(REQUEST_DELAY)
                if not download_image(url, tmp_path):
                    continue

                feat = embed_image(tmp_path, model, processor)
                all_features.append(feat)
                all_lats.append(cand["lat"])
                all_lons.append(cand["lon"])
                all_names.append(cand["name"][:NAME_MAX_LEN])
            except Exception as e:
                logger.debug(f"Skipping '{cand['name'][:40]}': {e}")
                continue
            finally:
                tmp_path.unlink(missing_ok=True)

            if len(all_features) % 10 == 0:
                logger.info(
                    f"[{len(all_features)} entries] "
                    f"last=({cand['lat']:.4f}, {cand['lon']:.4f}) "
                    f"'{cand['name'][:40]}' | examined={examined}"
                )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not all_features:
        return 0, examined

    np.savez_compressed(
        OUTPUT_PATH,
        features=np.array(all_features, dtype=np.float32),
        lats=np.array(all_lats, dtype=np.float64),
        lons=np.array(all_lons, dtype=np.float64),
        names=np.array(all_names, dtype="U50"),
    )
    return len(all_features), examined


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build CLIP retrieval DB from geotagged Wikimedia images"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the database already exists",
    )
    parser.add_argument(
        "--regions",
        action="store_true",
        help="Run only the regional queries (incremental rebuild)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Max images to process (default: 10000)",
    )
    args = parser.parse_args()

    if OUTPUT_PATH.exists() and not args.force:
        logger.info(f"Database already exists: {OUTPUT_PATH} (use --force to rebuild)")
        return

    if args.regions:
        queries: list[tuple[str, str] | str] = REGIONAL_QUERIES
    else:
        queries = [*SPARQL_QUERIES, *REGIONAL_QUERIES]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    entries, examined = build_database(args.limit, queries)

    if entries == 0:
        logger.error("No entries collected; database not saved")
        sys.exit(1)

    size_mb = OUTPUT_PATH.stat().st_size / 1e6
    with np.load(OUTPUT_PATH) as db:
        lat_range = (float(db["lats"].min()), float(db["lats"].max()))
        lon_range = (float(db["lons"].min()), float(db["lons"].max()))

    logger.info(f"Saved {entries} entries ({examined} examined) to {OUTPUT_PATH}")
    logger.info(f"File size: {size_mb:.1f} MB | dims: {entries} x 768")
    logger.info(f"Lat range: {lat_range[0]:.2f}..{lat_range[1]:.2f} | "
                f"Lon range: {lon_range[0]:.2f}..{lon_range[1]:.2f}")


if __name__ == "__main__":
    main()
