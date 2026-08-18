#!/usr/bin/env python3
"""Image download and synthetic image generator for geofind test suite.

Downloads test images from Wikimedia Commons with caching, and can generate
synthetic EXIF-embedded test images as a fallback.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("image_grabber")

DATASET_PATH = Path("W:/geofind/dev/test_dataset.json")
IMAGES_DIR = Path("W:/geofind/dev/test_images")


def load_dataset(path: Path = DATASET_PATH) -> dict[str, Any]:
    """Load the test dataset JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def download_image(
    url: str,
    dest: Path,
    timeout: int = 30,
    retries: int = 2,
    user_agent: str = "geofind-test-suite/1.0",
) -> bool:
    """Download a single image with retries. Returns True on success."""
    import requests  # lazy import

    if dest.exists() and dest.stat().st_size > 1024:
        logger.info(f"  Cached: {dest.name}")
        return True

    for attempt in range(1, retries + 1):
        try:
            logger.info(f"  Downloading (attempt {attempt}): {url[:80]}...")
            resp = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": user_agent},
                stream=True,
            )
            resp.raise_for_status()

            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            size_kb = dest.stat().st_size / 1024
            if size_kb < 1:
                logger.warning(f"  Suspiciously small: {size_kb:.1f} KB")
                dest.unlink(missing_ok=True)
                continue

            logger.info(f"  Saved: {dest.name} ({size_kb:.1f} KB)")
            return True

        except requests.exceptions.RequestException as e:
            logger.warning(f"  Attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(1.5 * attempt)

        except Exception as e:
            logger.error(f"  Unexpected error: {e}")
            break

    logger.error(f"  FAILED: {url[:80]}")
    return False


def download_all(dataset: dict[str, Any], force: bool = False) -> dict[str, bool]:
    """Download all images from the dataset. Returns id->success map."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, bool] = {}

    cases = dataset["test_cases"]
    total = len(cases)
    success = 0
    failed = 0

    for i, tc in enumerate(cases, 1):
        tid = tc["id"]
        url = tc.get("image_url", "")

        if not url:
            logger.warning(f"[{i}/{total}] {tid}: no image_url, skipping")
            results[tid] = False
            failed += 1
            continue

        dest = IMAGES_DIR / f"{tid}.jpg"

        if force and dest.exists():
            dest.unlink()

        ok = download_image(url, dest)
        results[tid] = ok
        if ok:
            success += 1
        else:
            failed += 1
        print(f"  [{i}/{total}] {tid}: {'OK' if ok else 'FAILED'}")

    print(f"\nDone: {success} downloaded, {failed} failed out of {total}")
    return results


def generate_synthetic_image(
    lat: float,
    lon: float,
    output_path: Path,
    width: int = 640,
    height: int = 480,
    seed: int | None = None,
    text_label: str = "",
) -> bool:
    """Generate a synthetic test image with embedded EXIF GPS data.

    Creates a simple geometric pattern image with EXIF metadata containing
    the specified GPS coordinates. Requires piexif and Pillow.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        logger.error("Pillow not installed: pip install Pillow")
        return False

    try:
        import piexif
        from piexif import TYPES
    except ImportError:
        logger.warning("piexif not installed: pip install piexif")
        logger.warning("Falling back to basic image without EXIF GPS")
        piexif = None

    if seed is None:
        seed = int(hashlib.md5(f"{lat},{lon}".encode()).hexdigest()[:8], 16)

    rng = __import__("random").Random(seed)

    # Create gradient background
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    r1 = rng.randint(30, 200)
    g1 = rng.randint(30, 200)
    b1 = rng.randint(30, 200)
    r2 = rng.randint(30, 200)
    g2 = rng.randint(30, 200)
    b2 = rng.randint(30, 200)

    for y in range(height):
        t = y / height
        r = int(r1 * (1 - t) + r2 * t)
        g = int(g1 * (1 - t) + g2 * t)
        b = int(b1 * (1 - t) + b2 * t)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Draw some random shapes for visual variety
    for _ in range(rng.randint(3, 8)):
        x0 = rng.randint(0, width - 50)
        y0 = rng.randint(0, height - 50)
        x1 = x0 + rng.randint(20, 150)
        y1 = y0 + rng.randint(20, 150)
        fill = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        draw.rectangle([x0, y0, x1, y1], fill=fill, outline=None)

    # Draw circles
    for _ in range(rng.randint(2, 5)):
        cx = rng.randint(0, width)
        cy = rng.randint(0, height)
        r = rng.randint(10, 60)
        fill = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)

    # Add text label if provided
    if text_label:
        try:
            draw.text(
                (width // 4, height - 40),
                text_label,
                fill=(255, 255, 255),
            )
        except Exception:
            pass

    # Embed EXIF GPS if piexif available
    exif_dict: dict[str, Any] | None = None
    if piexif is not None:
        try:
            def _dms(value: float) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
                """Convert decimal degrees to EXIF DMS format."""
                d = abs(value)
                deg = int(d)
                min_float = (d - deg) * 60
                minute = int(min_float)
                sec = int((min_float - minute) * 60 * 10000)
                return ((deg, 1), (minute, 1), (sec, 10000))

            lat_dms = _dms(lat)
            lon_dms = _dms(lon)

            gps_ifd: dict[bytes, Any] = {
                piexif.GPSIFD.GPSLatitudeRef: b"N" if lat >= 0 else b"S",
                piexif.GPSIFD.GPSLatitude: lat_dms,
                piexif.GPSIFD.GPSLongitudeRef: b"E" if lon >= 0 else b"W",
                piexif.GPSIFD.GPSLongitude: lon_dms,
                piexif.GPSIFD.GPSVersionID: (2, 3, 0, 0),
            }

            zeroth_ifd: dict[bytes, Any] = {
                piexif.ImageIFD.Make: b"geofind-test",
                piexif.ImageIFD.Software: b"synthetic-test-suite",
            }

            exif_dict = {"0th": zeroth_ifd, "GPS": gps_ifd}

        except Exception as e:
            logger.warning(f"  EXIF encoding failed: {e}")
            exif_dict = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "JPEG", quality=92)

    # Write EXIF separately if we have it
    if exif_dict is not None:
        try:
            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, str(output_path))
        except Exception as e:
            logger.warning(f"  EXIF insert failed: {e}")

    return True


# Synthetic dataset: 40 locations with coordinates and labels
SYNTHETIC_LOCATIONS: list[dict[str, Any]] = [
    {"id": "syn_tokyo", "lat": 35.6762, "lon": 139.6503, "label": "Tokyo, Japan"},
    {"id": "syn_paris", "lat": 48.8566, "lon": 2.3522, "label": "Paris, France"},
    {"id": "syn_nyc", "lat": 40.7128, "lon": -74.0060, "label": "New York, USA"},
    {"id": "syn_london", "lat": 51.5074, "lon": -0.1278, "label": "London, UK"},
    {"id": "syn_sydney", "lat": -33.8688, "lon": 151.2093, "label": "Sydney, Australia"},
    {"id": "syn_dubai", "lat": 25.2048, "lon": 55.2708, "label": "Dubai, UAE"},
    {"id": "syn_mumbai", "lat": 19.0760, "lon": 72.8777, "label": "Mumbai, India"},
    {"id": "syn_beijing", "lat": 39.9042, "lon": 116.4074, "label": "Beijing, China"},
    {"id": "syn_seoul", "lat": 37.5665, "lon": 126.9780, "label": "Seoul, South Korea"},
    {"id": "syn_rio", "lat": -22.9068, "lon": -43.1729, "label": "Rio de Janeiro, Brazil"},
    {"id": "syn_berlin", "lat": 52.5200, "lon": 13.4050, "label": "Berlin, Germany"},
    {"id": "syn_rome", "lat": 41.9028, "lon": 12.4964, "label": "Rome, Italy"},
    {"id": "syn_barcelona", "lat": 41.3874, "lon": 2.1686, "label": "Barcelona, Spain"},
    {"id": "syn_moscow", "lat": 55.7558, "lon": 37.6173, "label": "Moscow, Russia"},
    {"id": "syn_mexico_city", "lat": 19.4326, "lon": -99.1332, "label": "Mexico City, Mexico"},
    {"id": "syn_buenos_aires", "lat": -34.6037, "lon": -58.3816, "label": "Buenos Aires, Argentina"},
    {"id": "syn_cairo", "lat": 30.0444, "lon": 31.2357, "label": "Cairo, Egypt"},
    {"id": "syn_cape_town", "lat": -33.9249, "lon": 18.4241, "label": "Cape Town, South Africa"},
    {"id": "syn_nairobi", "lat": -1.2921, "lon": 36.8219, "label": "Nairobi, Kenya"},
    {"id": "syn_auckland", "lat": -36.8485, "lon": 174.7633, "label": "Auckland, New Zealand"},
    {"id": "syn_singapore", "lat": 1.3521, "lon": 103.8198, "label": "Singapore"},
    {"id": "syn_bangkok", "lat": 13.7563, "lon": 100.5018, "label": "Bangkok, Thailand"},
    {"id": "syn_toronto", "lat": 43.6532, "lon": -79.3832, "label": "Toronto, Canada"},
    {"id": "syn_vancouver", "lat": 49.2827, "lon": -123.1207, "label": "Vancouver, Canada"},
    {"id": "syn_lima", "lat": -12.0464, "lon": -77.0428, "label": "Lima, Peru"},
    {"id": "syn_santiago", "lat": -33.4489, "lon": -70.6693, "label": "Santiago, Chile"},
    {"id": "syn_stockholm", "lat": 59.3293, "lon": 18.0686, "label": "Stockholm, Sweden"},
    {"id": "syn_vienna", "lat": 48.2082, "lon": 16.3738, "label": "Vienna, Austria"},
    {"id": "syn_athens", "lat": 37.9838, "lon": 23.7275, "label": "Athens, Greece"},
    {"id": "syn_istanbul", "lat": 41.0082, "lon": 28.9784, "label": "Istanbul, Turkey"},
    {"id": "syn_lisbon", "lat": 38.7223, "lon": -9.1393, "label": "Lisbon, Portugal"},
    {"id": "syn_dublin", "lat": 53.3498, "lon": -6.2603, "label": "Dublin, Ireland"},
    {"id": "syn_helsinki", "lat": 60.1699, "lon": 24.9384, "label": "Helsinki, Finland"},
    {"id": "syn_warsaw", "lat": 52.2297, "lon": 21.0122, "label": "Warsaw, Poland"},
    {"id": "syn_prague", "lat": 50.0755, "lon": 14.4378, "label": "Prague, Czechia"},
    {"id": "syn_budapest", "lat": 47.4979, "lon": 19.0402, "label": "Budapest, Hungary"},
    {"id": "syn_reykjavik", "lat": 64.1466, "lon": -21.9426, "label": "Reykjavik, Iceland"},
    {"id": "syn_marrakech", "lat": 31.6295, "lon": -7.9811, "label": "Marrakech, Morocco"},
    {"id": "syn_luanda", "lat": -8.8390, "lon": 13.2894, "label": "Luanda, Angola"},
    {"id": "syn_la Paz", "lat": -16.5, "lon": -68.15, "label": "La Paz, Bolivia"},
]


def generate_synthetic_batch(
    output_dir: Path | None = None,
    locations: list[dict[str, Any]] | None = None,
) -> dict[str, bool]:
    """Generate a batch of synthetic test images."""
    if output_dir is None:
        output_dir = IMAGES_DIR / "synthetic"

    if locations is None:
        locations = SYNTHETIC_LOCATIONS

    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, bool] = {}
    total = len(locations)

    for i, loc in enumerate(locations, 1):
        tid = loc["id"]
        dest = output_dir / f"{tid}.jpg"
        ok = generate_synthetic_image(
            lat=loc["lat"],
            lon=loc["lon"],
            output_path=dest,
            text_label=loc.get("label", tid),
        )
        results[tid] = ok
        print(f"  [{i}/{total}] {tid}: {'OK' if ok else 'FAILED'}")

    success = sum(1 for v in results.values() if v)
    print(f"\nGenerated {success}/{total} synthetic images")
    return results


def create_synthetic_dataset(
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Create a test_dataset.json compatible dict from synthetic locations."""
    if output_path is None:
        output_path = DATASET_PATH.parent / "test_dataset_synthetic.json"

    categories = ["urban", "landmark", "nature", "coastal", "desert", "rural", "mountain"]

    dataset = {
        "version": "1.0",
        "description": "Synthetic test dataset — 40 generated images with embedded EXIF GPS",
        "created": "2026-08-18",
        "test_cases": [],
    }

    for i, loc in enumerate(SYNTHETIC_LOCATIONS):
        tc = {
            "id": loc["id"],
            "name": loc.get("label", loc["id"]),
            "expected_lat": loc["lat"],
            "expected_lon": loc["lon"],
            "category": categories[i % len(categories)],
            "continent": _guess_continent(loc["lat"], loc["lon"]),
            "image_url": "",
            "has_exif_gps": True,
            "difficulty": "easy",
        }
        dataset["test_cases"].append(tc)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    logger.info(f"Wrote synthetic dataset: {output_path}")
    return dataset


def _guess_continent(lat: float, lon: float) -> str:
    """Very rough continent guess from coordinates."""
    if lat > 60 and lon > -30 and lon < 180:
        return "europe"
    if lat > 15 and lon > -30 and lon < 60:
        return "africa"
    if lat > 15 and lon >= 60 and lon <= 180:
        return "asia"
    if lat > 15 and lon >= -180 and lon < -30:
        return "americas"
    if lat <= 15 and lon >= -85 and lon <= -30:
        return "americas"
    if lat <= 15 and lon > -30 and lon < 55:
        return "africa"
    if lat <= 15 and lon >= 55 and lon <= 180:
        return "asia"
    if lat < -60:
        return "antarctica"
    if lat < 0 and lon >= 100:
        return "oceania"
    return "unknown"


def main() -> None:
    """CLI entry point for image_grabber."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download test images or generate synthetic ones"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic EXIF-embedded images instead of downloading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to process",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=IMAGES_DIR,
        help="Output directory for images",
    )
    args = parser.parse_args()

    if args.synthetic:
        print("=== Generating Synthetic Test Images ===\n")
        results = generate_synthetic_batch(output_dir=args.output_dir)
        ok = sum(1 for v in results.values() if v)
        print(f"\n{ok} synthetic images ready")
    else:
        print("=== Downloading Test Images ===\n")
        dataset = load_dataset()
        cases = dataset["test_cases"]
        if args.limit:
            cases = cases[: args.limit]
            dataset["test_cases"] = cases

        results = download_all(dataset, force=args.force)
        ok = sum(1 for v in results.values() if v)
        print(f"\n{ok} images ready in {IMAGES_DIR}")


if __name__ == "__main__":
    main()
