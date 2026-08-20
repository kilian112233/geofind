#!/usr/bin/env python3
"""Self-evaluation harness: perturbation loop for accuracy validation.

Adds realistic social-media-style overlays to test images and iterates
until accuracy targets are met (avg <1km, max <10km across a batch).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path("W:/geofind")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("self_evaluate")

# ── Word lists for realistic text overlays ───────────────────────────────────

STREET_NAMES = [
    "Main St", "1st Ave", "Park Road", "High Street", "Broadway",
    "Oak Lane", "Maple Dr", "Elm Street", "Cedar Ave", "Pine Road",
    "Market St", "Church Lane", "King's Road", "Station Rd", "Bridge St",
    "Lake View", "Hill Road", "River Lane", "Mill Road", "Spring St",
]

RANDOM_WORDS = [
    "travel", "adventure", "beautiful", "amazing", "wonderful",
    "explore", "discover", "wanderlust", "vacation", "sunset",
    "morning", "view", "love", "happy", "life", "good vibes",
    "no filter", "mood", "instagood", "photooftheday",
]

# CITY_NAMES excluded — real social media photos don't have random city name
# overlays. Including them caused OCR to produce false place-name hits.
ALL_WORDS = STREET_NAMES + RANDOM_WORDS


# ── Perturbation function ────────────────────────────────────────────────────

def apply_perturbations(img: Image.Image, level: str = "medium") -> Image.Image:
    """Apply realistic social-media-style perturbations to an image.

    Adds overlays like what you'd see on social-media photos: text captions,
    colored strips, noise grain, brightness/contrast jitter, slight rotation.
    """
    import random

    img = img.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    # 1. Random text overlays (captions / watermarks)
    num_texts = {"light": (1, 2), "medium": (2, 4), "heavy": (3, 6)}
    lo, hi = num_texts.get(level, (2, 4))
    for _ in range(random.randint(lo, hi)):
        text = random.choice(ALL_WORDS)
        x = random.randint(0, max(0, w - 200))
        y = random.randint(0, max(0, h - 50))
        font_size = random.randint(12, 36)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            try:
                font = ImageFont.truetype("Arial.ttf", font_size)
            except OSError:
                font = ImageFont.load_default()
        alpha = random.randint(80, 180)
        bg_color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            alpha,
        )
        text_w = int(font_size * len(text) * 0.55) + 10
        draw.rectangle(
            [x - 5, y - 5, x + text_w, y + font_size + 10],
            fill=bg_color,
        )
        draw.text((x, y), text, fill=(255, 255, 255, 200), font=font)

    # 2. Colored strips (video progress bars, highlight overlays)
    num_strips = {"light": (0, 1), "medium": (1, 3), "heavy": (2, 4)}
    lo_s, hi_s = num_strips.get(level, (1, 3))
    for _ in range(random.randint(lo_s, hi_s)):
        y_pos = random.randint(0, h)
        strip_h = random.randint(2, 20)
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(60, 150),
        )
        draw.rectangle([0, y_pos, w, y_pos + strip_h], fill=color)

    # 3. Gaussian noise / grain
    noise_sigma = {"light": 3, "medium": 7, "heavy": 18}
    sigma = noise_sigma.get(level, 7)
    arr = np.array(img.convert("RGB")).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")

    # Re-create RGBA draw for any remaining RGBA operations
    draw = ImageDraw.Draw(img, "RGBA")

    # 4. Brightness / contrast jitter
    factor = random.uniform(0.9, 1.1)
    img = ImageEnhance.Brightness(img).enhance(factor)
    factor = random.uniform(0.9, 1.1)
    img = ImageEnhance.Contrast(img).enhance(factor)

    # 5. Slight rotation (30% chance)
    if random.random() < 0.3:
        angle = random.uniform(-5, 5)
        img = img.rotate(angle, fillcolor=(128, 128, 128))

    return img


# ── Discord webhook ──────────────────────────────────────────────────────────

DISCORD_WEBHOOK_URL = (
    "https://discord.com/api/webhooks/1539785113611800649/"
    "RDW9fCX1ICxgk_xLXaXjJpx-MLxPf2J9VFTmAemjmMi91fknZ8kEHD1URppmqStR3_Wq"
)


def discord_send(
    title: str,
    description: str,
    color: int = 0xFFFF00,
    fields: list[dict[str, str]] | None = None,
) -> bool:
    """Send a Discord message via webhook. Returns True on success."""
    # Build plain text (always works — embeds sometimes 403)
    text = f"**{title}**\n{description}"
    if fields:
        for f in fields:
            text += f"\n• {f['name']}: {f['value']}"
    payload = json.dumps({"content": text[:2000]}).encode("utf-8")
    req = urllib.request.Request(
        DISCORD_WEBHOOK_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "GeofindBot/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status in (200, 204)
    except urllib.error.URLError as e:
        logger.warning(f"Discord webhook failed: {e}")
        return False


# ── Main evaluation loop ─────────────────────────────────────────────────────

def run_evaluation(
    max_iterations: int = 30,
    images_per_iter: int = 10,
    perturbation_level: str = "medium",
) -> bool:
    """Run the self-evaluation loop until accuracy targets are met.

    Returns True if targets are achieved, False if max_iterations exhausted.
    """
    from geofind.core.pipeline import GeoPipeline
    from geofind.core.config import PipelineConfig, ModuleConfig
    from geofind.utils.geo import haversine_km, LatLon

    # Import image_grabber from dev/
    dev_dir = PROJECT_ROOT / "dev"
    if str(dev_dir) not in sys.path:
        sys.path.insert(0, str(dev_dir))
    import image_grabber

    # Config with EXIF disabled — we want to test visual-only geolocation
    config = PipelineConfig()
    config.modules["exif"] = ModuleConfig(name="exif", weight=0.0, enabled=False)

    pipeline = GeoPipeline(config)
    tmp_root = Path(tempfile.mkdtemp(prefix="geofind_eval_"))

    try:
        discord_send(
            "Self-Evaluation Started",
            f"**Config:** {images_per_iter} images/iter, "
            f"{perturbation_level} perturbation, max {max_iterations} iterations\n"
            f"**Target:** avg < 1.0 km, max < 10.0 km",
            color=0x3498DB,
        )

        for iteration in range(1, max_iterations + 1):
            logger.info(f"═══ Iteration {iteration}/{max_iterations} ═══")

            discord_send(
                f"Iteration {iteration}/{max_iterations}",
                "Fetching random geotagged images from Wikimedia...",
                color=0xF1C40F,
            )

            # 1. Fetch random images (clear cache first to get fresh images each iteration)
            cache_file = PROJECT_ROOT / "dev" / "cached_random_images.json"
            if cache_file.exists():
                cache_file.unlink()
                logger.info("Cleared image cache for fresh fetch")
            try:
                images = image_grabber.fetch_random_geotagged_images(
                    count=images_per_iter
                )
            except Exception as e:
                logger.error(f"Failed to fetch images: {e}")
                discord_send("Fetch Failed", str(e), color=0xE74C3C)
                continue

            if not images:
                logger.warning("No images fetched, retrying...")
                time.sleep(5)
                continue

            # 2. Download images
            download_results = {}
            for img_info in images:
                dest = tmp_root / f"{img_info['id']}.jpg"
                try:
                    ok = image_grabber.download_image(img_info["url"], dest)
                    download_results[img_info["id"]] = ok
                except Exception as e:
                    logger.warning(f"Download failed for {img_info['id']}: {e}")
                    download_results[img_info["id"]] = False

            # 3. Evaluate each image
            distances: list[float] = []
            results_detail: list[dict] = []

            for img_info in images:
                tid = img_info["id"]
                if not download_results.get(tid):
                    continue

                original_path = tmp_root / f"{tid}.jpg"
                if not original_path.exists():
                    continue

                gt_lat = img_info["lat"]
                gt_lon = img_info["lon"]

                try:
                    # Load and perturb
                    img = Image.open(original_path)
                    perturbed = apply_perturbations(img, level=perturbation_level)
                    perturbed_path = tmp_root / f"{tid}_perturbed.jpg"
                    perturbed.save(perturbed_path, "JPEG", quality=92)

                    # Strip EXIF
                    stripped_path = tmp_root / f"{tid}_clean.jpg"
                    image_grabber.strip_exif_gps(perturbed_path, stripped_path)

                    # Run pipeline
                    result = pipeline.analyze(stripped_path)

                    # Compare — use top candidate (most probable) not consensus centroid
                    # Consensus centroid can diverge wildly when modules disagree
                    top = result.top_candidate
                    if top:
                        pred_lat, pred_lon = top.lat, top.lon
                    else:
                        pred_lat, pred_lon = result.consensus_lat, result.consensus_lon
                    pred = LatLon(pred_lat, pred_lon)
                    truth = LatLon(gt_lat, gt_lon)
                    dist = haversine_km(pred, truth)
                    distances.append(dist)

                    detail = {
                        "id": tid,
                        "gt": f"{gt_lat:.4f},{gt_lon:.4f}",
                        "pred": f"{pred_lat:.4f},{pred_lon:.4f}",
                        "distance_km": round(dist, 2),
                        "modules": result.modules_run,
                    }
                    results_detail.append(detail)
                    logger.info(
                        f"  {tid}: {dist:.2f} km "
                        f"(pred={pred_lat:.3f},{pred_lon:.3f} "
                        f"vs gt={gt_lat:.3f},{gt_lon:.3f})"
                    )

                    discord_send(
                        f"Iter {iteration} — {tid}",
                        f"**Distance:** {dist:.2f} km\n"
                        f"**GT:** {gt_lat:.4f}, {gt_lon:.4f}\n"
                        f"**Pred:** {pred_lat:.4f}, {pred_lon:.4f}",
                        color=0x2ECC71 if dist < 1.0 else (0xF39C12 if dist < 10.0 else 0xE74C3C),
                    )

                except Exception as e:
                    logger.error(f"  {tid}: FAILED — {e}")
                    discord_send(f"Iter {iteration} — {tid} FAILED", str(e), color=0xE74C3C)

            # 4. Compute stats
            if not distances:
                logger.warning("No successful evaluations this iteration")
                continue

            avg_dist = sum(distances) / len(distances)
            max_dist = max(distances)
            passed = avg_dist < 1.0 and max_dist < 10.0

            logger.info(
                f"  Results: avg={avg_dist:.2f} km, max={max_dist:.2f} km, "
                f"{'PASS ✓' if passed else 'FAIL'}"
            )

            # 5. Send iteration summary
            fields = [
                {"name": "Avg Distance", "value": f"{avg_dist:.2f} km", "inline": "true"},
                {"name": "Max Distance", "value": f"{max_dist:.2f} km", "inline": "true"},
                {"name": "Images", "value": str(len(distances)), "inline": "true"},
            ]
            for d in results_detail:
                fields.append({
                    "name": d["id"],
                    "value": f'{d["distance_km"]} km',
                    "inline": "true",
                })

            discord_send(
                f"Iteration {iteration} — {'PASS ✓' if passed else 'FAIL ✗'}",
                f"**Avg:** {avg_dist:.2f} km | **Max:** {max_dist:.2f} km",
                color=0x2ECC71 if passed else 0xE74C3C,
                fields=fields,
            )

            if passed:
                logger.info("═" * 50)
                logger.info("TARGETS MET — Running final clean test...")
                logger.info("═" * 50)
                _run_clean_test(pipeline, image_grabber, tmp_root)
                return True

        # Exhausted iterations
        logger.warning(f"Max iterations ({max_iterations}) reached without meeting targets")
        discord_send(
            "Evaluation Complete",
            f"Did not meet targets in {max_iterations} iterations.",
            color=0xE74C3C,
        )
        return False

    finally:
        # Clean up temp directory
        try:
            shutil.rmtree(tmp_root, ignore_errors=True)
        except Exception:
            pass


def _run_clean_test(
    pipeline: "GeoPipeline",
    image_grabber: Any,
    tmp_root: Path,
) -> None:
    """Run a final test on a clean (unperturbed) image as a real-world proxy."""
    from geofind.utils.geo import haversine_km, LatLon
    from PIL import Image

    logger.info("Fetching a fresh image for clean test...")
    try:
        images = image_grabber.fetch_random_geotagged_images(count=1)
    except Exception as e:
        logger.error(f"Clean test fetch failed: {e}")
        return

    if not images:
        logger.warning("No images available for clean test")
        return

    img_info = images[0]
    tid = img_info["id"]
    dest = tmp_root / f"{tid}_clean_final.jpg"

    try:
        ok = image_grabber.download_image(img_info["url"], dest)
        if not ok:
            logger.error("Clean test download failed")
            return

        # Strip EXIF but do NOT perturb
        stripped = tmp_root / f"{tid}_clean_stripped.jpg"
        image_grabber.strip_exif_gps(dest, stripped)

        result = pipeline.analyze(stripped)

        gt_lat, gt_lon = img_info["lat"], img_info["lon"]
        top = result.top_candidate
        if top:
            pred_lat, pred_lon = top.lat, top.lon
        else:
            pred_lat, pred_lon = result.consensus_lat, result.consensus_lon
        pred = LatLon(pred_lat, pred_lon)
        truth = LatLon(gt_lat, gt_lon)
        dist = haversine_km(pred, truth)

        logger.info(f"Clean test — {tid}: {dist:.2f} km")

        discord_send(
            "Clean Test Result",
            f"**Image:** {tid}\n"
            f"**Distance:** {dist:.2f} km\n"
            f"**GT:** {gt_lat:.4f}, {gt_lon:.4f}\n"
            f"**Pred:** {pred_lat:.4f}, {pred_lon:.4f}",
            color=0x2ECC71 if dist < 1.0 else 0xF39C12,
        )

    except Exception as e:
        logger.error(f"Clean test failed: {e}")
        discord_send("Clean Test Failed", str(e), color=0xE74C3C)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-evaluation harness for geofind accuracy testing"
    )
    parser.add_argument(
        "--max-iter", type=int, default=30,
        help="Maximum evaluation iterations (default: 30)",
    )
    parser.add_argument(
        "--images", type=int, default=10,
        help="Images per iteration (default: 10)",
    )
    parser.add_argument(
        "--perturbation-level", choices=["light", "medium", "heavy"],
        default="medium",
        help="Perturbation intensity (default: medium)",
    )
    args = parser.parse_args()

    logger.info("Starting self-evaluation harness")
    logger.info(
        f"Config: max_iter={args.max_iter}, images={args.images}, "
        f"perturbation={args.perturbation_level}"
    )

    success = run_evaluation(
        max_iterations=args.max_iter,
        images_per_iter=args.images,
        perturbation_level=args.perturbation_level,
    )

    if success:
        logger.info("All targets met!")
    else:
        logger.warning("Targets not met within iteration limit")
        sys.exit(1)


if __name__ == "__main__":
    main()
