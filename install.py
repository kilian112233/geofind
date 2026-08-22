#!/usr/bin/env python
"""Geofind Installation Script

Sets up all dependencies, models, and data for geofind on a new machine.
Supports selective installation of optional features.

Usage:
    python install.py                  # Install everything
    python install.py --minimal        # Core modules only (no ML models)
    python install.py --features solar # Only add solar feature
    python install.py --list-features  # Show available features
    python install.py --check          # Check what's installed

Target: beefy PC with 80GB+ storage
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

# ── Feature definitions ──────────────────────────────────────────────
FEATURES = {
    "core": {
        "desc": "Core modules (EXIF, region, CLIP visual, OCR, Bayesian reranker)",
        "pip": [],  # core deps are in pyproject.toml
        "models": ["clip_vitl14", "retrieval_db"],
    },
    "geoclip": {
        "desc": "GeoCLIP geographic image encoder",
        "pip": ["geoclip"],
        "models": ["geoclip"],
        "requires": ["torch"],
    },
    "ocr": {
        "desc": "EasyOCR text extraction + text geocoding (online)",
        "pip": ["easyocr"],
        "models": [],
    },
    "solar": {
        "desc": "Shadow/sun analysis for latitude estimation",
        "pip": ["astral"],
        "models": [],
    },
    "terrain": {
        "desc": "Terrain/horizon matching with on-demand DEM downloads",
        "pip": [],
        "models": [],
        "note": "DEM tiles download on first use (~90MB per 5x5° tile)",
    },
    "places365": {
        "desc": "Scene classification (Places365 CNN)",
        "pip": ["torch", "torchvision"],
        "models": ["places365"],
    },
    "vehicle": {
        "desc": "Driving side detection (YOLO)",
        "pip": ["ultralytics"],
        "models": ["yolo_nas"],
    },
}

# ── Model downloads ──────────────────────────────────────────────────

def _ensure_clip_model(model_dir: Path) -> bool:
    """Download CLIP ViT-L/14 model if not present."""
    target = model_dir / "clip" / "ViT-L-14.pt"
    if target.exists():
        print(f"  [OK] CLIP ViT-L/14 already present")
        return True

    print("  > Downloading CLIP ViT-L/14 (~400MB)...")
    try:
        import torch
        from PIL import Image
        import clip as clip_lib

        model, _ = clip_lib.load("ViT-L/14", device="cpu", download_root=str(model_dir / "clip"))
        print(f"  [OK] CLIP ViT-L/14 saved")
        return True
    except Exception as e:
        print(f"  [!!] CLIP download failed: {e}")
        return False


def _ensure_geoclip_model(model_dir: Path) -> bool:
    """Download GeoCLIP model weights."""
    target = model_dir / "geoclip"
    if (target / "pytorch_model.bin").exists() or (target / "model.safetensors").exists():
        print(f"  [OK] GeoCLIP already present")
        return True

    print("  > Downloading GeoCLIP model (~400MB)...")
    try:
        target.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="m5049b/geoclip", local_dir=str(target))
        print(f"  [OK] GeoCLIP saved")
        return True
    except Exception as e:
        print(f"  [!!] GeoCLIP download failed: {e}")
        return False


def _ensure_places365_model(model_dir: Path) -> bool:
    """Download Places365 pretrained model."""
    target = model_dir / "places365"
    if (target / "resnet50_places365.pth").exists():
        print(f"  [OK] Places365 already present")
        return True

    print("  > Downloading Places365 ResNet50 (~100MB)...")
    try:
        import requests
        target.mkdir(parents=True, exist_ok=True)
        url = "https://huggingface.co/spaces/xianoppo/resnet50places365/resolve/main/resnet50_places365.pth.tar"
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        (target / "resnet50_places365.pth").write_bytes(resp.content)
        print(f"  [OK] Places365 saved")
        return True
    except Exception as e:
        print(f"  [!!] Places365 download failed: {e}")
        return False


def _ensure_retrieval_db(project_root: Path) -> bool:
    """Build the retrieval database if not present."""
    db_path = project_root / "models" / "retrieval" / "places.json"
    if db_path.exists():
        print(f"  [OK] Retrieval DB already present")
        return True

    print("  > Building retrieval DB (this takes a few minutes)...")
    try:
        subprocess.run(
            [PYTHON, str(project_root / "dev" / "build_retrieval_db.py")],
            check=True, cwd=str(project_root),
        )
        print(f"  [OK] Retrieval DB built")
        return True
    except Exception as e:
        print(f"  [!!] Retrieval DB build failed: {e}")
        return False


MODEL_ENSURE = {
    "clip_vitl14": _ensure_clip_model,
    "geoclip": _ensure_geoclip_model,
    "places365": _ensure_places365_model,
    "retrieval_db": _ensure_retrieval_db,
    "yolo_nas": lambda d: (print("  [OK] YOLO auto-downloads on first use"), True)[-1],
}


# ── Install logic ────────────────────────────────────────────────────

def install_pip_deps(features: list[str], extra_pip: list[str] | None = None) -> None:
    """Install pip dependencies for selected features."""
    # Always install core deps
    subprocess.run(
        [PYTHON, "-m", "pip", "install", "-e", f"{PROJECT_ROOT}[all]"],
        check=True,
    )
    print("  [OK] Core dependencies installed")

    # Install feature-specific deps
    seen = set()
    pkgs = []
    for feat in features:
        for pkg in FEATURES.get(feat, {}).get("pip", []):
            if pkg not in seen:
                pkgs.append(pkg)
                seen.add(pkg)
    for pkg in (extra_pip or []):
        if pkg not in seen:
            pkgs.append(pkg)
            seen.add(pkg)

    if pkgs:
        print(f"  > Installing: {', '.join(pkgs)}")
        subprocess.run(
            [PYTHON, "-m", "pip", "install"] + pkgs,
            check=True,
        )
        print(f"  [OK] Feature dependencies installed")


def install_models(features: list[str], model_dir: Path) -> None:
    """Download/verify models for selected features."""
    seen = set()
    for feat in features:
        for model in FEATURES.get(feat, {}).get("models", []):
            if model not in seen:
                ensure_fn = MODEL_ENSURE.get(model)
                if ensure_fn:
                    ensure_fn(model_dir)
                seen.add(model)


def check_installation(project_root: Path) -> None:
    """Report what's installed and what's missing."""
    model_dir = project_root / "models"

    print("\n+==================================================+")
    print("|          Geofind Installation Status              |")
    print("+==================================================+\n")

    # Check pip packages
    pkgs = {
        "torch": "PyTorch (Places365, CLIP)",
        "clip": "CLIP",
        "geoclip": "GeoCLIP",
        "easyocr": "EasyOCR",
        "astral": "Solar position",
        "ultralytics": "YOLO",
        "requests": "Online geocoding",
        "exifread": "EXIF reading",
        "langdetect": "Language detection",
        "cv2": "OpenCV",
        "scipy": "Bayesian reranker",
        "numpy": "Numerics",
        "PIL": "Image loading",
    }

    print("[PKG] Python Packages:")
    for pkg, desc in pkgs.items():
        try:
            __import__(pkg)
            print(f"  [OK]  {pkg:15s} -- {desc}")
        except ImportError:
            print(f"  [  ]  {pkg:15s} -- {desc}")

    # Check models
    print(f"\n[MDL] Models ({model_dir}):")
    checks = {
        "clip/ViT-L-14.pt": "CLIP ViT-L/14",
        "places365/resnet50_places365.pth": "Places365",
        "retrieval/places.json": "Retrieval DB",
    }
    for rel_path, desc in checks.items():
        p = model_dir / rel_path
        size = p.stat().st_size if p.exists() else 0
        if size > 0:
            size_mb = size / (1024 * 1024)
            print(f"  [OK]  {desc:25s} ({size_mb:.0f} MB)")
        else:
            print(f"  [  ]  {desc:25s} -- MISSING")

    # Check GeoCLIP
    geoclip_dir = model_dir / "geoclip"
    if geoclip_dir.exists() and any(geoclip_dir.iterdir()):
        size = sum(f.stat().st_size for f in geoclip_dir.rglob("*") if f.is_file())
        print(f"  [OK]  {'GeoCLIP':25s} ({size / 1024 / 1024:.0f} MB)")
    else:
        print(f"  [  ]  {'GeoCLIP':25s} -- MISSING")

    # Check DEM cache
    dem_dir = model_dir / "dem"
    if dem_dir.exists():
        tiles = list(dem_dir.glob("*.hgt"))
        total = sum(f.stat().st_size for f in tiles)
        print(f"\n[DEM] DEM Cache ({len(tiles)} tiles, {total / 1024 / 1024:.0f} MB)")
    else:
        print(f"\n[DEM] DEM Cache -- empty (tiles download on demand)")

    print()


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Geofind installer")
    parser.add_argument("--minimal", action="store_true",
                        help="Core modules only, no ML models")
    parser.add_argument("--features", nargs="+", default=None,
                        help=f"Features to install (choices: {list(FEATURES.keys())})")
    parser.add_argument("--list-features", action="store_true",
                        help="Show available features")
    parser.add_argument("--check", action="store_true",
                        help="Check current installation status")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models",
                        help="Model storage directory")
    args = parser.parse_args()

    if args.list_features:
        print("\nAvailable features:")
        for name, info in FEATURES.items():
            models = ", ".join(info["models"]) if info["models"] else "none"
            note = f" ({info['note']})" if "note" in info else ""
            print(f"  {name:15s} — {info['desc']}")
            print(f"  {'':15s}   Models: {models}{note}")
        print()
        return

    if args.check:
        check_installation(PROJECT_ROOT)
        return

    # Determine features to install
    if args.features:
        features = args.features
    elif args.minimal:
        features = ["core"]
    else:
        features = ["core", "geoclip", "ocr", "solar", "places365"]

    print("\n+==================================================+")
    print("|            Geofind Installer v0.1.0              |")
    print("+==================================================+")
    print(f"\nInstalling features: {', '.join(features)}\n")

    # Step 1: pip dependencies
    print("--- Step 1/3: Python Dependencies ---")
    try:
        install_pip_deps(features)
    except subprocess.CalledProcessError as e:
        print(f"\nx pip install failed: {e}")
        sys.exit(1)

    # Step 2: models
    print("\n--- Step 2/3: Models & Data ---")
    args.model_dir.mkdir(parents=True, exist_ok=True)
    install_models(features, args.model_dir)

    # Step 3: city DB (needed for OCR geocoding)
    if "ocr" in features:
        print("\n--- Step 3/3: City Database ---")
        city_db = PROJECT_ROOT / "dev" / "setup_city_db.py"
        if city_db.exists():
            print("  > Building city database...")
            try:
                subprocess.run(
                    [PYTHON, str(city_db)],
                    check=True, cwd=str(PROJECT_ROOT),
                )
                print(f"  [OK] City DB built")
            except Exception as e:
                print(f"  [FAIL] City DB build failed: {e}")

    # Summary
    print("\n--- Installation Complete ---\n")
    check_installation(PROJECT_ROOT)

    print("Quick start:")
    print(f"  python -m geofind.cli analyze --path <image>")
    print(f"  python dev/run_tests.py --random --limit 10 --no-exif\n")


if __name__ == "__main__":
    main()
