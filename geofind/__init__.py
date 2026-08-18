"""geofind - Offline multi-module media geolocation with Bayesian reranking."""

import subprocess
import sys

# Core deps — always required
_CORE_PACKAGES = {
    "typer": "typer[all]",
    "rich": "rich",
    "numpy": "numpy",
    "cv2": "opencv-python-headless",
    "scipy": "scipy",
    "PIL": "Pillow",
    "exifread": "exifread",
    "langdetect": "langdetect",
}

# Optional deps — powers every detection module
_OPTIONAL_PACKAGES = {
    "torch": "torch",
    "transformers": "transformers",
    "easyocr": "easyocr",
    "astral": "astral",
    "ultralytics": "ultralytics",
    "birdnet_analyzer": "birdnet-analyzer",
    "piexif": "piexif",
}


def _ensure_deps():
    """Auto-install any missing packages on first run."""
    all_missing = []
    # Core deps
    for module_name, pip_name in _CORE_PACKAGES.items():
        try:
            __import__(module_name)
        except ImportError:
            all_missing.append(pip_name)
    # Optional deps
    for module_name, pip_name in _OPTIONAL_PACKAGES.items():
        try:
            __import__(module_name)
        except ImportError:
            all_missing.append(pip_name)

    if all_missing:
        print(f"[geofind] First run — installing {len(all_missing)} missing packages...")
        print(f"[geofind] Packages: {', '.join(all_missing)}")
        print("[geofind] This may take a few minutes on first launch.\n")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet", "--user",
                 *all_missing],
            )
            print("[geofind] All packages installed successfully.\n")
        except subprocess.CalledProcessError as e:
            # Retry one-by-one so a single failure doesn't block the rest
            print(f"[geofind] Batch install failed ({e}), trying one-by-one...")
            for pkg in all_missing:
                try:
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", "--quiet",
                         "--user", pkg],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    print(f"  ✓ {pkg}")
                except subprocess.CalledProcessError:
                    print(f"  ✗ {pkg} (skipped)")
            print("[geofind] geofind will run with available modules.\n")


_ensure_deps()
