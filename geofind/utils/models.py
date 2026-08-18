"""Model download and cache management."""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Default models directory
_MODELS_DIR = Path("W:/geofind/models")


def get_models_dir() -> Path:
    """Get or create the models directory."""
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return _MODELS_DIR


def download_model(
    url: str,
    filename: str | None = None,
    models_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Download a model file with caching.

    Args:
        url: URL to download from.
        filename: Local filename. If None, extracted from URL.
        models_dir: Directory to cache models in.
        force: Re-download even if cached.

    Returns:
        Path to the cached model file.
    """
    if models_dir is None:
        models_dir = get_models_dir()

    if filename is None:
        filename = url.split("/")[-1].split("?")[0]

    target = models_dir / filename

    if target.exists() and not force:
        logger.info(f"Model cached: {target}")
        return target

    logger.info(f"Downloading model: {url}")
    models_dir.mkdir(parents=True, exist_ok=True)

    try:
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()

        with open(target, "wb") as f:
            shutil.copyfileobj(resp.raw, f)

        logger.info(f"Model saved: {target} ({target.stat().st_size / 1e6:.1f} MB)")
        return target

    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        if target.exists():
            target.unlink()
        raise


def hash_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_clip_model(models_dir: Path | None = None) -> str:
    """Ensure CLIP model is available. Returns model name for transformers."""
    # CLIP models are downloaded by HuggingFace transformers automatically
    # Just return the model name
    return "openai/clip-vit-base-patch32"


def ensure_landmark_model(models_dir: Path | None = None) -> str:
    """Ensure landmark recognition model is available."""
    return "google/vit-base-patch16-224"


def ensure_easyocr_langs() -> list[str]:
    """Default OCR languages.

    Easyocr has strict compatibility rules between languages. We use only
    English for OCR text detection — the Unicode script/language detection
    in OcrTextModule handles non-Latin script identification instead.
    """
    return ["en"]


# ── Lazy model loading with caching ─────────────────────────────────────────

_model_cache: dict[str, Any] = {}


def get_cached_model(key: str, loader: Any) -> Any:
    """Get or load a model with simple in-memory caching."""
    if key not in _model_cache:
        _model_cache[key] = loader()
    return _model_cache[key]


def clear_model_cache() -> None:
    """Clear all cached models (free memory)."""
    _model_cache.clear()


def get_cache_info() -> dict[str, bool]:
    """Report which models are currently cached in memory."""
    return {k: True for k in _model_cache}
