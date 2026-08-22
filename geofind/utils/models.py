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
    return "openai/clip-vit-large-patch14"


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


def get_clip_shared() -> tuple[Any, Any]:
    """Get a shared CLIP model+processor. All CLIP-based modules use this."""
    return get_cached_model("clip_shared", _load_clip)


def _load_clip():
    """Load CLIP model + processor once."""
    from transformers import CLIPProcessor, CLIPModel
    model_name = ensure_clip_model()
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def get_cache_info() -> dict[str, bool]:
    """Report which models are currently cached in memory."""
    return {k: True for k in _model_cache}


# ── Shared CLIP embedding caches ────────────────────────────────────────────
# Static prompt sets are encoded once and reused across images. Image
# embeddings are cached per unique image content so multiple modules running
# on the same frame share a single ViT-L/14 forward pass.

_clip_text_cache: dict[str, Any] = {}
_clip_image_cache: dict[str, Any] = {}


def _unwrap_clip_features(feats: Any, kind: str) -> Any:
    """Unwrap transformers 5.x model outputs to a plain tensor.

    `kind` is "text" or "image" — selects the right embeds attribute.
    """
    if hasattr(feats, "shape") and hasattr(feats, "norm"):
        return feats  # already a tensor
    primary = f"{kind}_embeds"
    for attr in (primary, "pooler_output", "last_hidden_state"):
        val = getattr(feats, attr, None)
        if val is not None:
            return val
    raise TypeError(f"Unexpected CLIP feature type: {type(feats)}")


def clip_text_embeddings(key: str, prompts: list[str]) -> Any:
    """Encode a static prompt list once; returns normalized tensor [N, D].

    Results are cached by `key` — subsequent calls are free. Prompt lists
    passed here must be deterministic per key.
    """
    if key not in _clip_text_cache:
        import torch

        model, processor = get_clip_shared()
        chunks: list[Any] = []
        with torch.no_grad():
            for i in range(0, len(prompts), 64):
                inputs = processor(
                    text=prompts[i : i + 64], return_tensors="pt", padding=True
                )
                feats = _unwrap_clip_features(
                    model.get_text_features(**inputs), "text"
                )
                chunks.append(feats / feats.norm(dim=-1, keepdim=True))
        _clip_text_cache[key] = torch.cat(chunks)
    return _clip_text_cache[key]


def clip_image_embedding(image: Any) -> Any:
    """Encode an image once per unique content; returns normalized tensor [D].

    Cached by content hash so repeated `.convert("RGB")` copies of the same
    frame across modules hit the cache instead of re-running the encoder.
    """
    import hashlib

    try:
        key = hashlib.md5(image.tobytes()).hexdigest()
    except Exception:
        key = f"id:{id(image)}"

    emb = _clip_image_cache.get(key)
    if emb is None:
        import torch

        model, processor = get_clip_shared()
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt")
            feats = _unwrap_clip_features(
                model.get_image_features(**inputs), "image"
            )

        feats = feats / feats.norm(dim=-1, keepdim=True)
        emb = feats[0]
        _clip_image_cache[key] = emb
        # Keep the cache small — only the most recent frames
        while len(_clip_image_cache) > 4:
            _clip_image_cache.pop(next(iter(_clip_image_cache)))
    return _clip_image_cache[key]


def clip_zero_shot_scores(
    image: Any, cache_key: str, prompts: list[str]
) -> list[float]:
    """Cosine similarity of an image against a cached prompt set.

    Values are comparable to HF `logits_per_image / 100`.
    """
    img_emb = clip_image_embedding(image)
    txt_embs = clip_text_embeddings(cache_key, prompts)
    return (img_emb @ txt_embs.T).tolist()


def clip_softmax_scores(
    image: Any, cache_key: str, prompts: list[str]
) -> list[float]:
    """Softmax probabilities over a prompt set (mirrors logits.softmax)."""
    import torch

    scores = clip_zero_shot_scores(image, cache_key, prompts)
    probs = torch.softmax(torch.tensor(scores) * 100.0, dim=0)
    return probs.tolist()


def clear_clip_image_cache() -> None:
    """Drop cached image embeddings (call when analyzing a new media file)."""
    _clip_image_cache.clear()


def get_streetclip() -> tuple[Any, Any]:
    """Get the StreetCLIP model+processor (geolocation-specialized CLIP).

    StreetCLIP (geolocal/StreetCLIP, CC-BY-NC-4.0) is a CLIP ViT-L/14
    fine-tuned on Street View imagery at 336px input. Zero-shot country/
    region classification beats supervised geo-models on Im2GPS/YFCC.
    """
    return get_cached_model(
        "streetclip",
        lambda: _load_hf_clip("geolocal/StreetCLIP"),
    )


def _load_hf_clip(model_name: str) -> tuple[Any, Any]:
    """Load any HF CLIP-compatible model + processor.

    Prefers the local cache (avoids slow/hanging hub etag checks on repeat
    loads); falls back to downloading when not cached yet.
    """
    from transformers import CLIPModel, CLIPProcessor

    try:
        model = CLIPModel.from_pretrained(model_name, local_files_only=True)
        processor = CLIPProcessor.from_pretrained(
            model_name, local_files_only=True
        )
    except Exception:
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


# ── Shared OCR cache ────────────────────────────────────────────────────────
# Multiple modules need the same EasyOCR pass over the same frame. The full
# preprocessing pipeline (CLAHE → denoise → sharpen → Otsu) plus inference is
# run ONCE per unique image content and the extracted text shared.

_ocr_text_cache: dict[str, str] = {}


def extract_ocr_text_cached(image: Any) -> str:
    """Extract all visible text from an image via the shared OCR pipeline.

    Results are cached per image content hash — repeated calls across modules
    are free.
    """
    import hashlib

    try:
        key = "ocr:" + hashlib.md5(image.tobytes()).hexdigest()
    except Exception:
        key = f"ocr:id:{id(image)}"

    if key in _ocr_text_cache:
        return _ocr_text_cache[key]

    import numpy as np

    img_array = np.array(image)
    text = ""
    try:
        import cv2
        from geofind.utils.models import get_cached_model, ensure_easyocr_langs

        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

        # Sharpen
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)

        # Otsu binarization
        _, binary = cv2.threshold(
            sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        langs = ensure_easyocr_langs()

        def _load():
            import easyocr
            return easyocr.Reader(langs, gpu=False)

        reader = get_cached_model("easyocr", _load)
        results = reader.readtext(binary)
        text = " ".join(r[1] for r in results if len(r) > 1)
    except Exception:
        text = ""

    _ocr_text_cache[key] = text
    return text


def clear_ocr_cache() -> None:
    """Drop cached OCR results (call when analyzing a new media file)."""
    _ocr_text_cache.clear()
