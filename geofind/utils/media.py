"""Media loading utilities: images, video frames, audio extraction."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Supported file extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp", ".heic", ".heif"}
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv", ".m4v", ".3gp"}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}


def classify_media(path: Path) -> str:
    """Classify a media file as image, video, or audio."""
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    elif ext in VIDEO_EXTS:
        return "video"
    elif ext in AUDIO_EXTS:
        return "audio"
    return "unknown"


def load_image(path: Path) -> Image.Image:
    """Load an image from path."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def load_image_array(path: Path) -> np.ndarray:
    """Load image as numpy array (H, W, 3) RGB."""
    img = load_image(path)
    return np.array(img)


def extract_video_frames(
    path: Path,
    max_frames: int = 60,
    interval_s: float = 2.0,
    use_scene_detection: bool = True,
) -> list[tuple[np.ndarray, float]]:
    """Extract frames from video file.

    Uses OpenCV BackgroundSubtractorMOG2 for scene detection when enabled.
    Falls back to fixed-interval extraction.

    Returns:
        List of (frame_array, timestamp_seconds) tuples.
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available, cannot extract video frames")
        return []

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps > 0 else 0

    logger.info(f"Video: {duration_s:.1f}s, {fps:.1f}fps, {total_frames} frames")

    frames: list[tuple[np.ndarray, float]] = []

    if use_scene_detection and total_frames > max_frames:
        frames = _extract_scene_frames(cap, fps, max_frames)
    else:
        frames = _extract_interval_frames(cap, fps, interval_s, max_frames)

    cap.release()
    logger.info(f"Extracted {len(frames)} frames")
    return frames


def _extract_scene_frames(
    cap: Any,
    fps: float,
    max_frames: int,
) -> list[tuple[np.ndarray, float]]:
    """Extract frames at scene changes using MOG2 background subtraction."""
    import cv2

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=50, detectShadows=False
    )

    frames: list[tuple[np.ndarray, float]] = []
    frame_idx = 0
    last_scene_frame = -100
    min_scene_gap = int(fps * 1.0)  # At least 1s between scenes

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps

        # Apply background subtraction
        mask = bg_subtractor.apply(frame)

        # Count foreground pixels
        fg_pixels = cv2.countNonZero(mask)
        frame_area = frame.shape[0] * frame.shape[1]
        fg_ratio = fg_pixels / frame_area if frame_area > 0 else 0

        # Scene change detected if significant foreground change
        is_scene_change = fg_ratio > 0.15 and (frame_idx - last_scene_frame) > min_scene_gap

        if is_scene_change or len(frames) == 0:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((rgb_frame, timestamp))
            last_scene_frame = frame_idx

            if len(frames) >= max_frames:
                break

        frame_idx += 1

    return frames


def _extract_interval_frames(
    cap: Any,
    fps: float,
    interval_s: float,
    max_frames: int,
) -> list[tuple[np.ndarray, float]]:
    """Extract frames at fixed intervals."""
    import cv2

    frames: list[tuple[np.ndarray, float]] = []
    interval_frames = int(fps * interval_s)
    if interval_frames < 1:
        interval_frames = 1

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval_frames == 0:
            timestamp = frame_idx / fps
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((rgb_frame, timestamp))

            if len(frames) >= max_frames:
                break

        frame_idx += 1

    return frames


def extract_audio_from_video(video_path: Path) -> Path | None:
    """Extract audio track from video to temporary WAV file."""
    try:
        import subprocess

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()

        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "44100", "-ac", "1",
            "-y", str(tmp_path),
        ]

        result = subprocess.run(
            cmd, capture_output=True, timeout=60
        )

        if result.returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size > 0:
            return tmp_path
        else:
            logger.warning(f"ffmpeg audio extraction failed: {result.stderr.decode()[:200]}")
            if tmp_path.exists():
                tmp_path.unlink()
            return None

    except FileNotFoundError:
        logger.warning("ffmpeg not found, cannot extract audio from video")
        return None
    except Exception as e:
        logger.warning(f"Audio extraction error: {e}")
        return None


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load audio file as numpy array.

    Returns:
        (audio_samples, sample_rate) tuple.
    """
    try:
        import subprocess
        import wave

        # Convert to WAV using ffmpeg if needed
        if path.suffix.lower() != ".wav":
            tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = Path(tmp_wav.name)
            tmp_wav.close()

            cmd = [
                "ffmpeg", "-i", str(path),
                "-ar", "44100", "-ac", "1",
                "-acodec", "pcm_s16le",
                "-y", str(tmp_path),
            ]
            subprocess.run(cmd, capture_output=True, timeout=60)
            path = tmp_path

        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        if n_channels > 1:
            audio = audio[::n_channels]  # Take first channel

        # Normalize to -1..1
        audio = audio / 32768.0

        return audio, sr

    except Exception as e:
        logger.error(f"Failed to load audio {path}: {e}")
        return np.array([], dtype=np.float32), 44100


def get_media_info(path: Path) -> dict[str, Any]:
    """Get basic info about a media file."""
    info: dict[str, Any] = {
        "path": str(path),
        "type": classify_media(path),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }

    if info["type"] == "image":
        try:
            img = Image.open(path)
            info["width"] = img.width
            info["height"] = img.height
            info["format"] = img.format
        except Exception:
            pass

    elif info["type"] == "video":
        try:
            import cv2
            cap = cv2.VideoCapture(str(path))
            info["fps"] = cap.get(cv2.CAP_PROP_FPS)
            info["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            info["duration_s"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
            cap.release()
        except Exception:
            pass

    elif info["type"] == "audio":
        try:
            import subprocess
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", str(path)],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                import json
                probe = json.loads(result.stdout)
                dur = probe.get("format", {}).get("duration")
                if dur:
                    info["duration_s"] = float(dur)
        except Exception:
            pass

    return info
