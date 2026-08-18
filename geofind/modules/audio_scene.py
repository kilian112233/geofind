"""Audio Scene Classification module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from geofind.core.candidate import ModuleHit
from geofind.core.config import PipelineConfig
from geofind.modules.base import BaseModule

logger = logging.getLogger(__name__)

_SCENE_GEOGRAPHIC_PRIORS: dict[str, dict[str, Any]] = {
    "urban": {
        "lat": 35.0, "lon": 0.0, "confidence": 0.25,
        "hint": "urban area",
    },
    "traffic": {
        "lat": 35.0, "lon": 0.0, "confidence": 0.20,
        "hint": "roadway/traffic",
    },
    "nature": {
        "lat": 0.0, "lon": 0.0, "confidence": 0.15,
        "hint": "natural/rural area",
    },
    "speech_english": {
        "lat": 40.0, "lon": -75.0, "confidence": 0.30,
        "hint": "English-speaking region",
    },
    "speech_spanish": {
        "lat": 20.0, "lon": -100.0, "confidence": 0.35,
        "hint": "Spanish-speaking region",
    },
    "speech_mandarin": {
        "lat": 35.86, "lon": 104.20, "confidence": 0.40,
        "hint": "Mandarin-speaking region",
    },
    "speech_hindi": {
        "lat": 20.59, "lon": 78.96, "confidence": 0.40,
        "hint": "Hindi-speaking region",
    },
    "speech_arabic": {
        "lat": 25.0, "lon": 45.0, "confidence": 0.40,
        "hint": "Arabic-speaking region",
    },
    "speech_japanese": {
        "lat": 36.20, "lon": 138.25, "confidence": 0.45,
        "hint": "Japanese-speaking region",
    },
    "speech_korean": {
        "lat": 35.91, "lon": 127.77, "confidence": 0.45,
        "hint": "Korean-speaking region",
    },
    "music_western": {
        "lat": 40.0, "lon": -75.0, "confidence": 0.10,
        "hint": "Western music",
    },
    "music_traditional_asian": {
        "lat": 30.0, "lon": 100.0, "confidence": 0.15,
        "hint": "Traditional Asian music",
    },
    "outdoor": {
        "lat": 0.0, "lon": 0.0, "confidence": 0.10,
        "hint": "outdoor environment",
    },
    "indoor": {
        "lat": 0.0, "lon": 0.0, "confidence": 0.10,
        "hint": "indoor environment",
    },
}


class AudioSceneModule(BaseModule):
    """Classify audio scene to infer geographic context."""

    name = "audio_scene"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)

    def is_available(self) -> bool:
        try:
            import numpy  # noqa: F401
            return True
        except ImportError:
            return False

    def detect(
        self,
        media_path: Path,
        *,
        frames: list[Any] | None = None,
        audio_path: Path | None = None,
    ) -> list[ModuleHit]:
        path = audio_path or media_path
        try:
            from geofind.utils.media import load_audio
            samples, sr = load_audio(path)
            if len(samples) < sr * 0.5:
                return []
        except Exception as e:
            self._log(f"Audio load failed: {e}", logging.WARNING)
            return []

        import numpy as np

        fft = np.abs(np.fft.rfft(samples))
        freqs = np.fft.rfftfreq(len(samples), d=1.0 / sr)

        features = self._extract_audio_features(samples, sr, fft, freqs)
        scene_scores = self._classify_scene(features)

        hits: list[ModuleHit] = []
        for scene, score in sorted(scene_scores.items(), key=lambda x: -x[1]):
            if score < 0.05:
                continue
            prior = _SCENE_GEOGRAPHIC_PRIORS.get(scene)
            if prior is None:
                continue

            confidence = min(score * prior["confidence"] * 3, 0.8)
            if confidence < 0.03:
                continue

            hits.append(self._make_hit(
                prior["lat"], prior["lon"], confidence,
                scene=scene,
                raw_score=score,
                hint=prior["hint"],
            ))

        return hits

    def _extract_audio_features(
        self, samples: Any, sr: int, fft: Any, freqs: Any
    ) -> dict[str, float]:
        import numpy as np

        features: dict[str, float] = {}

        total_energy = float(np.sum(fft ** 2)) + 1e-10

        bands = {
            "sub_bass": (20, 60),
            "bass": (60, 250),
            "low_mid": (250, 500),
            "mid": (500, 2000),
            "upper_mid": (2000, 4000),
            "high": (4000, 8000),
            "very_high": (8000, min(sr / 2, 20000)),
        }
        for name, (lo, hi) in bands.items():
            mask = (freqs >= lo) & (freqs <= hi)
            if np.any(mask):
                features[f"energy_{name}"] = float(np.sum(fft[mask] ** 2)) / total_energy
            else:
                features[f"energy_{name}"] = 0.0

        chunk_size = min(sr, len(samples))
        n_chunks = max(len(samples) // chunk_size, 1)
        energies = []
        for i in range(n_chunks):
            chunk = samples[i * chunk_size:(i + 1) * chunk_size]
            if len(chunk) > 0:
                energies.append(float(np.sqrt(np.mean(chunk ** 2))))
        features["rms_mean"] = float(np.mean(energies)) if energies else 0.0
        features["rms_std"] = float(np.std(energies)) if len(energies) > 1 else 0.0

        spectral_flatness = float(np.exp(np.mean(np.log(fft + 1e-10))) / (np.mean(fft) + 1e-10))
        features["spectral_flatness"] = spectral_flatness

        return features

    def _classify_scene(self, features: dict[str, float]) -> dict[str, float]:
        scores: dict[str, float] = {}

        energy_mid = features.get("energy_mid", 0)
        energy_high = features.get("energy_high", 0)
        energy_bass = features.get("energy_bass", 0)
        energy_low_mid = features.get("energy_low_mid", 0)
        rms = features.get("rms_mean", 0)
        flatness = features.get("spectral_flatness", 0)
        energy_very_high = features.get("energy_very_high", 0)

        speech_energy = energy_low_mid + energy_mid
        music_energy = energy_bass + energy_mid + energy_high
        noise_energy = flatness
        nature_energy = energy_high + energy_very_high

        if speech_energy > 0.4:
            if energy_mid > 0.2:
                scores["speech_english"] = speech_energy * 0.5
                scores["speech_spanish"] = speech_energy * 0.3
                scores["speech_mandarin"] = speech_energy * 0.2
                scores["speech_hindi"] = speech_energy * 0.15
                scores["speech_arabic"] = speech_energy * 0.15
                scores["speech_japanese"] = speech_energy * 0.1
                scores["speech_korean"] = speech_energy * 0.1

        if music_energy > 0.3 and energy_bass > 0.1:
            if energy_very_high > 0.05:
                scores["music_traditional_asian"] = music_energy * 0.3
            scores["music_western"] = music_energy * 0.5

        if rms > 0.01 and noise_energy < 0.3:
            if energy_bass > 0.15:
                scores["traffic"] = rms * 2
                scores["urban"] = rms * 1.5

        if nature_energy > 0.15 and rms < 0.02:
            scores["nature"] = nature_energy * 2
            scores["outdoor"] = nature_energy

        if flatness > 0.5:
            scores["urban"] = scores.get("urban", 0) + flatness * 0.3
            scores["traffic"] = scores.get("traffic", 0) + flatness * 0.2

        if energy_bass > 0.2 and energy_mid > 0.15:
            scores["indoor"] = energy_bass * 0.5

        if not scores:
            scores["outdoor"] = 0.1

        return scores
