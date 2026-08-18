"""BirdNET Species Geographic Range module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from geofind.core.candidate import ModuleHit
from geofind.core.config import PipelineConfig
from geofind.modules.base import BaseModule

logger = logging.getLogger(__name__)

_SAMPLE_RANGES: dict[str, dict[str, Any]] = {
    "American Robin": {"lat": 40.0, "lon": -95.0, "range_km": 2000, "confidence": 0.7},
    "European Robin": {"lat": 50.0, "lon": 5.0, "range_km": 1500, "confidence": 0.7},
    "House Sparrow": {"lat": 30.0, "lon": 30.0, "range_km": 5000, "confidence": 0.5},
    "Common Raven": {"lat": 55.0, "lon": -100.0, "range_km": 4000, "confidence": 0.6},
    "Northern Cardinal": {"lat": 38.0, "lon": -85.0, "range_km": 1500, "confidence": 0.7},
    "Bald Eagle": {"lat": 50.0, "lon": -110.0, "range_km": 3000, "confidence": 0.8},
    "Eurasian Magpie": {"lat": 50.0, "lon": 80.0, "range_km": 4000, "confidence": 0.6},
    "Indian Peafowl": {"lat": 20.0, "lon": 78.0, "range_km": 1500, "confidence": 0.8},
    "Kookaburra": {"lat": -28.0, "lon": 145.0, "range_km": 1500, "confidence": 0.9},
    "Common Myna": {"lat": 20.0, "lon": 78.0, "range_km": 3000, "confidence": 0.5},
    "Rainbow Lorikeet": {"lat": -28.0, "lon": 150.0, "range_km": 1000, "confidence": 0.9},
}


class BirdnetModule(BaseModule):
    """Detect bird species in audio to infer geographic range."""

    name = "birdnet"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)
        self._use_birdnet = False

    def is_available(self) -> bool:
        try:
            import numpy  # noqa: F401
            return True
        except ImportError:
            return False

    def prepare(self) -> None:
        try:
            from birdnet_analyzer import analyzer  # noqa: F401
            self._use_birdnet = True
            self._log("BirdNET analyzer available")
        except ImportError:
            self._use_birdnet = False
            self._log("BirdNET not installed, using fallback heuristic")

        from geofind.utils.models import get_cached_model
        import numpy as np

        if self._use_birdnet:
            def _load():
                from birdnet_analyzer import analyzer
                analyzer.load_model()
                return analyzer
            self._analyzer = get_cached_model("birdnet", _load)

        super().prepare()

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
            if len(samples) < sr:
                return []
        except Exception as e:
            self._log(f"Audio load failed: {e}", logging.WARNING)
            return []

        if self._use_birdnet:
            return self._detect_birdnet(samples, sr, path)
        return self._detect_heuristic(samples, sr, path)

    def _detect_birdnet(
        self, samples: Any, sr: int, path: Path
    ) -> list[ModuleHit]:
        import numpy as np

        hits: list[ModuleHit] = []
        try:
            config = {
                "input": str(path),
                "output": "",
                "min_confidence": 0.3,
            }
            results = self._analyzer.process_file(config)
            if not results:
                return hits

            for det in results:
                species = det.get("species", "")
                confidence = float(det.get("confidence", 0))
                if confidence < 0.3 or species not in _SAMPLE_RANGES:
                    continue

                info = _SAMPLE_RANGES[species]
                base_conf = info["confidence"]
                hits.append(self._make_hit(
                    info["lat"], info["lon"],
                    min(confidence * base_conf, 0.9),
                    species=species,
                    confidence_raw=confidence,
                ))
        except Exception as e:
            self._log(f"BirdNET analysis failed: {e}", logging.WARNING)

        return hits

    def _detect_heuristic(
        self, samples: Any, sr: int, path: Path
    ) -> list[ModuleHit]:
        import numpy as np

        hits: list[ModuleHit] = []

        n = len(samples)
        chunk_size = min(sr * 5, n)
        n_chunks = max(n // chunk_size, 1)

        bird_energy_bands: list[float] = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, n)
            chunk = samples[start:end]
            if len(chunk) == 0:
                continue

            fft = np.abs(np.fft.rfft(chunk))
            freqs = np.fft.rfftfreq(len(chunk), d=1.0 / sr)

            bird_mask = (freqs >= 2000) & (freqs <= 8000)
            if np.any(bird_mask):
                energy = float(np.sum(fft[bird_mask]))
                total = float(np.sum(fft)) + 1e-10
                bird_energy_bands.append(energy / total)

        if not bird_energy_bands:
            return hits

        avg_bird_energy = float(np.mean(bird_energy_bands))
        if avg_bird_energy < 0.01:
            return hits

        fft_full = np.abs(np.fft.rfft(samples))
        freqs_full = np.fft.rfftfreq(len(samples), d=1.0 / sr)

        bird_mask = (freqs_full >= 2000) & (freqs_full <= 8000)
        if not np.any(bird_mask):
            return hits

        bird_psd = fft_full[bird_mask]
        bird_freqs = freqs_full[bird_mask]
        peak_idx = np.argmax(bird_psd)
        peak_freq = float(bird_freqs[peak_idx])

        if 2000 <= peak_freq <= 4000:
            likely_tropical = True
            base_lat = 0.0
        elif 4000 < peak_freq <= 6000:
            likely_tropical = False
            base_lat = 40.0
        else:
            likely_tropical = False
            base_lat = 55.0

        confidence = min(avg_bird_energy * 10, 0.5)
        if confidence < 0.05:
            return hits

        hits.append(self._make_hit(
            base_lat, 0.0, confidence,
            method="heuristic",
            peak_freq=peak_freq,
            bird_energy=avg_bird_energy,
        ))

        return hits
