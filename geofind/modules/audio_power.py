"""Power Grid Frequency Detection module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from geofind.core.candidate import ModuleHit
from geofind.core.config import PipelineConfig
from geofind.modules.base import BaseModule

logger = logging.getLogger(__name__)

_COUNTRY_CENTROIDS: dict[str, tuple[float, float]] = {
    "GB": (55.38, -3.44), "DE": (51.17, 10.45), "FR": (46.23, 2.21),
    "IN": (20.59, 78.96), "AU": (-25.27, 133.78), "ZA": (-30.56, 22.94),
    "BR": (-14.24, -51.93), "EG": (26.82, 30.80), "TH": (15.87, 100.99),
    "TR": (38.96, 35.24), "PK": (30.38, 69.35), "NG": (9.08, 8.68),
    "SA": (23.89, 45.08), "AE": (23.42, 53.85), "ID": (-0.79, 113.92),
    "VN": (14.06, 108.28), "MY": (4.21, 101.98), "PH": (12.88, 121.77),
    "KR": (35.91, 127.77), "US": (37.09, -95.71), "CA": (56.13, -106.35),
    "MX": (23.63, -102.55), "JP": (36.20, 138.25), "TW": (23.70, 120.96),
    "CO": (4.57, -74.30), "AR": (-38.42, -63.62), "CL": (-35.68, -71.54),
}


class AudioPowerModule(BaseModule):
    """Detect power grid frequency (50/60 Hz) in audio to narrow country."""

    name = "audio_power"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)

    def is_available(self) -> bool:
        try:
            import numpy  # noqa: F401
            import scipy  # noqa: F401
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
        audio_data = self._load_audio(media_path, audio_path)
        if audio_data is None:
            return []

        samples, sr = audio_data
        if len(samples) < sr * 0.5:
            return []

        peak_freq, peak_power = self._find_power_freq(samples, sr)
        if peak_freq is None:
            return []

        self._log(f"Dominant frequency: {peak_freq:.1f} Hz (power: {peak_power:.4f})")

        detected_hz = 50.0 if abs(peak_freq - 50.0) < abs(peak_freq - 60.0) else 60.0
        confidence = min(peak_power * 5.0, 0.9)

        from geofind.utils.constants import POWER_GRID

        matching_countries = [
            cc for cc, (freq, _) in POWER_GRID.items()
            if freq == detected_hz
        ]

        if not matching_countries:
            return []

        hits: list[ModuleHit] = []
        n = len(matching_countries)
        for cc in matching_countries:
            lat, lon = _COUNTRY_CENTROIDS.get(cc, (0.0, 0.0))
            hits.append(self._make_hit(
                lat, lon, confidence / n,
                country=cc,
                frequency_hz=detected_hz,
                peak_freq=peak_freq,
            ))

        return hits

    def _load_audio(self, media_path: Path, audio_path: Path | None) -> tuple | None:
        import numpy as np

        path = audio_path or media_path
        try:
            from geofind.utils.media import load_audio
            samples, sr = load_audio(path)
            if len(samples) == 0:
                return None
            return samples, sr
        except Exception as e:
            self._log(f"Audio load failed: {e}", logging.WARNING)
            return None

    def _find_power_freq(
        self, samples: Any, sr: int
    ) -> tuple[float | None, float]:
        import numpy as np

        try:
            from scipy.signal import welch
            freqs, psd = welch(samples, fs=sr, nperseg=min(len(samples), sr * 2))
        except ImportError:
            n = len(samples)
            fft = np.abs(np.fft.rfft(samples))
            freqs = np.fft.rfftfreq(n, d=1.0 / sr)
            psd = fft ** 2 / n

        mask_50 = (freqs >= 45.0) & (freqs <= 55.0)
        mask_60 = (freqs >= 55.0) & (freqs <= 65.0)
        mask_total = mask_50 | mask_60

        if not np.any(mask_total):
            return None, 0.0

        band_freqs = freqs[mask_total]
        band_psd = psd[mask_total]
        idx = np.argmax(band_psd)
        peak_freq = float(band_freqs[idx])
        peak_power = float(band_psd[idx])

        total_power = float(np.sum(psd))
        if total_power > 0:
            peak_power /= total_power

        return peak_freq, peak_power
