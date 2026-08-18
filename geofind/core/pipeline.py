"""Main pipeline orchestrator for geofind media geolocation."""

from __future__ import annotations

import importlib
import logging
import time
from pathlib import Path
from typing import Any, Callable

from rich.console import Console

from geofind.core.candidate import GeoResult, ModuleHit
from geofind.core.config import PipelineConfig
from geofind.core.grid import GeoGrid
from geofind.core.reranker import BayesianReranker
from geofind.modules.base import BaseModule
from geofind.utils.media import (
    classify_media,
    extract_audio_from_video,
    extract_video_frames,
    get_media_info,
    load_audio,
    load_image,
)

logger = logging.getLogger(__name__)
console = Console(stderr=True)

# Map of module name → expected class name (PascalCase of the snake_case name)
_MODULE_CLASS_MAP: dict[str, str] = {
    "exif": "ExifModule",
    "clip_visual": "ClipVisualModule",
    "landmark": "LandmarkModule",
    "ocr_text": "OcrTextModule",
    "audio_power": "AudioPowerModule",
    "birdnet": "BirdnetModule",
    "sun_clock": "SunClockModule",
    "shadow_angle": "ShadowAngleModule",
    "vision_llm": "VisionLlmModule",
    "driving_side": "DrivingSideModule",
    "vegetation": "VegetationModule",
    "license_plate": "LicensePlateModule",
    "currency": "CurrencyModule",
    "audio_scene": "AudioSceneModule",
}

# Modules that only work on visual data (image/video frames)
_VISUAL_MODULES = {
    "exif", "clip_visual", "landmark", "ocr_text",
    "sun_clock", "shadow_angle", "vision_llm",
    "driving_side", "vegetation", "license_plate", "currency",
}

# Modules that only work on audio data
_AUDIO_MODULES = {"audio_power", "birdnet", "audio_scene"}


class GeoPipeline:
    """Orchestrates the full geolocation analysis pipeline.

    Loads media, runs all enabled detection modules, builds a probability
    heatmap on a geographic grid, and applies Bayesian reranking to produce
    ranked candidate locations.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.grid = GeoGrid(
            resolution=self.config.grid_resolution_deg,
            sigma_km=self.config.gaussian_sigma_km,
        )
        self.reranker = BayesianReranker(self.config)
        self._modules: list[BaseModule] = []

    def analyze(
        self,
        media_path: str | Path,
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> GeoResult:
        """Run full geolocation analysis on a media file.

        Args:
            media_path: Path to image, video, or audio file.
            progress_callback: Optional fn(module_name, status) for UI updates.
                Status is one of: "loading", "running", "done", "failed".

        Returns:
            GeoResult with ranked candidate locations.
        """
        media_path = Path(media_path)
        pipeline_start = time.perf_counter()

        if not media_path.exists():
            raise FileNotFoundError(f"Media file not found: {media_path}")

        console.print(f"[bold cyan]Analyzing:[/] {media_path.name}")

        # ── Classify and load media ──────────────────────────────────────
        media_type = classify_media(media_path)
        console.print(f"  Media type: [bold]{media_type}[/]")

        if progress_callback:
            progress_callback("_loading", "loading")

        media_data = self._load_media(media_path, media_type)

        if progress_callback:
            progress_callback("_loading", "done")

        # ── Load modules ─────────────────────────────────────────────────
        self._modules = self._load_modules()
        console.print(f"  Loaded [bold]{len(self._modules)}[/] modules")

        # ── Run modules ──────────────────────────────────────────────────
        all_hits: dict[str, list[ModuleHit]] = {}
        modules_run: list[str] = []
        modules_failed: list[str] = []

        for module in self._modules:
            name = module.name

            # Skip modules that can't handle this media type
            if media_type == "audio" and name in _VISUAL_MODULES:
                logger.debug(f"Skipping {name} (audio media, visual-only module)")
                continue
            if media_type == "image" and name in _AUDIO_MODULES:
                logger.debug(f"Skipping {name} (image media, audio-only module)")
                continue

            if progress_callback:
                progress_callback(name, "running")

            hits = self._run_module(module, media_data)
            all_hits[name] = hits

            if hits is not None:
                modules_run.append(name)
                console.print(
                    f"  [green]✓[/] {name}: [bold]{len(hits)}[/] hits"
                )
            else:
                modules_failed.append(name)
                console.print(f"  [red]✗[/] {name}: failed")

            if progress_callback:
                progress_callback(name, "done" if hits is not None else "failed")

        # ── Build heatmap ────────────────────────────────────────────────
        if progress_callback:
            progress_callback("_grid", "running")

        console.print("  Building probability grid...")
        for name in modules_run:
            self.grid.add_module_hits(name, all_hits[name])

        if progress_callback:
            progress_callback("_grid", "done")

        # ── Rerank ───────────────────────────────────────────────────────
        if progress_callback:
            progress_callback("_rerank", "running")

        console.print("  Running Bayesian reranking (3-pass)...")
        module_weights = {
            name: self.config.modules[name].weight
            for name in modules_run
            if name in self.config.modules
        }

        candidates = self.reranker.rerank(self.grid, all_hits, module_weights)

        if progress_callback:
            progress_callback("_rerank", "done")

        # ── Inject exact GPS coordinates from EXIF (bypass grid) ────────
        # When EXIF finds GPS data, we have exact coordinates (within ~5m).
        # The grid is 1°×1° (~111km per cell), so grid candidates are
        # always off by up to ~55km. Inject the exact EXIF coordinates as
        # the #1 candidate for sub-100m accuracy on GPS-enabled images.
        exact_injected = False
        if "exif" in all_hits and all_hits["exif"]:
            exif_hits = all_hits["exif"]
            # Take the highest-confidence EXIF hit
            best_exif = max(exif_hits, key=lambda h: h.confidence)
            if best_exif.confidence >= 0.5:
                from geofind.core.candidate import CandidateLocation
                exact_cand = CandidateLocation(
                    lat=best_exif.lat,
                    lon=best_exif.lon,
                    probability=1.0,  # Will be normalized later
                    log_posterior=0.0,
                    hits=[best_exif],
                    is_exact=True,
                )
                # Find the grid candidate closest to this EXIF point and
                # attach its module hits to the exact candidate for display
                from geofind.utils.geo import haversine_km, LatLon as _LL
                exif_point = _LL(best_exif.lat, best_exif.lon)
                best_grid_dist = float("inf")
                for c in candidates:
                    d = haversine_km(exif_point, _LL(c.lat, c.lon))
                    if d < best_grid_dist:
                        best_grid_dist = d
                        for h in c.hits:
                            if h.module != "exif":
                                exact_cand.add_hit(h)

                # Inject at position 0, shifting everything else down
                candidates.insert(0, exact_cand)
                exact_injected = True
                console.print(
                    f"  [bold green]✓[/] EXIF exact GPS injected: "
                    f"{best_exif.lat:.6f}, {best_exif.lon:.6f} "
                    f"(±~5m, vs grid cell {best_grid_dist:.0f}km away)"
                )

        # ── Normalize probabilities ────────────────────────────────────
        total_prob = sum(c.probability for c in candidates)
        if total_prob > 0:
            for c in candidates:
                c.probability /= total_prob

        # ── Compute consensus ────────────────────────────────────────
        consensus = self.reranker.compute_consensus(all_hits, module_weights)

        # ── Build result ─────────────────────────────────────────────────
        processing_time = time.perf_counter() - pipeline_start

        result = GeoResult(
            candidates=candidates,
            consensus_lat=consensus.get("centroid_lat", 0.0),
            consensus_lon=consensus.get("centroid_lon", 0.0),
            agreement_strength=consensus.get("strength", 0.0),
            modules_run=modules_run,
            modules_failed=modules_failed,
            processing_time_s=round(processing_time, 3),
        )

        self._print_summary(result)
        return result

    # ── Private helpers ──────────────────────────────────────────────────

    def _load_media(self, path: Path, media_type: str) -> dict[str, Any]:
        """Load media data into a dict passed to detection modules."""
        info = get_media_info(path)
        media_data: dict[str, Any] = {
            "type": media_type,
            "path": path,
            "info": info,
            "images": [],
            "audio": None,
            "audio_sr": None,
        }

        if media_type == "image":
            img = load_image(path)
            media_data["images"] = [img]
            console.print(f"  Image: {img.width}x{img.height}")

        elif media_type == "video":
            console.print("  Extracting video frames...")
            frames = extract_video_frames(
                path,
                max_frames=self.config.video_max_frames,
                interval_s=self.config.video_frame_interval_s,
            )
            # frames is list[(np.ndarray, float)] — convert arrays to PIL
            from PIL import Image as PILImage

            images = []
            for arr, ts in frames:
                img = PILImage.fromarray(arr)
                images.append(img)
            media_data["images"] = images
            console.print(f"  Extracted {len(images)} frames")

            # Also try extracting audio track from video
            audio_path = extract_audio_from_video(path)
            if audio_path is not None:
                audio, sr = load_audio(audio_path)
                if audio.size > 0:
                    media_data["audio"] = audio
                    media_data["audio_sr"] = sr
                    console.print("  Extracted audio track from video")
                # Clean up temp file
                try:
                    audio_path.unlink()
                except OSError:
                    pass

        elif media_type == "audio":
            audio, sr = load_audio(path)
            if audio.size > 0:
                media_data["audio"] = audio
                media_data["audio_sr"] = sr
                duration = len(audio) / sr if sr > 0 else 0
                console.print(f"  Audio: {duration:.1f}s @ {sr}Hz")
            else:
                console.print("  [yellow]Warning:[/] Could not load audio data")

        return media_data

    def _load_modules(self) -> list[BaseModule]:
        """Instantiate all configured modules that are available."""
        modules: list[BaseModule] = []

        for name, mod_cfg in self.config.modules.items():
            if not mod_cfg.enabled:
                continue

            try:
                module = self._instantiate_module(name, self.config)
                if module is None:
                    if not mod_cfg.optional:
                        console.print(
                            f"  [yellow]Warning:[/] Required module "
                            f"'{name}' not found, skipping"
                        )
                    continue

                if not module.is_available():
                    console.print(
                        f"  [yellow]⚠[/] {name}: dependencies not "
                        f"installed, skipping"
                    )
                    continue

                module.prepare()
                modules.append(module)

            except Exception as e:
                if mod_cfg.optional:
                    logger.debug(f"Optional module {name} failed to load: {e}")
                else:
                    console.print(
                        f"  [red]Error:[/] Failed to load required "
                        f"module '{name}': {e}"
                    )

        return modules

    def _instantiate_module(
        self, name: str, config: PipelineConfig
    ) -> BaseModule | None:
        """Try to import and instantiate a module by name.

        Looks for:
        1. ``geofind.modules.<name>`` submodule with a ``<Name>Module`` class
        2. Falls back to checking ``_MODULE_CLASS_MAP`` for the class name
        """
        class_name = _MODULE_CLASS_MAP.get(name)
        if class_name is None:
            # Guess: ExifModule, ClipVisualModule, etc.
            class_name = "".join(
                part.capitalize() for part in name.split("_")
            ) + "Module"

        # Try submodule import
        module_path = f"geofind.modules.{name}"
        try:
            mod = importlib.import_module(module_path)
        except ImportError:
            return None

        cls = getattr(mod, class_name, None)
        if cls is None:
            return None

        if not issubclass(cls, BaseModule):
            logger.warning(
                f"{class_name} in {module_path} is not a BaseModule subclass"
            )
            return None

        return cls(config)

    def _run_module(
        self, module: BaseModule, media_data: dict[str, Any]
    ) -> list[ModuleHit] | None:
        """Run a single module, returning hits or None on failure.

        Returns None (not []) to distinguish "ran and found nothing" from
        "failed to run". The caller uses this to track success vs failure.
        """
        media_path = media_data["path"]
        media_type = media_data["type"]

        try:
            if media_type == "image":
                frames = media_data.get("images")
                return module.detect(media_path, frames=frames)

            elif media_type == "video":
                frames = media_data.get("images")
                audio_path = None
                return module.detect(
                    media_path, frames=frames, audio_path=audio_path
                )

            elif media_type == "audio":
                # Audio-only: pass audio_path for audio modules,
                # skip visual modules (handled by caller via skip logic)
                return module.detect(media_path, audio_path=media_path)

            else:
                logger.warning(f"Unknown media type: {media_type}")
                return []

        except NotImplementedError:
            # Module explicitly says it can't handle this input
            return []
        except Exception as e:
            mod_cfg = self.config.modules.get(module.name)
            is_optional = mod_cfg.optional if mod_cfg else True
            level = logging.DEBUG if is_optional else logging.WARNING
            logger.log(level, f"Module {module.name} failed: {e}", exc_info=True)
            return None

    def _print_summary(self, result: GeoResult) -> None:
        """Print a summary of the pipeline results."""
        console.print()
        console.rule("[bold cyan]Results")

        if result.top_candidate:
            tc = result.top_candidate
            exact_tag = " [bold yellow](EXACT GPS)[/]" if tc.is_exact else ""
            console.print(
                f"  [bold green]Top candidate:[/] "
                f"{tc.lat:.6f}, {tc.lon:.6f} "
                f"(p={tc.probability:.6f}){exact_tag}"
            )
        else:
            console.print("  [yellow]No candidates found[/]")

        console.print(
            f"  Consensus: {result.consensus_lat:.4f}, "
            f"{result.consensus_lon:.4f}"
        )
        console.print(
            f"  Agreement strength: {result.agreement_strength:.2%}"
        )
        console.print(
            f"  Modules: [green]{len(result.modules_run)}[/] run, "
            f"[red]{len(result.modules_failed)}[/] failed"
        )
        console.print(
            f"  Time: [bold]{result.processing_time_s:.2f}s[/]"
        )
