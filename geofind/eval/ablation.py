"""Ablation study framework for geofind pipeline.

Defines pipeline variants that disable specific components, then measures
the impact on accuracy. Each variant modifies PipelineConfig or the
reranking logic to isolate the contribution of individual features.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from geofind.core.config import PipelineConfig, ModuleConfig
from geofind.core.pipeline import GeoPipeline
from geofind.core.reranker import BayesianReranker
from geofind.core.candidate import CandidateLocation, GeoResult, ModuleHit
from geofind.core.grid import GeoGrid
from geofind.utils.geo import LatLon, haversine_km
from geofind.eval.metrics import EvalMetrics, ImageResult

logger = logging.getLogger(__name__)


@dataclass
class AblationVariant:
    """A single ablation variant — a modified pipeline configuration."""
    name: str
    description: str
    modify_config: Callable[[PipelineConfig], PipelineConfig] | None = None
    modify_reranker: bool = False  # If True, use custom reranker
    reranker_class: type | None = None


# ── Variant definitions ────────────────────────────────────────────────────

VARIANT_BASELINE = AblationVariant(
    name="baseline",
    description="Full pipeline — all modules, 3-pass reranking, hierarchical refinement",
)

VARIANT_NO_CONSENSUS = AblationVariant(
    name="no_consensus",
    description="Disable Pass 2 consensus clustering buff (set buff multipliers to 1.0)",
    modify_reranker=True,
)

VARIANT_NO_FUZZY_CAL = AblationVariant(
    name="no_fuzzy_cal",
    description="Disable Pass 3 fuzzy module calibration",
    modify_reranker=True,
)

VARIANT_NO_HIERARCHICAL = AblationVariant(
    name="no_hierarchical",
    description="Disable hierarchical fine-grid refinement",
    modify_config=lambda c: setattr(c, "hierarchical_enabled", False) or c,
)

VARIANT_EQUAL_WEIGHTS = AblationVariant(
    name="equal_weights",
    description="All modules at weight=1.0 (remove weight bias)",
    modify_config=lambda c: _set_all_weights(c, 1.0),
)

VARIANT_EXIF_ONLY = AblationVariant(
    name="exif_only",
    description="Only EXIF module (upper bound for GPS-enabled images)",
    modify_config=lambda c: _enable_only(c, {"exif"}),
)

VARIANT_NO_EXIF = AblationVariant(
    name="no_exif",
    description="All modules except EXIF (visual-only accuracy)",
    modify_config=lambda c: _disable_modules(c, {"exif"}),
)

VARIANT_LOW_CONSENSUS_BUFF = AblationVariant(
    name="low_consensus_buff",
    description="Reduced consensus buff (1.1x instead of 3.0x for top-N)",
    modify_reranker=True,
)

VARIANT_NO_PROXIMITY_PENALTY = AblationVariant(
    name="no_proximity_penalty",
    description="Disable Pass 3 proximity penalty for distant candidates",
    modify_reranker=True,
)

VARIANT_NO_MULTI_MODULE_BONUS = AblationVariant(
    name="no_multi_module_bonus",
    description="Disable Pass 3 multi-module agreement bonus",
    modify_reranker=True,
)


# All named variants (used by the ablation runner)
ALL_VARIANTS: dict[str, AblationVariant] = {
    v.name: v for v in [
        VARIANT_BASELINE,
        VARIANT_NO_CONSENSUS,
        VARIANT_NO_FUZZY_CAL,
        VARIANT_NO_HIERARCHICAL,
        VARIANT_EQUAL_WEIGHTS,
        VARIANT_EXIF_ONLY,
        VARIANT_NO_EXIF,
        VARIANT_LOW_CONSENSUS_BUFF,
        VARIANT_NO_PROXIMITY_PENALTY,
        VARIANT_NO_MULTI_MODULE_BONUS,
    ]
}

# Core ablation variants (the key ones for understanding reranking)
CORE_VARIANTS = ["baseline", "no_consensus", "no_fuzzy_cal", "no_hierarchical"]


# ── Custom rerankers for ablation ──────────────────────────────────────────

class NoConsensusReranker(BayesianReranker):
    """Reranker with consensus buff disabled."""

    def _apply_consensus_buff(self, candidates, weights):
        for c in candidates:
            c.buff_multiplier = 1.0
        return candidates, {"centroid_lat": 0.0, "centroid_lon": 0.0, "strength": 0.0, "definitive_count": 0}


class NoFuzzyCalReranker(BayesianReranker):
    """Reranker with fuzzy calibration disabled."""

    def _fuzzy_calibration(self, candidates, consensus, weights):
        return candidates


class LowConsensusBuffReranker(BayesianReranker):
    """Reranker with reduced consensus buff."""

    def _apply_consensus_buff(self, candidates, weights):
        if not candidates:
            return candidates, {"centroid": None, "strength": 0.0}

        definitive = [c for c in candidates[:100] if len(c.hits) >= 2]
        if not definitive:
            definitive = candidates[:20]

        total_weight = sum(
            max(h.confidence, 0.01) for c in definitive for h in c.hits
        )
        if total_weight == 0:
            return candidates, {"centroid": None, "strength": 0.0}

        centroid_lat = sum(
            c.lat * sum(max(h.confidence, 0.01) for h in c.hits)
            for c in definitive
        ) / total_weight
        centroid_lon = sum(
            c.lon * sum(max(h.confidence, 0.01) for h in c.hits)
            for c in definitive
        ) / total_weight
        centroid = LatLon(centroid_lat, centroid_lon)

        dists = [haversine_km(centroid, LatLon(c.lat, c.lon)) for c in candidates]
        sorted_by_dist = sorted(range(len(candidates)), key=lambda i: dists[i])

        top_dists = dists[:min(20, len(dists))]
        avg_dist = sum(top_dists) / len(top_dists) if top_dists else 1000
        strength = max(0.0, min(1.0, 1.0 - (avg_dist - 100) / 1900))

        top_n = self.config.consensus_top_n
        top_half = len(candidates) // 2

        for c in candidates:
            c.buff_multiplier = 1.0

        for rank, idx in enumerate(sorted_by_dist):
            if rank < top_n:
                candidates[idx].buff_multiplier = 1.1  # Much lower than default 3.0
            elif rank < top_half:
                candidates[idx].buff_multiplier = 1.05
            candidates[idx].buff_multiplier = 1.0 + (candidates[idx].buff_multiplier - 1.0) * strength

        return candidates, {
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
            "strength": strength,
            "definitive_count": len(definitive),
        }


class NoProximityPenaltyReranker(BayesianReranker):
    """Reranker with Pass 3 proximity penalty disabled."""

    def _fuzzy_calibration(self, candidates, consensus, weights):
        if not candidates or not consensus.get("centroid_lat"):
            return candidates

        for candidate in candidates:
            if len(candidate.hits) >= 3:
                candidate.probability *= 1.2
            elif len(candidate.hits) >= 2:
                candidate.probability *= 1.1

        total = sum(c.probability for c in candidates)
        if total > 0:
            for c in candidates:
                c.probability /= total

        return candidates


class NoMultiModuleBonusReranker(BayesianReranker):
    """Reranker with multi-module bonus disabled."""

    def _fuzzy_calibration(self, candidates, consensus, weights):
        if not candidates or not consensus.get("centroid_lat"):
            return candidates

        centroid = LatLon(consensus["centroid_lat"], consensus["centroid_lon"])

        for candidate in candidates:
            cand_point = LatLon(candidate.lat, candidate.lon)
            dist_to_centroid = haversine_km(cand_point, centroid)

            if dist_to_centroid > 3000:
                proximity_factor = math.exp(
                    -((dist_to_centroid - 3000) ** 2) / (2 * 5000 ** 2)
                )
                candidate.probability *= (0.8 + 0.2 * proximity_factor)

        total = sum(c.probability for c in candidates)
        if total > 0:
            for c in candidates:
                c.probability /= total

        return candidates


# Map variant names to custom reranker classes
_RERANKER_MAP: dict[str, type] = {
    "no_consensus": NoConsensusReranker,
    "no_fuzzy_cal": NoFuzzyCalReranker,
    "low_consensus_buff": LowConsensusBuffReranker,
    "no_proximity_penalty": NoProximityPenaltyReranker,
    "no_multi_module_bonus": NoMultiModuleBonusReranker,
}


# ── Helper functions ────────────────────────────────────────────────────────

def _set_all_weights(config: PipelineConfig, weight: float) -> PipelineConfig:
    for name in config.modules:
        config.modules[name].weight = weight
    return config


def _enable_only(config: PipelineConfig, names: set[str]) -> PipelineConfig:
    for name in config.modules:
        config.modules[name].enabled = name in names
    return config


def _disable_modules(config: PipelineConfig, names: set[str]) -> PipelineConfig:
    for name in names:
        if name in config.modules:
            config.modules[name].enabled = False
    return config


# ── Ablation runner ────────────────────────────────────────────────────────

def run_ablation(
    images: list[Any],
    pipeline_fn: Callable[[Path, PipelineConfig], GeoResult],
    variants: list[str] | None = None,
    image_dir: Path | None = None,
    strip_exif: bool = True,
    progress_callback: Callable[[str, str, int, int], None] | None = None,
) -> dict[str, EvalMetrics]:
    """Run ablation study across multiple pipeline variants.

    Args:
        images: List of EvalImage objects (or similar with id, lat, lon, local_path).
        pipeline_fn: Function(image_path, config) -> GeoResult.
        variants: List of variant names. None = core variants only.
        image_dir: Where images are stored.
        strip_exif: Whether to use EXIF-stripped images.
        progress_callback: fn(variant_name, image_id, current, total).

    Returns:
        Dict mapping variant name -> EvalMetrics.
    """
    if variants is None:
        variants = CORE_VARIANTS

    results: dict[str, EvalMetrics] = {}

    for variant_name in variants:
        if variant_name not in ALL_VARIANTS:
            logger.warning(f"Unknown variant: {variant_name}, skipping")
            continue

        variant = ALL_VARIANTS[variant_name]
        logger.info(f"Ablation: {variant.name} — {variant.description}")

        metrics = EvalMetrics(variant=variant_name)
        total = len(images)

        for i, img in enumerate(images):
            if progress_callback:
                progress_callback(variant_name, img.id, i + 1, total)

            # Get image path
            if strip_exif and hasattr(img, "stripped_path") and img.stripped_path:
                img_path = img.stripped_path
            elif hasattr(img, "local_path") and img.local_path:
                img_path = img.local_path
            else:
                metrics.record_error(img.id, meta={"name": img.name})
                continue

            if not img_path.exists():
                metrics.record_error(img.id, meta={"name": img.name})
                continue

            # Build config for this variant
            config = PipelineConfig()
            if variant.modify_config:
                config = variant.modify_config(config)

            try:
                result = pipeline_fn(img_path, config)
                metrics.record(
                    img.id,
                    expected_lat=img.lat,
                    expected_lon=img.lon,
                    result=result,
                    meta={"name": img.name, "source": getattr(img, "source", "unknown")},
                )
            except Exception as e:
                logger.debug(f"Pipeline failed for {img.id} [{variant_name}]: {e}")
                metrics.record_error(img.id, meta={"name": img.name}, error_message=str(e))

        results[variant_name] = metrics
        s = metrics.summary()
        if "error" not in s:
            logger.info(
                f"  {variant_name}: avg={s['avg_distance_km']:.2f}km, "
                f"1km={s['accuracy_1km']:.1%}, 10km={s['accuracy_10km']:.1%}"
            )

    return results


def default_pipeline_fn(
    image_path: Path, config: PipelineConfig, variant_name: str = "baseline"
) -> GeoResult:
    """Default pipeline function that applies variant-specific reranker."""
    pipeline = GeoPipeline(config)

    # Inject custom reranker if needed
    if variant_name in _RERANKER_MAP:
        pipeline.reranker = _RERANKER_MAP[variant_name](config)

    return pipeline.analyze(image_path)
