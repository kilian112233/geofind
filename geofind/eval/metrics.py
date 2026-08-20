"""Evaluation metrics — distance statistics, bootstrap CI, CSV export.

Wraps the existing AccuracyTracker from dev/accuracy.py and adds
statistical tools for comparing pipeline variants.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from geofind.utils.geo import LatLon, haversine_km

# Re-export from dev accuracy for compatibility
from geofind.core.candidate import GeoResult

# Thresholds in km
THRESHOLDS = [0.1, 1.0, 10.0, 50.0, 100.0]
THRESHOLD_NAMES = ["100m", "1km", "10km", "50km", "100km"]


@dataclass
class ImageResult:
    """Result for a single evaluation image."""
    id: str
    name: str
    expected_lat: float
    expected_lon: float
    predicted_lat: float = 0.0
    predicted_lon: float = 0.0
    distance_km: float = -1.0
    category: str = "unknown"
    continent: str = "unknown"
    difficulty: str = "unknown"
    source: str = "unknown"

    # Threshold flags
    within_100m: bool = False
    within_1km: bool = False
    within_10km: bool = False
    within_50km: bool = False
    within_100km: bool = False

    # Module analysis
    modules_run: list[str] = field(default_factory=list)
    modules_failed: list[str] = field(default_factory=list)
    module_contributions: list[dict[str, Any]] = field(default_factory=list)
    processing_time_s: float = 0.0

    # Status
    success: bool = False
    error_message: str = ""
    variant: str = "baseline"


@dataclass
class EvalMetrics:
    """Aggregate metrics across all evaluation images."""
    results: list[ImageResult] = field(default_factory=list)
    variant: str = "baseline"

    def record(
        self,
        image_id: str,
        expected_lat: float,
        expected_lon: float,
        result: GeoResult,
        meta: dict[str, Any] | None = None,
    ) -> ImageResult:
        """Record a single pipeline result."""
        meta = meta or {}
        expected = LatLon(expected_lat, expected_lon)

        ir = ImageResult(
            id=image_id,
            name=meta.get("name", image_id),
            expected_lat=expected_lat,
            expected_lon=expected_lon,
            category=meta.get("category", "unknown"),
            continent=meta.get("continent", "unknown"),
            difficulty=meta.get("difficulty", "unknown"),
            source=meta.get("source", "unknown"),
            modules_run=result.modules_run,
            modules_failed=result.modules_failed,
            processing_time_s=result.processing_time_s,
            variant=self.variant,
        )

        if result.top_candidate:
            tc = result.top_candidate
            ir.predicted_lat = tc.lat
            ir.predicted_lon = tc.lon
            ir.distance_km = haversine_km(expected, LatLon(tc.lat, tc.lon))
            ir.within_100m = ir.distance_km <= 0.1
            ir.within_1km = ir.distance_km <= 1.0
            ir.within_10km = ir.distance_km <= 10.0
            ir.within_50km = ir.distance_km <= 50.0
            ir.within_100km = ir.distance_km <= 100.0

            for cand in result.candidates[:20]:
                cand_dist = haversine_km(expected, LatLon(cand.lat, cand.lon))
                for hit in cand.hits:
                    ir.module_contributions.append({
                        "module": hit.module,
                        "candidate_dist_km": round(cand_dist, 3),
                        "confidence": hit.confidence,
                    })

        ir.success = True
        self.results.append(ir)
        return ir

    def record_error(
        self,
        image_id: str,
        meta: dict[str, Any] | None = None,
        error_message: str = "",
    ) -> ImageResult:
        """Record a failed image."""
        meta = meta or {}
        ir = ImageResult(
            id=image_id,
            name=meta.get("name", image_id),
            expected_lat=meta.get("expected_lat", 0.0),
            expected_lon=meta.get("expected_lon", 0.0),
            category=meta.get("category", "unknown"),
            continent=meta.get("continent", "unknown"),
            difficulty=meta.get("difficulty", "unknown"),
            source=meta.get("source", "unknown"),
            success=False,
            error_message=error_message,
            variant=self.variant,
        )
        self.results.append(ir)
        return ir

    @property
    def successful(self) -> list[ImageResult]:
        return [r for r in self.results if r.success]

    def summary(self) -> dict[str, Any]:
        """Compute aggregate statistics."""
        ok = self.successful
        n = len(ok)
        if n == 0:
            return {"error": "no successful results", "variant": self.variant}

        errors = [r.distance_km for r in ok]

        thresholds_met = {}
        for t, name in zip(THRESHOLDS, THRESHOLD_NAMES):
            count = sum(1 for e in errors if e <= t)
            thresholds_met[f"within_{name}"] = count
            thresholds_met[f"accuracy_{name}"] = count / n

        # Module effectiveness
        mod_hits: dict[str, int] = defaultdict(int)
        mod_total: dict[str, int] = defaultdict(int)
        for r in ok:
            seen: set[str] = set()
            for mc in r.module_contributions:
                mod_total[mc["module"]] += 1
                if mc["candidate_dist_km"] <= 100 and mc["module"] not in seen:
                    mod_hits[mc["module"]] += 1
                    seen.add(mc["module"])

        return {
            "variant": self.variant,
            "total": n,
            "failed": len(self.results) - n,
            "avg_distance_km": round(statistics.mean(errors), 4),
            "median_distance_km": round(statistics.median(errors), 4),
            "std_distance_km": round(statistics.stdev(errors), 4) if n > 1 else 0.0,
            "min_distance_km": round(min(errors), 4),
            "max_distance_km": round(max(errors), 4),
            "avg_processing_time_s": round(
                statistics.mean(r.processing_time_s for r in ok), 3
            ),
            **thresholds_met,
            "module_hit_rate": {
                mod: {
                    "hits": mod_hits.get(mod, 0),
                    "total": total,
                    "rate": mod_hits.get(mod, 0) / total if total else 0,
                }
                for mod, total in sorted(mod_total.items())
            },
        }

    def bootstrap_ci(
        self,
        metric_fn: Any = None,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
        seed: int = 42,
    ) -> dict[str, float]:
        """Compute bootstrap confidence interval for a metric.

        Args:
            metric_fn: Function(metrics_list) -> float. Defaults to mean distance.
            n_bootstrap: Number of bootstrap samples.
            ci: Confidence level (e.g. 0.95 for 95% CI).
            seed: Random seed for reproducibility.

        Returns:
            Dict with 'point_estimate', 'ci_lower', 'ci_upper'.
        """
        ok = self.successful
        if not ok:
            return {"point_estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

        if metric_fn is None:
            metric_fn = lambda rs: statistics.mean(r.distance_km for r in rs)

        point = metric_fn(ok)
        rng = __import__("random").Random(seed)
        boot_values: list[float] = []

        for _ in range(n_bootstrap):
            sample = rng.choices(ok, k=len(ok))
            boot_values.append(metric_fn(sample))

        boot_values.sort()
        lower_idx = int((1 - ci) / 2 * n_bootstrap)
        upper_idx = int((1 + ci) / 2 * n_bootstrap) - 1

        return {
            "point_estimate": round(point, 4),
            "ci_lower": round(boot_values[lower_idx], 4),
            "ci_upper": round(boot_values[min(upper_idx, len(boot_values) - 1)], 4),
            "ci_level": ci,
            "n_bootstrap": n_bootstrap,
        }

    def accuracy_ci(
        self, threshold_km: float = 1.0, n_bootstrap: int = 1000, ci: float = 0.95
    ) -> dict[str, float]:
        """Bootstrap CI for accuracy at a given threshold."""
        ok = self.successful
        if not ok:
            return {"point_estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

        point = sum(1 for r in ok if r.distance_km <= threshold_km) / len(ok)
        rng = __import__("random").Random(42)
        boot_values: list[float] = []

        for _ in range(n_bootstrap):
            sample = rng.choices(ok, k=len(ok))
            acc = sum(1 for r in sample if r.distance_km <= threshold_km) / len(sample)
            boot_values.append(acc)

        boot_values.sort()
        lower_idx = int((1 - ci) / 2 * n_bootstrap)
        upper_idx = int((1 + ci) / 2 * n_bootstrap) - 1

        return {
            "point_estimate": round(point, 4),
            "ci_lower": round(boot_values[lower_idx], 4),
            "ci_upper": round(boot_values[min(upper_idx, len(boot_values) - 1)], 4),
            "threshold_km": threshold_km,
        }

    def to_csv(self, path: Path) -> None:
        """Export per-image results to CSV."""
        ok = self.successful
        if not ok:
            return

        fieldnames = [
            "id", "name", "expected_lat", "expected_lon",
            "predicted_lat", "predicted_lon", "distance_km",
            "within_100m", "within_1km", "within_10km",
            "within_50km", "within_100km",
            "category", "continent", "difficulty", "source",
            "modules_run", "processing_time_s", "variant",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in ok:
                writer.writerow({
                    "id": r.id,
                    "name": r.name,
                    "expected_lat": r.expected_lat,
                    "expected_lon": r.expected_lon,
                    "predicted_lat": r.predicted_lat,
                    "predicted_lon": r.predicted_lon,
                    "distance_km": round(r.distance_km, 4),
                    "within_100m": r.within_100m,
                    "within_1km": r.within_1km,
                    "within_10km": r.within_10km,
                    "within_50km": r.within_50km,
                    "within_100km": r.within_100km,
                    "category": r.category,
                    "continent": r.continent,
                    "difficulty": r.difficulty,
                    "source": r.source,
                    "modules_run": ",".join(r.modules_run),
                    "processing_time_s": round(r.processing_time_s, 3),
                    "variant": r.variant,
                })

    def to_json(self, path: Path | None = None) -> dict[str, Any]:
        """Export results as JSON-serializable dict."""
        out = {
            "variant": self.variant,
            "summary": self.summary(),
            "results": [
                {
                    "id": r.id,
                    "name": r.name,
                    "expected_lat": r.expected_lat,
                    "expected_lon": r.expected_lon,
                    "predicted_lat": r.predicted_lat,
                    "predicted_lon": r.predicted_lon,
                    "distance_km": round(r.distance_km, 4) if r.success else None,
                    "within_100m": r.within_100m,
                    "within_1km": r.within_1km,
                    "within_10km": r.within_10km,
                    "category": r.category,
                    "continent": r.continent,
                    "difficulty": r.difficulty,
                    "modules_run": r.modules_run,
                    "processing_time_s": r.processing_time_s,
                    "success": r.success,
                    "error_message": r.error_message,
                }
                for r in self.results
            ],
        }
        if path:
            path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        return out

    @classmethod
    def from_json(cls, path: Path) -> EvalMetrics:
        """Load metrics from a JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        m = cls(variant=data.get("variant", "loaded"))
        for rd in data.get("results", []):
            ir = ImageResult(
                id=rd["id"],
                name=rd.get("name", rd["id"]),
                expected_lat=rd["expected_lat"],
                expected_lon=rd["expected_lon"],
                predicted_lat=rd.get("predicted_lat", 0.0),
                predicted_lon=rd.get("predicted_lon", 0.0),
                distance_km=rd.get("distance_km") or -1.0,
                category=rd.get("category", "unknown"),
                continent=rd.get("continent", "unknown"),
                difficulty=rd.get("difficulty", "unknown"),
                within_100m=rd.get("within_100m", False),
                within_1km=rd.get("within_1km", False),
                within_10km=rd.get("within_10km", False),
                modules_run=rd.get("modules_run", []),
                processing_time_s=rd.get("processing_time_s", 0.0),
                success=rd.get("success", False),
                error_message=rd.get("error_message", ""),
                variant=rd.get("variant", "loaded"),
            )
            m.results.append(ir)
        return m

    def compare(self, other: EvalMetrics) -> dict[str, Any]:
        """Compare this variant against another (usually baseline)."""
        s1 = self.summary()
        s2 = other.summary()
        if "error" in s1 or "error" in s2:
            return {"error": "one or both variants have no results"}

        return {
            "baseline": other.variant,
            "variant": self.variant,
            "avg_distance_delta_km": round(
                s1["avg_distance_km"] - s2["avg_distance_km"], 4
            ),
            "median_distance_delta_km": round(
                s1["median_distance_km"] - s2["median_distance_km"], 4
            ),
            "accuracy_deltas": {
                name: round(
                    s1[f"accuracy_{name}"] - s2[f"accuracy_{name}"], 4
                )
                for name in THRESHOLD_NAMES
            },
            "baseline_summary": s2,
            "variant_summary": s1,
        }
