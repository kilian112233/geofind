#!/usr/bin/env python3
"""Accuracy measurement and reporting for geofind test harness.

Tracks per-image results, computes distance errors, near-miss analysis,
and generates aggregate statistics with per-category breakdowns.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Import from geofind ───────────────────────────────────────────────────────

from geofind.core.candidate import CandidateLocation, GeoResult, ModuleHit
from geofind.utils.geo import LatLon, haversine_km


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ImageResult:
    """Result for a single test image."""
    test_id: str
    name: str
    expected_lat: float
    expected_lon: float
    category: str
    continent: str
    difficulty: str

    # Computed metrics
    distance_error_km: float = -1.0
    distance_error_m: float = -1.0
    within_100m: bool = False
    within_1km: bool = False
    within_10km: bool = False
    within_50km: bool = False
    within_100km: bool = False

    # Top-N analysis
    top_n_rank: int = -1  # rank of closest candidate (1-indexed), -1 if none within 100m
    near_miss: bool = False  # correct location was outranked
    num_candidates: int = 0

    # Module analysis
    module_contributions: list[dict[str, Any]] = field(default_factory=list)
    outranked_modules: list[str] = field(default_factory=list)
    top_candidate_lat: float = 0.0
    top_candidate_lon: float = 0.0
    top_candidate_prob: float = 0.0
    is_exact_gps: bool = False  # True if top candidate came from exact EXIF GPS

    # Pipeline info
    modules_run: list[str] = field(default_factory=list)
    modules_failed: list[str] = field(default_factory=list)
    processing_time_s: float = 0.0

    # Success / failure
    success: bool = False
    error_message: str = ""


@dataclass
class AccuracyTracker:
    """Collects and computes accuracy metrics across all test images."""

    results: list[ImageResult] = field(default_factory=list)
    is_no_exif: bool = False

    # With-EXIF targets
    target_accuracy_100m: float = 0.50  # 50% target within 100m
    target_accuracy_1km: float = 0.75   # 75% target within 1km
    # Without-EXIF targets (wider thresholds)
    target_accuracy_10km: float = 0.20  # 20% target within 10km
    target_accuracy_100km: float = 0.50 # 50% target within 100km

    def __post_init__(self) -> None:
        if self.is_no_exif:
            self.target_accuracy_100m = 0.05
            self.target_accuracy_1km = 0.10

    def record(
        self,
        test_id: str,
        expected_lat: float,
        expected_lon: float,
        result: GeoResult,
        meta: dict[str, Any] | None = None,
    ) -> ImageResult:
        """Record a single test result and compute all metrics.

        Args:
            test_id: Unique test identifier.
            expected_lat: Expected latitude.
            expected_lon: Expected longitude.
            result: Pipeline GeoResult.
            meta: Optional metadata (name, category, continent, difficulty).

        Returns:
            The ImageResult with all metrics populated.
        """
        if meta is None:
            meta = {}

        expected = LatLon(expected_lat, expected_lon)
        ir = ImageResult(
            test_id=test_id,
            name=meta.get("name", test_id),
            expected_lat=expected_lat,
            expected_lon=expected_lon,
            category=meta.get("category", "unknown"),
            continent=meta.get("continent", "unknown"),
            difficulty=meta.get("difficulty", "unknown"),
            modules_run=result.modules_run,
            modules_failed=result.modules_failed,
            processing_time_s=result.processing_time_s,
            num_candidates=len(result.candidates),
        )

        # ── Top candidate distance ───────────────────────────────────────
        if result.top_candidate:
            tc = result.top_candidate
            ir.top_candidate_lat = tc.lat
            ir.top_candidate_lon = tc.lon
            ir.top_candidate_prob = tc.probability
            ir.is_exact_gps = getattr(tc, "is_exact", False)

            tc_point = LatLon(tc.lat, tc.lon)
            ir.distance_error_km = haversine_km(expected, tc_point)
            ir.distance_error_m = ir.distance_error_km * 1000.0
            ir.within_100m = ir.distance_error_m <= 100
            ir.within_1km = ir.distance_error_km <= 1.0
            ir.within_10km = ir.distance_error_km <= 10.0
            ir.within_50km = ir.distance_error_km <= 50.0
            ir.within_100km = ir.distance_error_km <= 100.0

        # ── Top-N rank (closest candidate within 100m) ──────────────────
        best_rank = -1
        best_dist = float("inf")
        for rank, cand in enumerate(result.candidates[:20], 1):
            cand_point = LatLon(cand.lat, cand.lon)
            dist = haversine_km(expected, cand_point)
            if dist < best_dist:
                best_dist = dist
            if dist <= 0.1:  # within 100m
                if best_rank == -1 or rank < best_rank:
                    best_rank = rank
                    break

        ir.top_n_rank = best_rank
        ir.near_miss = (best_rank > 1) and (best_rank != -1)

        # ── Module contributions ─────────────────────────────────────────
        for cand in result.candidates[:20]:
            cand_point = LatLon(cand.lat, cand.lon)
            cand_dist = haversine_km(expected, cand_point)
            for hit in cand.hits:
                ir.module_contributions.append({
                    "module": hit.module,
                    "candidate_lat": cand.lat,
                    "candidate_lon": cand.lon,
                    "candidate_dist_km": round(cand_dist, 3),
                    "hit_lat": hit.lat,
                    "hit_lon": hit.lon,
                    "confidence": hit.confidence,
                })

        # ── Outranked modules (found right area but were outranked) ──────
        if ir.near_miss:
            # Modules that hit near expected but weren't in the top candidate
            outranked: set[str] = set()
            for cand in result.candidates[1:20]:
                cand_point = LatLon(cand.lat, cand.lon)
                if haversine_km(expected, cand_point) <= 0.1:
                    for hit in cand.hits:
                        outranked.add(hit.module)
            ir.outranked_modules = sorted(outranked)

        ir.success = True
        self.results.append(ir)
        return ir

    def record_error(
        self,
        test_id: str,
        meta: dict[str, Any] | None = None,
        error_message: str = "",
    ) -> ImageResult:
        """Record a failed test (pipeline error, image not found, etc.)."""
        if meta is None:
            meta = {}
        ir = ImageResult(
            test_id=test_id,
            name=meta.get("name", test_id),
            expected_lat=meta.get("expected_lat", 0.0),
            expected_lon=meta.get("expected_lon", 0.0),
            category=meta.get("category", "unknown"),
            continent=meta.get("continent", "unknown"),
            difficulty=meta.get("difficulty", "unknown"),
            success=False,
            error_message=error_message,
        )
        self.results.append(ir)
        return ir

    def summary(self) -> dict[str, Any]:
        """Compute aggregate accuracy statistics."""
        ok = [r for r in self.results if r.success]
        if not ok:
            return {"error": "no successful results"}

        total = len(ok)
        errors = [r.distance_error_km for r in ok]

        passed_100m = sum(1 for r in ok if r.within_100m)
        passed_1km = sum(1 for r in ok if r.within_1km)
        passed_10km = sum(1 for r in ok if r.within_10km)
        passed_50km = sum(1 for r in ok if r.within_50km)
        passed_100km = sum(1 for r in ok if r.within_100km)
        exact_gps_count = sum(1 for r in ok if r.is_exact_gps)

        near_miss_count = sum(1 for r in ok if r.near_miss)

        # Worst cases
        worst = sorted(ok, key=lambda r: r.distance_error_km, reverse=True)[:5]

        # Per-category accuracy
        cat_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"total": 0, "pass_100m": 0, "pass_1km": 0})
        for r in ok:
            cat_stats[r.category]["total"] += 1
            if r.within_100m:
                cat_stats[r.category]["pass_100m"] += 1
            if r.within_1km:
                cat_stats[r.category]["pass_1km"] += 1

        per_category = {}
        for cat, s in cat_stats.items():
            t = s["total"]
            per_category[cat] = {
                "total": t,
                "accuracy_100m": s["pass_100m"] / t if t else 0,
                "accuracy_1km": s["pass_1km"] / t if t else 0,
            }

        # Per-difficulty accuracy
        diff_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"total": 0, "pass_100m": 0, "pass_1km": 0})
        for r in ok:
            diff_stats[r.difficulty]["total"] += 1
            if r.within_100m:
                diff_stats[r.difficulty]["pass_100m"] += 1
            if r.within_1km:
                diff_stats[r.difficulty]["pass_1km"] += 1

        per_difficulty = {}
        for diff, s in diff_stats.items():
            t = s["total"]
            per_difficulty[diff] = {
                "total": t,
                "accuracy_100m": s["pass_100m"] / t if t else 0,
                "accuracy_1km": s["pass_1km"] / t if t else 0,
            }

        # Per-continent accuracy
        cont_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"total": 0, "pass_100m": 0, "pass_1km": 0})
        for r in ok:
            cont_stats[r.continent]["total"] += 1
            if r.within_100m:
                cont_stats[r.continent]["pass_100m"] += 1
            if r.within_1km:
                cont_stats[r.continent]["pass_1km"] += 1

        per_continent = {}
        for cont, s in cont_stats.items():
            t = s["total"]
            per_continent[cont] = {
                "total": t,
                "accuracy_100m": s["pass_100m"] / t if t else 0,
                "accuracy_1km": s["pass_1km"] / t if t else 0,
            }

        # Module hit rate: how often each module contributed to the closest candidate
        mod_hits: dict[str, int] = defaultdict(int)
        mod_total: dict[str, int] = defaultdict(int)
        for r in ok:
            seen_modules: set[str] = set()
            for mc in r.module_contributions:
                mod_total[mc["module"]] += 1
                if mc["candidate_dist_km"] <= 100:  # within 100km
                    if mc["module"] not in seen_modules:
                        mod_hits[mc["module"]] += 1
                        seen_modules.add(mc["module"])

        module_hit_rate = {}
        for mod in sorted(mod_total.keys()):
            ht = mod_total[mod]
            module_hit_rate[mod] = {
                "total_hits": ht,
                "area_hits": mod_hits.get(mod, 0),
                "hit_rate": mod_hits.get(mod, 0) / ht if ht else 0,
            }

        return {
            "total_tests": total,
            "failed_tests": len(self.results) - total,
            "passed_100m": passed_100m,
            "accuracy_100m": passed_100m / total,
            "passed_1km": passed_1km,
            "accuracy_1km": passed_1km / total,
            "passed_10km": passed_10km,
            "accuracy_10km": passed_10km / total,
            "passed_50km": passed_50km,
            "accuracy_50km": passed_50km / total,
            "passed_100km": passed_100km,
            "accuracy_100km": passed_100km / total,
            "avg_distance_error_km": statistics.mean(errors) if errors else 0,
            "median_distance_error_km": statistics.median(errors) if errors else 0,
            "std_distance_error_km": statistics.stdev(errors) if len(errors) > 1 else 0,
            "worst_cases": [
                {
                    "test_id": r.test_id,
                    "name": r.name,
                    "distance_km": round(r.distance_error_km, 2),
                    "category": r.category,
                    "difficulty": r.difficulty,
                }
                for r in worst
            ],
            "near_miss_count": near_miss_count,
            "near_miss_rate": near_miss_count / total if total else 0,
            "per_category_accuracy": per_category,
            "per_difficulty_accuracy": per_difficulty,
            "per_continent_accuracy": per_continent,
            "module_hit_rate": module_hit_rate,
            "exact_gps_count": exact_gps_count,
            "target_accuracy_100m": self.target_accuracy_100m,
            "target_accuracy_1km": self.target_accuracy_1km,
            "target_accuracy_10km": self.target_accuracy_10km,
            "target_accuracy_100km": self.target_accuracy_100km,
            "meets_100m_target": (passed_100m / total) >= self.target_accuracy_100m if total else False,
            "meets_1km_target": (passed_1km / total) >= self.target_accuracy_1km if total else False,
            "meets_10km_target": (passed_10km / total) >= self.target_accuracy_10km if total else False,
            "meets_100km_target": (passed_100km / total) >= self.target_accuracy_100km if total else False,
            "is_no_exif": self.is_no_exif,
        }

    def detailed_report(self) -> str:
        """Generate a human-readable detailed accuracy report."""
        s = self.summary()
        if "error" in s:
            return f"No results to report: {s['error']}"

        lines: list[str] = []
        lines.append("=" * 72)
        lines.append("           GEOFIND ACCURACY REPORT")
        if self.is_no_exif:
            lines.append("           (EXIF GPS DISABLED)")
        lines.append("=" * 72)
        lines.append("")

        # Overall
        lines.append(f"Total tests: {s['total_tests']} ({s['failed_tests']} failed)")
        lines.append("")
        lines.append("─── Accuracy by Threshold ─────────────────────────────────────────")
        lines.append(f"  Within  100m:  {s['passed_100m']:3d}/{s['total_tests']}  ({s['accuracy_100m']:.1%})")
        lines.append(f"  Within    1km: {s['passed_1km']:3d}/{s['total_tests']}  ({s['accuracy_1km']:.1%})")
        lines.append(f"  Within   10km: {s['passed_10km']:3d}/{s['total_tests']}  ({s['accuracy_10km']:.1%})")
        lines.append(f"  Within   50km: {s['passed_50km']:3d}/{s['total_tests']}  ({s['accuracy_50km']:.1%})")
        lines.append(f"  Within  100km: {s['passed_100km']:3d}/{s['total_tests']}  ({s['accuracy_100km']:.1%})")
        lines.append("")

        lines.append(f"  Avg error:     {s['avg_distance_error_km']:.2f} km")
        lines.append(f"  Median error:  {s['median_distance_error_km']:.2f} km")
        lines.append(f"  Std dev:       {s['std_distance_error_km']:.2f} km")
        lines.append("")

        # Targets
        m100 = "PASS" if s["meets_100m_target"] else "FAIL"
        m1k = "PASS" if s["meets_1km_target"] else "FAIL"
        lines.append(f"  100m target ({s['target_accuracy_100m']:.0%}): [{m100}]")
        lines.append(f"   1km target ({s['target_accuracy_1km']:.0%}): [{m1k}]")
        if self.is_no_exif:
            m10k = "PASS" if s["meets_10km_target"] else "FAIL"
            m100k = "PASS" if s["meets_100km_target"] else "FAIL"
            lines.append(f"  10km target ({s['target_accuracy_10km']:.0%}): [{m10k}]")
            lines.append(f" 100km target ({s['target_accuracy_100km']:.0%}): [{m100k}]")
        lines.append("")

        # Near misses
        lines.append(f"Near-misses: {s['near_miss_count']} ({s['near_miss_rate']:.1%})")
        lines.append("")

        # Per-category
        lines.append("─── Accuracy by Category ──────────────────────────────────────────")
        for cat, stats in sorted(s["per_category_accuracy"].items()):
            lines.append(f"  {cat:15s}: {stats['accuracy_100m']:.0%} 100m / {stats['accuracy_1km']:.0%} 1km  (n={stats['total']})")
        lines.append("")

        # Per-difficulty
        lines.append("─── Accuracy by Difficulty ────────────────────────────────────────")
        for diff, stats in sorted(s["per_difficulty_accuracy"].items()):
            lines.append(f"  {diff:10s}: {stats['accuracy_100m']:.0%} 100m / {stats['accuracy_1km']:.0%} 1km  (n={stats['total']})")
        lines.append("")

        # Per-continent
        lines.append("─── Accuracy by Continent ─────────────────────────────────────────")
        for cont, stats in sorted(s["per_continent_accuracy"].items()):
            lines.append(f"  {cont:12s}: {stats['accuracy_100m']:.0%} 100m / {stats['accuracy_1km']:.0%} 1km  (n={stats['total']})")
        lines.append("")

        # Module hit rates
        lines.append("─── Module Hit Rates ──────────────────────────────────────────────")
        for mod, stats in sorted(s["module_hit_rate"].items()):
            lines.append(f"  {mod:20s}: {stats['hit_rate']:.0%} area hit rate  ({stats['area_hits']}/{stats['total_hits']})")
        lines.append("")

        # Worst cases
        lines.append("─── Worst Cases ───────────────────────────────────────────────────")
        for wc in s["worst_cases"]:
            lines.append(f"  {wc['test_id']:25s}: {wc['distance_km']:8.2f} km  [{wc['difficulty']}]")
        lines.append("")

        # Per-image detail
        lines.append("─── Per-Image Results ─────────────────────────────────────────────")
        lines.append(f"  {'ID':25s} {'Err(km)':>10s} {'100m':>5s} {'1km':>5s} {'Rank':>5s} {'Diff':>8s}")
        lines.append("  " + "-" * 65)
        for r in sorted(self.results, key=lambda x: x.distance_error_km if x.success else 999999):
            if not r.success:
                lines.append(f"  {r.test_id:25s} {'ERROR':>10s} {'-':>5s} {'-':>5s} {'-':>5s} {r.difficulty:>8s}")
            else:
                ok_100m = "Y" if r.within_100m else "n"
                ok_1km = "Y" if r.within_1km else "n"
                rank_s = str(r.top_n_rank) if r.top_n_rank > 0 else "-"
                lines.append(
                    f"  {r.test_id:25s} {r.distance_error_km:10.2f} {ok_100m:>5s} {ok_1km:>5s} {rank_s:>5s} {r.difficulty:>8s}"
                )

        lines.append("")
        lines.append("=" * 72)
        return "\n".join(lines)

    def to_json(self, path: Path | None = None) -> dict[str, Any]:
        """Export full results as a JSON-serializable dict."""
        out: dict[str, Any] = {
            "summary": self.summary(),
            "results": [],
        }
        for r in self.results:
            out["results"].append({
                "test_id": r.test_id,
                "name": r.name,
                "expected_lat": r.expected_lat,
                "expected_lon": r.expected_lon,
                "top_candidate_lat": r.top_candidate_lat,
                "top_candidate_lon": r.top_candidate_lon,
                "top_candidate_prob": r.top_candidate_prob,
                "distance_error_km": round(r.distance_error_km, 4) if r.success else None,
                "distance_error_m": round(r.distance_error_m, 2) if r.success else None,
                "within_100m": r.within_100m,
                "within_1km": r.within_1km,
                "within_10km": r.within_10km,
                "top_n_rank": r.top_n_rank,
                "near_miss": r.near_miss,
                "num_candidates": r.num_candidates,
                "is_exact_gps": r.is_exact_gps,
                "module_contributions": r.module_contributions,
                "outranked_modules": r.outranked_modules,
                "modules_run": r.modules_run,
                "modules_failed": r.modules_failed,
                "processing_time_s": r.processing_time_s,
                "category": r.category,
                "continent": r.continent,
                "difficulty": r.difficulty,
                "success": r.success,
                "error_message": r.error_message,
            })

        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)

        return out

    @classmethod
    def from_json(cls, path: Path) -> AccuracyTracker:
        """Load a tracker from a previously saved JSON results file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tracker = cls()
        for rd in data.get("results", []):
            ir = ImageResult(
                test_id=rd["test_id"],
                name=rd.get("name", rd["test_id"]),
                expected_lat=rd["expected_lat"],
                expected_lon=rd["expected_lon"],
                category=rd.get("category", "unknown"),
                continent=rd.get("continent", "unknown"),
                difficulty=rd.get("difficulty", "unknown"),
                distance_error_km=rd.get("distance_error_km") or -1.0,
                distance_error_m=rd.get("distance_error_m") or -1.0,
                within_100m=rd.get("within_100m", False),
                within_1km=rd.get("within_1km", False),
                within_10km=rd.get("within_10km", False),
                top_n_rank=rd.get("top_n_rank", -1),
                near_miss=rd.get("near_miss", False),
                num_candidates=rd.get("num_candidates", 0),
                module_contributions=rd.get("module_contributions", []),
                outranked_modules=rd.get("outranked_modules", []),
                top_candidate_lat=rd.get("top_candidate_lat", 0.0),
                top_candidate_lon=rd.get("top_candidate_lon", 0.0),
                top_candidate_prob=rd.get("top_candidate_prob", 0.0),
                modules_run=rd.get("modules_run", []),
                modules_failed=rd.get("modules_failed", []),
                processing_time_s=rd.get("processing_time_s", 0.0),
                success=rd.get("success", False),
                error_message=rd.get("error_message", ""),
            )
            tracker.results.append(ir)

        return tracker
