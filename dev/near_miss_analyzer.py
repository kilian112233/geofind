#!/usr/bin/env python3
"""Near-miss analyzer for geofind test harness.

Analyzes cases where the correct location was outranked — identifies which
modules found the right area, what "outvoted" the correct answer, and
generates actionable improvement suggestions per module.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from geofind.utils.geo import LatLon, haversine_km

logger = logging.getLogger("near_miss")


@dataclass
class ModuleBlame:
    """Tracks whether a module helped or hurt a specific test case."""
    module: str
    hit_lat: float = 0.0
    hit_lon: float = 0.0
    confidence: float = 0.0
    distance_to_expected_km: float = -1.0
    distance_to_winner_km: float = -1.0
    helped_correct: bool = False  # found the right area
    hurt_correct: bool = False    # pulled probability away from correct answer
    was_in_winner: bool = False   # was in the top candidate


@dataclass
class NearMissCase:
    """Detailed analysis of a near-miss test case."""
    test_id: str
    name: str
    expected_lat: float
    expected_lon: float
    top_candidate_lat: float
    top_candidate_lon: float
    top_candidate_dist_km: float
    correct_rank: int
    correct_dist_km: float
    category: str
    continent: str
    difficulty: str

    helping_modules: list[ModuleBlame] = field(default_factory=list)
    hurting_modules: list[ModuleBlame] = field(default_factory=list)
    winner_modules: list[ModuleBlame] = field(default_factory=list)

    suggestions: list[str] = field(default_factory=list)


@dataclass
class NearMissReport:
    """Complete near-miss analysis report."""
    total_cases: int = 0
    near_miss_count: int = 0
    cases: list[NearMissCase] = field(default_factory=list)
    module_blame_summary: dict[str, dict[str, Any]] = field(default_factory=dict)
    top_hurting_modules: list[tuple[str, float]] = field(default_factory=list)
    improvement_suggestions: list[str] = field(default_factory=list)


def analyze_near_misses(
    results_path: Path,
    dataset_path: Path | None = None,
    threshold_km: float = 0.1,
    outrank_margin_km: float = 0.5,
) -> NearMissReport:
    """Analyze all near-miss cases from a test results JSON.

    Args:
        results_path: Path to test_results.json from run_tests.
        dataset_path: Optional path to test_dataset.json for metadata.
        threshold_km: Distance threshold for "correct" (default 0.1km = 100m).
        outrank_margin_km: How far the winner can be from expected to count as "outranked".

    Returns:
        NearMissReport with full analysis.
    """
    # Load results
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])

    # Load dataset for metadata
    dataset_meta: dict[str, dict[str, Any]] = {}
    if dataset_path and dataset_path.exists():
        with open(dataset_path, "r", encoding="utf-8") as f:
            ds = json.load(f)
        for tc in ds.get("test_cases", []):
            dataset_meta[tc["id"]] = tc

    report = NearMissReport()
    blame_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"helped": 0, "hurted": 0, "total": 0})

    for r in results:
        if not r.get("success", False):
            continue

        tid = r["test_id"]
        expected_lat = r["expected_lat"]
        expected_lon = r["expected_lon"]
        expected = LatLon(expected_lat, expected_lon)

        report.total_cases += 1
        meta = dataset_meta.get(tid, {})

        # Check if this is a near-miss
        correct_dist_km = r.get("distance_error_km", -1)
        if correct_dist_km < 0:
            continue

        # Is the top candidate near the expected location?
        top_near = correct_dist_km <= outrank_margin_km

        # Was the correct answer outranked?
        correct_rank = r.get("top_n_rank", -1)
        is_near_miss = r.get("near_miss", False) or (top_near and correct_rank > 1)

        if not is_near_miss:
            continue

        report.near_miss_count += 1

        # Build the near-miss case
        nmc = NearMissCase(
            test_id=tid,
            name=r.get("name", tid),
            expected_lat=expected_lat,
            expected_lon=expected_lon,
            top_candidate_lat=r.get("top_candidate_lat", 0.0),
            top_candidate_lon=r.get("top_candidate_lon", 0.0),
            top_candidate_dist_km=r.get("distance_error_km", 0.0),
            correct_rank=correct_rank,
            correct_dist_km=correct_dist_km,
            category=meta.get("category", r.get("category", "unknown")),
            continent=meta.get("continent", r.get("continent", "unknown")),
            difficulty=meta.get("difficulty", r.get("difficulty", "unknown")),
        )

        # Analyze module contributions
        module_contribs = r.get("module_contributions", [])
        top_cand_lat = nmc.top_candidate_lat
        top_cand_lon = nmc.top_candidate_lon
        top_cand = LatLon(top_cand_lat, top_cand_lon)

        for mc in module_contribs:
            mod = mc["module"]
            hit_lat = mc["hit_lat"]
            hit_lon = mc["hit_lon"]
            confidence = mc["confidence"]
            cand_lat = mc["candidate_lat"]
            cand_lon = mc["candidate_lon"]

            hit_point = LatLon(hit_lat, hit_lon)
            cand_point = LatLon(cand_lat, cand_lon)

            dist_to_expected = haversine_km(expected, cand_point)
            dist_to_winner = haversine_km(top_cand, cand_point)

            blame = ModuleBlame(
                module=mod,
                hit_lat=hit_lat,
                hit_lon=hit_lon,
                confidence=confidence,
                distance_to_expected_km=dist_to_expected,
                distance_to_winner_km=dist_to_winner,
                helped_correct=dist_to_expected <= threshold_km,
                hurt_correct=dist_to_winner <= threshold_km and dist_to_expected > outrank_margin_km,
                was_in_winner=dist_to_winner <= threshold_km,
            )

            blame_counts[mod]["total"] += 1
            if blame.helped_correct:
                nmc.helping_modules.append(blame)
                blame_counts[mod]["helped"] += 1
            elif blame.hurt_correct:
                nmc.hurting_modules.append(blame)
                blame_counts[mod]["hurted"] += 1

            if blame.was_in_winner:
                nmc.winner_modules.append(blame)

        # Generate suggestions
        nmc.suggestions = _generate_suggestions(nmc)
        report.cases.append(nmc)

    # Compute module blame summary
    for mod, counts in blame_counts.items():
        total = counts["total"]
        helped = counts["helped"]
        hurted = counts["hurted"]
        if total > 0:
            report.module_blame_summary[mod] = {
                "total_participations": total,
                "helped_correct": helped,
                "hurt_correct": hurted,
                "help_rate": helped / total,
                "blame_rate": hurted / total,
            }

    # Rank modules by blame rate
    report.top_hurting_modules = sorted(
        [(mod, s["blame_rate"]) for mod, s in report.module_blame_summary.items()],
        key=lambda x: x[1],
        reverse=True,
    )

    # Generate overall improvement suggestions
    report.improvement_suggestions = _generate_overall_suggestions(report)

    return report


def _generate_suggestions(nmc: NearMissCase) -> list[str]:
    """Generate improvement suggestions for a single near-miss case."""
    suggestions: list[str] = []

    # If the correct answer was outranked by fuzzy modules
    fuzzy_outranking = [
        m for m in nmc.winner_modules
        if m.module in {"clip_visual", "vegetation", "vision_llm", "ocr_text"}
    ]
    if fuzzy_outranking and nmc.helping_modules:
        mod_names = ", ".join(m.module for m in fuzzy_outranking)
        suggestions.append(
            f"Fuzzy modules [{mod_names}] pulled probability to wrong location. "
            f"Consider increasing weight of precise modules or adding "
            f"confidence-gating to fuzzy modules."
        )

    # If no modules helped
    if not nmc.helping_modules:
        suggestions.append(
            "No modules correctly identified the expected location. "
            "Consider adding new modules or improving existing coverage."
        )

    # If only one module helped but was outranked
    if len(nmc.helping_modules) == 1:
        mod = nmc.helping_modules[0].module
        suggestions.append(
            f"Only '{mod}' found the correct area. Increase its weight "
            f"or add corroborating modules for this region."
        )

    # If winner has many low-confidence hits vs one high-confidence correct
    if nmc.winner_modules and nmc.helping_modules:
        winner_avg_conf = sum(m.confidence for m in nmc.winner_modules) / len(nmc.winner_modules)
        helper_avg_conf = sum(m.confidence for m in nmc.helping_modules) / len(nmc.helping_modules)
        if helper_avg_conf > winner_avg_conf * 1.5:
            suggestions.append(
                f"Correct location had higher average confidence ({helper_avg_conf:.2f}) "
                f"than winner ({winner_avg_conf:.2f}). Consider re-weighting by confidence."
            )

    # Distance-based suggestion
    if nmc.correct_dist_km > 0 and nmc.top_candidate_dist_km > 10:
        suggestions.append(
            f"Winner was {nmc.top_candidate_dist_km:.1f}km from expected. "
            f"Consider post-filtering by geographic consistency."
        )

    return suggestions


def _generate_overall_suggestions(report: NearMissReport) -> list[str]:
    """Generate overall improvement suggestions from the full report."""
    suggestions: list[str] = []

    if not report.cases:
        return suggestions

    # Most common outranking pattern
    outranking_modules: dict[str, int] = defaultdict(int)
    for nmc in report.cases:
        for m in nmc.winner_modules:
            outranking_modules[m.module] += 1

    if outranking_modules:
        top_outranker = max(outranking_modules, key=outranking_modules.get)
        count = outranking_modules[top_outranker]
        suggestions.append(
            f"Module '{top_outranker}' was in the winner for {count}/{report.near_miss_count} "
            f"near-misses. Consider reducing its weight or adding spatial confidence gates."
        )

    # Module with highest blame rate
    if report.top_hurting_modules:
        worst_mod, blame_rate = report.top_hurting_modules[0]
        if blame_rate > 0.3:
            suggestions.append(
                f"Module '{worst_mod}' has {blame_rate:.0%} blame rate across near-misses. "
                f"Consider disabling it, reducing weight, or adding output validation."
            )

    # Category-specific suggestions
    cat_misses: dict[str, int] = defaultdict(int)
    for nmc in report.cases:
        cat_misses[nmc.category] += 1
    if cat_misses:
        worst_cat = max(cat_misses, key=cat_misses.get)
        suggestions.append(
            f"Category '{worst_cat}' has {cat_misses[worst_cat]} near-misses. "
            f"Consider adding module-specific tuning for this category."
        )

    # Difficulty-based
    hard_misses = [nmc for nmc in report.cases if nmc.difficulty == "hard"]
    if hard_misses:
        suggestions.append(
            f"{len(hard_misses)} near-misses are in 'hard' difficulty. "
            f"These may require better training data or new detection modules."
        )

    return suggestions


def print_near_miss_report(report: NearMissReport) -> None:
    """Print a human-readable near-miss analysis report."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box
        HAS_RICH = True
    except ImportError:
        HAS_RICH = False

    console = Console(stderr=True) if HAS_RICH else None

    def _print(msg: str, **kwargs: Any) -> None:
        if console:
            console.print(msg, **kwargs)
        else:
            clean = msg.replace("[bold ", "").replace("[/", "]").replace("[", "").replace("]", "")
            print(clean)

    _print("\n" + "=" * 72)
    _print("           NEAR-MISS ANALYSIS REPORT")
    _print("=" * 72)
    _print(f"\nTotal cases: {report.total_cases}")
    _print(f"Near-misses: {report.near_miss_count}")
    if report.total_cases > 0:
        _print(f"Near-miss rate: {report.near_miss_count / report.total_cases:.1%}")
    _print("")

    if not report.cases:
        _print("[green]No near-misses found! All tests either passed or failed completely.[/]")
        return

    # Module blame summary
    if HAS_RICH:
        table = Table(title="Module Blame Summary", box=box.ROUNDED, show_lines=True)
        table.add_column("Module", style="bold")
        table.add_column("Participations", justify="right")
        table.add_column("Helped", justify="right", style="green")
        table.add_column("Hurted", justify="right", style="red")
        table.add_column("Help Rate", justify="right")
        table.add_column("Blame Rate", justify="right")

        for mod, s in sorted(report.module_blame_summary.items(), key=lambda x: x[1].get("blame_rate", 0), reverse=True):
            blame_style = "red" if s["blame_rate"] > 0.3 else "yellow" if s["blame_rate"] > 0.1 else "green"
            table.add_row(
                mod,
                str(s["total_participations"]),
                str(s["helped_correct"]),
                str(s["hurt_correct"]),
                f"{s['help_rate']:.0%}",
                f"[{blame_style}]{s['blame_rate']:.0%}[/{blame_style}]",
            )
        console.print(table)
    else:
        print("\nModule Blame Summary:")
        for mod, s in sorted(report.module_blame_summary.items(), key=lambda x: x[1].get("blame_rate", 0), reverse=True):
            print(f"  {mod:20s}: {s['helped_correct']} helped, {s['hurt_correct']} hurted, {s['blame_rate']:.0%} blame")

    # Individual near-miss cases
    _print("\n─── Near-Miss Details ──────────────────────────────────────────────")
    for nmc in report.cases[:20]:
        _print(f"\n  [bold]{nmc.test_id}[/] ({nmc.name})")
        _print(f"    Expected: {nmc.expected_lat:.4f}, {nmc.expected_lon:.4f}")
        _print(f"    Winner:   {nmc.top_candidate_lat:.4f}, {nmc.top_candidate_lon:.4f} ({nmc.top_candidate_dist_km:.2f}km away)")
        _print(f"    Correct rank: #{nmc.correct_rank} ({nmc.correct_dist_km:.2f}km)")

        if nmc.helping_modules:
            helpers = ", ".join(m.module for m in nmc.helping_modules)
            _print(f"    [green]Helping:[/] {helpers}")

        if nmc.hurting_modules:
            hurters = ", ".join(m.module for m in nmc.hurting_modules)
            _print(f"    [red]Hurting:[/] {hurters}")

        if nmc.suggestions:
            for s in nmc.suggestions:
                _print(f"    [yellow]Suggestion:[/] {s}")

    # Overall suggestions
    if report.improvement_suggestions:
        _print("\n─── Improvement Suggestions ────────────────────────────────────────")
        for i, s in enumerate(report.improvement_suggestions, 1):
            _print(f"  {i}. {s}")

    _print("\n" + "=" * 72)


def main() -> None:
    """CLI entry point for near_miss_analyzer."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze near-misses in geofind test results"
    )
    parser.add_argument(
        "results",
        nargs="?",
        default="W:/geofind/dev/test_results.json",
        help="Path to test_results.json (default: dev/test_results.json)",
    )
    parser.add_argument(
        "--dataset",
        default="W:/geofind/dev/test_dataset.json",
        help="Path to test_dataset.json",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Distance threshold in km for 'correct' (default: 0.1 = 100m)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.5,
        help="Outrank margin in km (default: 0.5)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save JSON report to this path",
    )
    args = parser.parse_args()

    report = analyze_near_misses(
        results_path=Path(args.results),
        dataset_path=Path(args.dataset),
        threshold_km=args.threshold,
        outrank_margin_km=args.margin,
    )

    print_near_miss_report(report)

    if args.output:
        out_data = {
            "total_cases": report.total_cases,
            "near_miss_count": report.near_miss_count,
            "module_blame_summary": report.module_blame_summary,
            "top_hurting_modules": report.top_hurting_modules,
            "improvement_suggestions": report.improvement_suggestions,
            "cases": [
                {
                    "test_id": c.test_id,
                    "name": c.name,
                    "correct_rank": c.correct_rank,
                    "correct_dist_km": c.correct_dist_km,
                    "top_candidate_dist_km": c.top_candidate_dist_km,
                    "helping_modules": [m.module for m in c.helping_modules],
                    "hurting_modules": [m.module for m in c.hurting_modules],
                    "suggestions": c.suggestions,
                    "category": c.category,
                    "difficulty": c.difficulty,
                }
                for c in report.cases
            ],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
