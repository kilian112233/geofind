#!/usr/bin/env python3
"""geofind Dev Test Runner

Downloads test images, runs the geofind pipeline, measures accuracy,
and generates reports. Supports both downloaded and synthetic test images.

Usage:
    python dev/run_tests.py                     # Run all tests with downloaded images
    python dev/run_tests.py --synthetic         # Use synthetic EXIF images
    python dev/run_tests.py --limit 5 --verbose # Quick smoke test
    python dev/run_tests.py --modules exif      # Test specific modules only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on path
PROJECT_ROOT = Path("W:/geofind")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geofind.core.candidate import GeoResult
from geofind.core.config import PipelineConfig
from geofind.core.pipeline import GeoPipeline

# Local imports
from accuracy import AccuracyTracker
from image_grabber import (
    IMAGES_DIR,
    load_dataset,
    generate_synthetic_batch,
    download_all,
    SYNTHETIC_LOCATIONS,
)

# ── Rich console output ───────────────────────────────────────────────────────

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.panel import Panel
    from rich.text import Text
    from rich.tree import Tree
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

console = Console(stderr=True) if HAS_RICH else None

logger = logging.getLogger("run_tests")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def _rich_print(msg: str, **kwargs: Any) -> None:
    if console:
        console.print(msg, **kwargs)
    else:
        # Strip rich markup
        clean = msg.replace("[bold ", "").replace("[/", "]").replace("[", "").replace("]", "")
        print(clean)


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_pipeline(
    image_path: Path,
    modules: str | None = None,
    config: PipelineConfig | None = None,
) -> GeoResult:
    """Run the geofind pipeline on a single image."""
    pipeline = GeoPipeline(config)

    module_list = None
    if modules:
        module_list = [m.strip() for m in modules.split(",")]

    if module_list:
        # Override enabled modules
        for name in pipeline.config.modules:
            pipeline.config.modules[name].enabled = name in module_list

    result = pipeline.analyze(image_path)
    return result


# ── Main test loop ────────────────────────────────────────────────────────────

def run_all_tests(
    dataset_path: Path,
    use_synthetic: bool = False,
    modules: str | None = None,
    limit: int | None = None,
    output_path: Path | None = None,
    verbose: bool = False,
) -> AccuracyTracker:
    """Run the full test suite and return accuracy results."""

    # ── Load dataset ─────────────────────────────────────────────────────
    _rich_print("\n[bold cyan]=== geofind Test Runner ===[/]")
    _rich_print(f"Dataset: {dataset_path}")

    # When using synthetic mode, create a matching synthetic dataset
    if use_synthetic:
        from image_grabber import create_synthetic_dataset
        dataset = create_synthetic_dataset()
        # Override image directory to synthetic folder
        image_dir = IMAGES_DIR / "synthetic"
    else:
        dataset = load_dataset(dataset_path)
        image_dir = IMAGES_DIR

    cases = dataset["test_cases"]
    if limit:
        cases = cases[:limit]
    total = len(cases)
    _rich_print(f"Test cases: [bold]{total}[/]")

    # ── Prepare images ───────────────────────────────────────────────────
    if use_synthetic:
        _rich_print("\n[bold yellow]Mode: Synthetic EXIF images[/]")
        # image_dir already set above for synthetic mode
        generate_synthetic_batch(output_dir=image_dir)
    else:
        _rich_print("\n[bold yellow]Mode: Download from Wikimedia[/]")
        image_dir = IMAGES_DIR
        # Check which images we already have
        existing = 0
        for tc in cases:
            if (image_dir / f"{tc['id']}.jpg").exists():
                existing += 1
        _rich_print(f"  Cached: {existing}/{total}")

        if existing < total:
            _rich_print("  Downloading missing images...")
            download_all(dataset, force=False)

    # ── Configure pipeline ───────────────────────────────────────────────
    config = PipelineConfig()
    if modules:
        mod_list = [m.strip() for m in modules.split(",")]
        for name in config.modules:
            config.modules[name].enabled = name in mod_list
        _rich_print(f"Modules: [bold]{', '.join(mod_list)}[/]")
    else:
        enabled = [n for n, c in config.modules.items() if c.enabled]
        _rich_print(f"Modules: [bold]{', '.join(enabled)}[/]")

    # ── Run tests ────────────────────────────────────────────────────────
    tracker = AccuracyTracker()
    passed = 0
    failed = 0
    errors = 0
    start_time = time.perf_counter()

    if HAS_RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Testing...", total=total)

            for i, tc in enumerate(cases):
                tid = tc["id"]
                image_path = image_dir / f"{tid}.jpg"

                progress.update(task, description=f"[cyan]{tid}[/]")

                # Check image exists
                if not image_path.exists():
                    progress.update(task, description=f"[red]{tid} (no image)[/]")
                    tracker.record_error(
                        tid,
                        meta=tc,
                        error_message=f"Image not found: {image_path}",
                    )
                    errors += 1
                    progress.advance(task)
                    continue

                # Run pipeline
                try:
                    result = run_pipeline(image_path, modules=modules, config=config)
                    ir = tracker.record(
                        tid,
                        expected_lat=tc["expected_lat"],
                        expected_lon=tc["expected_lon"],
                        result=result,
                        meta=tc,
                    )

                    if ir.within_1km:
                        progress.update(task, description=f"[green]{tid} ✓[/]")
                        passed += 1
                    else:
                        progress.update(task, description=f"[yellow]{tid} ({ir.distance_error_km:.1f}km)[/]")
                        failed += 1

                except Exception as e:
                    progress.update(task, description=f"[red]{tid} (error)[/]")
                    tracker.record_error(tid, meta=tc, error_message=str(e))
                    errors += 1
                    logger.error(f"Pipeline failed for {tid}: {e}", exc_info=verbose)

                progress.advance(task)

    else:
        for i, tc in enumerate(cases, 1):
            tid = tc["id"]
            image_path = image_dir / f"{tid}.jpg"
            print(f"  [{i}/{total}] {tid}...", end=" ", flush=True)

            if not image_path.exists():
                tracker.record_error(tid, meta=tc, error_message="Image not found")
                errors += 1
                print("SKIP (no image)")
                continue

            try:
                result = run_pipeline(image_path, modules=modules, config=config)
                ir = tracker.record(
                    tid,
                    expected_lat=tc["expected_lat"],
                    expected_lon=tc["expected_lon"],
                    result=result,
                    meta=tc,
                )
                if ir.within_1km:
                    passed += 1
                    print(f"OK ({ir.distance_error_km:.2f}km)")
                else:
                    failed += 1
                    print(f"MISS ({ir.distance_error_km:.2f}km)")
            except Exception as e:
                tracker.record_error(tid, meta=tc, error_message=str(e))
                errors += 1
                print(f"ERROR: {e}")

    elapsed = time.perf_counter() - start_time

    # ── Print summary ────────────────────────────────────────────────────
    _rich_print("\n")
    _rich_print("[bold cyan]═══ Test Run Complete ═══[/]")
    _rich_print(f"  Time:   [bold]{elapsed:.1f}s[/]")
    _rich_print(f"  Passed: [green]{passed}[/] | Failed: [yellow]{failed}[/] | Errors: [red]{errors}[/]")
    _rich_print("")

    # Print accuracy table
    s = tracker.summary()
    if "error" not in s:
        if HAS_RICH:
            table = Table(title="Accuracy Summary", box=box.ROUNDED, show_lines=True)
            table.add_column("Threshold", style="bold")
            table.add_column("Pass", justify="right")
            table.add_column("Total", justify="right")
            table.add_column("Accuracy", justify="right")
            table.add_column("Target", justify="right")
            table.add_column("Status", justify="center")

            thresholds = [
                ("100m", s["passed_100m"], s["total_tests"], s["accuracy_100m"], s["target_accuracy_100m"]),
                ("1km", s["passed_1km"], s["total_tests"], s["accuracy_1km"], s["target_accuracy_1km"]),
                ("10km", s["passed_10km"], s["total_tests"], s["accuracy_10km"], 0.90),
                ("50km", s["passed_50km"], s["total_tests"], s["accuracy_50km"], 0.95),
            ]

            for name, passed_n, total_n, acc, target in thresholds:
                met = "PASS" if acc >= target else "FAIL"
                style = "green" if acc >= target else "red"
                table.add_row(
                    f"Within {name}",
                    str(passed_n),
                    str(total_n),
                    f"{acc:.1%}",
                    f"{target:.0%}",
                    f"[{style}]{met}[/{style}]",
                )

            console.print(table)

            # Category breakdown
            cat_table = Table(title="By Category", box=box.SIMPLE)
            cat_table.add_column("Category", style="bold")
            cat_table.add_column("N", justify="right")
            cat_table.add_column("100m", justify="right")
            cat_table.add_column("1km", justify="right")

            for cat, stats in sorted(s["per_category_accuracy"].items()):
                cat_table.add_row(
                    cat,
                    str(stats["total"]),
                    f"{stats['accuracy_100m']:.0%}",
                    f"{stats['accuracy_1km']:.0%}",
                )
            console.print(cat_table)

            # Difficulty breakdown
            diff_table = Table(title="By Difficulty", box=box.SIMPLE)
            diff_table.add_column("Difficulty", style="bold")
            diff_table.add_column("N", justify="right")
            diff_table.add_column("100m", justify="right")
            diff_table.add_column("1km", justify="right")

            for diff, stats in sorted(s["per_difficulty_accuracy"].items()):
                diff_table.add_row(
                    diff,
                    str(stats["total"]),
                    f"{stats['accuracy_100m']:.0%}",
                    f"{stats['accuracy_1km']:.0%}",
                )
            console.print(diff_table)

            # Near misses
            _rich_print(f"\nNear-misses: [yellow]{s['near_miss_count']}[/] ({s['near_miss_rate']:.1%})")

            # Worst cases
            _rich_print("\n[bold red]Worst Cases:[/]")
            for wc in s["worst_cases"]:
                _rich_print(
                    f"  {wc['test_id']:30s}  {wc['distance_km']:8.2f} km  [{wc['difficulty']}]"
                )
        else:
            print(f"\nAccuracy:")
            print(f"  100m: {s['accuracy_100m']:.1%}")
            print(f"  1km:  {s['accuracy_1km']:.1%}")
            print(f"  10km: {s['accuracy_10km']:.1%}")

    # ── Save results ─────────────────────────────────────────────────────
    if output_path is None:
        output_path = PROJECT_ROOT / "dev" / "test_results.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tracker.to_json(output_path)
    _rich_print(f"\nResults saved: [bold]{output_path}[/]")

    # Save detailed text report
    report_path = output_path.with_suffix(".txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(tracker.detailed_report())
    _rich_print(f"Report saved:  [bold]{report_path}[/]")

    return tracker


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="geofind Dev Test Runner — download, analyze, measure, report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic EXIF-embedded test images instead of downloading",
    )
    parser.add_argument(
        "--modules",
        default=None,
        help="Comma-separated module list (default: all available)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of test images to process",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: dev/test_results.json)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("W:/geofind/dev/test_dataset.json"),
        help="Path to test_dataset.json",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    tracker = run_all_tests(
        dataset_path=args.dataset,
        use_synthetic=args.synthetic,
        modules=args.modules,
        limit=args.limit,
        output_path=args.output,
        verbose=args.verbose,
    )

    # Exit code based on accuracy
    s = tracker.summary()
    if "error" not in s and s["meets_1km_target"]:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
