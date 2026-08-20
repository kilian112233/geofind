"""Main evaluation orchestrator.

Ties together image sources, pipeline execution, metrics collection,
and ablation studies into a single high-level API.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from geofind.core.config import PipelineConfig
from geofind.core.pipeline import GeoPipeline
from geofind.eval.ablation import (
    AblationVariant,
    ALL_VARIANTS,
    CORE_VARIANTS,
    run_ablation,
    default_pipeline_fn,
)
from geofind.eval.metrics import EvalMetrics, ImageResult
from geofind.eval.sources import EvalImage, ImageSource, get_source

logger = logging.getLogger(__name__)


class EvalRunner:
    """High-level evaluation runner.

    Usage:
        runner = EvalRunner()
        # Single evaluation
        metrics = runner.evaluate(source="wikimedia", count=25, strip_exif=True)
        # Ablation study
        results = runner.ablate(source="wikimedia", count=25, variants=["baseline", "no_consensus"])
    """

    def __init__(
        self,
        output_dir: Path | None = None,
        modules: str | None = None,
        disable_modules: str | None = None,
        verbose: bool = False,
    ):
        self.output_dir = output_dir or Path("W:/geofind/output/eval")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.modules = modules
        self.disable_modules = disable_modules
        self.verbose = verbose

    def evaluate(
        self,
        source: str = "wikimedia",
        count: int = 25,
        strip_exif: bool = True,
        source_kwargs: dict[str, Any] | None = None,
    ) -> EvalMetrics:
        """Run a single evaluation pass.

        Args:
            source: Image source name ("wikimedia" or "local").
            count: Number of images to evaluate.
            strip_exif: Strip EXIF GPS data before analysis.
            source_kwargs: Extra args passed to get_source().

        Returns:
            EvalMetrics with per-image results.
        """
        from rich.console import Console
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

        console = Console(stderr=True)
        source_kwargs = source_kwargs or {}

        # Fetch images
        console.print(f"\n[bold cyan]Fetching {count} images from {source}...[/]")
        img_source = get_source(source, **source_kwargs)
        images = img_source.fetch(count)

        if not images:
            console.print("[red]No images fetched. Aborting.[/]")
            return EvalMetrics()

        # Download and optionally strip EXIF
        image_dir = self.output_dir / "images"
        if strip_exif:
            console.print("[bold magenta]Mode: EXIF GPS stripped[/]")

        downloaded = img_source.download(images, image_dir, strip_exif=strip_exif)
        console.print(f"  Downloaded: [green]{len(downloaded)}[/] / {len(images)}")

        # Build config
        config = self._build_config()

        # Run pipeline on each image
        metrics = EvalMetrics(variant="baseline")
        pipeline = GeoPipeline(config)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Evaluating...", total=len(downloaded))

            for img in downloaded:
                progress.update(task, description=f"[cyan]{img.id}[/]")

                img_path = img.stripped_path or img.local_path
                if img_path is None or not img_path.exists():
                    metrics.record_error(img.id, meta={"name": img.name})
                    progress.advance(task)
                    continue

                try:
                    result = pipeline.analyze(img_path)
                    ir = metrics.record(
                        img.id,
                        expected_lat=img.lat,
                        expected_lon=img.lon,
                        result=result,
                        meta={"name": img.name, "source": img.source},
                    )
                    status = (
                        f"[green]{ir.distance_km:.1f}km[/]"
                        if ir.distance_km <= 10
                        else f"[yellow]{ir.distance_km:.1f}km[/]"
                    )
                    progress.update(task, description=f"{img.id}: {status}")
                except Exception as e:
                    metrics.record_error(
                        img.id, meta={"name": img.name}, error_message=str(e)
                    )
                    if self.verbose:
                        console.print(f"  [red]Error:[/] {img.id}: {e}")

                progress.advance(task)

        # Print summary
        self._print_summary(metrics, console)

        # Save results
        ts = time.strftime("%Y%m%d_%H%M%S")
        json_path = self.output_dir / f"eval_{ts}.json"
        csv_path = self.output_dir / f"eval_{ts}.csv"
        metrics.to_json(json_path)
        metrics.to_csv(csv_path)
        console.print(f"\n  Results: [cyan]{json_path}[/]")
        console.print(f"  CSV:     [cyan]{csv_path}[/]")

        return metrics

    def ablate(
        self,
        source: str = "wikimedia",
        count: int = 25,
        strip_exif: bool = True,
        variants: list[str] | None = None,
        source_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, EvalMetrics]:
        """Run an ablation study across multiple pipeline variants.

        Args:
            source: Image source name.
            count: Number of images per variant.
            strip_exif: Strip EXIF GPS data.
            variants: List of variant names to test. None = core variants.
            source_kwargs: Extra args passed to get_source().

        Returns:
            Dict mapping variant name -> EvalMetrics.
        """
        from rich.console import Console
        from rich.table import Table
        from rich import box

        console = Console(stderr=True)
        source_kwargs = source_kwargs or {}

        if variants is None:
            variants = CORE_VARIANTS

        # Validate variants
        for v in variants:
            if v not in ALL_VARIANTS:
                console.print(f"[red]Unknown variant:[/] {v}")
                console.print(f"  Available: {', '.join(ALL_VARIANTS.keys())}")
                return {}

        # Fetch images (shared across all variants)
        console.print(f"\n[bold cyan]Ablation Study — {len(variants)} variants[/]")
        console.print(f"  Images: {count} from {source}")

        img_source = get_source(source, **source_kwargs)
        images = img_source.fetch(count)

        if not images:
            console.print("[red]No images fetched. Aborting.[/]")
            return {}

        # Download images once
        image_dir = self.output_dir / "images"
        downloaded = img_source.download(images, image_dir, strip_exif=strip_exif)
        console.print(f"  Downloaded: [green]{len(downloaded)}[/] / {len(images)}")

        # Run ablation
        def pipeline_fn(img_path: Path, config: PipelineConfig) -> Any:
            pipeline = GeoPipeline(config)
            return pipeline.analyze(img_path)

        console.print()

        def progress_cb(variant_name: str, img_id: str, current: int, total: int):
            console.print(
                f"  [{variant_name}] {current}/{total} {img_id}",
                end="\r" if current < total else "\n",
            )

        results = run_ablation(
            images=downloaded,
            pipeline_fn=pipeline_fn,
            variants=variants,
            image_dir=image_dir,
            strip_exif=strip_exif,
            progress_callback=progress_cb,
        )

        # Print comparison table
        self._print_ablation_table(results, console)

        # Save all results
        ts = time.strftime("%Y%m%d_%H%M%S")
        for variant_name, metrics in results.items():
            json_path = self.output_dir / f"ablation_{variant_name}_{ts}.json"
            metrics.to_json(json_path)

        # Save comparison
        from geofind.eval.report import save_ablation_comparison
        comparison_path = self.output_dir / f"ablation_comparison_{ts}.json"
        save_ablation_comparison(results, comparison_path)
        console.print(f"\n  Comparison: [cyan]{comparison_path}[/]")

        return results

    # ── Private helpers ──────────────────────────────────────────────────

    def _build_config(self) -> PipelineConfig:
        """Build PipelineConfig with optional module overrides."""
        config = PipelineConfig()

        if self.modules:
            allow = {m.strip() for m in self.modules.split(",")}
            for name, mod_cfg in config.modules.items():
                mod_cfg.enabled = name in allow

        if self.disable_modules:
            skip = {m.strip() for m in self.disable_modules.split(",")}
            for name in skip:
                if name in config.modules:
                    config.modules[name].enabled = False

        return config

    def _print_summary(self, metrics: EvalMetrics, console: Any) -> None:
        """Print evaluation summary."""
        from rich.table import Table
        from rich import box

        s = metrics.summary()
        if "error" in s:
            console.print(f"[yellow]No results: {s['error']}[/]")
            return

        console.print("\n")
        console.rule("[bold cyan]Evaluation Results")

        table = Table(title="Accuracy Summary", box=box.ROUNDED, show_lines=True)
        table.add_column("Threshold", style="bold")
        table.add_column("Pass", justify="right")
        table.add_column("Total", justify="right")
        table.add_column("Accuracy", justify="right")

        for name in ["100m", "1km", "10km", "50km", "100km"]:
            passed = s[f"within_{name}"]
            acc = s[f"accuracy_{name}"]
            table.add_row(
                f"Within {name}",
                str(passed),
                str(s["total"]),
                f"{acc:.1%}",
            )

        console.print(table)
        console.print(f"\n  Avg distance:  [bold]{s['avg_distance_km']:.2f} km[/]")
        console.print(f"  Median:        {s['median_distance_km']:.2f} km")
        console.print(f"  Std dev:       {s['std_distance_km']:.2f} km")
        console.print(f"  Processing:    {s['avg_processing_time_s']:.2f}s avg")
        console.print()

    def _print_ablation_table(
        self, results: dict[str, EvalMetrics], console: Any
    ) -> None:
        """Print ablation comparison table."""
        from rich.table import Table
        from rich import box

        console.print("\n")
        console.rule("[bold cyan]Ablation Comparison")

        table = Table(title="Variant Comparison", box=box.ROUNDED, show_lines=True)
        table.add_column("Variant", style="bold")
        table.add_column("Avg (km)", justify="right")
        table.add_column("Median (km)", justify="right")
        table.add_column("1km", justify="right")
        table.add_column("10km", justify="right")
        table.add_column("100km", justify="right")
        table.add_column("Time (s)", justify="right")
        table.add_column("N", justify="right")

        baseline_avg = None
        for variant_name in sorted(results.keys()):
            s = results[variant_name].summary()
            if "error" in s:
                continue

            if variant_name == "baseline":
                baseline_avg = s["avg_distance_km"]

            # Delta from baseline
            delta_str = ""
            if baseline_avg is not None and variant_name != "baseline":
                delta = s["avg_distance_km"] - baseline_avg
                sign = "+" if delta > 0 else ""
                color = "red" if delta > 0 else "green"
                delta_str = f" [{color}]({sign}{delta:.1f}km)[/{color}]"

            table.add_row(
                f"{variant_name}{delta_str}",
                f"{s['avg_distance_km']:.2f}",
                f"{s['median_distance_km']:.2f}",
                f"{s['accuracy_1km']:.1%}",
                f"{s['accuracy_10km']:.1%}",
                f"{s['accuracy_100km']:.1%}",
                f"{s['avg_processing_time_s']:.2f}",
                str(s["total"]),
            )

        console.print(table)
