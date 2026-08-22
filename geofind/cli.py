"""CLI entry point for geofind."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

app = typer.Typer(name="geofind", help="Offline multi-module media geolocation system")
console = Console(stderr=True)

LOGO = r"""
   ___                _______ _______
  / _ \__      _____|  ___|  _____|
 / /_)/ \ \ /\ / / _ \ |_  | |___
/ ___/ \ V  V /  __/  _| |___    \
\/    \_/\_/\_/ \___|_|     |____/
"""


def _build_config(
    output: Path | None,
    modules: str | None,
    disable_modules: str | None,
    models_dir: Path | None,
    grid_resolution: float,
    json_output: bool,
    html_output: bool,
    top_n: int,
):
    from geofind.core.config import PipelineConfig

    config_kwargs: dict = {}
    if output is not None:
        config_kwargs["output_dir"] = output
    if models_dir is not None:
        config_kwargs["models_dir"] = models_dir
    if grid_resolution != 1.0:
        config_kwargs["grid_resolution_deg"] = grid_resolution
    if top_n != 20:
        config_kwargs["top_display"] = top_n
    config_kwargs["output_json"] = json_output
    config_kwargs["output_html"] = html_output

    config = PipelineConfig(**config_kwargs)

    if modules:
        allow = {m.strip() for m in modules.split(",")}
        for name, mod_cfg in config.modules.items():
            mod_cfg.enabled = name in allow
    if disable_modules:
        skip = {m.strip() for m in disable_modules.split(",")}
        for name, mod_cfg in config.modules.items():
            if name in skip:
                mod_cfg.enabled = False

    return config


def _print_header():
    logo = Text(LOGO, style="bold cyan")
    console.print(logo)
    console.print(
        Panel(
            "[bold]Offline multi-module media geolocation with Bayesian reranking[/]",
            style="cyan",
            padding=(0, 2),
        )
    )
    console.print()


def _print_module_table(config):
    from geofind.core.pipeline import _VISUAL_MODULES, _AUDIO_MODULES

    table = Table(title="Detection Modules", show_lines=False, title_style="bold")
    table.add_column("Module", style="bold")
    table.add_column("Enabled", justify="center")
    table.add_column("Type")
    table.add_column("Weight", justify="right")

    for name, mod_cfg in config.modules.items():
        if name in _VISUAL_MODULES:
            mod_type = "[blue]visual[/]"
        elif name in _AUDIO_MODULES:
            mod_type = "[magenta]audio[/]"
        else:
            mod_type = "—"
        enabled_str = "[green]✓[/]" if mod_cfg.enabled else "[red]✗[/]"
        table.add_row(name, enabled_str, mod_type, f"{mod_cfg.weight:.1f}")

    console.print(table)
    console.print()


def _results_table(candidates, top_n: int) -> Table:
    table = Table(title="Top Candidates", show_lines=True, title_style="bold green")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Latitude", justify="right")
    table.add_column("Longitude", justify="right")
    table.add_column("Probability", justify="right", style="bold")
    table.add_column("Country")
    table.add_column("Modules")

    for i, c in enumerate(candidates[:top_n], 1):
        module_names = ",".join(h.module for h in c.hits) if c.hits else "—"
        prob_pct = f"{c.probability * 100:.2f}%"
        country = c.country_hint or "—"
        style = "bold green" if i == 1 else ""
        table.add_row(
            str(i),
            f"{c.lat:.4f}",
            f"{c.lon:.4f}",
            prob_pct,
            country,
            module_names,
            style=style,
        )

    return table


def _export_json(result, output_dir: Path, top_n: int) -> Path:
    out_path = output_dir / "geofind_result.json"
    data = {
        "consensus": {
            "lat": result.consensus_lat,
            "lon": result.consensus_lon,
            "agreement_strength": result.agreement_strength,
        },
        "modules_run": result.modules_run,
        "modules_failed": result.modules_failed,
        "processing_time_s": result.processing_time_s,
        "candidates": [],
    }
    for c in result.candidates[:top_n]:
        data["candidates"].append(
            {
                "lat": c.lat,
                "lon": c.lon,
                "probability": c.probability,
                "log_posterior": c.log_posterior,
                "country_hint": c.country_hint,
                "modules": [
                    {
                        "name": h.module,
                        "confidence": h.confidence,
                        "metadata": h.metadata,
                    }
                    for h in c.hits
                ],
            }
        )
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


@app.command()
def analyze(
    media: Path = typer.Argument(..., help="Path to image, video, or audio file"),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Output directory"
    ),
    top_n: int = typer.Option(20, "--top-n", "-n", help="Number of top results to show"),
    json_output: bool = typer.Option(False, "--json", help="Export results as JSON"),
    html_output: bool = typer.Option(True, "--html/--no-html", help="Generate HTML heatmap"),
    modules: Optional[str] = typer.Option(
        None, "--modules", "-m", help="Comma-separated list of modules to enable (default: all)"
    ),
    disable_modules: Optional[str] = typer.Option(
        None, "--disable", help="Comma-separated list of modules to disable"
    ),
    models_dir: Optional[Path] = typer.Option(
        None, "--models-dir", help="Directory for cached models"
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
    grid_resolution: float = typer.Option(1.0, "--resolution", help="Grid resolution in degrees"),
    hierarchical: bool = typer.Option(True, "--hierarchical/--no-hierarchical", help="Enable hierarchical fine-grid refinement"),
    fine_top_n: int = typer.Option(10, "--fine-top-n", help="Number of coarse candidates to refine on fine grid"),
):
    """Analyze media files to determine geographic origin."""
    try:
        _print_header()

        if not media.exists():
            console.print(f"[red]Error:[/] File not found: {media}")
            raise typer.Exit(1)

        config = _build_config(
            output, modules, disable_modules, models_dir,
            grid_resolution, json_output, html_output, top_n,
        )

        if output is not None:
            config.output_dir.mkdir(parents=True, exist_ok=True)

        _print_module_table(config)

        if verbose:
            console.print(f"  Input:   {media}")
            console.print(f"  Output:  {config.output_dir}")
            console.print(f"  Models:  {config.models_dir}")
            console.print(f"  Grid:    {config.grid_resolution_deg}°")
            console.print()

        from geofind.core.pipeline import GeoPipeline

        pipeline = GeoPipeline(config)

        module_status: dict[str, str] = {}

        def progress_callback(module_name: str, status: str) -> None:
            if not module_name.startswith("_"):
                module_status[module_name] = status

        t0 = time.perf_counter()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Running analysis...", total=None)

            result = pipeline.analyze(media, progress_callback=progress_callback)
            progress.update(task, completed=True)

        elapsed = result.processing_time_s
        console.print()
        console.print(
            f"  [bold]Analysis complete in [cyan]{elapsed:.2f}s[/cyan][/bold]"
        )
        console.print()

        console.print(_results_table(result.candidates, top_n))
        console.print()

        console.print(
            f"  Consensus: [bold]{result.consensus_lat:.4f}°, {result.consensus_lon:.4f}°[/]"
        )
        console.print(
            f"  Agreement: [bold]{result.agreement_strength:.1%}[/]"
        )
        console.print(
            f"  Modules: [green]{len(result.modules_run)}[/] run, "
            f"[red]{len(result.modules_failed)}[/] failed"
        )
        console.print()

        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        if json_output:
            json_path = _export_json(result, output_dir, top_n)
            console.print(f"  JSON exported: [cyan]{json_path}[/]")

        if html_output:
            try:
                from geofind.utils.heatmap import generate_heatmap_html

                html_path = output_dir / "geofind_heatmap.html"
                generate_heatmap_html(result, html_path)
                console.print(f"  Heatmap exported: [cyan]{html_path}[/]")
            except Exception as e:
                console.print(f"  [yellow]Warning:[/] Failed to generate heatmap: {e}")

        console.print()

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/]")
        raise typer.Exit(130)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def modules():
    """List all available detection modules and their status."""
    _print_header()

    from geofind.core.config import PipelineConfig
    from geofind.core.pipeline import GeoPipeline, _VISUAL_MODULES, _AUDIO_MODULES

    config = PipelineConfig()
    pipeline = GeoPipeline(config)

    table = Table(title="Detection Modules", show_lines=True, title_style="bold")
    table.add_column("Module", style="bold")
    table.add_column("Type")
    table.add_column("Weight", justify="right")
    table.add_column("Available", justify="center")
    table.add_column("Description")

    descriptions = {
        "exif": "EXIF/GPS metadata extraction",
        "geoclip": "GeoCLIP pre-trained geo-vision model (4.7M images)",
        "clip_visual": "CLIP visual embedding similarity",
        "landmark": "Famous landmark detection",
        "ocr_text": "Text detection and OCR geolocation",
        "audio_power": "Audio power spectrum analysis",
        "birdnet": "Bird species identification",
        "sun_clock": "Sun position / time-of-day estimation",
        "shadow_angle": "Shadow angle geolocation",

        "driving_side": "Left/right traffic side detection",
        "vegetation": "Vegetation type classification",
        "license_plate": "License plate format detection",
        "currency": "Currency identification",
        "audio_scene": "Environmental audio scene classification",
    }

    for name, mod_cfg in config.modules.items():
        if name in _VISUAL_MODULES:
            mod_type = "[blue]visual[/]"
        elif name in _AUDIO_MODULES:
            mod_type = "[magenta]audio[/]"
        else:
            mod_type = "—"

        module = pipeline._instantiate_module(name, config)
        if module is not None:
            try:
                available = module.is_available()
            except Exception:
                available = False
        else:
            available = False

        avail_str = "[green]✓[/]" if available else "[red]✗[/]"
        desc = descriptions.get(name, "")
        table.add_row(name, mod_type, f"{mod_cfg.weight:.1f}", avail_str, desc)

    console.print(table)
    console.print()


@app.command()
def info():
    """Show system info and dependency status."""
    _print_header()

    import platform

    table = Table(title="System Info", show_lines=False, title_style="bold")
    table.add_column("Key", style="bold")
    table.add_column("Value")

    table.add_row("Python", platform.python_version())
    table.add_row("Platform", f"{platform.system()} {platform.release()}")
    table.add_row("Architecture", platform.machine())

    try:
        import numpy
        table.add_row("NumPy", numpy.__version__)
    except ImportError:
        table.add_row("NumPy", "[red]not installed[/]")

    try:
        import cv2
        table.add_row("OpenCV", cv2.__version__)
    except ImportError:
        table.add_row("OpenCV", "[red]not installed[/]")

    try:
        import scipy
        table.add_row("SciPy", scipy.__version__)
    except ImportError:
        table.add_row("SciPy", "[red]not installed[/]")

    try:
        import PIL
        table.add_row("Pillow", PIL.__version__)
    except ImportError:
        table.add_row("Pillow", "[red]not installed[/]")

    try:
        import typer
        table.add_row("Typer", typer.__version__)
    except ImportError:
        table.add_row("Typer", "[red]not installed[/]")

    try:
        from importlib.metadata import version as _get_ver
        table.add_row("Rich", _get_ver("rich"))
    except Exception:
        try:
            import rich
            table.add_row("Rich", getattr(rich, "__version__", "unknown"))
        except ImportError:
            table.add_row("Rich", "[red]not installed[/]")

    console.print(table)
    console.print()

    try:
        from geofind.core.pipeline import GeoPipeline
        from geofind.core.config import PipelineConfig

        config = PipelineConfig()
        pipeline = GeoPipeline(config)
        avail_table = Table(title="Module Availability", title_style="bold")
        avail_table.add_column("Module", style="bold")
        avail_table.add_column("Status")
        avail_table.add_column("Notes")

        for name in config.modules:
            module = pipeline._instantiate_module(name, config)
            if module is None:
                avail_table.add_row(name, "[red]✗ import failed[/]", "module code missing")
                continue
            try:
                if module.is_available():
                    avail_table.add_row(name, "[green]✓ available[/]", "")
                else:
                    avail_table.add_row(
                        name, "[yellow]⚠ deps missing[/]", "install optional dependencies"
                    )
            except Exception as e:
                avail_table.add_row(name, "[red]✗ error[/]", str(e)[:40])

        console.print(avail_table)
    except Exception as e:
        console.print(f"[yellow]Could not check module availability: {e}[/]")

    console.print()


@app.command()
def evaluate(
    source: str = typer.Option(
        "wikimedia", "--source", "-s", help="Image source: wikimedia or local"
    ),
    count: int = typer.Option(25, "--count", "-n", help="Number of images to evaluate"),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Output directory"
    ),
    modules: Optional[str] = typer.Option(
        None, "--modules", "-m", help="Comma-separated modules to enable"
    ),
    disable_modules: Optional[str] = typer.Option(
        None, "--disable", help="Comma-separated modules to disable"
    ),
    no_strip_exif: bool = typer.Option(
        False, "--no-strip-exif", help="Keep EXIF GPS data in images"
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
):
    """Evaluate geofind accuracy on random geotagged images.

    Downloads random images from Wikimedia Commons, strips EXIF GPS,
    runs the full pipeline, and measures distance error from ground truth.
    """
    try:
        _print_header()

        from geofind.eval.runner import EvalRunner

        runner = EvalRunner(
            output_dir=output,
            modules=modules,
            disable_modules=disable_modules,
            verbose=verbose,
        )

        metrics = runner.evaluate(
            source=source,
            count=count,
            strip_exif=not no_strip_exif,
        )

        s = metrics.summary()
        if "error" in s:
            console.print("[yellow]No results to evaluate[/]")
            raise typer.Exit(1)

        # Generate HTML report
        from geofind.eval.report import generate_html_report
        html_path = (output or Path("W:/geofind/output/eval")) / "eval_report.html"
        generate_html_report({"baseline": metrics}, html_path, title="geofind Evaluation")
        console.print(f"  HTML report: [cyan]{html_path}[/]")

        # Exit code based on accuracy
        if not no_strip_exif:
            if s.get("accuracy_1km", 0) >= 0.10:
                raise typer.Exit(0)
        else:
            if s.get("accuracy_100m", 0) >= 0.50:
                raise typer.Exit(0)
        raise typer.Exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/]")
        raise typer.Exit(130)
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def ablate(
    source: str = typer.Option(
        "wikimedia", "--source", "-s", help="Image source: wikimedia or local"
    ),
    count: int = typer.Option(25, "--count", "-n", help="Number of images per variant"),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Output directory"
    ),
    variants: Optional[str] = typer.Option(
        None, "--variants", "-v",
        help="Comma-separated variant names (default: core variants)",
    ),
    all_variants: bool = typer.Option(
        False, "--all", help="Run all available variants"
    ),
    no_strip_exif: bool = typer.Option(
        False, "--no-strip-exif", help="Keep EXIF GPS data in images"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
):
    """Run an ablation study on pipeline components.

    Tests multiple pipeline variants (e.g., no consensus buff, no fuzzy
    calibration, equal weights) to measure the impact of each component
    on geolocation accuracy.
    """
    try:
        _print_header()

        from geofind.eval.ablation import ALL_VARIANTS, CORE_VARIANTS
        from geofind.eval.runner import EvalRunner

        if variants:
            variant_list = [v.strip() for v in variants.split(",")]
        elif all_variants:
            variant_list = list(ALL_VARIANTS.keys())
        else:
            variant_list = CORE_VARIANTS

        # Validate
        for v in variant_list:
            if v not in ALL_VARIANTS:
                console.print(f"[red]Unknown variant:[/] {v}")
                console.print(f"  Available: {', '.join(sorted(ALL_VARIANTS.keys()))}")
                raise typer.Exit(1)

        runner = EvalRunner(
            output_dir=output,
            verbose=verbose,
        )

        results = runner.ablate(
            source=source,
            count=count,
            strip_exif=not no_strip_exif,
            variants=variant_list,
        )

        if not results:
            console.print("[yellow]No results[/]")
            raise typer.Exit(1)

        # Generate HTML report
        from geofind.eval.report import generate_html_report
        html_path = (output or Path("W:/geofind/output/eval")) / "ablation_report.html"
        generate_html_report(results, html_path, title="geofind Ablation Study")
        console.print(f"  HTML report: [cyan]{html_path}[/]")

        raise typer.Exit(0)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/]")
        raise typer.Exit(130)
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
