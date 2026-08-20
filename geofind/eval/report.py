"""Report generation for evaluation results.

Generates JSON, CSV, and HTML reports comparing pipeline variants.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from geofind.eval.metrics import EvalMetrics, THRESHOLD_NAMES


def save_ablation_comparison(
    results: dict[str, EvalMetrics],
    path: Path,
) -> None:
    """Save ablation comparison as JSON."""
    data = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "variants": {},
    }

    baseline = results.get("baseline")
    baseline_summary = baseline.summary() if baseline else None

    for name, metrics in sorted(results.items()):
        s = metrics.summary()
        ci = metrics.accuracy_ci(threshold_km=1.0) if "error" not in s else None

        entry: dict[str, Any] = {"summary": s}

        if ci:
            entry["accuracy_1km_ci"] = ci

        # Delta from baseline
        if baseline_summary and "error" not in baseline_summary and "error" not in s:
            entry["delta"] = {
                "avg_distance_km": round(
                    s["avg_distance_km"] - baseline_summary["avg_distance_km"], 4
                ),
                "accuracy_1km": round(
                    s["accuracy_1km"] - baseline_summary["accuracy_1km"], 4
                ),
                "accuracy_10km": round(
                    s["accuracy_10km"] - baseline_summary["accuracy_10km"], 4
                ),
            }

        data["variants"][name] = entry

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv_comparison(
    results: dict[str, EvalMetrics],
    path: Path,
) -> None:
    """Save per-image comparison CSV (one row per image, columns per variant)."""
    import csv

    # Collect all image IDs
    all_ids: set[str] = set()
    for metrics in results.values():
        for r in metrics.results:
            all_ids.add(r.id)

    # Build lookup: variant -> id -> result
    lookups: dict[str, dict[str, Any]] = {}
    for variant_name, metrics in results.items():
        lookups[variant_name] = {r.id: r for r in metrics.results}

    variant_names = sorted(results.keys())
    fieldnames = ["id", "name", "expected_lat", "expected_lon"]
    for v in variant_names:
        fieldnames.extend([
            f"{v}_distance_km",
            f"{v}_within_1km",
            f"{v}_within_10km",
        ])

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for img_id in sorted(all_ids):
            row: dict[str, Any] = {"id": img_id}
            for v in variant_names:
                r = lookups[v].get(img_id)
                if r:
                    row["name"] = r.name
                    row["expected_lat"] = r.expected_lat
                    row["expected_lon"] = r.expected_lon
                    row[f"{v}_distance_km"] = round(r.distance_km, 4) if r.success else ""
                    row[f"{v}_within_1km"] = r.within_1km if r.success else ""
                    row[f"{v}_within_10km"] = r.within_10km if r.success else ""
            writer.writerow(row)


def generate_html_report(
    results: dict[str, EvalMetrics],
    path: Path,
    title: str = "geofind Ablation Study",
) -> None:
    """Generate an HTML report with interactive comparison charts."""
    baseline = results.get("baseline")
    baseline_summary = baseline.summary() if baseline else None

    # Build chart data
    variant_names = sorted(results.keys())
    avg_distances = []
    accuracies_1km = []
    accuracies_10km = []
    processing_times = []

    for v in variant_names:
        s = results[v].summary()
        if "error" in s:
            avg_distances.append(0)
            accuracies_1km.append(0)
            accuracies_10km.append(0)
            processing_times.append(0)
        else:
            avg_distances.append(s["avg_distance_km"])
            accuracies_1km.append(round(s["accuracy_1km"] * 100, 1))
            accuracies_10km.append(round(s["accuracy_10km"] * 100, 1))
            processing_times.append(s["avg_processing_time_s"])

    # Build per-image table rows
    table_rows = ""
    if baseline:
        for r in sorted(baseline.results, key=lambda x: x.distance_km if x.success else 9999):
            if not r.success:
                continue
            err_class = "good" if r.distance_km <= 1 else ("ok" if r.distance_km <= 10 else "bad")
            table_rows += f"""
            <tr>
                <td>{r.id}</td>
                <td>{r.name[:40]}</td>
                <td class="{err_class}">{r.distance_km:.2f}</td>
                <td>{'Y' if r.within_1km else '-'}</td>
                <td>{'Y' if r.within_10km else '-'}</td>
                <td>{r.category}</td>
                <td>{r.source}</td>
            </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #1a1a2e; color: #e0e0e0; }}
        h1 {{ color: #00d4ff; }}
        h2 {{ color: #a0a0ff; margin-top: 30px; }}
        .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
        .chart-container {{ background: #16213e; border-radius: 12px; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th {{ background: #16213e; color: #00d4ff; padding: 10px; text-align: left; border-bottom: 2px solid #0f3460; }}
        td {{ padding: 8px 10px; border-bottom: 1px solid #0f3460; }}
        tr:hover {{ background: #16213e44; }}
        .good {{ color: #00ff88; }}
        .ok {{ color: #ffaa00; }}
        .bad {{ color: #ff4444; }}
        .summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
        .stat {{ background: #16213e; border-radius: 8px; padding: 15px; text-align: center; }}
        .stat .value {{ font-size: 2em; color: #00d4ff; font-weight: bold; }}
        .stat .label {{ color: #888; margin-top: 5px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p>Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>

    <div class="charts">
        <div class="chart-container">
            <canvas id="distanceChart"></canvas>
        </div>
        <div class="chart-container">
            <canvas id="accuracyChart"></canvas>
        </div>
    </div>

    <h2>Variant Summary</h2>
    <table>
        <tr>
            <th>Variant</th>
            <th>Avg Distance (km)</th>
            <th>Median (km)</th>
            <th>Within 1km</th>
            <th>Within 10km</th>
            <th>Within 100km</th>
            <th>Avg Time (s)</th>
            <th>N</th>
        </tr>
        {"".join(_variant_row(v, results[v], baseline_summary) for v in variant_names)}
    </table>

    <h2>Per-Image Results (Baseline)</h2>
    <table>
        <tr>
            <th>ID</th><th>Name</th><th>Error (km)</th>
            <th>1km</th><th>10km</th><th>Category</th><th>Source</th>
        </tr>
        {table_rows}
    </table>

    <script>
        const ctx1 = document.getElementById('distanceChart').getContext('2d');
        new Chart(ctx1, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(variant_names)},
                datasets: [{{
                    label: 'Avg Distance (km)',
                    data: {json.dumps(avg_distances)},
                    backgroundColor: 'rgba(0, 212, 255, 0.6)',
                    borderColor: 'rgba(0, 212, 255, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ title: {{ display: true, text: 'Average Distance Error', color: '#e0e0e0' }} }},
                scales: {{ y: {{ ticks: {{ color: '#e0e0e0' }}, grid: {{ color: '#0f3460' }} }},
                           x: {{ ticks: {{ color: '#e0e0e0' }}, grid: {{ color: '#0f3460' }} }} }}
            }}
        }});

        const ctx2 = document.getElementById('accuracyChart').getContext('2d');
        new Chart(ctx2, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(variant_names)},
                datasets: [
                    {{ label: 'Within 1km (%)', data: {json.dumps(accuracies_1km)}, backgroundColor: 'rgba(0,255,136,0.6)' }},
                    {{ label: 'Within 10km (%)', data: {json.dumps(accuracies_10km)}, backgroundColor: 'rgba(255,170,0,0.6)' }}
                ]
            }},
            options: {{
                responsive: true,
                plugins: {{ title: {{ display: true, text: 'Accuracy by Threshold', color: '#e0e0e0' }} }},
                scales: {{ y: {{ ticks: {{ color: '#e0e0e0' }}, grid: {{ color: '#0f3460' }} }},
                           x: {{ ticks: {{ color: '#e0e0e0' }}, grid: {{ color: '#0f3460' }} }} }}
            }}
        }});
    </script>
</body>
</html>"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def _variant_row(name: str, metrics: EvalMetrics, baseline_summary: dict | None) -> str:
    s = metrics.summary()
    if "error" in s:
        return f"<tr><td>{name}</td><td colspan='7'>No results</td></tr>"

    delta = ""
    if baseline_summary and name != "baseline":
        d = s["avg_distance_km"] - baseline_summary["avg_distance_km"]
        sign = "+" if d > 0 else ""
        cls = "bad" if d > 0 else "good"
        delta = f' <span class="{cls}">({sign}{d:.1f})</span>'

    return f"""<tr>
        <td><b>{name}</b>{delta}</td>
        <td>{s['avg_distance_km']:.2f}</td>
        <td>{s['median_distance_km']:.2f}</td>
        <td>{s['accuracy_1km']:.1%}</td>
        <td>{s['accuracy_10km']:.1%}</td>
        <td>{s['accuracy_100km']:.1%}</td>
        <td>{s['avg_processing_time_s']:.2f}</td>
        <td>{s['total']}</td>
    </tr>"""
