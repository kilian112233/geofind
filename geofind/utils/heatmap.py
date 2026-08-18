"""HTML heatmap output using Leaflet.js."""

from __future__ import annotations

from pathlib import Path

from geofind.core.candidate import GeoResult


def generate_heatmap_html(result: GeoResult, output_path: Path) -> Path:
    """Generate a self-contained HTML file with an interactive Leaflet.js heatmap.

    Args:
        result: GeoResult with ranked candidates.
        output_path: Where to write the HTML file.

    Returns:
        Path to the written file.
    """
    lat = result.consensus_lat
    lon = result.consensus_lon
    strength = result.agreement_strength
    modules_run = result.modules_run
    top = result.top_candidate

    # Build heat data: [lat, lon, intensity]
    max_prob = max((c.probability for c in result.candidates), default=1.0) or 1.0
    heat_points = []
    for c in result.candidates:
        intensity = c.probability / max_prob
        if intensity > 0.001:
            heat_points.append([c.lat, c.lon, round(intensity, 6)])

    # Build markers for top candidates
    markers = []
    for i, c in enumerate(result.candidates[:20]):
        if c.probability <= 0:
            continue
        if i < 1:
            color = "red"
            rank_label = "#1"
        elif i < 5:
            color = "orange"
            rank_label = f"#{i + 1}"
        else:
            color = "gold"
            rank_label = f"#{i + 1}"

        module_list = ", ".join(h.module for h in c.hits) if c.hits else "—"
        popup_html = (
            f"<b>{rank_label}</b><br>"
            f"Lat: {c.lat:.4f} Lon: {c.lon:.4f}<br>"
            f"Probability: {c.probability * 100:.2f}%<br>"
            f"Modules: {module_list}"
        ).replace('"', "&quot;")
        markers.append(
            {
                "lat": c.lat,
                "lon": c.lon,
                "color": color,
                "rank": rank_label,
                "prob": c.probability * 100,
                "popup": popup_html,
                "weight": c.probability / max_prob,
            }
        )

    top_info = ""
    if top:
        top_info = (
            f"Top candidate: {top.lat:.4f}, {top.lon:.4f} "
            f"({top.probability * 100:.2f}%)"
        )

    modules_str = ", ".join(modules_run) if modules_run else "none"

    html = _HTML_TEMPLATE.format(
        center_lat=lat,
        center_lon=lon,
        consensus_lat=lat,
        consensus_lon=lon,
        agreement=f"{strength:.1%}",
        top_info=top_info,
        modules_str=modules_str,
        heat_data=str(heat_points),
        markers_json=str(markers),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>geofind Heatmap</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; }}
  #info-panel {{
    background: #1a1a2e;
    color: #e0e0e0;
    padding: 12px 20px;
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
    font-size: 13px;
    border-bottom: 2px solid #0f3460;
  }}
  #info-panel .label {{ color: #888; }}
  #info-panel .value {{ color: #53d8fb; font-weight: bold; }}
  #info-panel .title {{ font-size: 16px; font-weight: bold; color: #53d8fb; }}
  #map {{ height: calc(100vh - 52px); width: 100%; }}
  .marker-popup b {{ font-size: 14px; }}
</style>
</head>
<body>
<div id="info-panel">
  <div>
    <span class="title">geofind</span>
  </div>
  <div>
    <span class="label">Consensus: </span>
    <span class="value">{consensus_lat:.4f}, {consensus_lon:.4f}</span>
  </div>
  <div>
    <span class="label">Agreement: </span>
    <span class="value">{agreement}</span>
  </div>
  <div>
    <span class="value">{top_info}</span>
  </div>
  <div>
    <span class="label">Modules: </span>
    <span class="value">{modules_str}</span>
  </div>
</div>
<div id="map"></div>
<script>
  var map = L.map('map').setView([{center_lat}, {center_lon}], 5);
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    maxZoom: 18,
    attribution: '&copy; OpenStreetMap contributors'
  }}).addTo(map);

  // Heatmap layer
  var heatData = {heat_data};
  L.heatLayer(heatData, {{
    radius: 25,
    blur: 15,
    maxZoom: 10,
    max: 1.0,
    gradient: {{0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'}}
  }}).addTo(map);

  // Markers for top candidates
  var markers = {markers_json};
  markers.forEach(function(m) {{
    var circleMarker = L.circleMarker([m.lat, m.lon], {{
      radius: 5 + m.weight * 8,
      fillColor: m.color,
      color: '#fff',
      weight: 1.5,
      fillOpacity: 0.85
    }}).addTo(map);
    circleMarker.bindPopup('<div class="marker-popup">' + m.popup + '</div>');
  }});

  // Consensus marker
  var consensusIcon = L.divIcon({{
    html: '<div style="width:16px;height:16px;border:3px solid #fff;border-radius:50%;background:#e94560;box-shadow:0 0 8px #e94560;"></div>',
    iconSize: [16, 16],
    iconAnchor: [8, 8],
    className: ''
  }});
  L.marker([{center_lat}, {{center_lon}}], {{icon: consensusIcon}})
    .addTo(map)
    .bindPopup('<b>Consensus</b><br>{consensus_lat:.4f}, {consensus_lon:.4f}<br>Agreement: {agreement}');
</script>
</body>
</html>
"""
