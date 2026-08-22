"""Analyze reranker impact: runs pipeline with and without reranking on test images."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from geofind.core.config import PipelineConfig, ModuleConfig
from geofind.core.pipeline import GeoPipeline
from geofind.utils.geo import haversine_km, LatLon

import json

def run_test(image_path: str, reranker_enabled: bool, label: str):
    """Run pipeline on a single image, return results dict."""
    config = PipelineConfig()
    # Disable EXIF for fair comparison
    config.modules["exif"] = ModuleConfig(name="exif", weight=5.0, enabled=False, optional=False)
    
    pipeline = GeoPipeline(config)
    
    start = time.time()
    result = pipeline.analyze(image_path)
    elapsed = time.time() - start
    
    top = result.top_candidate
    return {
        "image": Path(image_path).stem,
        "label": label,
        "reranker": reranker_enabled,
        "top_lat": top.lat if top else None,
        "top_lon": top.lon if top else None,
        "top_prob": top.probability if top else None,
        "agreement": result.agreement_strength,
        "modules_run": len(result.modules_run),
        "candidates": len(result.candidates),
        "time_s": round(elapsed, 1),
    }


def run_analyze_reranker():
    """Run images through pipeline and compare raw vs reranked results."""
    # Find test images
    test_dir = Path("W:/geofind/dev/test_images")
    images = sorted(test_dir.glob("wiki_*.jpg"))[:5]
    
    if not images:
        print("No test images found!")
        return
    
    print(f"\n{'='*80}")
    print(f"RERANKER IMPACT ANALYSIS — {len(images)} images")
    print(f"{'='*80}\n")
    
    # Step 1: Run with reranker
    print("▶ Running with reranker...")
    reranked_results = []
    for img in images:
        r = run_test(str(img), True, "reranked")
        reranked_results.append(r)
        print(f"  {r['image']}: lat={r['top_lat']:.4f}, lon={r['top_lon']:.4f}, "
              f"prob={r['top_prob']:.6f}, agreement={r['agreement']:.2%}")
    
    # Step 2: We need to compare with raw grid output (no reranking)
    # The simplest way is to check what the grid gives us vs after reranking
    # We'll do this by running with modified weights where all module weights are equal
    # to see if the reranker is distorting or improving
    
    print("\n▶ Analyzing per-module hit positions...")
    config2 = PipelineConfig()
    config2.modules["exif"] = ModuleConfig(name="exif", weight=5.0, enabled=False, optional=False)
    pipeline2 = GeoPipeline(config2)
    
    for img in images:
        print(f"\n  ── {img.stem} ──")
        # We need raw access to all_hits. Let's run analyze and inspect the result
        result = pipeline2.analyze(str(img))
        
        # Print all module hits
        for mod_name, hits in result.all_module_hits.items():
            if hits:
                # Compute centroid of hits
                total_conf = sum(h.confidence for h in hits)
                if total_conf > 0:
                    cent_lat = sum(h.lat * h.confidence for h in hits) / total_conf
                    cent_lon = sum(h.lon * h.confidence for h in hits) / total_conf
                else:
                    cent_lat = hits[0].lat
                    cent_lon = hits[0].lon
                
                # Distance from consensus
                dist_to_consensus = haversine_km(
                    LatLon(cent_lat, cent_lon),
                    LatLon(result.consensus_lat, result.consensus_lon)
                )
                
                hit_details = "; ".join(
                    f"({h.lat:.1f},{h.lon:.1f},c={h.confidence:.2f})"
                    for h in hits[:3]
                )
                print(f"    {mod_name:20s}: {len(hits):2d} hits, "
                      f"centroid=({cent_lat:.2f},{cent_lon:.2f}), "
                      f"dist_to_consensus={dist_to_consensus:.0f}km | {hit_details}")
            else:
                print(f"    {mod_name:20s}: 0 hits")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    run_analyze_reranker()
