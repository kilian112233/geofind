"""Local Vision LLM (LLaVA) module with Llava15ChatHandler for multimodal inference."""

from __future__ import annotations

import gc
import logging
import re
from pathlib import Path
from typing import Any

from geofind.core.candidate import ModuleHit
from geofind.core.config import PipelineConfig
from geofind.modules.base import BaseModule

logger = logging.getLogger(__name__)

# ── Expanded location database: country centroids + ~200 cities ──────────────
_LOCATION_COORDS: dict[str, tuple[float, float]] = {
    # ── Countries ──
    "united states": (37.09, -95.71), "usa": (37.09, -95.71),
    "canada": (56.13, -106.35), "mexico": (23.63, -102.55),
    "brazil": (-14.24, -51.93), "argentina": (-38.42, -63.62),
    "united kingdom": (55.38, -3.44), "england": (52.36, -1.17),
    "scotland": (56.49, -4.20), "wales": (52.13, -3.78),
    "france": (46.23, 2.21), "germany": (51.17, 10.45),
    "italy": (41.88, 12.57), "spain": (40.46, -3.75),
    "portugal": (39.40, -8.22), "netherlands": (52.13, 5.29),
    "belgium": (50.50, 4.47), "switzerland": (46.82, 8.23),
    "austria": (47.52, 14.55), "sweden": (60.13, 18.64),
    "norway": (60.47, 8.47), "denmark": (56.26, 9.50),
    "finland": (61.92, 25.75), "poland": (51.92, 19.15),
    "czech republic": (49.82, 15.47), "czechia": (49.82, 15.47),
    "hungary": (47.16, 19.50), "romania": (45.94, 24.97),
    "greece": (39.07, 21.82), "turkey": (38.96, 35.24),
    "russia": (61.52, 105.32), "china": (35.86, 104.20),
    "japan": (36.20, 138.25), "south korea": (35.91, 127.77),
    "korea": (35.91, 127.77), "india": (20.59, 78.96),
    "thailand": (15.87, 100.99), "vietnam": (14.06, 108.28),
    "indonesia": (-0.79, 113.92), "philippines": (12.88, 121.77),
    "malaysia": (4.21, 101.98), "singapore": (1.35, 103.82),
    "australia": (-25.27, 133.78), "new zealand": (-40.90, 174.89),
    "south africa": (-30.56, 22.94), "egypt": (26.82, 30.80),
    "nigeria": (9.08, 8.68), "kenya": (-0.02, 37.91),
    "ethiopia": (9.15, 40.49), "morocco": (31.79, -7.09),
    "saudi arabia": (23.89, 45.08), "uae": (23.42, 53.85),
    "united arab emirates": (23.42, 53.85), "israel": (31.05, 34.85),
    "pakistan": (30.38, 69.35), "bangladesh": (23.68, 90.36),
    "nepal": (28.39, 84.12), "sri lanka": (7.87, 80.77),
    "colombia": (4.57, -74.30), "chile": (-35.68, -71.54),
    "peru": (-9.19, -75.02), "venezuela": (6.42, -66.59),
    "cuba": (21.52, -77.78), "jamaica": (18.11, -77.30),
    "ireland": (53.14, -7.69), "iceland": (64.96, -19.02),
    "ukraine": (48.38, 31.17), "croatia": (45.10, 15.20),
    "serbia": (44.02, 21.01), "bulgaria": (42.73, 25.49),
    "taiwan": (23.70, 120.96), "hong kong": (22.40, 114.11),
    "ireland": (53.14, -7.69), "iceland": (64.96, -19.02),
    "croatia": (45.10, 15.20), "slovenia": (46.15, 14.99),
    "slovakia": (48.67, 19.70), "lithuania": (55.17, 23.88),
    "latvia": (56.88, 24.60), "estonia": (58.60, 25.01),
    "albania": (41.15, 20.17), "north macedonia": (41.51, 21.75),
    "bosnia": (43.92, 17.68), "montenegro": (42.71, 19.37),
    "kosovo": (42.60, 20.90), "moldova": (47.41, 28.37),
    "belarus": (53.71, 27.95), "georgia": (42.32, 43.36),
    "armenia": (40.07, 45.04), "azerbaijan": (40.14, 47.58),
    "kazakhstan": (48.02, 66.92), "uzbekistan": (41.38, 64.59),
    "mongolia": (46.86, 103.85), "myanmar": (21.91, 95.96),
    "cambodia": (12.57, 104.99), "laos": (19.86, 102.50),
    "brunei": (4.54, 114.73), "papua new guinea": (-6.31, 143.96),
    "tanzania": (-6.37, 34.89), "uganda": (1.37, 32.29),
    "ghana": (7.95, -1.02), "senegal": (14.50, -14.45),
    "cameroon": (7.37, 12.35), "ivory coast": (7.54, -5.55),
    "madagascar": (-18.77, 46.87), "mozambique": (-18.67, 35.53),
    "zambia": (-13.13, 27.85), "zimbabwe": (-19.02, 29.15),
    "botswana": (-22.33, 24.68), "namibia": (-22.96, 18.49),
    "paraguay": (-23.44, -58.44), "uruguay": (-32.52, -55.77),
    "bolivia": (-16.29, -63.59), "ecuador": (-1.83, -78.18),
    "panama": (8.54, -80.78), "costa rica": (9.75, -83.75),
    "honduras": (15.20, -86.24), "guatemala": (15.78, -90.23),
    "el salvador": (13.79, -88.90), "nicaragua": (12.87, -85.21),
    "dominican republic": (18.74, -70.16), "haiti": (18.97, -72.29),
    "trinidad": (10.69, -61.22), "puerto rico": (18.22, -66.59),
    "lebanon": (33.85, 35.86), "jordan": (30.59, 36.24),
    "iraq": (33.22, 43.68), "iran": (32.43, 53.69),
    "syria": (34.80, 38.00), "yemen": (15.55, 48.52),
    "oman": (21.47, 55.98), "qatar": (25.35, 51.18),
    "kuwait": (29.31, 47.48), "bahrain": (26.07, 50.55),
    "cyprus": (35.13, 33.43), "malta": (35.94, 14.38),
    "luxembourg": (49.82, 6.13), "andorra": (42.55, 1.60),
    "monaco": (43.73, 7.42), "san marino": (43.94, 12.46),
    "liechtenstein": (47.17, 9.56), "bermuda": (32.31, -64.75),
    # ── US Cities ──
    "new york": (40.71, -74.01), "los angeles": (34.05, -118.24),
    "chicago": (41.88, -87.63), "houston": (29.76, -95.37),
    "phoenix": (33.45, -112.07), "philadelphia": (39.95, -75.17),
    "san antonio": (29.42, -98.49), "san diego": (32.72, -117.16),
    "dallas": (32.78, -96.80), "san jose": (37.34, -121.89),
    "austin": (30.27, -97.74), "jacksonville": (30.33, -81.66),
    "fort worth": (32.75, -97.33), "columbus": (39.96, -82.99),
    "charlotte": (35.23, -80.84), "san francisco": (37.77, -122.42),
    "indianapolis": (39.77, -86.16), "seattle": (47.61, -122.33),
    "denver": (39.74, -104.99), "washington": (38.91, -77.04),
    "nashville": (36.16, -86.78), "oklahoma city": (35.47, -97.52),
    "el paso": (31.76, -106.49), "boston": (42.36, -71.06),
    "portland": (45.52, -122.68), "las vegas": (36.17, -115.14),
    "memphis": (35.15, -90.05), "louisville": (38.25, -85.76),
    "baltimore": (39.29, -76.61), "milwaukee": (43.04, -87.91),
    "albuquerque": (35.08, -106.65), "tucson": (32.22, -110.97),
    "fresno": (36.74, -119.79), "sacramento": (38.58, -121.49),
    "mesa": (33.42, -111.83), "kansas city": (39.10, -94.58),
    "atlanta": (33.75, -84.39), "omaha": (41.26, -95.94),
    "miami": (25.76, -80.19), "minneapolis": (44.98, -93.27),
    "new orleans": (29.95, -90.07), "detroit": (42.33, -83.05),
    "honolulu": (21.31, -157.86), "pittsburgh": (40.44, -79.99),
    "st louis": (38.63, -90.20), "cincinnati": (39.10, -84.51),
    "minneapolis": (44.98, -93.27), "raleigh": (35.78, -78.64),
    "tampa": (27.95, -82.46), "charleston": (32.78, -79.93),
    "savannah": (32.08, -81.09), "salt lake city": (40.76, -111.89),
    "honolulu": (21.31, -157.86), "anchorage": (61.22, -149.90),
    "boise": (43.62, -116.21), "spokane": (47.66, -117.43),
    "buffalo": (42.89, -78.88), "rochester": (43.16, -77.61),
    "milwaukee": (43.04, -87.91), "madison": (43.07, -89.40),
    "des moines": (41.59, -93.62), "wichita": (37.69, -97.34),
    "lubbock": (33.58, -101.85), "amarillo": (35.22, -101.83),
    "little rock": (34.75, -92.29), "baton rouge": (30.45, -91.19),
    "jackson": (32.30, -90.18), "montgomery": (32.38, -86.30),
    "richmond": (37.54, -77.44), "norfolk": (36.85, -76.29),
    "lexington": (38.04, -84.50), "knoxville": (35.96, -83.92),
    "chattanooga": (35.05, -85.31), "birmingham": (33.52, -86.81),
    "huntsville": (34.73, -86.59), "mobile": (30.69, -88.04),
    "tulsa": (36.15, -95.99), "springfield": (39.78, -89.65),
    "providence": (41.82, -71.41), "hartford": (41.76, -72.68),
    "new haven": (41.31, -72.92), "bridgeport": (41.19, -73.19),
    # ── European Cities ──
    "london": (51.51, -0.13), "paris": (48.86, 2.35),
    "berlin": (52.52, 13.41), "rome": (41.90, 12.50),
    "madrid": (40.42, -3.70), "amsterdam": (52.37, 4.90),
    "vienna": (48.21, 16.37), "prague": (50.08, 14.44),
    "brussels": (50.85, 4.35), "zurich": (47.38, 8.54),
    "munich": (48.14, 11.58), "hamburg": (53.55, 9.99),
    "frankfurt": (50.11, 8.68), "cologne": (50.94, 6.96),
    "milan": (45.46, 9.19), "naples": (40.85, 14.27),
    "turin": (45.07, 7.69), "florence": (43.77, 11.25),
    "barcelona": (41.39, 2.17), "valencia": (39.47, -0.38),
    "seville": (37.39, -5.98), "lisbon": (38.72, -9.14),
    "porto": (41.15, -8.61), "stockholm": (59.33, 18.07),
    "copenhagen": (55.68, 12.57), "oslo": (59.91, 10.75),
    "helsinki": (60.17, 24.94), "dublin": (53.35, -6.26),
    "edinburgh": (55.95, -3.19), "glasgow": (55.86, -4.25),
    "manchester": (53.48, -2.24), "birmingham england": (52.49, -1.89),
    "warsaw": (52.23, 21.01), "krakow": (50.06, 19.94),
    "budapest": (47.50, 19.04), "athens": (37.98, 23.73),
    "istanbul": (41.01, 28.98), "moscow": (55.76, 37.62),
    "saint petersburg": (59.93, 30.32), "kyiv": (50.45, 30.52),
    "bucharest": (44.43, 26.10), "sofia": (42.70, 23.32),
    "belgrade": (44.79, 20.47), "zagreb": (45.81, 15.98),
    "ljubljana": (46.06, 14.51), "bratislava": (48.15, 17.11),
    "tallinn": (59.44, 24.75), "riga": (56.95, 24.11),
    "vilnius": (54.69, 25.28), "reykjavik": (64.15, -21.94),
    "luxembourg city": (49.61, 6.13), "monaco city": (43.73, 7.42),
    "nice": (43.71, 7.26), "lyon": (45.76, 4.84),
    "marseille": (43.30, 5.37), "bordeaux": (44.84, -0.58),
    "strasbourg": (48.57, 7.75), "lille": (50.63, 3.06),
    "cannes": (43.55, 7.02), "venice": (45.44, 12.34),
    "genoa": (44.41, 8.93), "palermo": (38.12, 13.36),
    "catania": (37.50, 15.09), "bologna": (44.49, 11.34),
    "verona": (45.44, 10.99), "dubrovnik": (42.65, 18.09),
    "split": (43.51, 16.44), "sarajevo": (43.86, 18.41),
    "tirana": (41.33, 19.82), "thessaloniki": (40.64, 22.94),
    "heraklion": (35.34, 25.14), "mykonos": (37.45, 25.33),
    "santorini": (36.39, 25.46), "corfu": (39.62, 19.92),
    "budva": (42.29, 18.84), "kotor": (42.42, 18.77),
    "Mostar": (43.34, 17.81),
    # ── Asian Cities ──
    "tokyo": (35.68, 139.69), "osaka": (34.69, 135.50),
    "kyoto": (35.01, 135.77), "yokohama": (35.44, 139.64),
    "nagoya": (35.18, 136.91), "hiroshima": (34.40, 132.46),
    "fukuoka": (33.59, 130.40), "sapporo": (43.06, 141.35),
    "beijing": (39.90, 116.40), "shanghai": (31.23, 121.47),
    "guangzhou": (23.13, 113.26), "shenzhen": (22.54, 114.06),
    "chengdu": (30.57, 104.07), "hangzhou": (30.27, 120.15),
    "xian": (34.26, 108.94), "wuhan": (30.59, 114.31),
    "chongqing": (29.56, 106.55), "tianjin": (39.14, 117.18),
    "nanjing": (32.06, 118.80), "suzhou": (31.30, 120.62),
    "hong kong": (22.40, 114.11), "macau": (22.20, 113.55),
    "taipei": (25.03, 121.57), "kaohsiung": (22.63, 120.30),
    "seoul": (37.57, 126.98), "busan": (35.18, 129.08),
    "bangkok": (13.76, 100.50), "chiang mai": (18.79, 98.98),
    "phuket": (7.88, 98.39), "pattaya": (12.92, 100.88),
    "hanoi": (21.03, 105.85), "ho chi minh": (10.82, 106.63),
    "da nang": (16.05, 108.22), "siem reap": (13.36, 103.86),
    "phnom penh": (11.56, 104.92), "kuala lumpur": (3.14, 101.69),
    "penang": (5.41, 100.33), "singapore": (1.35, 103.82),
    "jakarta": (-6.21, 106.85), "bali": (-8.34, 115.09),
    "yogyakarta": (-7.80, 110.36), "manila": (14.60, 120.98),
    "cebu": (10.31, 123.89), "mumbai": (19.08, 72.88),
    "delhi": (28.61, 77.21), "bangalore": (12.97, 77.59),
    "chennai": (13.08, 80.27), "kolkata": (22.57, 88.36),
    "jaipur": (26.91, 75.79), "agra": (27.18, 78.02),
    "varanasi": (25.32, 83.01), "goa": (15.49, 73.83),
    "kochi": (9.93, 76.27), "lahore": (31.55, 74.35),
    "karachi": (24.86, 67.01), "islamabad": (33.69, 73.04),
    "dubai": (25.20, 55.27), "abu dhabi": (24.45, 54.65),
    "doha": (25.29, 51.53), "kuwait city": (29.38, 47.99),
    "riyadh": (24.71, 46.68), "jeddah": (21.49, 39.19),
    "cairo": (30.04, 31.24), "alexandria": (31.20, 29.92),
    "marrakech": (31.63, -8.01), "casablanca": (33.57, -7.59),
    "cape town": (-33.93, 18.42), "johannesburg": (-26.20, 28.05),
    "nairobi": (-1.29, 36.82), "dar es salaam": (-6.79, 39.28),
    "accra": (5.60, -0.19), "lagos": (6.52, 3.38),
    "addis ababa": (9.02, 38.75), "casablanca": (33.57, -7.59),
    "tel aviv": (32.09, 34.78), "jerusalem": (31.77, 35.23),
    "amman": (31.95, 35.93), "beirut": (33.89, 35.50),
    "baghdad": (33.31, 44.37), "tehran": (35.69, 51.39),
    "kabul": (34.53, 69.17), "colombo": (6.93, 79.84),
    "kathmandu": (27.72, 85.32), "thimphu": (27.47, 89.64),
    "dhaka": (23.81, 90.41), "yangon": (16.87, 96.20),
    "ulaanbaatar": (47.91, 106.91), "bishkek": (42.87, 74.59),
    "tashkent": (41.30, 69.28), "dushanbe": (38.56, 68.77),
    "ashgabat": (37.95, 58.38), "tbilisi": (41.72, 44.79),
    "yerevan": (40.18, 44.51), "baku": (40.41, 49.87),
    # ── Oceania ──
    "sydney": (-33.87, 151.21), "melbourne": (-37.81, 144.96),
    "brisbane": (-27.47, 153.03), "perth": (-31.95, 115.86),
    "adelaide": (-34.93, 138.60), "auckland": (-36.85, 174.76),
    "wellington": (-41.29, 174.78), "christchurch": (-43.53, 172.64),
    "queenstown": (-45.03, 168.66),
    # ── South America ──
    "buenos aires": (-34.60, -58.38), "santiago": (-33.45, -70.67),
    "lima": (-12.05, -77.04), "bogota": (4.71, -74.07),
    "rio de janeiro": (-22.91, -43.17), "sao paulo": (-23.55, -46.63),
    "brasilia": (-15.78, -47.93), "medellin": (6.24, -75.57),
    "quito": (-0.18, -78.47), "caracas": (10.49, -66.88),
    "montevideo": (-34.90, -56.19), "asuncion": (-25.26, -57.58),
    "la paz": (-16.50, -68.15), "cusco": (-13.53, -71.97),
    "mendoza": (-32.89, -68.83), "cartagena": (10.39, -75.51),
    "cusco": (-13.53, -71.97),
    # ── Canada ──
    "toronto": (43.65, -79.38), "vancouver": (49.28, -123.12),
    "montreal": (45.50, -73.57), "calgary": (51.05, -114.07),
    "ottawa": (45.42, -75.70), "edmonton": (53.55, -113.49),
    "winnipeg": (49.90, -97.14), "halifax": (44.65, -63.57),
    "quebec city": (46.81, -71.21), "victoria": (48.43, -123.37),
    "saskatoon": (52.13, -106.67), "regina": (50.45, -104.62),
    "kelowna": (49.88, -119.49),
}


class VisionLlmModule(BaseModule):
    """Use a local LLaVA vision model to identify locations in images.

    Uses Llava15ChatHandler for proper multimodal inference with mmproj.
    Loads model on-demand, unloads after each image to save RAM.
    """

    name = "vision_llm"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)
        self._model = None
        self._model_path: Path | None = None
        self._mmproj_path: Path | None = None

    def is_available(self) -> bool:
        try:
            from llama_cpp import Llama  # noqa: F401
            from llama_cpp.llama_chat_format import Llava15ChatHandler  # noqa: F401
            return True
        except ImportError:
            return False

    def prepare(self) -> None:
        models_dir = self.config.models_dir
        all_gguf = list(models_dir.glob("*.gguf"))

        # Find the LLM model (not mmproj)
        llm_candidates = [f for f in all_gguf
                          if "mmproj" not in f.name.lower()]
        llava_candidates = [f for f in llm_candidates
                           if "llava" in f.stem.lower()]

        if llava_candidates:
            self._model_path = llava_candidates[0]
        elif llm_candidates:
            self._model_path = llm_candidates[0]

        # Find mmproj (vision projector)
        mmproj = [f for f in all_gguf if "mmproj" in f.name.lower()]
        if mmproj:
            self._mmproj_path = mmproj[0]

        if self._model_path and self._mmproj_path:
            self._log(
                f"LLM: {self._model_path.name}, "
                f"mmproj: {self._mmproj_path.name}"
            )
        elif self._model_path:
            self._log(
                f"LLM: {self._model_path.name} (no mmproj found — "
                f"vision may not work)",
                logging.WARNING,
            )
        else:
            self._log("No GGUF model found in models directory", logging.WARNING)

        super().prepare()

    def _load_model(self) -> bool:
        """Load the LLaVA model with mmproj. Returns True on success."""
        if self._model is not None:
            return True
        if self._model_path is None:
            return False

        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Llava15ChatHandler

            if self._mmproj_path:
                chat_handler = Llava15ChatHandler(
                    clip_model_path=str(self._mmproj_path),
                    verbose=False,
                )
                self._model = Llama(
                    model_path=str(self._model_path),
                    chat_handler=chat_handler,
                    n_ctx=2048,
                    n_gpu_layers=0,
                    verbose=False,
                    embedding=False,
                )
                self._log("LLaVA model loaded with mmproj vision")
            else:
                self._model = Llama(
                    model_path=str(self._model_path),
                    n_ctx=2048,
                    n_gpu_layers=0,
                    verbose=False,
                )
                self._log("LLaVA model loaded (no vision projector)")
            return True
        except Exception as e:
            self._log(f"Failed to load LLaVA: {e}", logging.WARNING)
            return False

    def _unload_model(self) -> None:
        """Explicitly release model memory."""
        if self._model is not None:
            try:
                del self._model
            except Exception:
                pass
            self._model = None
            gc.collect()
            self._log("LLaVA model unloaded to free RAM")

    def detect(
        self,
        media_path: Path,
        *,
        frames: list[Any] | None = None,
        audio_path: Path | None = None,
    ) -> list[ModuleHit]:
        if self._model_path is None:
            self._log("No vision LLM model available, skipping")
            return []

        if not self._load_model():
            return []

        image = self._get_image(media_path, frames)
        if image is None:
            return []

        hits: list[ModuleHit] = []
        prompt = (
            "Look at this image carefully. What specific country, city, "
            "or region is this photo taken in? Consider any visible text, "
            "signs, architecture, landscape, license plates, or other clues. "
            "Be as specific as possible. "
            "Answer with just the location name (e.g. 'Tokyo, Japan' or "
            "'Paris, France' or 'California, United States')."
        )

        try:
            import base64
            import io

            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            response = self._model.create_chat_completion(
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
                max_tokens=150,
                temperature=0.1,
            )
            raw_text = response["choices"][0]["message"]["content"].strip()
            # Strip markdown artifacts (### headers, **, *, etc.)
            text = re.sub(r'#{1,6}\s*', '', raw_text)
            text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
            text = text.split('\n')[0].strip()  # Take first line only
            # Remove trailing punctuation artifacts
            text = re.sub(r'[#*_`~]+$', '', text).strip()
            self._log(f"LLM response: {text}")

            hits = self._parse_location(text)

        except Exception as e:
            self._log(f"LLaVA inference failed: {e}", logging.WARNING)

        # Unload after inference to save RAM
        self._unload_model()

        return hits

    def _parse_location(self, text: str) -> list[ModuleHit]:
        """Parse LLM text into location hits. Tries longest-match first."""
        hits: list[ModuleHit] = []
        text_lower = text.lower().strip()

        # Clean up common LLM artifacts
        text_clean = re.sub(r'[^\w\s,.-]', '', text_lower)

        # Sort locations longest-first for greedy matching
        sorted_locations = sorted(
            _LOCATION_COORDS.keys(), key=len, reverse=True,
        )
        matched: list[str] = []

        for loc in sorted_locations:
            if loc in text_clean:
                # Don't match sub-strings of already-matched locations
                if not any(loc in m or m in loc for m in matched):
                    matched.append(loc)

        # Fallback: try capitalized words if no match
        if not matched:
            words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            for w in words:
                wl = w.lower()
                if wl in _LOCATION_COORDS:
                    matched.append(wl)

        for loc in matched:
            lat, lon = _LOCATION_COORDS[loc]
            # More specific locations get higher confidence
            confidence = min(0.5 + len(loc) * 0.02, 0.9)
            hits.append(self._make_hit(
                lat, lon, confidence,
                location_mention=loc,
                llm_response=text[:500],
            ))

        return hits

    def _get_image(self, media_path: Path, frames: list[Any] | None) -> Any | None:
        from PIL import Image

        if frames:
            f = frames[0]
            if isinstance(f, Image.Image):
                return f.convert("RGB")
            try:
                import numpy as np
                return Image.fromarray(f).convert("RGB")
            except Exception:
                pass

        try:
            from geofind.utils.media import load_image
            return load_image(media_path)
        except Exception:
            return None
