"""OCR + Script/Language Detection module with place-name geocoding."""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

from geofind.core.candidate import ModuleHit
from geofind.core.config import PipelineConfig
from geofind.modules.base import BaseModule

logger = logging.getLogger(__name__)

_COUNTRY_CENTROIDS: dict[str, tuple[float, float]] = {
    "TH": (13.75, 100.50), "JP": (36.20, 138.25), "KR": (35.91, 127.77),
    "CN": (35.86, 104.20), "TW": (23.70, 120.96), "HK": (22.40, 114.11),
    "MO": (22.20, 113.55), "SA": (23.89, 45.08), "AE": (23.42, 53.85),
    "EG": (26.82, 30.80), "MA": (31.79, -7.09), "TN": (33.89, 9.54),
    "IQ": (33.22, 43.68), "IR": (32.43, 53.69), "JO": (30.59, 36.24),
    "LB": (33.85, 35.86), "RU": (61.52, 105.32), "UA": (48.38, 31.17),
    "BG": (42.73, 25.49), "RS": (44.02, 21.01), "BY": (53.71, 27.95),
    "KZ": (48.02, 66.92), "KG": (41.20, 74.77), "IN": (20.59, 78.96),
    "NP": (28.39, 84.12), "BD": (23.68, 90.36), "LK": (7.87, 80.77),
    "TR": (38.96, 35.24), "VN": (14.06, 108.28), "ID": (-0.79, 113.92),
    "PH": (12.88, 121.77), "MY": (4.21, 101.98), "GR": (39.07, 21.82),
    "IL": (31.05, 34.85), "GE": (42.32, 43.36), "AM": (40.07, 45.04),
    "ET": (9.15, 40.49), "MM": (21.91, 95.96), "KH": (12.57, 104.99),
    "LA": (19.86, 102.50),
}

_SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    "thai": [(0x0E00, 0x0E7F)],
    "japanese_hiragana": [(0x3040, 0x309F)],
    "japanese_katakana": [(0x30A0, 0x30FF)],
    "korean": [(0xAC00, 0xD7AF), (0x1100, 0x11FF)],
    "chinese_cjk": [(0x4E00, 0x9FFF)],
    "arabic": [(0x0600, 0x06FF), (0x0750, 0x077F)],
    "cyrillic": [(0x0400, 0x04FF)],
    "devanagari": [(0x0900, 0x097F)],
    "bengali": [(0x0980, 0x09FF)],
    "tamil": [(0x0B80, 0x0BFF)],
    "telugu": [(0x0C00, 0x0C7F)],
    "kannada": [(0x0C80, 0x0CFF)],
    "malayalam": [(0x0D00, 0x0D7F)],
    "gurmukhi": [(0x0A00, 0x0A7F)],
    "greek": [(0x0370, 0x03FF)],
    "hebrew": [(0x0590, 0x05FF)],
    "georgian": [(0x10A0, 0x10FF)],
    "armenian": [(0x0530, 0x058F)],
    "ethiopic": [(0x1200, 0x137F)],
    "myanmar": [(0x1000, 0x109F)],
    "khmer": [(0x1780, 0x17FF)],
    "lao": [(0x0E80, 0x0EFF)],
    "tibetan": [(0x0F00, 0x0FFF)],
}

_SCRIPT_TO_HINT_KEY: dict[str, str] = {
    "thai": "thai", "japanese_hiragana": "japanese", "japanese_katakana": "japanese",
    "korean": "korean", "chinese_cjk": "chinese_similar", "arabic": "arabic",
    "cyrillic": "cyrillic", "devanagari": "devanagari", "bengali": "bengali",
    "tamil": "tamil", "telugu": "telugu", "kannada": "kannada",
    "malayalam": "malayalam", "gurmukhi": "gurmukhi", "greek": "greek",
    "hebrew": "hebrew", "georgian": "georgian", "armenian": "armenian",
    "ethiopic": "ethiopic", "myanmar": "myanmar", "khmer": "khmer",
    "lao": "lao", "tibetan": "tibetan",
}

_LATIN_EXTENDED_RANGES: list[tuple[int, int]] = [
    (0x00C0, 0x024F), (0x1E00, 0x1EFF), (0x0100, 0x017F),
    (0x0180, 0x024F), (0x0250, 0x02AF),
]

# ---------------------------------------------------------------------------
# Place-name geocoding lookup  (hardcoded, no network calls)
# Format:  "lowercase name" -> (lat, lon, "type")
#   type: "city", "capital", "country", "admin"
# For ambiguous names (multiple cities share the same name), we store the
# most-populated city and add _alt variants for the others.
# ---------------------------------------------------------------------------

_PLACE_NAMES: dict[str, tuple[float, float, str]] = {
    # --- World capitals & major cities: Africa ---
    "abidjan": (5.36, -4.01, "city"), "abuja": (9.06, 7.49, "capital"),
    "accra": (5.60, -0.19, "capital"), "addis ababa": (9.03, 38.75, "capital"),
    "algiers": (36.75, 3.04, "capital"), "amman": (31.95, 35.93, "capital"),
    "ankara": (39.93, 32.86, "capital"), "antananarivo": (-18.91, 47.54, "capital"),
    "asmara": (15.34, 38.93, "capital"), "bamako": (12.64, -8.0, "capital"),
    "bangui": (4.37, 18.56, "capital"), "banjul": (13.45, -16.58, "capital"),
    "bissau": (11.80, -15.18, "capital"), "blantyre": (-15.79, 35.01, "city"),
    "brazzaville": (-4.26, 15.28, "capital"), "bujumbura": (-3.38, 29.36, "city"),
    "cairo": (30.04, 31.24, "capital"), "casablanca": (33.57, -7.59, "city"),
    "conakry": (9.64, -13.58, "capital"), "dakar": (14.72, -17.47, "capital"),
    "dar es salaam": (-6.79, 39.28, "city"), "djibouti": (11.59, 43.15, "capital"),
    "doha": (25.29, 51.53, "capital"), "douala": (4.05, 9.70, "city"),
    "el fasher": (13.63, 25.35, "city"), "freetown": (8.48, -13.23, "capital"),
    "gaborone": (-24.63, 25.91, "capital"), "harare": (-17.83, 31.05, "capital"),
    "johannesburg": (-26.20, 28.04, "city"), "kampala": (0.35, 32.58, "capital"),
    "khartoum": (15.50, 32.56, "capital"), "kigali": (-1.94, 30.06, "capital"),
    "kinshasa": (-4.44, 15.27, "capital"),     "kisangani": (0.52, 25.19, "city"),
    "kolkata": (22.57, 88.36, "city"),
    "laayoune": (27.15, -13.20, "city"),
    "libreville": (0.38, 9.46, "capital"), "lilongwe": (-13.97, 33.79, "capital"),
    "lomé": (6.13, 1.23, "capital"), "luanda": (-8.84, 13.23, "capital"),
    "lubumbashi": (-11.68, 27.50, "city"), "lusaka": (-15.42, 28.28, "capital"),
    "malabo": (3.75, 8.77, "capital"), "maputo": (-25.97, 32.57, "capital"),
    "marrakech": (31.63, -8.01, "city"), "marrakesh": (31.63, -8.01, "city"),
    "maseru": (-29.32, 27.48, "capital"),     "mbabane": (-26.31, 31.13, "capital"),
    "mogadishu": (2.05, 45.32, "capital"), "moroni": (-11.70, 43.24, "capital"),
    "nairobi": (-1.29, 36.82, "capital"),
    "ndjamena": (12.13, 15.07, "capital"), "niamey": (13.51, 2.11, "capital"),
    "nouakchott": (18.09, -15.98, "capital"), "ouagadougou": (12.37, -1.52, "capital"),
    "prétoria": (-25.75, 28.19, "capital"), "pretoria": (-25.75, 28.19, "capital"),
    "tripoli": (32.90, 13.18, "capital"), "tunis": (36.81, 10.18, "capital"),
    "windhoek": (-22.56, 17.08, "capital"), "yaoundé": (3.87, 11.52, "capital"),
    "yaounde": (3.87, 11.52, "capital"),
    # --- World capitals & major cities: Asia ---
    "abu dhabi": (24.45, 54.65, "capital"), "almaty": (43.24, 76.95, "city"),
    "amritsar": (31.63, 74.87, "city"),     "astana": (51.13, 71.43, "capital"), "bahrain": (26.23, 50.59, "capital"),
    "baku": (40.41, 49.87, "capital"), "bandar seri begawan": (4.93, 114.95, "capital"),
    "bangkok": (13.76, 100.50, "city"), "bangalore": (12.97, 77.59, "city"),
    "bengaluru": (12.97, 77.59, "city"), "beirut": (33.89, 35.50, "capital"),
    "bhopal": (23.26, 77.41, "city"), "bhubaneswar": (20.30, 85.82, "city"),
    "brasilia": (-15.79, -47.88, "capital"),
    "chennai": (13.08, 80.27, "city"), "chittagong": (22.36, 91.78, "city"),
    "chongqing": (29.43, 106.91, "city"), "colombo": (6.93, 79.85, "capital"),
    "damascus": (33.51, 36.29, "capital"), "delhi": (28.61, 77.23, "city"),
    "dhaka": (23.81, 90.41, "capital"), "dubai": (25.20, 55.27, "city"),
    "dushanbe": (38.56, 68.77, "capital"), "faisalabad": (31.42, 73.08, "city"),
    "gaza": (31.50, 34.47, "city"), "guangzhou": (23.13, 113.26, "city"),
    "hangzhou": (30.27, 120.15, "city"), "hanoi": (21.03, 105.85, "capital"),
    "havana": (23.11, -82.37, "capital"),
    "ho chi minh city": (10.82, 106.63, "city"),
    "hong kong": (22.40, 114.11, "city"),
    "hyderabad": (17.39, 78.49, "city"),
    "islamabad": (33.69, 73.04, "capital"),
    "istanbul": (41.01, 28.98, "city"), "izmir": (38.42, 27.14, "city"),
    "jaipur": (26.91, 75.79, "city"), "jakarta": (-6.21, 106.85, "capital"),
    "jerusalem": (31.77, 35.23, "capital"), "jeddah": (21.49, 39.19, "city"),
    "johor bahru": (1.49, 103.74, "city"),
    "kabul": (34.53, 69.17, "capital"), "kathmandu": (27.72, 85.32, "capital"),
    "kawasaki": (35.53, 139.70, "city"),
    "kolkata": (22.57, 88.36, "city"), "kuala lumpur": (3.14, 101.69, "capital"),
    "kuwait city": (29.38, 47.99, "capital"), "kyoto": (35.01, 135.77, "city"),
    "lagos": (6.52, 3.38, "city"), "lahore": (31.55, 74.35, "city"),
    "las vegas": (36.17, -115.14, "city"),
    "macau": (22.20, 113.55, "city"), "manila": (14.60, 120.98, "capital"),
    "mashhad": (36.30, 59.60, "city"), "medina": (24.47, 39.61, "city"),
    "mecca": (21.39, 39.86, "city"),
    "muscat": (23.59, 58.54, "capital"), "nagoya": (35.18, 136.91, "city"),
    "nanning": (22.82, 108.32, "city"), "new delhi": (28.61, 77.23, "capital"),
    "novosibirsk": (55.01, 82.93, "city"),
    "osaka": (34.69, 135.50, "city"),
    "phnom penh": (11.56, 104.92, "capital"),
    "peshawar": (34.01, 71.58, "city"),
    "prague": (50.08, 14.44, "city"),  # also Europe
    "pyongyang": (39.02, 125.75, "capital"),
    "qatar": (25.35, 51.18, "capital"),  # country name used as city hint
    "riyadh": (24.71, 46.68, "capital"),
    "sapporo": (43.06, 141.35, "city"), "seoul": (37.57, 126.98, "capital"),
    "shanghai": (31.23, 121.47, "city"), "shenzhen": (22.54, 114.06, "city"),
    "singapore": (1.35, 103.82, "capital"), "stockholm": (59.33, 18.07, "capital"),
    "suzhou": (31.30, 120.62, "city"),
    "taipei": (25.03, 121.57, "capital"), "tashkent": (41.30, 69.28, "capital"),
    "tbilisi": (41.72, 44.78, "capital"),
    "tianjin": (39.14, 117.18, "city"),
    "tokyo": (35.68, 139.69, "capital"),
    "tunis": (36.81, 10.18, "capital"),
    "ube": (33.91, 131.25, "city"),
    "vientiane": (17.97, 102.63, "capital"),
    "vijayawada": (16.51, 80.63, "city"),
    "washington": (38.91, -77.04, "city"),  # DC
    "wenzhou": (27.99, 120.70, "city"),
    "wuhan": (30.59, 114.31, "city"), "xiamen": (24.48, 118.09, "city"),
    "xi'an": (34.26, 108.94, "city"), "xian": (34.26, 108.94, "city"),
    "yangon": (16.87, 96.20, "city"), "yerevan": (40.18, 44.51, "capital"),
    "yokohama": (35.44, 139.64, "city"),
    "zhengzhou": (34.75, 113.65, "city"),
    "fukuoka": (33.59, 130.40, "city"),
    "sendai": (38.27, 140.87, "city"),
    "kobe": (34.69, 135.20, "city"),
    "okinawa": (26.34, 127.80, "city"),
    "da nang": (16.05, 108.22, "city"),
    "hoi an": (15.88, 108.33, "city"),
    "siem reap": (13.37, 103.86, "city"),
    "luang prabang": (19.89, 102.13, "city"),
    "bali": (-8.34, 115.09, "city"),
    "yogyakarta": (-7.80, 110.36, "city"),
    "surabaya": (-7.25, 112.75, "city"),
    "bandung": (-6.92, 107.61, "city"),
    "cebu": (10.31, 123.89, "city"),
    "davao": (7.19, 125.45, "city"),
    "pattaya": (12.92, 100.88, "city"),
    "chiang mai": (18.79, 98.98, "city"),
    "phuket": (7.88, 98.39, "city"),
    "kota kinabalu": (5.98, 116.07, "city"),
    "penang": (5.41, 100.33, "city"),
    "jeju": (33.50, 126.53, "city"),
    "busan": (35.18, 129.08, "city"),
    "incheon": (37.46, 126.71, "city"),
    "foshan": (23.02, 113.12, "city"),
    "chengdu": (30.57, 104.07, "city"),
    "nanjing": (32.06, 118.80, "city"),
    "shaanxi": (34.26, 108.94, "admin"),
    "hebei": (38.04, 114.51, "admin"),
    "sichuan": (30.57, 104.07, "admin"),
    "guangdong": (23.13, 113.26, "admin"),
    "zhejiang": (30.27, 120.15, "admin"),
    "jiangsu": (32.06, 118.80, "admin"),
    "shandong": (36.67, 117.00, "admin"),
    "hubei": (30.59, 114.31, "admin"),
    "hunan": (28.23, 112.94, "admin"),
    "henan": (34.76, 113.65, "admin"),
    "fujian": (26.07, 119.30, "admin"),
    "anhui": (31.86, 117.28, "admin"),
    "yunnan": (25.04, 102.71, "admin"),
    "guizhou": (26.82, 106.71, "admin"),
    "liaoning": (41.80, 123.43, "admin"),
    "jilin": (43.88, 125.32, "admin"),
    "heilongjiang": (45.74, 126.66, "admin"),
    "gansu": (36.06, 103.83, "admin"),
    "bergen": (60.39, 5.32, "city"),
    "tromsø": (69.65, 18.96, "city"),
    "nice": (43.71, 7.26, "city"),
    "montpellier": (43.61, 3.88, "city"),
    "strasbourg": (48.57, 7.75, "city"),
    "porto": (41.15, -8.61, "city"),
    "seville": (37.39, -5.98, "city"),
    "valencia": (39.47, -0.38, "city"),
    "malaga": (36.72, -4.42, "city"),
    "granada": (37.18, -3.60, "city"),
    "bilbao": (43.26, -2.93, "city"),
    "san sebastian": (43.32, -1.98, "city"),
    "ghent": (51.05, 3.72, "city"),
    "antwerp": (51.22, 4.40, "city"),
    "delft": (52.01, 4.36, "city"),
    "the hague": (52.08, 4.30, "city"),
    "utrecht": (52.09, 5.12, "city"),
    "groningen": (53.22, 6.57, "city"),
    # --- World capitals & major cities: Europe ---
    "amsterdam": (52.37, 4.90, "capital"), "athens": (37.98, 23.73, "capital"),
    "barcelona": (41.39, 2.17, "city"), "belgrade": (44.79, 20.47, "capital"),
    "berlin": (52.52, 13.41, "capital"), "bern": (46.95, 7.45, "capital"),
    "bratislava": (48.15, 17.11, "capital"), "brussels": (50.85, 4.35, "capital"),
    "bucharest": (44.43, 26.10, "capital"), "budapest": (47.50, 19.04, "capital"),
    "chisinau": (47.01, 28.86, "capital"), "cologne": (50.94, 6.96, "city"),
    "copenhagen": (55.68, 12.57, "capital"), "dublin": (53.35, -6.26, "capital"),
    "edinburgh": (55.95, -3.19, "capital"), "florence": (43.77, 11.25, "city"),
    "frankfurt": (50.11, 8.68, "city"), "geneva": (46.20, 6.14, "city"),
    "glasgow": (55.86, -4.25, "city"), "hamburg": (53.55, 10.00, "city"),
    "helsinki": (60.17, 24.94, "capital"), "istanbul": (41.01, 28.98, "city"),
    "kiev": (50.45, 30.52, "capital"), "kyiv": (50.45, 30.52, "capital"),
    "lansing": (42.73, -84.56, "city"),  # also US
    "leipzig": (51.34, 12.37, "city"), "lisbon": (38.72, -9.14, "capital"),
    "london": (51.51, -0.13, "capital"), "lyon": (45.76, 4.84, "city"),
    "madrid": (40.42, -3.70, "capital"), "manchester": (53.48, -2.24, "city"),
    "marseille": (43.30, 5.37, "city"), "milan": (45.46, 9.19, "city"),
    "minsk": (53.90, 27.57, "capital"), "moscow": (55.76, 37.62, "capital"),
    "munich": (48.14, 11.58, "city"), "naples": (40.85, 14.27, "city"),
    "nicosia": (35.17, 33.37, "capital"), "oslo": (59.91, 10.75, "capital"),
    "paris": (48.86, 2.35, "capital"), "podgorica": (42.44, 19.26, "capital"),
    "prague": (50.08, 14.44, "capital"), "reykjavik": (64.13, -21.90, "capital"),
    "riga": (56.95, 24.11, "capital"), "rome": (41.90, 12.50, "capital"),
    "rotterdam": (51.92, 4.48, "city"), "saint petersburg": (59.93, 30.32, "city"),
    "sofia": (42.70, 23.32, "capital"), "split": (43.51, 16.44, "city"),
    "stockholm": (59.33, 18.07, "capital"), "tbilisi": (41.72, 44.78, "capital"),
    "vienna": (48.21, 16.37, "capital"), "vilnius": (54.69, 25.28, "capital"),
    "warsaw": (52.23, 21.01, "capital"), "zagreb": (45.81, 15.98, "capital"),
    "zurich": (47.37, 8.54, "city"),
    # --- World capitals & major cities: Americas ---
    "belo horizonte": (-19.92, -43.94, "city"), "bogota": (4.71, -74.07, "capital"),
    "buenos aires": (-34.60, -58.38, "capital"),
    "cali": (3.44, -76.52, "city"), "campinas": (-22.90, -47.06, "city"),
    "caracas": (10.48, -66.90, "capital"), "chicago": (41.88, -87.63, "city"),
    "cleveland": (41.50, -81.69, "city"),
    "curitiba": (-25.43, -49.27, "city"),
    "dallas": (32.78, -96.80, "city"), "denver": (39.74, -104.99, "city"),
    "fortaleza": (-3.72, -38.53, "city"),
    "guadalajara": (20.67, -103.35, "city"),
    "guatemala city": (14.63, -90.51, "capital"),
    "guayaquil": (-2.17, -79.92, "city"), "houston": (29.76, -95.37, "city"),
    "havana": (23.11, -82.37, "capital"), "kingston": (17.97, -76.79, "capital"),
    "la paz": (-16.50, -68.12, "capital"),
    "lima": (-12.05, -77.04, "capital"), "los angeles": (34.05, -118.24, "city"),
    "managua": (12.11, -86.24, "capital"), "medellin": (6.24, -75.57, "city"),
    "mexico city": (19.43, -99.13, "capital"),
    "miami": (25.76, -80.19, "city"),
    "montevideo": (-34.88, -56.18, "capital"),
    "montreal": (45.50, -73.57, "city"),
    "new york": (40.71, -74.01, "city"),
    "panama city": (8.98, -79.52, "capital"),
    "paramaribo": (5.85, -55.20, "capital"),
    "port au prince": (18.54, -72.34, "capital"),
    "port of spain": (10.65, -61.50, "capital"),
    "quito": (-0.18, -78.47, "capital"),
    "rio de janeiro": (-22.91, -43.17, "city"),
    "salvador": (-12.97, -38.51, "city"),
    "san jose": (9.93, -84.09, "capital"),
    "san salvador": (13.69, -89.19, "capital"),
    "santiago": (-33.45, -70.67, "capital"),
    "sao paulo": (-23.55, -46.63, "city"),
    "seattle": (47.61, -122.33, "city"),
    "st louis": (38.63, -90.20, "city"),
    "toronto": (43.65, -79.38, "city"),
    "tucson": (32.22, -110.93, "city"),
    "vancouver": (49.28, -123.12, "city"),
    "washington dc": (38.91, -77.04, "capital"),
    # --- World capitals & major cities: Oceania ---
    "auckland": (-36.85, 174.76, "city"),
    "canberra": (-35.28, 149.13, "capital"),
    "christchurch": (-43.53, 172.64, "city"),
    "melbourne": (-37.81, 144.96, "city"),
    "perth": (-31.95, 115.86, "city"),
    "sydney": (-33.87, 151.21, "city"),
    "wellington": (-41.29, 174.78, "capital"),
    # --- Additional major cities worldwide (population > 1M or tourist) ---
    "baltimore": (39.29, -76.61, "city"),
    "boston": (42.36, -71.06, "city"),
    "charlotte": (35.23, -80.84, "city"),
    "detroit": (42.33, -83.05, "city"),
    "indianapolis": (39.77, -86.16, "city"),
    "milwaukee": (43.04, -87.91, "city"),
    "minneapolis": (44.98, -93.27, "city"),
    "nashville": (36.16, -86.78, "city"),
    "new orleans": (29.95, -90.07, "city"),
    "philadelphia": (39.95, -75.17, "city"),
    "phoenix": (33.45, -112.07, "city"),
    "portland": (45.52, -122.68, "city"),
    "san antonio": (29.42, -98.49, "city"),
    "san diego": (32.72, -117.16, "city"),
    "san francisco": (37.77, -122.42, "city"),
    "atlanta": (33.75, -84.39, "city"),
    "dubai": (25.20, 55.27, "city"),
    "bangkok": (13.76, 100.50, "city"),
    "beijing": (39.90, 116.40, "capital"),
    "birmingham": (52.49, -1.89, "city"),
    "bordeaux": (44.84, -0.58, "city"),
    "bruges": (51.21, 3.22, "city"),
    "cannes": (43.55, 7.02, "city"),
    "dresden": (51.05, 13.74, "city"),
    "dusseldorf": (51.23, 6.78, "city"),
    "erfurt": (50.98, 11.03, "city"),
    "fes": (34.03, -5.00, "city"),
    "florence": (43.77, 11.25, "city"),
    "goa": (15.40, 73.83, "city"),
    "havana": (23.11, -82.37, "capital"),
    "innsbruck": (47.26, 11.39, "city"),
    "istanbul": (41.01, 28.98, "city"),
    "kamakura": (35.32, 139.55, "city"),
    "koya": (34.21, 135.60, "city"),
    "kyoto": (35.01, 135.77, "city"),
    "lhasa": (29.65, 91.14, "city"),
    "lyon": (45.76, 4.84, "city"),
    "marrakech": (31.63, -8.01, "city"),
    "nara": (34.69, 135.80, "city"),
    "palermo": (38.12, 13.36, "city"),
    "pisa": (43.72, 10.40, "city"),
    "portland": (45.52, -122.68, "city"),
    "salzburg": (47.80, 13.05, "city"),
    "siena": (43.32, 11.33, "city"),
    "singapore": (1.35, 103.82, "capital"),
    "venice": (45.44, 12.32, "city"),
    "vienna": (48.21, 16.37, "capital"),
    "york": (53.96, -1.08, "city"),
    # --- Countries (as fallback hints) ---
    "afghanistan": (33.94, 67.71, "country"),
    "albania": (41.15, 20.17, "country"),
    "algeria": (28.03, 1.66, "country"),
    "angola": (-11.20, 17.87, "country"),
    "argentina": (-38.42, -63.62, "country"),
    "armenia": (40.07, 45.04, "country"),
    "australia": (-25.27, 133.78, "country"),
    "austria": (47.52, 14.55, "country"),
    "azerbaijan": (40.14, 47.58, "country"),
    "bangladesh": (23.68, 90.36, "country"),
    "belarus": (53.71, 27.95, "country"),
    "belgium": (50.50, 4.47, "country"),
    "bolivia": (-16.29, -63.59, "country"),
    "botswana": (-22.33, 24.68, "country"),
    "brazil": (-14.24, -51.93, "country"),
    "brunei": (4.54, 114.73, "country"),
    "bulgaria": (42.73, 25.49, "country"),
    "cambodia": (12.57, 104.99, "country"),
    "cameroon": (7.37, 12.35, "country"),
    "canada": (56.13, -106.35, "country"),
    "chad": (15.45, 18.73, "country"),
    "chile": (-35.68, -71.54, "country"),
    "china": (35.86, 104.20, "country"),
    "colombia": (4.57, -74.30, "country"),
    "congo": (-0.23, 15.90, "country"),
    "costa rica": (9.75, -83.75, "country"),
    "croatia": (45.10, 15.20, "country"),
    "cuba": (21.52, -77.78, "country"),
    "cyprus": (35.13, 33.43, "country"),
    "czech republic": (49.82, 15.47, "country"),
    "czechia": (49.82, 15.47, "country"),
    "denmark": (56.26, 9.50, "country"),
    "dominican republic": (18.74, -70.16, "country"),
    "ecuador": (-1.83, -78.18, "country"),
    "egypt": (26.82, 30.80, "country"),
    "ethiopia": (9.15, 40.49, "country"),
    "finland": (61.92, 25.75, "country"),
    "france": (46.23, 2.21, "country"),
    "georgia": (42.32, 43.36, "country"),
    "germany": (51.17, 10.45, "country"),
    "ghana": (7.95, -1.02, "country"),
    "greece": (39.07, 21.82, "country"),
    "guatemala": (15.78, -90.23, "country"),
    "guinea": (9.95, -11.75, "country"),
    "guyana": (4.86, -58.93, "country"),
    "haiti": (18.97, -72.29, "country"),
    "honduras": (15.20, -86.24, "country"),
    "hungary": (47.16, 19.50, "country"),
    "iceland": (64.96, -19.02, "country"),
    "india": (20.59, 78.96, "country"),
    "indonesia": (-0.79, 113.92, "country"),
    "iran": (32.43, 53.69, "country"),
    "iraq": (33.22, 43.68, "country"),
    "ireland": (53.14, -7.69, "country"),
    "israel": (31.05, 34.85, "country"),
    "italy": (41.87, 12.57, "country"),
    "ivory coast": (7.54, -5.55, "country"),
    "jamaica": (18.11, -77.30, "country"),
    "japan": (36.20, 138.25, "country"),
    "jordan": (30.59, 36.24, "country"),
    "kazakhstan": (48.02, 66.92, "country"),
    "kenya": (-0.02, 37.91, "country"),
    "kuwait": (29.31, 47.48, "country"),
    "kyrgyzstan": (41.20, 74.77, "country"),
    "laos": (19.86, 102.50, "country"),
    "latvia": (56.88, 24.60, "country"),
    "lebanon": (33.85, 35.86, "country"),
    "libya": (26.34, 17.23, "country"),
    "lithuania": (55.17, 23.88, "country"),
    "madagascar": (-18.77, 46.87, "country"),
    "malawi": (-13.25, 34.30, "country"),
    "malaysia": (4.21, 101.98, "country"),
    "mali": (17.57, -4.00, "country"),
    "mexico": (23.63, -102.55, "country"),
    "mongolia": (46.86, 103.85, "country"),
    "morocco": (31.79, -7.09, "country"),
    "mozambique": (-18.67, 35.53, "country"),
    "myanmar": (21.91, 95.96, "country"),
    "namibia": (-22.96, 18.49, "country"),
    "nepal": (28.39, 84.12, "country"),
    "netherlands": (52.13, 5.29, "country"),
    "new zealand": (-40.90, 174.89, "country"),
    "nicaragua": (12.87, -85.21, "country"),
    "niger": (17.61, 8.08, "country"),
    "nigeria": (9.08, 8.68, "country"),
    "north korea": (40.34, 127.51, "country"),
    "norway": (60.47, 8.47, "country"),
    "oman": (21.47, 55.98, "country"),
    "pakistan": (30.38, 69.35, "country"),
    "palestine": (31.95, 35.23, "country"),
    "panama": (8.54, -80.78, "country"),
    "papua new guinea": (-6.31, 143.95, "country"),
    "paraguay": (-23.44, -58.44, "country"),
    "peru": (-9.19, -75.02, "country"),
    "philippines": (12.88, 121.77, "country"),
    "poland": (51.92, 19.15, "country"),
    "portugal": (39.40, -8.22, "country"),
    "qatar": (25.35, 51.18, "country"),
    "romania": (45.94, 24.97, "country"),
    "russia": (61.52, 105.32, "country"),
    "rwanda": (-1.94, 29.87, "country"),
    "saudi arabia": (23.89, 45.08, "country"),
    "senegal": (14.50, -14.45, "country"),
    "serbia": (44.02, 21.01, "country"),
    "singapore": (1.35, 103.82, "country"),
    "slovakia": (48.67, 19.70, "country"),
    "slovenia": (46.15, 14.99, "country"),
    "somalia": (5.15, 46.20, "country"),
    "south africa": (-30.56, 22.94, "country"),
    "south korea": (35.91, 127.77, "country"),
    "south sudan": (6.88, 31.31, "country"),
    "spain": (40.46, -3.75, "country"),
    "sri lanka": (7.87, 80.77, "country"),
    "sudan": (12.86, 30.22, "country"),
    "sweden": (60.13, 18.64, "country"),
    "switzerland": (46.82, 8.23, "country"),
    "syria": (34.80, 38.99, "country"),
    "taiwan": (23.70, 120.96, "country"),
    "tajikistan": (38.86, 71.28, "country"),
    "tanzania": (-6.37, 34.89, "country"),
    "thailand": (15.87, 100.99, "country"),
    "tunisia": (33.89, 9.54, "country"),
    "turkey": (38.96, 35.24, "country"),
    "turkmenistan": (38.97, 59.56, "country"),
    "uganda": (1.37, 32.29, "country"),
    "ukraine": (48.38, 31.17, "country"),
    "united arab emirates": (23.42, 53.85, "country"),
    "united kingdom": (55.38, -3.44, "country"),
    "united states": (37.09, -95.71, "country"),
    "uruguay": (-32.52, -55.77, "country"),
    "uzbekistan": (41.38, 64.59, "country"),
    "venezuela": (6.42, -66.59, "country"),
    "vietnam": (14.06, 108.28, "country"),
    "yemen": (15.55, 48.52, "country"),
    "zambia": (-13.13, 27.85, "country"),
    "zimbabwe": (-19.02, 29.15, "country"),
    # --- US states ---
    "alabama": (32.81, -86.79, "admin"),
    "alaska": (64.24, -152.50, "admin"),
    "arizona": (34.05, -111.09, "admin"),
    "arkansas": (35.20, -91.83, "admin"),
    "california": (36.78, -119.42, "admin"),
    "colorado": (39.55, -105.78, "admin"),
    "connecticut": (41.60, -72.70, "admin"),
    "delaware": (39.00, -75.52, "admin"),
    "florida": (27.66, -81.52, "admin"),
    "georgia": (32.17, -82.90, "admin"),
    "hawaii": (19.90, -155.58, "admin"),
    "idaho": (44.07, -114.74, "admin"),
    "illinois": (40.63, -89.40, "admin"),
    "indiana": (40.27, -86.13, "admin"),
    "iowa": (42.01, -93.62, "admin"),
    "kansas": (39.01, -98.48, "admin"),
    "kentucky": (37.84, -85.76, "admin"),
    "louisiana": (30.98, -91.96, "admin"),
    "maine": (45.25, -69.45, "admin"),
    "maryland": (39.05, -76.64, "admin"),
    "massachusetts": (42.41, -71.38, "admin"),
    "michigan": (44.31, -85.60, "admin"),
    "minnesota": (46.73, -94.69, "admin"),
    "mississippi": (32.35, -89.40, "admin"),
    "missouri": (38.46, -92.57, "admin"),
    "montana": (47.05, -110.36, "admin"),
    "nebraska": (41.50, -100.00, "admin"),
    "nevada": (38.80, -116.42, "admin"),
    "new hampshire": (43.19, -71.57, "admin"),
    "new jersey": (40.06, -74.41, "admin"),
    "new mexico": (34.52, -105.87, "admin"),
    "new york": (42.17, -74.95, "admin"),
    "north carolina": (35.76, -79.02, "admin"),
    "north dakota": (47.55, -101.00, "admin"),
    "ohio": (40.42, -82.91, "admin"),
    "oklahoma": (35.57, -97.51, "admin"),
    "oregon": (43.80, -120.55, "admin"),
    "pennsylvania": (41.20, -77.19, "admin"),
    "rhode island": (41.58, -71.48, "admin"),
    "south carolina": (34.00, -81.03, "admin"),
    "south dakota": (44.50, -100.23, "admin"),
    "tennessee": (35.52, -86.58, "admin"),
    "texas": (31.97, -99.90, "admin"),
    "utah": (39.32, -111.09, "admin"),
    "vermont": (44.56, -72.58, "admin"),
    "virginia": (37.77, -78.17, "admin"),
    "washington": (47.75, -120.74, "admin"),
    "west virginia": (38.60, -80.63, "admin"),
    "wisconsin": (44.57, -89.78, "admin"),
    "wyoming": (43.08, -107.29, "admin"),
    # --- Canadian provinces ---
    "alberta": (53.93, -116.58, "admin"),
    "british columbia": (53.73, -127.65, "admin"),
    "manitoba": (53.76, -98.81, "admin"),
    "new brunswick": (46.57, -66.45, "admin"),
    "newfoundland": (47.56, -52.71, "admin"),
    "nova scotia": (44.68, -63.57, "admin"),
    "ontario": (51.25, -85.32, "admin"),
    "quebec": (52.94, -73.58, "admin"),
    "saskatchewan": (52.94, -106.45, "admin"),
    # --- Australian states ---
    "queensland": (-20.92, 142.70, "admin"),
    "victoria": (-37.81, 144.96, "admin"),
    "south australia": (-30.00, 136.21, "admin"),
    "western australia": (-25.27, 121.77, "admin"),
    "tasmania": (-41.45, 145.37, "admin"),
    "new south wales": (-32.17, 147.22, "admin"),
    # --- Regions & landmarks (for geo-guessing context) ---
    "amazon": (-3.47, -62.22, "region"),
    "alps": (46.50, 10.00, "region"),
    "amazonas": (-3.47, -62.22, "region"),
    "andes": (-13.20, -72.00, "region"),
    "arctic": (82.50, 20.00, "region"),
    "balkans": (42.00, 20.00, "region"),
    "caribbean": (15.00, -70.00, "region"),
    "caucasus": (42.00, 44.00, "region"),
    "himalayas": (28.00, 84.00, "region"),
    "middle east": (29.00, 47.00, "region"),
    "pacific": (0.00, -160.00, "region"),
    "sahara": (23.00, 10.00, "region"),
    "silk road": (39.00, 60.00, "region"),
}

# Common words that look like place names but aren't — false-positive blacklist
_PLACE_BLACKLIST: set[str] = {
    "the", "photo", "image", "test", "hello", "welcome", "hotel", "restaurant",
    "street", "road", "avenue", "map", "google", "instagram", "facebook",
    "twitter", "youtube", "tiktok", "snapchat", "whatsapp", "wifi",
    "password", "open", "close", "entrance", "exit", "stop", "go",
    "north", "south", "east", "west", "left", "right", "up", "down",
    "yes", "no", "please", "thank", "sorry", "help", "info", "data",
    "free", "new", "old", "big", "small", "hot", "cold", "fast", "slow",
    "good", "bad", "best", "top", "low", "high", "live", "love",
    "time", "day", "night", "morning", "evening", "today", "tomorrow",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "red", "blue", "green", "yellow", "black", "white", "brown", "orange",
    "park", "garden", "square", "center", "centre", "place", "house", "home",
    "car", "bus", "train", "taxi", "airport", "station", "port", "harbor",
    "museum", "church", "temple", "mosque", "palace", "tower", "bridge",
    "river", "lake", "mountain", "island", "beach", "sea", "ocean",
    "food", "menu", "coffee", "tea", "water", "beer", "wine", "bar",
    "shop", "store", "market", "supermarket", "mall", "bank", "hospital",
    "school", "university", "college", "library", "office", "factory",
    "club", "disco", "party", "dance", "music", "art", "design",
    "sun", "moon", "star", "rain", "snow", "wind", "fire",
    "baby", "boy", "girl", "man", "woman", "people", "person",
    "king", "queen", "prince", "president", "minister", "mayor",
    "phone", "computer", "camera", "screen", "digital", "electronic",
    "security", "alarm", "cctv", "notice", "warning",
    "area", "zone", "district", "section", "block", "floor", "level",
    "corner", "edge", "side", "front", "back",
    "special", "official", "private", "public", "national", "international",
    "care", "pet", "animal", "tree", "flower", "plant",
    "fresh", "natural", "classic", "modern", "style", "beauty",
    "speed", "power", "max", "mini", "ultra", "super", "mega",
    "view", "scene", "shot", "snap", "pic", "selfie",
    "original", "version", "edition", "series", "season", "episode",
    "like", "share", "follow", "subscribe", "post", "story",
    "hotspot", "internet", "online", "web", "site", "page",
    "safe", "alert", "danger", "caution", "keep", "watch", "look",
    "buy", "sell", "rent", "sale", "price", "cost", "pay", "cash",
    "bill", "ticket", "pass", "card", "gift", "prize", "bonus",
    "happy", "birthday", "congratulations", "celebrate", "holiday",
    "vacation", "travel", "trip", "tour", "visit", "explore",
    "adventure", "discover", "journey", "destination",
    "landscape", "panorama", "sunset", "sunrise", "horizon",
    "cloud", "sky", "earth", "world", "space", "universe",
    "a", "an", "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "into", "about", "above", "below", "under", "over",
    "and", "or", "but", "if", "then", "else", "when", "where",
    "how", "what", "which", "who", "why", "this", "that", "it",
    "you", "we", "they", "he", "she", "my", "your", "our", "their",
    "do", "is", "are", "was", "were", "be", "been", "has", "have",
    "can", "could", "will", "would", "shall", "should", "may", "might",
    "not", "no", "nor", "so", "too", "very", "just", "only", "also",
    "now", "here", "there", "all", "some", "any", "each", "every",
    "more", "most", "less", "least", "first", "last", "next", "same",
    "other", "another", "such", "own", "different", "important",
    "main", "key", "basic", "standard", "normal", "general", "specific",
    "real", "true", "false", "wrong", "sure", "ok", "okay",
    "note", "tip", "hint", "example", "sample", "demo", "draft",
    "copy", "final", "summary", "detail",
    "video", "audio", "text", "file", "folder", "document",
    "logo", "icon", "banner", "header", "footer", "button",
    "link", "url", "http", "https", "www", "com", "org", "net",
    "app", "software", "hardware", "system", "program", "code",
    "feature", "function", "method", "class", "object", "variable",
    "server", "client", "database", "network", "host",
    "root", "admin", "user", "guest", "member", "staff", "team",
    "company", "business", "enterprise", "corporation", "limited",
    "copyright", "trademark", "patent", "license", "terms", "privacy",
    "policy", "rule", "regulation", "law", "act", "statute",
    "order", "command", "instruction", "manual", "guide", "tutorial",
    "course", "lesson", "chapter", "volume",
    "report", "review", "analysis", "research", "study", "survey",
    "news", "update", "announcement", "message",
    "event", "meeting", "conference", "workshop", "seminar",
    "project", "plan", "strategy", "goal", "target", "objective",
    "result", "output", "input", "process", "step", "stage",
    "problem", "solution", "answer", "question", "issue", "topic",
    "subject", "theme", "category", "type", "kind", "sort",
    "size", "length", "width", "height", "depth", "weight",
    "number", "amount", "total", "average", "maximum", "minimum",
    "percent", "rate", "ratio", "index", "scale", "range",
    "degree", "grade", "rank", "score", "points",
    "hour", "minute", "second", "week", "month", "year", "decade",
    "age", "era", "period", "phase", "cycle", "round",
    "hand", "head", "eye", "face", "body", "foot", "leg", "arm",
    "heart", "mind", "soul", "spirit", "life", "death", "birth",
    "peace", "war", "fight", "battle", "attack", "defense", "guard",
    "flag", "anthem", "border", "boundary", "line", "mark", "sign",
    "signal", "key", "lock", "gate", "door", "window",
    "wall", "roof", "ceiling", "room", "hall", "corridor",
    "castle", "fort", "citadel", "monument", "statue",
    "fountain", "pool", "pond", "stream", "creek", "canal",
    "dock", "wharf", "pier", "jetty", "quay",
    "hill", "valley", "plain", "plateau", "desert", "forest", "jungle",
    "cave", "canyon", "cliff", "glacier", "volcano", "geyser",
    "reef", "coast", "shore", "bay", "gulf", "strait", "channel",
    "cape", "point", "headland", "peninsula", "isthmus",
    "field", "farm", "ranch", "village", "town",
    "country", "state", "province", "region", "territory", "zone",
    "parish", "county", "municipality", "borough", "township",
    "boulevard", "drive", "lane", "alley", "court",
    "plaza", "terrace", "crescent", "loop",
    "tunnel", "overpass", "underpass", "crossing",
    "parking", "garage", "lot", "yard", "driveway",
    "motel", "inn", "hostel", "resort", "lodge",
    "bistro", "diner", "pub", "tavern", "saloon",
    "bakery", "butcher", "grocer", "pharmacy", "clinic", "ward",
    "theater", "theatre", "cinema", "arena", "stadium", "gallery",
    "booth", "kiosk", "stand", "cart", "truck", "van",
    "tram", "metro", "subway", "ferry", "cable",
    "lift", "escalator", "elevator", "ramp", "stair",
    "wire", "pipe", "tube", "duct",
    "tank", "silo", "bunker", "vault",
    "bench", "chair", "table", "desk", "shelf", "rack",
    "lamp", "bulb", "led", "neon", "beam",
    "bass", "treble", "volume", "frequency",
    "velocity", "acceleration", "momentum", "force",
    "energy", "heat", "temperature", "climate",
    "weather", "forecast", "storm", "hurricane", "typhoon", "cyclone",
    "hail", "sleet", "fog", "mist", "smog",
    "thunder", "lightning", "rainbow", "aurora", "eclipse",
    "spring", "summer", "autumn", "winter", "equinox",
    "tropic", "antarctic", "equator", "meridian", "latitude",
    "longitude", "altitude", "elevation", "distance",
    "direction", "bearing", "heading", "course", "route", "path",
    "track", "trail", "highway", "freeway", "expressway", "motorway", "interstate",
    "junction", "intersection", "crossroads", "roundabout", "terminus",
    "origin", "destination", "departure", "arrival", "terminal",
    "platform", "service", "express", "local", "direct", "through", "connecting",
    "transfer", "exchange", "connection", "waterway", "passage",
    "aisle", "doorway", "gateway", "portal", "threshold", "barrier", "obstacle",
    "fence", "hedge", "screen", "partition", "divider",
    "curtain", "blind", "shade", "awning", "canopy",
    "dome", "arch", "column", "pillar", "post", "pole",
    "rafter", "truss", "joist", "plank", "board",
    "brick", "stone", "concrete", "cement", "mortar", "plaster",
    "glass", "mirror", "lens", "prism", "crystal",
    "iron", "steel", "copper", "brass", "bronze",
    "gold", "silver", "platinum", "aluminum", "titanium",
    "plastic", "rubber", "latex", "foam", "sponge", "fiber",
    "timber", "lumber", "log",
    "paper", "card", "sheet", "page", "leaf",
    "cloth", "fabric", "textile", "cotton", "silk", "wool",
    "leather", "fur", "feather", "scale", "shell", "bone",
    "rock", "pebble", "sand", "gravel", "dust", "ash",
    "soil", "mud", "clay", "dirt", "ground",
    "seed", "root", "stem", "fruit",
    "branch", "twig", "bark", "trunk", "crown",
    "grass", "moss", "fern", "vine", "shrub", "bush",
    "bird", "fish", "insect", "worm", "snake",
    "mammal", "reptile", "amphibian", "crustacean", "mollusk",
    "predator", "prey", "herbivore", "carnivore", "omnivore",
    "herd", "flock", "pack", "swarm", "school", "colony",
    "nest", "den", "burrow", "hive", "web", "cocoon",
    "breed", "species", "genus", "family", "phylum", "domain", "organism", "cell",
    "atom", "molecule", "particle", "quark", "photon", "electron",
    "proton", "neutron", "nucleus", "orbital",
    "mass", "charge", "spin", "field",
    "wave", "pulse", "vibration", "oscillation", "resonance",
    "spectrum", "wavelength", "amplitude", "phase",
    "inertia", "friction",
    "gravity", "magnetism", "electricity", "current", "voltage",
    "resistance", "capacitance", "inductance", "impedance",
    "noise", "bit", "byte", "word",
    "memory", "storage", "cache", "buffer", "stack", "queue",
    "graph", "mesh", "grid", "lattice",
    "algorithm", "procedure", "routine", "module",
    "constant", "parameter", "argument", "value",
    "struct", "record", "tuple",
    "array", "list", "vector", "matrix", "tensor", "set",
    "dictionary", "hash", "index", "pointer",
    "address", "reference", "edge", "node",
    "parent", "child", "sibling",
    "count", "null", "void", "none", "undefined",
    "while", "switch",
    "case", "break", "continue", "return", "yield", "throw",
    "try", "catch", "finally", "raise", "assert",
    "import", "export", "package", "library",
    "interface", "trait", "mixin", "abstract",
    "public", "private", "protected", "internal", "static",
    "final", "const", "var", "let", "mutable", "immutable",
    "async", "await", "promise", "future", "callback",
    "listener", "handler", "delegate",
    "slot", "connector",
    "factory",
    "builder", "adapter", "wrapper", "proxy", "facade",
    "decorator", "modifier", "extension", "plugin", "addon",
    "middleware", "interceptor", "filter", "transform", "mapper",
    "reducer", "combiner", "selector", "action", "dispatch",
    "store", "model", "view", "controller",
    "presenter", "binder", "injector", "provider", "resolver",
    "validator", "serializer", "deserializer", "encoder", "decoder",
    "parser", "lexer", "tokenizer", "scanner", "compiler",
    "interpreter", "runtime", "vm", "jit", "gc",
    "cali",
}

# Pre-compiled regex: lowercase word-boundary search for each place name key.
# Built once at module load to avoid re-compiling per detect() call.
_PLACE_NAME_KEYS = sorted(
    (k for k in _PLACE_NAMES if _PLACE_NAMES[k][2] not in ("_skip", "_alias_ho chi minh city")),
    key=len, reverse=True,
)


def _find_place_matches(text: str) -> list[tuple[float, float, float]]:
    """Return ``(lat, lon, confidence)`` for every city/capital/region found in *text*.

    Matching is case-insensitive.  Whole-word matches score higher than
    substring matches.  Blacklisted words are skipped.
    Country and admin names are handled separately by ``_find_admin_matches``.
    """
    lower = text.lower()
    seen: set[str] = set()
    results: list[tuple[float, float, float]] = []

    for name in _PLACE_NAME_KEYS:
        kind = _PLACE_NAMES[name][2]
        if kind in ("country", "admin"):
            continue
        if name in seen:
            continue
        if name in _PLACE_BLACKLIST:
            continue
        # Whole-word match via regex word boundaries
        pattern = r"(?<!\w)" + re.escape(name) + r"(?!\w)"
        if re.search(pattern, lower):
            lat, lon, _kind = _PLACE_NAMES[name]
            confidence = 0.85  # whole-word exact match
            results.append((lat, lon, confidence))
            seen.add(name)
            continue
        # Substring containment (partial match, e.g. "york" in "New York")
        if name in lower and len(name) >= 4:
            lat, lon, _kind = _PLACE_NAMES[name]
            confidence = 0.55  # partial / substring match
            results.append((lat, lon, confidence))
            seen.add(name)

    return results


def _find_admin_matches(text: str) -> list[tuple[float, float, float]]:
    """Match country and state/province names for additional geo hints."""
    lower = text.lower()
    seen: set[str] = set()
    results: list[tuple[float, float, float]] = []

    for name in _PLACE_NAME_KEYS:
        kind = _PLACE_NAMES[name][2]
        if kind not in ("country", "admin"):
            continue
        if name in seen:
            continue
        if name in _PLACE_BLACKLIST:
            continue
        pattern = r"(?<!\w)" + re.escape(name) + r"(?!\w)"
        if re.search(pattern, lower):
            lat, lon, _ = _PLACE_NAMES[name]
            confidence = 0.70 if kind == "country" else 0.65
            results.append((lat, lon, confidence))
            seen.add(name)

    return results


class OcrTextModule(BaseModule):
    """Extract text via OCR, detect script/language, map to country hints."""

    name = "ocr_text"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)

    def is_available(self) -> bool:
        try:
            import easyocr  # noqa: F401
            return True
        except ImportError:
            return False

    def prepare(self) -> None:
        from geofind.utils.models import get_cached_model, ensure_easyocr_langs

        langs = ensure_easyocr_langs()

        def _load():
            import easyocr
            return easyocr.Reader(langs, gpu=False)

        self._reader = get_cached_model("easyocr", _load)
        super().prepare()

    def detect(
        self,
        media_path: Path,
        *,
        frames: list[Any] | None = None,
        audio_path: Path | None = None,
    ) -> list[ModuleHit]:
        if not self._ready:
            return []

        from PIL import Image

        image = self._get_image(media_path, frames)
        if image is None:
            return []

        try:
            import numpy as np
            img_array = np.array(image)
            results = self._reader.readtext(img_array)
        except Exception as e:
            self._log(f"OCR failed: {e}", logging.WARNING)
            return []

        if not results:
            return []

        full_text = " ".join(r[1] for r in results if len(r) > 1)
        if not full_text.strip():
            return []

        self._log(f"OCR text: {full_text[:100]}")

        scripts = self._detect_scripts(full_text)
        country_votes: dict[str, float] = {}

        from geofind.utils.constants import SCRIPT_COUNTRY_HINTS

        for script, count in scripts.items():
            hint_key = _SCRIPT_TO_HINT_KEY.get(script, script)
            countries = SCRIPT_COUNTRY_HINTS.get(hint_key, [])
            weight = count / max(len(full_text), 1)
            for cc in countries:
                country_votes[cc] = country_votes.get(cc, 0.0) + weight

        hits: list[ModuleHit] = []
        if country_votes:
            total = max(sum(country_votes.values()), 1e-9)
            for cc, score in sorted(country_votes.items(), key=lambda x: -x[1]):
                if score / total < 0.01:
                    continue
                lat, lon = _COUNTRY_CENTROIDS.get(cc, (0.0, 0.0))
                # Script detection is COUNTRY-level, not city-level.
                # Use wide sigma so it covers the whole country, not just
                # the centroid point.
                hits.append(self._make_hit(
                    lat, lon, min(score / total, 1.0),
                    sigma_km=800.0,
                    country=cc,
                    ocr_text=full_text[:500],
                    scripts=list(scripts.keys()),
                    hint_level="country",
                ))
        else:
            # Latin extended script detection is too generic (half the world
            # uses Latin script) and the centroid (37°N 15°E) is actively
            # harmful — it pulls probability toward the Mediterranean for
            # images from US, UK, Germany, France, etc.  Only produce a
            # very weak hint when the text is clearly non-European Latin.
            latin_score = self._count_latin_extended(full_text) / max(len(full_text), 1)
            if latin_score > 0.1:
                latin_text = full_text.lower()
                # Check if text looks like a Latin-script country in
                # Africa/SE-Asia/Americas rather than Europe
                non_eu_hints = [
                    "bra", "mex", "arg", "col", "per", "ven", "chi",
                    "vie", "tha", "ind", "phi", "cam", "laos", "mya",
                    "nig", "gha", "ken", "tan", "eth", "zim", "nam",
                    "cub", "jam", "dom", "hon", "gua", "pan", "nic",
                    "bol", "par", "uru", "ecu", "guy", "sur",
                ]
                has_non_eu = any(h in latin_text for h in non_eu_hints)
                if has_non_eu:
                    hits.append(self._make_hit(
                        10.0, -2.0, 0.05,
                        scripts=["latin_extended_weak"],
                        ocr_text=full_text[:500],
                    ))

        # --- Place-name geocoding (city / admin / country names) ---
        place_matches = _find_place_matches(full_text)
        admin_matches = _find_admin_matches(full_text)

        for lat, lon, conf in place_matches:
            hits.append(self._make_hit(
                lat, lon, conf,
                match_type="city",
                ocr_text=full_text[:500],
            ))

        for lat, lon, conf in admin_matches:
            hits.append(self._make_hit(
                lat, lon, conf,
                sigma_km=800.0,  # Country/admin level — wide spread
                match_type="admin",
                ocr_text=full_text[:500],
                hint_level="country",
            ))

        return hits

    def _detect_scripts(self, text: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for ch in text:
            cp = ord(ch)
            for script_name, ranges in _SCRIPT_RANGES.items():
                for start, end in ranges:
                    if start <= cp <= end:
                        counts[script_name] = counts.get(script_name, 0) + 1
                        break
        return counts

    def _count_latin_extended(self, text: str) -> int:
        count = 0
        for ch in text:
            cp = ord(ch)
            for start, end in _LATIN_EXTENDED_RANGES:
                if start <= cp <= end:
                    count += 1
                    break
        return count

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
