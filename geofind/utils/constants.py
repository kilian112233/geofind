"""Constant data: country info, power frequencies, biome regions, landmarks."""

from __future__ import annotations

from dataclasses import dataclass, field


# ── Power Grid Frequencies ──────────────────────────────────────────────────

# Country → (frequency_hz, plug_type_hint)
POWER_GRID: dict[str, tuple[float, str]] = {
    # 50 Hz countries (sample)
    "GB": (50.0, "G"), "DE": (50.0, "C/F"), "FR": (50.0, "C/E"),
    "IN": (50.0, "C/D/M"), "AU": (50.0, "I"), "ZA": (50.0, "M"),
    "BR": (50.0, "N"), "EG": (50.0, "C"), "TH": (50.0, "A/B/C"),
    "TR": (50.0, "C/F"), "PK": (50.0, "C/D"), "NG": (50.0, "D/G"),
    "SA": (50.0, "G"), "AE": (50.0, "G"), "ID": (50.0, "C/F"),
    "VN": (50.0, "A/C"), "MY": (50.0, "G"), "PH": (50.0, "A/B/C"),
    "KR": (60.0, "C/F"),  # SK is 60Hz!
    # 60 Hz countries
    "US": (60.0, "A/B"), "CA": (60.0, "A/B"), "MX": (60.0, "A/B"),
    "JP": (60.0, "A/B"),  # East Japan is 60Hz
    "KR": (60.0, "C/F"),
    "TW": (60.0, "A/B"),
    "PH": (60.0, "A/B/C"),
    "SA": (60.0, "A/B/G"),
    "CO": (60.0, "A/B"), "AR": (60.0, "C/I"), "CL": (60.0, "C/L"),
}

# Region → (frequency_hz) for broader matching
REGION_FREQUENCY: dict[str, float] = {
    "europe": 50.0,
    "africa": 50.0,
    "asia_pacific": 50.0,
    "middle_east": 50.0,
    "north_america": 60.0,
    "central_america": 60.0,
    "south_america": 60.0,
    "east_asia": 60.0,
}

# Frequency → country list (reversed mapping)
FREQ_TO_COUNTRIES: dict[float, list[str]] = {}
for _cc, (_freq, _) in POWER_GRID.items():
    FREQ_TO_COUNTRIES.setdefault(_freq, []).append(_cc)


# ── Driving Side ────────────────────────────────────────────────────────────

# Left-hand traffic countries (ISO 3166-1 alpha-2)
LEFT_DRIVING: set[str] = {
    "AU", "NZ", "GB", "IE", "IN", "PK", "BD", "LK", "JP", "TH",
    "ID", "MY", "SG", "HK", "MO", "ZA", "KE", "TZ", "UG", "NG",
    "JM", "TT", "BS", "BB", "BZ", "JM", "TT", "FJ", "SB", "PG",
    "MY", "TH", "ID", "IN", "PK", "LK", "BD", "AU", "NZ", "GB",
    "IE", "JA", "JP",
}

RIGHT_DRIVING: set[str] = {
    "US", "CA", "MX", "BR", "AR", "CL", "CO", "PE", "VE",
    "DE", "FR", "IT", "ES", "PT", "NL", "BE", "AT", "CH",
    "PL", "CZ", "SK", "HU", "RO", "BG", "GR", "TR", "RU",
    "CN", "KR", "TW", "PH", "VN", "EG", "SA", "AE", "IL",
    "SE", "NO", "FI", "DK", "IS",
}


# ── License Plate Patterns ──────────────────────────────────────────────────

@dataclass
class PlatePattern:
    """Expected license plate format for a country."""
    country: str
    format_regex: str
    description: str
    plate_colors: list[str] = field(default_factory=lambda: ["white"])


PLATE_PATTERNS: dict[str, PlatePattern] = {
    "US": PlatePattern("US", r"[A-Z0-9]{1,8}", "Varies by state"),
    "GB": PlatePattern("GB", r"[A-Z]{2}\d{2}\s?[A-Z]{3}", "AB12 CDE"),
    "DE": PlatePattern("DE", r"[A-Z]{1,3}-[A-Z]{1,2}\s?\d{1,4}", "M-AB 1234"),
    "FR": PlatePattern("FR", r"\d{3}\s?[A-Z]{3}\s?\d{2}", "123 ABC 45"),
    "JP": PlatePattern("JP", r"\d{3,4}[あ-ん][\d-]{4}", "Regional kana"),
    "IN": PlatePattern("IN", r"[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}", "State code prefix"),
    "BR": PlatePattern("BR", r"[A-Z]{3}\d[A-Z0-9]\d{2}", "Mercosul format"),
    "AU": PlatePattern("AU", r"[A-Z0-9]{1,6}", "State varies"),
    "IT": PlatePattern("IT", r"[A-Z]{2}\d{3}[A-Z]{2}", "White plate"),
    "ES": PlatePattern("ES", r"\d{4}\s?[A-Z]{3}", "Four digits three letters"),
    "RU": PlatePattern("RU", r"[A-Z]\d{3}[A-Z]{2}\d{2,3}", "Region code suffix"),
    "KR": PlatePattern("KR", r"\d{2,3}[가-힣]\d{4}", "Korean format"),
    "CN": PlatePattern("CN", r"[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁][A-Z][A-HJ-NP-Z0-9]{4,5}[A-HJ-NP-Z0-9挂学警]", "Chinese province prefix"),
}


# ── Driving Side Detection Countries ────────────────────────────────────────

# Countries where YOLO vehicle detection + steering wheel position matters
DRIVING_SIDE_COUNTRIES: dict[str, str] = {
    "AU": "left", "NZ": "left", "GB": "left", "IE": "left",
    "IN": "left", "PK": "left", "TH": "left", "ID": "left",
    "JP": "left", "MY": "left", "SG": "left", "HK": "left",
    "ZA": "left", "KE": "left", "TZ": "left", "NG": "left",
    "US": "right", "CA": "right", "MX": "right", "BR": "right",
    "DE": "right", "FR": "right", "IT": "right", "ES": "right",
    "RU": "right", "CN": "right", "KR": "right", "SA": "right",
}


# ── Biome / Vegetation Regions ──────────────────────────────────────────────

# Rough biome classification boundaries
BIOME_ZONES: list[tuple[str, float, float, float, float]] = [
    # (name, min_lat, max_lat, min_lon, max_lon) - very rough
    ("tropical_rainforest", -10, 10, -80, 140),
    ("tropical_savanna", -20, 20, -60, 150),
    ("desert", 15, 35, -20, 80),     # Sahara/Arabian
    ("temperate_forest", 30, 55, -130, 140),
    ("boreal_forest", 50, 70, -170, 180),
    ("tundra", 65, 83, -180, 180),
    ("mediterranean", 30, 45, -10, 40),
    ("grassland", 30, 50, -110, 90),
    ("subtropical", 20, 35, -100, 130),
]

# CLIP-style biome prompts for zero-shot classification
BIOME_PROMPTS: dict[str, list[str]] = {
    "tropical_rainforest": [
        "a dense tropical rainforest with thick canopy",
        "lush green jungle with vines and humidity",
        "tropical broadleaf forest with high rainfall",
    ],
    "desert": [
        "an arid desert landscape with sand dunes",
        "dry rocky desert with sparse vegetation",
        "sandy desert with cacti or drought plants",
    ],
    "temperate_forest": [
        "a deciduous forest with seasonal color changes",
        "temperate woodland with oak or maple trees",
        "mixed forest with broadleaf and coniferous trees",
    ],
    "boreal_forest": [
        "a boreal taiga forest with coniferous trees",
        "spruce and pine forest in cold climate",
        "northern evergreen forest",
    ],
    "tundra": [
        "barren tundra landscape with permafrost",
        "treeless arctic tundra with moss and lichen",
        "frozenundra with sparse low vegetation",
    ],
    "mediterranean": [
        "Mediterranean landscape with olive trees",
        "dry sunny climate with shrubland",
        "coastal Mediterranean vegetation",
    ],
    "grassland": [
        "open grassland prairie or steppe",
        "vast grassy plains with few trees",
        "savanna or pampas grassland",
    ],
    "subtropical": [
        "subtropical vegetation with palm trees",
        "warm humid climate with lush greenery",
        "subtropical garden or parkland",
    ],
}


# ── Landmark Categories ─────────────────────────────────────────────────────

# Common landmark categories that map to specific locations
LANDMARK_CATEGORIES: dict[str, dict[str, float]] = {
    "eiffel_tower": {"lat": 48.8584, "lon": 2.2945, "confidence": 0.95},
    "statue_of_liberty": {"lat": 40.6892, "lon": -74.0445, "confidence": 0.95},
    "big_ben": {"lat": 51.5007, "lon": -0.1246, "confidence": 0.90},
    "taj_mahal": {"lat": 27.1751, "lon": 78.0421, "confidence": 0.95},
    "sydney_opera_house": {"lat": -33.8568, "lon": 151.2153, "confidence": 0.95},
    "colosseum": {"lat": 41.8902, "lon": 12.4922, "confidence": 0.95},
    "golden_gate_bridge": {"lat": 37.8199, "lon": -122.4783, "confidence": 0.90},
    "mount_fuji": {"lat": 35.3606, "lon": 138.7274, "confidence": 0.90},
    "christ_redeemer": {"lat": -22.9519, "lon": -43.2105, "confidence": 0.95},
    "pyramids_of_giza": {"lat": 29.9792, "lon": 31.1342, "confidence": 0.95},
    "great_wall": {"lat": 40.4319, "lon": 116.5704, "confidence": 0.90},
    "petra": {"lat": 30.3285, "lon": 35.4444, "confidence": 0.95},
    "machu_picchu": {"lat": -13.1631, "lon": -72.5450, "confidence": 0.95},
    "angkor_wat": {"lat": 13.4125, "lon": 103.8670, "confidence": 0.95},
    "borobudur": {"lat": -7.6079, "lon": 110.2038, "confidence": 0.90},
    "kremlin": {"lat": 55.7520, "lon": 37.6175, "confidence": 0.90},
    "brandenburg_gate": {"lat": 52.5163, "lon": 13.3777, "confidence": 0.90},
    "parliament_london": {"lat": 51.4995, "lon": -0.1248, "confidence": 0.90},
    "white_house": {"lat": 38.8977, "lon": -77.0365, "confidence": 0.90},
    "un_symposium": {"lat": 40.7489, "lon": -73.9680, "confidence": 0.85},
}


# ── Currency Identification ─────────────────────────────────────────────────

CURRENCY_PROMPTS: dict[str, list[str]] = {
    "USD": ["US dollar bill", "American currency", "United States money"],
    "EUR": ["Euro banknote", "European currency", "Euro bill"],
    "GBP": ["British pound note", "UK currency", "Sterling banknote"],
    "JPY": ["Japanese yen note", "Japanese currency", "Yen banknote"],
    "INR": ["Indian rupee note", "Indian currency", "Rupee banknote"],
    "BRL": ["Brazilian real note", "Brazilian currency"],
    "AUD": ["Australian dollar note", "Australian currency"],
    "CAD": ["Canadian dollar note", "Canadian currency"],
    "CNY": ["Chinese yuan note", "Chinese currency", "Renminbi banknote"],
    "KRW": ["South Korean won note", "Korean currency"],
    "RUB": ["Russian ruble note", "Russian currency"],
    "TRY": ["Turkish lira note", "Turkish currency"],
    "MXN": ["Mexican peso note", "Mexican currency"],
    "ZAR": ["South African rand note", "South African currency"],
    "THB": ["Thai baht note", "Thai currency"],
    "IDR": ["Indonesian rupiah note", "Indonesian currency"],
    "MYR": ["Malaysian ringgit note", "Malaysian currency"],
    "PHP": ["Philippine peso note", "Philippine currency"],
    "SAR": ["Saudi riyal note", "Saudi currency"],
    "AED": ["UAE dirham note", "Emirati currency"],
}

# Currency → Country(s)
CURRENCY_TO_COUNTRIES: dict[str, list[str]] = {
    "USD": ["US"],
    "EUR": ["DE", "FR", "IT", "ES", "PT", "NL", "BE", "AT", "GR", "IE", "FI"],
    "GBP": ["GB"],
    "JPY": ["JP"],
    "INR": ["IN"],
    "BRL": ["BR"],
    "AUD": ["AU"],
    "CAD": ["CA"],
    "CNY": ["CN"],
    "KRW": ["KR"],
    "RUB": ["RU"],
    "TRY": ["TR"],
    "MXN": ["MX"],
    "ZAR": ["ZA"],
    "THB": ["TH"],
    "IDR": ["ID"],
    "MYR": ["MY"],
    "PHP": ["PH"],
    "SAR": ["SA"],
    "AED": ["AE"],
}


# ── OCR Language → Country Hints ────────────────────────────────────────────

# Scripts / languages detected by OCR → probable countries
SCRIPT_COUNTRY_HINTS: dict[str, list[str]] = {
    "thai": ["TH"],
    "japanese": ["JP"],
    "korean": ["KR"],
    "chinese_similar": ["CN", "TW", "HK", "MO"],
    "arabic": ["SA", "AE", "EG", "MA", "TN", "IQ", "IR", "JO", "LB"],
    "cyrillic": ["RU", "UA", "BG", "RS", "BY", "KZ", "KG"],
    "devanagari": ["IN", "NP"],
    "bengali": ["BD", "IN"],
    "tamil": ["IN", "LK"],
    "telugu": ["IN"],
    "kannada": ["IN"],
    "malayalam": ["IN"],
    "gurmukhi": ["IN"],
    "latin_extended": ["TR", "VN", "ID", "PH", "MY"],
    "greek": ["GR"],
    "hebrew": ["IL"],
    "georgian": ["GE"],
    "armenian": ["AM"],
    "ethiopic": ["ET"],
    "myanmar": ["MM"],
    "khmer": ["KH"],
    "lao": ["LA"],
    "tibetan": ["CN", "IN", "NP"],
}


# ── Sun Clock Timezone Hints ────────────────────────────────────────────────

# UTC offset ranges by region
REGION_UTC_OFFSETS: dict[str, tuple[float, float]] = {
    "US_eastern": (-5, -4),
    "US_central": (-6, -5),
    "US_mountain": (-7, -6),
    "US_pacific": (-8, -7),
    "UK": (0, 1),
    "central_europe": (1, 2),
    "eastern_europe": (2, 3),
    "russia_moscow": (3, 3),
    "india": (5.5, 5.5),
    "china": (8, 8),
    "japan": (9, 9),
    "korea": (9, 9),
    "australia_east": (10, 11),
    "australia_west": (8, 8),
}
