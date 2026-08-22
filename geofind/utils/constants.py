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
# confidence: 0.95 = iconic/unique, 0.85 = well-known, 0.75 = regional
LANDMARK_CATEGORIES: dict[str, dict[str, float]] = {
    # ── Europe ───────────────────────────────────────────────────────────
    "eiffel_tower": {"lat": 48.8584, "lon": 2.2945, "confidence": 0.95},
    "colosseum": {"lat": 41.8902, "lon": 12.4922, "confidence": 0.95},
    "big_ben": {"lat": 51.5007, "lon": -0.1246, "confidence": 0.95},
    "brandenburg_gate": {"lat": 52.5163, "lon": 13.3777, "confidence": 0.95},
    "sagrada_familia": {"lat": 41.4036, "lon": 2.1744, "confidence": 0.95},
    "parthenon": {"lat": 37.9715, "lon": 23.7267, "confidence": 0.95},
    "st_basils_cathedral": {"lat": 55.7525, "lon": 37.6231, "confidence": 0.95},
    "statue_of_david": {"lat": 43.7696, "lon": 11.2558, "confidence": 0.85},
    "tower_of_london": {"lat": 51.5081, "lon": -0.0759, "confidence": 0.90},
    "edinburgh_castle": {"lat": 55.9486, "lon": -3.1875, "confidence": 0.90},
    "charles_bridge": {"lat": 50.0865, "lon": 14.4114, "confidence": 0.85},
    "neuschwanstein_castle": {"lat": 47.5576, "lon": 10.7498, "confidence": 0.90},
    "alhambra": {"lat": 37.1760, "lon": -3.5881, "confidence": 0.90},
    "park_guell": {"lat": 41.4145, "lon": 2.1527, "confidence": 0.85},
    "moulin_rouge": {"lat": 48.8841, "lon": 2.3321, "confidence": 0.85},
    "tower_of_pisa": {"lat": 43.7230, "lon": 10.3966, "confidence": 0.95},
    "mont_saint_michel": {"lat": 48.6361, "lon": -1.5115, "confidence": 0.90},
    "sistine_chapel": {"lat": 41.9029, "lon": 12.4545, "confidence": 0.85},
    "acropolis_athens": {"lat": 37.9715, "lon": 23.7257, "confidence": 0.95},
    "stonehenge": {"lat": 51.1789, "lon": -1.8262, "confidence": 0.95},
    "windsor_castle": {"lat": 51.4791, "lon": -0.6095, "confidence": 0.85},
    "amsterdam_canal": {"lat": 52.3676, "lon": 4.9041, "confidence": 0.75},
    "little_mermaid_copenhagen": {"lat": 55.6929, "lon": 12.5996, "confidence": 0.85},
    "romeo_and_juliet_verona": {"lat": 45.4401, "lon": 10.9989, "confidence": 0.75},
    "michelangelo_david_florence": {"lat": 43.7764, "lon": 11.2566, "confidence": 0.80},
    "pont_d_avignon": {"lat": 43.9542, "lon": 4.8052, "confidence": 0.80},
    "santorini": {"lat": 36.3932, "lon": 25.4615, "confidence": 0.90},
    "dubrovnik_walls": {"lat": 42.6415, "lon": 18.1107, "confidence": 0.85},
    "hallstatt": {"lat": 47.5622, "lon": 13.6493, "confidence": 0.75},
    "meteora": {"lat": 39.7217, "lon": 21.6306, "confidence": 0.85},
    "plitvice_lakes": {"lat": 44.8654, "lon": 15.5820, "confidence": 0.80},
    "louvre_pyramid": {"lat": 48.8611, "lon": 2.3359, "confidence": 0.90},
    "notre_dame_paris": {"lat": 48.8530, "lon": 2.3499, "confidence": 0.90},
    "champs_elysees": {"lat": 48.8698, "lon": 2.3075, "confidence": 0.75},
    "arc_de_triomphe": {"lat": 48.8738, "lon": 2.2950, "confidence": 0.90},
    "versailles": {"lat": 48.8049, "lon": 2.1204, "confidence": 0.90},
    "sagrada_familia_interior": {"lat": 41.4036, "lon": 2.1744, "confidence": 0.85},
    "vatican_st_peters": {"lat": 41.9022, "lon": 12.4539, "confidence": 0.90},
    "pantheon_rome": {"lat": 41.8986, "lon": 12.4769, "confidence": 0.85},
    "trevi_fountain": {"lat": 41.9009, "lon": 12.4833, "confidence": 0.85},

    # ── Asia ─────────────────────────────────────────────────────────────
    "taj_mahal": {"lat": 27.1751, "lon": 78.0421, "confidence": 0.95},
    "great_wall": {"lat": 40.4319, "lon": 116.5704, "confidence": 0.95},
    "angkor_wat": {"lat": 13.4125, "lon": 103.8670, "confidence": 0.95},
    "mount_fuji": {"lat": 35.3606, "lon": 138.7274, "confidence": 0.95},
    "petronas_towers": {"lat": 3.1578, "lon": 101.7116, "confidence": 0.95},
    "marina_bay_sands": {"lat": 1.2834, "lon": 103.8607, "confidence": 0.90},
    "temple_emerald_buddha": {"lat": 13.7516, "lon": 100.4927, "confidence": 0.90},
    "forbidden_city": {"lat": 39.9163, "lon": 116.3972, "confidence": 0.95},
    "terracotta_army": {"lat": 34.3842, "lon": 109.2785, "confidence": 0.90},
    "ha_long_bay": {"lat": 20.9101, "lon": 107.1839, "confidence": 0.90},
    "bagan_temples": {"lat": 21.1717, "lon": 94.8585, "confidence": 0.85},
    "borobudur": {"lat": -7.6079, "lon": 110.2038, "confidence": 0.90},
    "sigiriya": {"lat": 7.9572, "lon": 80.7603, "confidence": 0.85},
    "swaminarayan_akshardham": {"lat": 28.6127, "lon": 77.2773, "confidence": 0.80},
    "golden_temple_amritsar": {"lat": 31.6200, "lon": 74.8765, "confidence": 0.90},
    "mount_everest_base": {"lat": 28.0025, "lon": 86.8528, "confidence": 0.90},
    "mountain_fuji_chureito_pagoda": {"lat": 35.3150, "lon": 138.8075, "confidence": 0.85},
    "kinkakuji_golden_pavilion": {"lat": 35.0394, "lon": 135.7292, "confidence": 0.85},
    "tokyo_tower": {"lat": 35.6586, "lon": 139.7455, "confidence": 0.90},
    "mt_olympus_japan": {"lat": 36.5786, "lon": 138.5225, "confidence": 0.80},
    "great_buddha_kamakura": {"lat": 35.3167, "lon": 139.5356, "confidence": 0.80},
    "meiji_shrine": {"lat": 35.6764, "lon": 139.6993, "confidence": 0.80},
    "itsukushima_shrine": {"lat": 34.2960, "lon": 132.3199, "confidence": 0.85},
    "haeundae_beach_busan": {"lat": 35.1586, "lon": 129.1604, "confidence": 0.75},
    "gyeongbokgung_palace": {"lat": 37.5796, "lon": 126.9770, "confidence": 0.85},
    "chuang_yan_monastery_hanging": {"lat": 39.8766, "lon": 113.6186, "confidence": 0.80},
    "potala_palace": {"lat": 29.6575, "lon": 91.1170, "confidence": 0.90},
    "mount_huangshan": {"lat": 30.1375, "lon": 118.1694, "confidence": 0.80},
    "li_river_guilin": {"lat": 24.7829, "lon": 110.2990, "confidence": 0.75},
    "jatiluwih_rice_terrace": {"lat": -8.3744, "lon": 115.1346, "confidence": 0.75},
    "ulun_danu_bratan": {"lat": -8.2752, "lon": 115.1667, "confidence": 0.80},
    "petra_treasury": {"lat": 30.3285, "lon": 35.4444, "confidence": 0.95},
    "mount_kinabalu": {"lat": 6.0753, "lon": 116.5535, "confidence": 0.80},
    "bangkok_grand_palace": {"lat": 13.7510, "lon": 100.4915, "confidence": 0.85},

    # ── Americas ─────────────────────────────────────────────────────────
    "statue_of_liberty": {"lat": 40.6892, "lon": -74.0445, "confidence": 0.95},
    "golden_gate_bridge": {"lat": 37.8199, "lon": -122.4783, "confidence": 0.95},
    "christ_redeemer": {"lat": -22.9519, "lon": -43.2105, "confidence": 0.95},
    "machu_picchu": {"lat": -13.1631, "lon": -72.5450, "confidence": 0.95},
    "chichen_itza": {"lat": 20.6843, "lon": -88.5678, "confidence": 0.95},
    "cn_tower": {"lat": 43.6426, "lon": -79.3871, "confidence": 0.95},
    "hollywood_sign": {"lat": 34.1341, "lon": -118.3215, "confidence": 0.95},
    "gateway_arch": {"lat": 38.6247, "lon": -90.1848, "confidence": 0.95},
    "mount_rushmore": {"lat": 43.8791, "lon": -103.4591, "confidence": 0.95},
    "christ_of_the_ozarks": {"lat": 36.4326, "lon": -93.3446, "confidence": 0.75},
    "stone_mountain": {"lat": 33.8062, "lon": -84.1667, "confidence": 0.75},
    "white_house": {"lat": 38.8977, "lon": -77.0365, "confidence": 0.95},
    "united_nations_ny": {"lat": 40.7489, "lon": -73.9680, "confidence": 0.85},
    "times_square": {"lat": 40.7580, "lon": -73.9855, "confidence": 0.85},
    "empire_state_building": {"lat": 40.7484, "lon": -73.9857, "confidence": 0.90},
    "brooklyn_bridge": {"lat": 40.7061, "lon": -73.9969, "confidence": 0.90},
    "transamerica_pyramid": {"lat": 37.7952, "lon": -122.4028, "confidence": 0.80},
    "space_needle_seattle": {"lat": 47.6205, "lon": -122.3493, "confidence": 0.90},
    "willis_tower": {"lat": 41.8789, "lon": -87.6359, "confidence": 0.85},
    "niagara_falls": {"lat": 43.0896, "lon": -79.0849, "confidence": 0.90},
    "petra_beach_brazil": {"lat": -22.9838, "lon": -43.2096, "confidence": 0.75},
    "iguazu_falls": {"lat": -25.6953, "lon": -54.4367, "confidence": 0.90},
    "san_giglio_basilica_rio": {"lat": -22.9133, "lon": -43.1808, "confidence": 0.80},
    "nazca_lines": {"lat": -14.7350, "lon": -75.1300, "confidence": 0.85},
    "galapagos_tortoise": {"lat": -0.9538, "lon": -90.9656, "confidence": 0.80},
    "banff_lake_louise": {"lat": 51.4254, "lon": -116.1771, "confidence": 0.80},
    "grand_canyon": {"lat": 36.1069, "lon": -112.1129, "confidence": 0.90},
    "yellowstone_old_faithful": {"lat": 44.4605, "lon": -110.8281, "confidence": 0.85},
    "yosemite_el_capitan": {"lat": 37.7459, "lon": -119.5332, "confidence": 0.80},
    "french_quarter_new_orleans": {"lat": 29.9581, "lon": -90.0640, "confidence": 0.75},

    # ── Africa ───────────────────────────────────────────────────────────
    "pyramids_of_giza": {"lat": 29.9792, "lon": 31.1342, "confidence": 0.95},
    "sphinx_giza": {"lat": 29.9753, "lon": 31.1376, "confidence": 0.95},
    "table_mountain": {"lat": -33.9625, "lon": 18.4039, "confidence": 0.90},
    "kilimanjaro": {"lat": -3.0674, "lon": 37.3556, "confidence": 0.95},
    "great_zimbabwe": {"lat": -20.2674, "lon": 30.9337, "confidence": 0.75},
    "abu_simbel": {"lat": 22.3360, "lon": 31.6256, "confidence": 0.90},
    "victoria_falls": {"lat": -17.9243, "lon": 25.8572, "confidence": 0.90},
    "serengeti": {"lat": -2.3333, "lon": 34.8333, "confidence": 0.80},
    "sahara_dunes": {"lat": 23.4162, "lon": 25.6628, "confidence": 0.80},
    "luxor_temple": {"lat": 25.6997, "lon": 32.6390, "confidence": 0.85},
    "valley_of_kings": {"lat": 25.7402, "lon": 32.6014, "confidence": 0.85},
    "bab_el_mansour_meknes": {"lat": 33.8932, "lon": -5.5537, "confidence": 0.75},
    "djenné_mosque": {"lat": 13.9054, "lon": -4.5556, "confidence": 0.80},
    "cape_of_good_hope": {"lat": -34.3568, "lon": 18.4741, "confidence": 0.85},
    "kruger_national_park": {"lat": -24.0128, "lon": 31.4914, "confidence": 0.80},
    "atlas_mountains": {"lat": 31.8000, "lon": -7.0000, "confidence": 0.75},

    # ── Middle East ──────────────────────────────────────────────────────
    "petra": {"lat": 30.3285, "lon": 35.4444, "confidence": 0.95},
    "burj_khalifa": {"lat": 25.1972, "lon": 55.2744, "confidence": 0.95},
    "burj_al_arab": {"lat": 25.1412, "lon": 55.1854, "confidence": 0.95},
    "dome_of_the_rock": {"lat": 31.7784, "lon": 35.2353, "confidence": 0.95},
    "hagia_sophia": {"lat": 41.0086, "lon": 28.9802, "confidence": 0.95},
    "blue_mosque": {"lat": 41.0054, "lon": 28.9768, "confidence": 0.90},
    "wailing_wall": {"lat": 31.7767, "lon": 35.2345, "confidence": 0.90},
    "palm_jumeirah": {"lat": 25.0960, "lon": 55.1390, "confidence": 0.85},
    "cappadocia": {"lat": 38.6431, "lon": 34.8289, "confidence": 0.80},
    "ephesus": {"lat": 37.9394, "lon": 27.3416, "confidence": 0.80},
    "persepolis": {"lat": 29.9354, "lon": 52.8914, "confidence": 0.85},
    "masada": {"lat": 31.3167, "lon": 35.3547, "confidence": 0.80},
    "dead_sea": {"lat": 31.5000, "lon": 35.5000, "confidence": 0.80},

    # ── Oceania ──────────────────────────────────────────────────────────
    "sydney_opera_house": {"lat": -33.8568, "lon": 151.2153, "confidence": 0.95},
    "auckland_sky_tower": {"lat": -36.8485, "lon": 174.7622, "confidence": 0.85},
    "great_barrier_reef": {"lat": -18.2871, "lon": 147.6992, "confidence": 0.90},
    "uluru": {"lat": -25.3444, "lon": 131.0369, "confidence": 0.95},
    "twelve_apostles_australia": {"lat": -38.6613, "lon": 143.1049, "confidence": 0.80},
    "milford_sound": {"lat": -44.6414, "lon": 167.9250, "confidence": 0.85},
    "hobbiton_nz": {"lat": -37.8722, "lon": 175.6813, "confidence": 0.75},
    "sydney_harbour_bridge": {"lat": -33.8523, "lon": 151.2108, "confidence": 0.90},
    "bora_bora": {"lat": -16.5004, "lon": -151.7415, "confidence": 0.85},
    "tongariro": {"lat": -39.2890, "lon": 175.6358, "confidence": 0.75},

    # ── Russia / Central Asia ────────────────────────────────────────────
    "kremlin": {"lat": 55.7520, "lon": 37.6175, "confidence": 0.95},
    "red_square": {"lat": 55.7539, "lon": 37.6208, "confidence": 0.90},
    "winter_palace": {"lat": 59.9398, "lon": 30.3146, "confidence": 0.90},
    "lake_baikal": {"lat": 53.5587, "lon": 108.1650, "confidence": 0.85},
    "kamchatka_volcano": {"lat": 56.0553, "lon": 160.6430, "confidence": 0.75},
    "statue_of_almaty_kazakhstan": {"lat": 43.2380, "lon": 76.9455, "confidence": 0.75},
    "registan_samarkand": {"lat": 39.6547, "lon": 66.9780, "confidence": 0.85},

    # ── Additional Icons ─────────────────────────────────────────────────
    "parliament_london": {"lat": 51.4995, "lon": -0.1248, "confidence": 0.90},
    "london_eye": {"lat": 51.5033, "lon": -0.1195, "confidence": 0.90},
    "london_bridge": {"lat": 51.5079, "lon": -0.0877, "confidence": 0.80},
    "buckingham_palace": {"lat": 51.5014, "lon": -0.1419, "confidence": 0.85},
    "stanley_park_vancouver": {"lat": 49.2958, "lon": -123.1391, "confidence": 0.75},
    "bryce_canyon": {"lat": 37.5930, "lon": -112.1871, "confidence": 0.80},
    "monument_valley": {"lat": 36.9837, "lon": -110.1152, "confidence": 0.85},
    "walt_disney_world_cinderella": {"lat": 28.4177, "lon": -81.5812, "confidence": 0.80},
    "statue_of_christ_tenerife": {"lat": 28.1042, "lon": -16.5275, "confidence": 0.75},
    "pompeii": {"lat": 40.7488, "lon": 14.4849, "confidence": 0.85},
    "amalfi_coast": {"lat": 40.6340, "lon": 14.6027, "confidence": 0.75},
    "venice_canals": {"lat": 45.4408, "lon": 12.3155, "confidence": 0.80},
    "cinque_terre": {"lat": 44.1461, "lon": 9.6572, "confidence": 0.75},
    "black_forest_germany": {"lat": 48.3665, "lon": 8.1506, "confidence": 0.75},
    "fjord_norway": {"lat": 61.5000, "lon": 6.8000, "confidence": 0.80},
    "trolltunga": {"lat": 60.1241, "lon": 6.7400, "confidence": 0.75},
    "preikestolen_pulpit_rock": {"lat": 58.9863, "lon": 6.1908, "confidence": 0.80},
    "wadi_rum": {"lat": 29.5321, "lon": 35.4179, "confidence": 0.80},
    "king_canyon_australia": {"lat": -24.2575, "lon": 131.5442, "confidence": 0.75},
    "blue_mountain_australia": {"lat": -33.7150, "lon": 150.3114, "confidence": 0.75},
    "everglades": {"lat": 25.2866, "lon": -80.8987, "confidence": 0.80},
    "pacific_coast_highway": {"lat": 36.2704, "lon": -121.8081, "confidence": 0.75},

    # ── Africa (expanded) ────────────────────────────────────────────────
    "zanzibar_stone_town": {"lat": -6.1622, "lon": 39.1919, "confidence": 0.75},
    "robben_island": {"lat": -33.8073, "lon": 18.3663, "confidence": 0.85},
    "karnak_temple": {"lat": 25.7188, "lon": 32.6573, "confidence": 0.80},
    "apartheid_museum": {"lat": -26.2353, "lon": 28.0085, "confidence": 0.75},
    "blyde_river_canyon": {"lat": -24.5741, "lon": 30.8117, "confidence": 0.75},
    "chefchaouen_medina": {"lat": 35.1714, "lon": -5.2697, "confidence": 0.75},
    "nile_river_cairo": {"lat": 30.0444, "lon": 31.2357, "confidence": 0.80},

    # ── South America (expanded) ─────────────────────────────────────────
    "salar_de_uyuni": {"lat": -20.1338, "lon": -67.4891, "confidence": 0.90},
    "angel_falls": {"lat": 5.9701, "lon": -62.5362, "confidence": 0.85},
    "cartagena_walled_city": {"lat": 10.3910, "lon": -75.5144, "confidence": 0.80},
    "moai_easter_island": {"lat": -27.1127, "lon": -109.3497, "confidence": 0.90},
    "copacabana_beach": {"lat": -22.9711, "lon": -43.1823, "confidence": 0.80},
    "sugarloaf_mountain": {"lat": -22.9486, "lon": -43.1560, "confidence": 0.85},
    "torres_del_paine": {"lat": -51.2536, "lon": -72.3456, "confidence": 0.85},
    "los_glaciares_perito_moreno": {"lat": -50.3402, "lon": -72.2647, "confidence": 0.80},
    "teotihuacan": {"lat": 19.6925, "lon": -98.8438, "confidence": 0.90},
    "palenque": {"lat": 17.4838, "lon": -92.0461, "confidence": 0.85},
    "cusco_historic_center": {"lat": -13.5319, "lon": -71.9675, "confidence": 0.80},

    # ── Southeast Asia (expanded) ────────────────────────────────────────
    "shwedagon_pagoda": {"lat": 16.8714, "lon": 96.1497, "confidence": 0.90},
    "komodo_island": {"lat": -8.5503, "lon": 119.4833, "confidence": 0.80},
    "boracay_island": {"lat": 11.9674, "lon": 121.9248, "confidence": 0.75},
    "cu_chi_tunnels": {"lat": 11.0522, "lon": 106.5364, "confidence": 0.75},
    "patong_beach_phuket": {"lat": 7.8804, "lon": 98.2920, "confidence": 0.75},

    # ── Middle East (expanded) ───────────────────────────────────────────
    "al_aqsa_mosque": {"lat": 31.7767, "lon": 35.2353, "confidence": 0.90},
    "sheikh_zayed_mosque": {"lat": 24.4128, "lon": 54.4751, "confidence": 0.85},
    "king_fahd_fountain": {"lat": 21.6168, "lon": 39.1073, "confidence": 0.80},
    "mount_sinai": {"lat": 28.5356, "lon": 33.9747, "confidence": 0.80},
    "rock_of_gibraltar": {"lat": 36.1408, "lon": -5.3536, "confidence": 0.85},

    # ── Eastern Europe (expanded) ────────────────────────────────────────
    "bran_castle": {"lat": 45.5152, "lon": 25.3672, "confidence": 0.85},
    "lake_bled": {"lat": 46.3639, "lon": 14.0940, "confidence": 0.80},
    "mostar_bridge": {"lat": 43.3373, "lon": 17.8153, "confidence": 0.80},
    "palace_of_parliament_bucharest": {"lat": 44.4275, "lon": 26.0875, "confidence": 0.80},
    "krakow_wawel_castle": {"lat": 50.0540, "lon": 19.9354, "confidence": 0.85},
    "trinity_lavra_st_sergius": {"lat": 56.3153, "lon": 38.1324, "confidence": 0.75},

    # ── India (expanded) ─────────────────────────────────────────────────
    "gateway_of_india": {"lat": 18.9220, "lon": 72.8347, "confidence": 0.85},
    "red_fort_delhi": {"lat": 28.6562, "lon": 77.2410, "confidence": 0.85},
    "qutub_minar": {"lat": 28.5244, "lon": 77.1855, "confidence": 0.85},
    "hawa_mahal": {"lat": 26.9239, "lon": 75.8267, "confidence": 0.85},
    "lotus_temple_delhi": {"lat": 28.5535, "lon": 77.2588, "confidence": 0.80},
    "meenakshi_temple": {"lat": 9.9195, "lon": 78.1193, "confidence": 0.80},
    "hampi_ruins": {"lat": 15.3350, "lon": 76.4600, "confidence": 0.80},
    "konark_sun_temple": {"lat": 19.8876, "lon": 86.0947, "confidence": 0.80},
    "brihadeeswara_temple": {"lat": 10.7828, "lon": 79.1318, "confidence": 0.80},

    # ── China (expanded) ─────────────────────────────────────────────────
    "zhangjiajie_pillars": {"lat": 29.3249, "lon": 110.4342, "confidence": 0.85},
    "jiuzhaigou_valley": {"lat": 33.2600, "lon": 103.9170, "confidence": 0.80},
    "west_lake_hangzhou": {"lat": 30.2421, "lon": 120.1483, "confidence": 0.80},
    "giant_buddha_leshan": {"lat": 29.9470, "lon": 103.7721, "confidence": 0.80},
    "summer_palace_beijing": {"lat": 39.9999, "lon": 116.2755, "confidence": 0.85},
    "temple_of_heaven_beijing": {"lat": 39.8822, "lon": 116.4065, "confidence": 0.85},
    "mogao_caves_dunhuang": {"lat": 40.0420, "lon": 94.8090, "confidence": 0.80},

    # ── Japan (expanded) ─────────────────────────────────────────────────
    "fushimi_inari_shrine": {"lat": 34.9671, "lon": 135.7727, "confidence": 0.90},
    "senso_ji_temple": {"lat": 35.7148, "lon": 139.7967, "confidence": 0.85},
    "hiroshima_peace_memorial": {"lat": 34.3955, "lon": 132.4537, "confidence": 0.85},
    "arashiyama_bamboo_grove": {"lat": 35.0094, "lon": 135.6670, "confidence": 0.85},
    "nijo_castle_kyoto": {"lat": 35.0142, "lon": 135.7481, "confidence": 0.80},
    "todai_ji_nara": {"lat": 34.6891, "lon": 135.8398, "confidence": 0.80},
    "osaka_castle": {"lat": 34.6873, "lon": 135.5262, "confidence": 0.85},
    "himeji_castle": {"lat": 34.8394, "lon": 134.6939, "confidence": 0.85},
    "matsumoto_castle": {"lat": 36.2384, "lon": 137.9720, "confidence": 0.80},
    "nikko_toshogu_shrine": {"lat": 36.7580, "lon": 139.5998, "confidence": 0.80},

    # ── Central Asia (expanded) ──────────────────────────────────────────
    "silk_road_bukhara": {"lat": 39.7747, "lon": 64.4175, "confidence": 0.80},
    "issyk_kul_lake": {"lat": 42.4500, "lon": 77.2500, "confidence": 0.75},
    "tian_shan_mountains": {"lat": 42.0000, "lon": 80.0000, "confidence": 0.75},

    # ── Oceania (expanded) ───────────────────────────────────────────────
    "waitomo_glowworm_caves": {"lat": -38.2614, "lon": 175.1061, "confidence": 0.75},
    "pinnacles_desert_australia": {"lat": -30.3463, "lon": 115.1514, "confidence": 0.75},

    # ── Additional Global Icons ──────────────────────────────────────────
    "tower_bridge_london": {"lat": 51.5055, "lon": -0.0754, "confidence": 0.90},
    "westminster_abbey": {"lat": 51.4993, "lon": -0.1273, "confidence": 0.85},
    "atomium_brussels": {"lat": 50.8949, "lon": 4.3417, "confidence": 0.80},
    "manneken_pis": {"lat": 50.8450, "lon": 4.3498, "confidence": 0.75},
    "berlin_wall_memorial": {"lat": 52.5351, "lon": 13.3900, "confidence": 0.80},
    "sachsenhausen_memorial": {"lat": 52.7493, "lon": 13.2631, "confidence": 0.75},
    "auschwitz_birkenau": {"lat": 50.0343, "lon": 19.1783, "confidence": 0.85},
    "mykonos": {"lat": 37.4467, "lon": 25.3289, "confidence": 0.75},
    "olympia_greece": {"lat": 37.6388, "lon": 21.6303, "confidence": 0.80},
    "alcatraz_island": {"lat": 37.8267, "lon": -122.4230, "confidence": 0.85},
    "central_park_nyc": {"lat": 40.7829, "lon": -73.9654, "confidence": 0.80},
    "one_world_trade_center": {"lat": 40.7127, "lon": -74.0134, "confidence": 0.85},
    "pentagon_memorial": {"lat": 38.8719, "lon": -77.0563, "confidence": 0.75},
    "liberty_bell": {"lat": 39.9496, "lon": -75.1503, "confidence": 0.85},
    "independence_hall": {"lat": 39.9489, "lon": -75.1501, "confidence": 0.80},
    "lincoln_memorial": {"lat": 38.8893, "lon": -77.0502, "confidence": 0.90},
    "washington_monument": {"lat": 38.8895, "lon": -77.0353, "confidence": 0.90},
    "jefferson_memorial": {"lat": 38.8810, "lon": -77.0365, "confidence": 0.80},
    "capitol_building_dc": {"lat": 38.8899, "lon": -77.0091, "confidence": 0.85},
    "smithsonian_institution": {"lat": 38.8888, "lon": -77.0260, "confidence": 0.80},

    # ── Additional Europe ────────────────────────────────────────────────
    "cologne_cathedral": {"lat": 50.9413, "lon": 6.9583, "confidence": 0.85},
    "reichstag_berlin": {"lat": 52.5186, "lon": 13.3762, "confidence": 0.85},
    "prague_astronomical_clock": {"lat": 50.0870, "lon": 14.4213, "confidence": 0.85},
    "hungarian_parliament_budapest": {"lat": 47.5071, "lon": 19.0456, "confidence": 0.85},
    "grand_place_brussels": {"lat": 50.8467, "lon": 4.3525, "confidence": 0.85},
    "bruges_markt": {"lat": 51.2092, "lon": 3.2247, "confidence": 0.75},
    "tivoli_gardens_copenhagen": {"lat": 55.6761, "lon": 12.5683, "confidence": 0.75},
    "vasa_museum_stockholm": {"lat": 59.3289, "lon": 18.0894, "confidence": 0.80},
    "vigeland_park_oslo": {"lat": 59.9269, "lon": 10.7007, "confidence": 0.75},
    "hallgrimskirkja_iceland": {"lat": 64.1417, "lon": -21.9267, "confidence": 0.80},
    "florence_cathedral_duomo": {"lat": 43.7731, "lon": 11.2560, "confidence": 0.90},
    "piazza_san_marco_venice": {"lat": 45.4343, "lon": 12.3388, "confidence": 0.80},
    "portofino_italy": {"lat": 44.3034, "lon": 9.2097, "confidence": 0.75},
    "lake_geneva": {"lat": 46.4600, "lon": 6.5700, "confidence": 0.75},
    "kotor_montenegro": {"lat": 42.4247, "lon": 18.7712, "confidence": 0.75},

    # ── Additional Americas ──────────────────────────────────────────────
    "las_vegas_strip": {"lat": 36.1147, "lon": -115.1728, "confidence": 0.85},
    "hoover_dam": {"lat": 36.0160, "lon": -114.7377, "confidence": 0.85},
    "antelope_canyon": {"lat": 36.8619, "lon": -111.3743, "confidence": 0.80},
    "sedona_red_rocks": {"lat": 34.8697, "lon": -111.7610, "confidence": 0.80},
    "jasper_national_park": {"lat": 52.8737, "lon": -117.7361, "confidence": 0.75},
    "art_institute_chicago": {"lat": 41.8796, "lon": -87.6237, "confidence": 0.75},
    "navy_pier_chicago": {"lat": 41.8917, "lon": -87.6063, "confidence": 0.75},
    "pike_place_market_seattle": {"lat": 47.6101, "lon": -122.3424, "confidence": 0.80},
    "bonneville_salt_flats": {"lat": 40.7561, "lon": -113.8979, "confidence": 0.75},
    "los_angeles_coliseum": {"lat": 34.0141, "lon": -118.2879, "confidence": 0.75},
    "washington_square_park_nyc": {"lat": 40.7308, "lon": -73.9973, "confidence": 0.75},
    "gettysburg_battlefield": {"lat": 39.8109, "lon": -77.2275, "confidence": 0.80},
    "pearl_harbor_memorial": {"lat": 21.3649, "lon": -157.9500, "confidence": 0.85},

    # ── Additional Asia ──────────────────────────────────────────────────
    "chiang_mai_old_city": {"lat": 18.7883, "lon": 98.9853, "confidence": 0.75},
    "jeju_hallasan": {"lat": 33.3617, "lon": 126.5292, "confidence": 0.75},
    "seoul_namsan_tower": {"lat": 37.5512, "lon": 126.9882, "confidence": 0.80},
    "gyeongju_bulguksa_temple": {"lat": 35.7900, "lon": 129.3360, "confidence": 0.75},
    "varanasi_ghats": {"lat": 25.3044, "lon": 83.0109, "confidence": 0.80},
    "kathmandu_durbar_square": {"lat": 27.7045, "lon": 85.3070, "confidence": 0.80},
    "pokhara_lake_phewa": {"lat": 28.2096, "lon": 83.9856, "confidence": 0.75},

    # ── Additional Africa ────────────────────────────────────────────────
    "okavango_delta": {"lat": -19.5000, "lon": 22.9667, "confidence": 0.75},
    "ngorongoro_crater": {"lat": -3.1800, "lon": 35.5800, "confidence": 0.80},
    "fish_river_canyon": {"lat": -27.5833, "lon": 17.5833, "confidence": 0.75},
    "drakensberg_mountains": {"lat": -29.1000, "lon": 29.5000, "confidence": 0.75},
    "cairo_khan_el_khalili": {"lat": 30.0475, "lon": 31.2627, "confidence": 0.75},

    # ── Additional Oceania ───────────────────────────────────────────────
    "franz_josef_glacier": {"lat": -43.3885, "lon": 170.1834, "confidence": 0.75},
    "rotorua_geysers_nz": {"lat": -38.3436, "lon": 176.2661, "confidence": 0.75},
    "bay_of_islands_nz": {"lat": -35.2274, "lon": 174.1072, "confidence": 0.75},

    # ── Additional Middle East ───────────────────────────────────────────
    "madain_saleh": {"lat": 26.7753, "lon": 37.9536, "confidence": 0.75},
    "jerash_jordan": {"lat": 32.2747, "lon": 35.8928, "confidence": 0.75},

    # ── North America (expanded): US cities & icons ──────────────────────
    "chrysler_building_nyc": {"lat": 40.7516, "lon": -73.9755, "confidence": 0.85},
    "flatiron_building_nyc": {"lat": 40.7400, "lon": -73.9898, "confidence": 0.85},
    "grand_central_terminal": {"lat": 40.7527, "lon": -73.9772, "confidence": 0.80},
    "rockefeller_center_nyc": {"lat": 40.7587, "lon": -73.9787, "confidence": 0.80},
    "wall_street_bull": {"lat": 40.7049, "lon": -74.0134, "confidence": 0.80},
    "brooklyn_heights_promenade": {"lat": 40.6975, "lon": -73.9935, "confidence": 0.75},
    "central_park_bow_bridge": {"lat": 40.7715, "lon": -73.9693, "confidence": 0.75},
    "santa_monica_pier": {"lat": 34.0083, "lon": -118.4988, "confidence": 0.85},
    "griffith_observatory": {"lat": 34.1184, "lon": -118.3004, "confidence": 0.85},
    "venice_beach_sign": {"lat": 33.9850, "lon": -118.4695, "confidence": 0.80},
    "lombard_street_sf": {"lat": 37.8021, "lon": -122.4187, "confidence": 0.80},
    "painted_ladies_sf": {"lat": 37.7762, "lon": -122.4328, "confidence": 0.75},
    "fenway_park_boston": {"lat": 42.3467, "lon": -71.0972, "confidence": 0.80},
    "miami_south_beach": {"lat": 25.7907, "lon": -80.1300, "confidence": 0.80},
    "sphere_las_vegas": {"lat": 36.1160, "lon": -115.1736, "confidence": 0.85},
    "alamo_san_antonio": {"lat": 29.4260, "lon": -98.4861, "confidence": 0.85},
    "palm_springs_windmills": {"lat": 33.9200, "lon": -116.5000, "confidence": 0.75},
    "cadillac_ranch_texas": {"lat": 35.1872, "lon": -101.9819, "confidence": 0.75},
    "cloud_gate_chicago": {"lat": 41.8827, "lon": -87.6227, "confidence": 0.85},
    "graceland_memphis": {"lat": 35.0479, "lon": -90.0259, "confidence": 0.75},
    "iwo_jima_memorial": {"lat": 38.8889, "lon": -77.0694, "confidence": 0.75},
    "hearst_castle_california": {"lat": 35.6850, "lon": -121.1681, "confidence": 0.75},
    "white_sands_new_mexico": {"lat": 32.7831, "lon": -106.1706, "confidence": 0.80},
    "kennedy_space_center": {"lat": 28.5729, "lon": -80.6490, "confidence": 0.80},
    "multnomah_falls_oregon": {"lat": 45.5762, "lon": -122.1158, "confidence": 0.75},
    "crater_lake_oregon": {"lat": 42.8684, "lon": -122.1685, "confidence": 0.80},
    "bixby_bridge_big_sur": {"lat": 36.3714, "lon": -121.9028, "confidence": 0.75},

    # ── North America (expanded): Canada, Mexico, Caribbean ──────────────
    "chateau_frontenac_quebec": {"lat": 46.8131, "lon": -71.2075, "confidence": 0.85},
    "notre_dame_basilica_montreal": {"lat": 45.5045, "lon": -73.5565, "confidence": 0.80},
    "peggys_cove_lighthouse": {"lat": 44.4986, "lon": -63.9156, "confidence": 0.80},
    "hopewell_rocks_new_brunswick": {"lat": 45.8434, "lon": -64.6064, "confidence": 0.75},
    "moraine_lake_canada": {"lat": 51.3217, "lon": -116.1860, "confidence": 0.80},
    "havana_malecon": {"lat": 23.1440, "lon": -82.3830, "confidence": 0.80},
    "panama_canal_miraflores": {"lat": 8.9967, "lon": -79.5897, "confidence": 0.80},
    "tulum_ruins_mexico": {"lat": 20.2149, "lon": -87.4290, "confidence": 0.80},
    "diamond_head_hawaii": {"lat": 21.2608, "lon": -157.8060, "confidence": 0.85},
    "na_pali_coast_kauai": {"lat": 22.1667, "lon": -159.6500, "confidence": 0.80},

    # ── South America (expanded) ─────────────────────────────────────────
    "lencois_maranhenses": {"lat": -2.4833, "lon": -43.1333, "confidence": 0.75},
    "amazon_theatre_manaus": {"lat": -3.1190, "lon": -60.0217, "confidence": 0.75},
    "pelourinho_salvador": {"lat": -12.9714, "lon": -38.5014, "confidence": 0.75},
    "obelisco_buenos_aires": {"lat": -34.6037, "lon": -58.3816, "confidence": 0.80},
    "caminito_la_boca": {"lat": -34.6345, "lon": -58.3632, "confidence": 0.75},
    "monte_fitz_roy": {"lat": -49.2736, "lon": -73.0133, "confidence": 0.75},
    "valparaiso_chile": {"lat": -33.0472, "lon": -71.6127, "confidence": 0.75},
    "valle_de_la_luna_atacama": {"lat": -22.9167, "lon": -68.2833, "confidence": 0.75},
    "uros_floating_islands": {"lat": -15.8200, "lon": -69.9500, "confidence": 0.75},
    "colca_canyon_condor": {"lat": -15.6086, "lon": -71.9983, "confidence": 0.75},
    "quito_old_town": {"lat": -0.2202, "lon": -78.5123, "confidence": 0.75},

    # ── Europe (expanded): France & Alps ─────────────────────────────────
    "sacre_coeur_paris": {"lat": 48.8867, "lon": 2.3431, "confidence": 0.85},
    "pont_du_gard": {"lat": 43.9481, "lon": 4.5350, "confidence": 0.85},
    "palais_des_papes_avignon": {"lat": 43.9509, "lon": 4.8079, "confidence": 0.75},
    "etretat_cliffs": {"lat": 49.7075, "lon": 0.2044, "confidence": 0.75},
    "mont_blanc_chamonix": {"lat": 45.8326, "lon": 6.8652, "confidence": 0.80},
    "matterhorn_zermatt": {"lat": 45.9763, "lon": 7.6586, "confidence": 0.90},
    "chapel_bridge_lucerne": {"lat": 47.0517, "lon": 8.3093, "confidence": 0.80},

    # ── Europe (expanded): Italy ─────────────────────────────────────────
    "spanish_steps_rome": {"lat": 41.9056, "lon": 12.4823, "confidence": 0.85},
    "vittoriano_rome": {"lat": 41.8959, "lon": 12.4823, "confidence": 0.75},
    "roman_forum": {"lat": 41.8925, "lon": 12.4853, "confidence": 0.80},
    "milan_duomo": {"lat": 45.4642, "lon": 9.1900, "confidence": 0.90},
    "doges_palace_venice": {"lat": 45.4344, "lon": 12.3390, "confidence": 0.80},
    "rialto_bridge_venice": {"lat": 45.4380, "lon": 12.3358, "confidence": 0.80},
    "burano_colorful_houses": {"lat": 45.4853, "lon": 12.4160, "confidence": 0.75},
    "san_gimignano_towers": {"lat": 43.4678, "lon": 11.0442, "confidence": 0.75},
    "matera_sassi": {"lat": 40.6664, "lon": 16.6043, "confidence": 0.80},
    "faraglioni_capri": {"lat": 40.5497, "lon": 14.2433, "confidence": 0.75},

    # ── Europe (expanded): Central & Eastern ─────────────────────────────
    "st_vitus_cathedral_prague": {"lat": 50.0909, "lon": 14.4004, "confidence": 0.85},
    "dancing_house_prague": {"lat": 50.0749, "lon": 14.4072, "confidence": 0.75},
    "chain_bridge_budapest": {"lat": 47.4979, "lon": 19.0402, "confidence": 0.80},
    "fishermans_bastion_budapest": {"lat": 47.5069, "lon": 19.0324, "confidence": 0.80},
    "schonbrunn_palace_vienna": {"lat": 48.1855, "lon": 16.3122, "confidence": 0.85},
    "st_stephens_cathedral_vienna": {"lat": 48.2085, "lon": 16.3730, "confidence": 0.85},
    "hohensalzburg_fortress": {"lat": 47.7950, "lon": 13.0472, "confidence": 0.80},
    "rothenburg_ob_der_tauber": {"lat": 49.3767, "lon": 10.1797, "confidence": 0.75},
    "frauenkirche_dresden": {"lat": 51.0518, "lon": 13.7415, "confidence": 0.80},
    "east_side_gallery_berlin": {"lat": 52.5048, "lon": 13.4394, "confidence": 0.80},
    "berlin_tv_tower": {"lat": 52.5208, "lon": 13.4094, "confidence": 0.85},
    "sanssouci_potsdam": {"lat": 52.4041, "lon": 13.0396, "confidence": 0.80},
    "malbork_castle_poland": {"lat": 54.0397, "lon": 19.0278, "confidence": 0.75},
    "warsaw_old_town": {"lat": 52.2497, "lon": 21.0122, "confidence": 0.75},
    "diocletian_palace_split": {"lat": 43.5081, "lon": 16.4402, "confidence": 0.75},

    # ── Europe (expanded): Nordic & Baltic ───────────────────────────────
    "nyhavn_copenhagen": {"lat": 55.6798, "lon": 12.5912, "confidence": 0.85},
    "oslo_opera_house": {"lat": 59.9075, "lon": 10.7530, "confidence": 0.80},
    "bryggen_bergen": {"lat": 60.3975, "lon": 5.3236, "confidence": 0.75},
    "lofoten_islands": {"lat": 67.9333, "lon": 13.0833, "confidence": 0.75},
    "geirangerfjord": {"lat": 62.1005, "lon": 7.2065, "confidence": 0.80},
    "gamla_stan_stockholm": {"lat": 59.3251, "lon": 18.0711, "confidence": 0.80},
    "helsinki_cathedral": {"lat": 60.1699, "lon": 24.9522, "confidence": 0.80},
    "tallinn_old_town": {"lat": 59.4370, "lon": 24.7536, "confidence": 0.75},
    "house_of_blackheads_riga": {"lat": 56.9489, "lon": 24.1072, "confidence": 0.75},
    "hill_of_crosses_lithuania": {"lat": 56.1719, "lon": 23.6017, "confidence": 0.75},

    # ── Europe (expanded): UK & Ireland ──────────────────────────────────
    "glencoe_scotland": {"lat": 56.6717, "lon": -5.0106, "confidence": 0.75},
    "old_man_of_storr_skye": {"lat": 57.2464, "lon": -6.2283, "confidence": 0.80},
    "giants_causeway": {"lat": 55.2408, "lon": -6.5116, "confidence": 0.85},
    "cliffs_of_moher": {"lat": 52.9715, "lon": -9.4309, "confidence": 0.85},
    "white_cliffs_of_dover": {"lat": 51.1279, "lon": 1.3134, "confidence": 0.80},
    "brighton_palace_pier": {"lat": 50.8161, "lon": -0.1370, "confidence": 0.75},
    "housesteads_hadrians_wall": {"lat": 55.0117, "lon": -2.3050, "confidence": 0.75},
    "york_minster": {"lat": 53.9610, "lon": -1.0816, "confidence": 0.80},
    "greenwich_observatory": {"lat": 51.4769, "lon": -0.0005, "confidence": 0.75},

    # ── Europe (expanded): Iberia & Greece ───────────────────────────────
    "belem_tower_lisbon": {"lat": 38.6916, "lon": -9.2160, "confidence": 0.85},
    "pena_palace_sintra": {"lat": 38.7876, "lon": -9.3906, "confidence": 0.85},
    "dom_luis_bridge_porto": {"lat": 41.1408, "lon": -8.6110, "confidence": 0.80},
    "santiago_compostela_cathedral": {"lat": 42.8805, "lon": -8.5447, "confidence": 0.75},
    "guggenheim_bilbao": {"lat": 43.2687, "lon": -2.9340, "confidence": 0.85},
    "casa_batllo_barcelona": {"lat": 41.3917, "lon": 2.1649, "confidence": 0.80},
    "plaza_de_espana_seville": {"lat": 37.3826, "lon": -5.9866, "confidence": 0.85},
    "mezquita_cordoba": {"lat": 37.8792, "lon": -4.7793, "confidence": 0.85},
    "puente_nuevo_ronda": {"lat": 36.7406, "lon": -5.1661, "confidence": 0.80},
    "toledo_old_town": {"lat": 39.8628, "lon": -4.0273, "confidence": 0.75},
    "segovia_aqueduct": {"lat": 40.9485, "lon": -4.1180, "confidence": 0.85},

    # ── Asia (expanded): South & Central Asia ────────────────────────────
    "tigers_nest_bhutan": {"lat": 27.4917, "lon": 89.3633, "confidence": 0.85},
    "amber_fort_jaipur": {"lat": 26.9855, "lon": 75.8513, "confidence": 0.80},
    "jal_mahal_jaipur": {"lat": 26.9535, "lon": 75.8465, "confidence": 0.75},
    "india_gate_delhi": {"lat": 28.6129, "lon": 77.2295, "confidence": 0.85},
    "humayuns_tomb_delhi": {"lat": 28.5933, "lon": 77.2507, "confidence": 0.80},
    "fatehpur_sikri": {"lat": 27.0940, "lon": 77.6610, "confidence": 0.75},
    "charminar_hyderabad": {"lat": 17.3616, "lon": 78.4747, "confidence": 0.80},
    "mysore_palace": {"lat": 12.3052, "lon": 76.6552, "confidence": 0.80},
    "dal_lake_srinagar": {"lat": 34.1200, "lon": 74.8500, "confidence": 0.75},
    "pangong_lake_ladakh": {"lat": 33.7500, "lon": 78.6667, "confidence": 0.75},
    "nine_arch_bridge_ella": {"lat": 6.8669, "lon": 81.0469, "confidence": 0.75},
    "maldives_overwater_villas": {"lat": 4.1755, "lon": 73.5093, "confidence": 0.75},

    # ── Asia (expanded): Middle East & Caucasus & Turkey ─────────────────
    "kuwait_towers": {"lat": 29.3833, "lon": 48.4833, "confidence": 0.75},
    "sultan_qaboos_mosque_muscat": {"lat": 23.5859, "lon": 58.4059, "confidence": 0.75},
    "flame_towers_baku": {"lat": 40.3647, "lon": 49.8350, "confidence": 0.75},
    "mount_ararat": {"lat": 39.7020, "lon": 44.2988, "confidence": 0.75},
    "pamukkale_travertines": {"lat": 37.9203, "lon": 29.1211, "confidence": 0.85},
    "galata_tower_istanbul": {"lat": 41.0256, "lon": 28.9744, "confidence": 0.80},
    "sumela_monastery": {"lat": 40.6622, "lon": 39.6600, "confidence": 0.75},
    "khiva_itchan_kala": {"lat": 41.3775, "lon": 60.3639, "confidence": 0.75},

    # ── Asia (expanded): China, Hong Kong, Macau ─────────────────────────
    "pingyao_ancient_city": {"lat": 37.2010, "lon": 112.1760, "confidence": 0.75},
    "yungang_grottoes": {"lat": 40.1117, "lon": 113.1283, "confidence": 0.75},
    "longmen_grottoes": {"lat": 34.5590, "lon": 112.4720, "confidence": 0.80},
    "shaolin_temple": {"lat": 34.5070, "lon": 112.9290, "confidence": 0.75},
    "the_bund_shanghai": {"lat": 31.2336, "lon": 121.4906, "confidence": 0.85},
    "shanghai_tower": {"lat": 31.2336, "lon": 121.5055, "confidence": 0.80},
    "yu_garden_shanghai": {"lat": 31.2272, "lon": 121.4921, "confidence": 0.75},
    "hong_kong_skyline": {"lat": 22.2908, "lon": 114.1501, "confidence": 0.85},
    "victoria_peak_hong_kong": {"lat": 22.2759, "lon": 114.1450, "confidence": 0.85},
    "tian_tan_buddha_hong_kong": {"lat": 22.2575, "lon": 113.9050, "confidence": 0.85},
    "ruins_st_pauls_macau": {"lat": 22.1979, "lon": 113.5404, "confidence": 0.80},

    # ── Asia (expanded): Taiwan & Korea ──────────────────────────────────
    "taipei_101": {"lat": 25.0330, "lon": 121.5654, "confidence": 0.90},
    "taroko_gorge_taiwan": {"lat": 24.1783, "lon": 121.5033, "confidence": 0.75},
    "jiufen_taiwan": {"lat": 25.1097, "lon": 121.8450, "confidence": 0.75},
    "gamcheon_village_busan": {"lat": 35.0976, "lon": 129.0053, "confidence": 0.75},

    # ── Asia (expanded): Japan ───────────────────────────────────────────
    "shibuya_crossing_tokyo": {"lat": 35.6595, "lon": 139.7005, "confidence": 0.85},
    "tokyo_skytree": {"lat": 35.7101, "lon": 139.8107, "confidence": 0.85},
    "kiyomizudera_kyoto": {"lat": 34.9949, "lon": 135.7850, "confidence": 0.85},
    "shirakawa_go_village": {"lat": 36.2560, "lon": 136.9020, "confidence": 0.80},
    "jigokudani_snow_monkeys": {"lat": 36.6833, "lon": 138.5000, "confidence": 0.75},
    "blue_pond_hokkaido": {"lat": 43.5453, "lon": 142.5683, "confidence": 0.75},
    "shuri_castle_okinawa": {"lat": 26.2170, "lon": 127.7194, "confidence": 0.75},

    # ── Asia (expanded): Southeast Asia ──────────────────────────────────
    "chocolate_hills_bohol": {"lat": 9.8333, "lon": 124.1333, "confidence": 0.80},
    "mayon_volcano": {"lat": 13.2543, "lon": 123.6867, "confidence": 0.80},
    "el_nido_palawan": {"lat": 11.1800, "lon": 119.3900, "confidence": 0.75},
    "golden_bridge_da_nang": {"lat": 15.9990, "lon": 107.9160, "confidence": 0.85},
    "hoi_an_ancient_town": {"lat": 15.8801, "lon": 108.3380, "confidence": 0.80},
    "hue_imperial_city": {"lat": 16.4700, "lon": 107.5780, "confidence": 0.75},
    "bayon_temple_angkor": {"lat": 13.4410, "lon": 103.8580, "confidence": 0.80},
    "ta_prohm_temple": {"lat": 13.4340, "lon": 103.8890, "confidence": 0.80},
    "luang_prabang_laos": {"lat": 19.8834, "lon": 102.1350, "confidence": 0.75},
    "kuang_si_falls": {"lat": 19.2833, "lon": 102.1417, "confidence": 0.75},
    "wat_arun_bangkok": {"lat": 13.7437, "lon": 100.4889, "confidence": 0.85},
    "white_temple_chiang_rai": {"lat": 19.8242, "lon": 99.7633, "confidence": 0.80},
    "river_kwai_bridge_kanchanaburi": {"lat": 14.0430, "lon": 99.5030, "confidence": 0.75},
    "maya_bay_phi_phi": {"lat": 7.6783, "lon": 98.7667, "confidence": 0.80},
    "batu_caves_kuala_lumpur": {"lat": 3.2379, "lon": 101.6840, "confidence": 0.85},
    "george_town_penang": {"lat": 5.4141, "lon": 100.3288, "confidence": 0.75},
    "tanah_lot_bali": {"lat": -8.6212, "lon": 115.0868, "confidence": 0.85},
    "lempuyang_gates_of_heaven": {"lat": -8.3740, "lon": 115.5140, "confidence": 0.80},
    "raja_ampat": {"lat": -0.2346, "lon": 130.5079, "confidence": 0.75},
    "u_bein_bridge_mandalay": {"lat": 21.4250, "lon": 96.0980, "confidence": 0.80},
    "kyaiktiyo_golden_rock": {"lat": 16.9930, "lon": 97.6300, "confidence": 0.75},

    # ── Africa (expanded): North Africa ──────────────────────────────────
    "jemaa_el_fna_marrakech": {"lat": 31.6258, "lon": -7.9891, "confidence": 0.80},
    "hassan_ii_mosque_casablanca": {"lat": 33.6083, "lon": -7.6325, "confidence": 0.85},
    "ait_ben_haddou": {"lat": 31.0470, "lon": -7.1329, "confidence": 0.80},
    "erg_chebbi_dunes_merzouga": {"lat": 31.1000, "lon": -4.0000, "confidence": 0.75},
    "el_jem_amphitheatre_tunisia": {"lat": 35.2958, "lon": 10.7050, "confidence": 0.75},
    "carthage_ruins_tunis": {"lat": 36.8528, "lon": 10.3236, "confidence": 0.75},
    "sidi_bou_said": {"lat": 36.8704, "lon": 10.3474, "confidence": 0.80},
    "leptis_magna_libya": {"lat": 32.6428, "lon": 14.3367, "confidence": 0.75},
    "bibliotheca_alexandrina": {"lat": 31.2089, "lon": 29.9092, "confidence": 0.75},

    # ── Africa (expanded): East, Southern & West Africa ──────────────────
    "meroe_pyramids_sudan": {"lat": 16.9382, "lon": 33.7492, "confidence": 0.75},
    "lalibela_rock_churches": {"lat": 12.0317, "lon": 39.0417, "confidence": 0.80},
    "maasai_mara_kenya": {"lat": -1.4061, "lon": 35.0080, "confidence": 0.75},
    "sossusvlei_namibia": {"lat": -24.7333, "lon": 15.3333, "confidence": 0.80},
    "boulders_beach_penguins": {"lat": -34.1975, "lon": 18.4517, "confidence": 0.80},
    "bo_kaap_cape_town": {"lat": -33.9207, "lon": 18.4145, "confidence": 0.75},
    "avenue_of_the_baobabs": {"lat": -20.2506, "lon": 44.4183, "confidence": 0.80},
    "tsingy_de_bemaraha": {"lat": -19.1333, "lon": 44.7833, "confidence": 0.75},
    "anse_source_dargent_seychelles": {"lat": -4.3167, "lon": 55.7333, "confidence": 0.75},
    "le_morne_mauritius": {"lat": -20.4500, "lon": 57.3167, "confidence": 0.75},
    "cape_coast_castle_ghana": {"lat": 5.1054, "lon": -1.2466, "confidence": 0.75},
    "goree_island_senegal": {"lat": 14.6672, "lon": -17.3986, "confidence": 0.75},
    "timbuktu_mali": {"lat": 16.7735, "lon": -3.0074, "confidence": 0.75},

    # ── Middle East (expanded) ───────────────────────────────────────────
    "bahai_terraces_haifa": {"lat": 32.7045, "lon": 34.9900, "confidence": 0.80},
    "kingdom_centre_riyadh": {"lat": 24.7115, "lon": 46.6745, "confidence": 0.75},
    "naqsh_e_jahan_isfahan": {"lat": 32.6572, "lon": 51.6776, "confidence": 0.85},
    "si_o_se_pol_bridge_isfahan": {"lat": 32.6447, "lon": 51.6672, "confidence": 0.75},

    # ── Oceania (expanded): Australia ────────────────────────────────────
    "bondi_beach_sydney": {"lat": -33.8908, "lon": 151.2743, "confidence": 0.85},
    "kata_tjuta_olgas": {"lat": -25.2986, "lon": 131.1600, "confidence": 0.75},
    "great_ocean_road_memorial_arch": {"lat": -38.6622, "lon": 143.6722, "confidence": 0.75},
    "phillip_island_penguin_parade": {"lat": -38.5106, "lon": 145.1489, "confidence": 0.75},
    "cradle_mountain_tasmania": {"lat": -41.6608, "lon": 145.9500, "confidence": 0.75},
    "wineglass_bay_tasmania": {"lat": -42.1522, "lon": 148.2986, "confidence": 0.75},
    "whitehaven_beach_whitsundays": {"lat": -20.2822, "lon": 149.0333, "confidence": 0.80},
    "heart_reef_whitsundays": {"lat": -19.7500, "lon": 149.1667, "confidence": 0.75},
    "fraser_island_kgari": {"lat": -25.2500, "lon": 153.1500, "confidence": 0.75},
    "byron_bay_lighthouse": {"lat": -28.6386, "lon": 153.6375, "confidence": 0.75},
    "wave_rock_western_australia": {"lat": -32.4422, "lon": 118.8972, "confidence": 0.75},
    "natures_window_kalbarri": {"lat": -28.0722, "lon": 114.1472, "confidence": 0.75},
    "rottnest_island_quokkas": {"lat": -32.0083, "lon": 115.5000, "confidence": 0.75},

    # ── Oceania (expanded): New Zealand & Pacific ────────────────────────
    "church_good_shepherd_tekapo": {"lat": -44.0048, "lon": 170.4786, "confidence": 0.80},
    "wanaka_tree": {"lat": -44.7050, "lon": 169.1250, "confidence": 0.75},
    "moeraki_boulders": {"lat": -45.3453, "lon": 170.8261, "confidence": 0.75},
    "cathedral_cove_coromandel": {"lat": -36.8333, "lon": 175.8000, "confidence": 0.75},
    "mount_yasur_vanuatu": {"lat": -19.5300, "lon": 169.4400, "confidence": 0.75},
    "to_sua_ocean_trench_samoa": {"lat": -13.8333, "lon": -171.9333, "confidence": 0.75},
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

# ── Language-based geolocation hints ────────────────────────────────────────
# Maps detected OCR language → countries where it is an official/primary
# language, weighted by specificity. English is deliberately EXCLUDED (it is
# a global lingua franca and carries almost no location signal).
# Weight semantics: 1.0 = language defines the country, lower = shared.

LANGUAGE_COUNTRY_HINTS: dict[str, list[tuple[str, float]]] = {
    # Germanic (non-English)
    "de": [("DE", 1.0), ("AT", 0.8), ("CH", 0.6), ("LI", 0.5), ("LU", 0.4), ("BE", 0.3)],
    "nl": [("NL", 1.0), ("BE", 0.7), ("SR", 0.3)],
    "sv": [("SE", 1.0)],
    "no": [("NO", 1.0)],
    "nb": [("NO", 1.0)],
    "da": [("DK", 1.0)],
    "is": [("IS", 1.0)],
    "af": [("ZA", 0.7), ("NA", 0.5)],
    # Romance
    "fr": [("FR", 1.0), ("BE", 0.6), ("CH", 0.5), ("LU", 0.4), ("MC", 0.5)],
    "it": [("IT", 1.0), ("CH", 0.3), ("SM", 0.5)],
    "es": [("ES", 1.0), ("MX", 0.4), ("AR", 0.3), ("CO", 0.3), ("CL", 0.3)],
    "pt": [("PT", 1.0), ("BR", 0.6)],
    "ro": [("RO", 1.0), ("MD", 0.7)],
    "ca": [("ES", 0.8)],
    # Slavic
    "pl": [("PL", 1.0)],
    "cs": [("CZ", 1.0)],
    "sk": [("SK", 1.0)],
    "ru": [("RU", 1.0), ("BY", 0.7), ("KZ", 0.5)],
    "uk": [("UA", 1.0)],
    "bg": [("BG", 1.0)],
    "sr": [("RS", 1.0), ("BA", 0.5)],
    "hr": [("HR", 1.0), ("BA", 0.4)],
    "sl": [("SI", 1.0)],
    "mk": [("MK", 1.0)],
    "bs": [("BA", 1.0)],
    # Baltic
    "et": [("EE", 1.0)],
    "lv": [("LV", 1.0)],
    "lt": [("LT", 1.0)],
    # Other European
    "el": [("GR", 1.0), ("CY", 0.6)],
    "hu": [("HU", 1.0)],
    "fi": [("FI", 1.0)],
    "et_": [],  # placeholder guard
    "sq": [("AL", 1.0)],
    "tr": [("TR", 1.0)],
    "ka": [("GE", 1.0)],
    "hy": [("AM", 1.0)],
    "az": [("AZ", 1.0)],
    "cy": [],  # Welsh — too regional-ambiguous, skip
    # Middle East
    "he": [("IL", 1.0)],
    "fa": [("IR", 1.0)],
    "ar": [("EG", 0.5), ("SA", 0.5), ("AE", 0.4), ("MA", 0.4), ("DZ", 0.4),
           ("TN", 0.4), ("JO", 0.4), ("IQ", 0.4), ("LY", 0.4), ("KW", 0.4),
           ("QA", 0.4), ("OM", 0.4), ("LB", 0.4), ("SY", 0.4)],
    "ur": [("PK", 1.0)],
    # Asia
    "ja": [("JP", 1.0)],
    "ko": [("KR", 1.0)],
    "zh-cn": [("CN", 1.0), ("SG", 0.2)],
    "zh-tw": [("TW", 1.0), ("HK", 0.4)],
    "th": [("TH", 1.0)],
    "vi": [("VN", 1.0)],
    "id": [("ID", 1.0)],
    "ms": [("MY", 0.8), ("ID", 0.3), ("BN", 0.6)],
    "tl": [("PH", 1.0)],
    "kn": [("IN", 1.0)],
    "ta": [("IN", 0.5), ("LK", 0.5)],
    "te": [("IN", 1.0)],
    "ml": [("IN", 1.0)],
    "mr": [("IN", 1.0)],
    "gu": [("IN", 1.0)],
    "pa": [("IN", 0.7), ("PK", 0.5)],
    "bn": [("BD", 0.9), ("IN", 0.4)],
    "ne": [("NP", 1.0)],
    "si": [("LK", 1.0)],
    "my": [("MM", 1.0)],
    "km": [("KH", 1.0)],
    "lo": [("LA", 1.0)],
    "mn": [("MN", 1.0)],
    "kk": [("KZ", 1.0)],
    "uz": [("UZ", 1.0)],
}

# Weight multiplier applied to language votes overall — OCR text is short and
# langdetect can misfire on fragments, so keep these hints modest.
_LANGUAGE_VOTE_SCALE = 0.8


def language_country_votes(text: str) -> dict[str, float]:
    """Detect the language of `text` and return weighted country-code votes.

    Returns {} when detection is unreliable (too little text, detection
    failure, or English/global languages).
    """
    if not text or len(text.strip()) < 20:
        return {}

    try:
        from langdetect import detect_langs, DetectorFactory
        DetectorFactory.seed = 42  # deterministic
        candidates = detect_langs(text)
    except Exception:
        return {}

    votes: dict[str, float] = {}
    for cand in candidates[:2]:
        lang_code = getattr(cand, "lang", None) or (
            cand[0] if isinstance(cand, tuple) else str(cand)
        )
        prob = float(getattr(cand, "prob", None) or (
            cand[1] if isinstance(cand, tuple) and len(cand) > 1 else 0.0
        ))
        if prob < 0.5:
            continue
        hints = LANGUAGE_COUNTRY_HINTS.get(lang_code)
        if not hints:
            continue  # includes "en" — deliberately unmapped
        for cc, weight in hints:
            votes[cc] = votes.get(cc, 0.0) + weight * prob * _LANGUAGE_VOTE_SCALE
    return votes

# ── Country centroids (ISO2 → lat, lon) ─────────────────────────────────────
# Comprehensive table for language/plate/currency hints. Approximate centers.

COUNTRY_CENTROIDS: dict[str, tuple[float, float]] = {
    "DE": (51.17, 10.45), "AT": (47.52, 14.55), "CH": (46.82, 8.23),
    "LI": (47.17, 9.56), "LU": (49.82, 6.13), "BE": (50.50, 4.47),
    "FR": (46.23, 2.21), "MC": (43.73, 7.42), "NL": (52.13, 5.29),
    "IT": (41.87, 12.57), "SM": (43.94, 12.46), "VA": (41.90, 12.45),
    "ES": (40.46, -3.75), "PT": (39.40, -8.22), "BR": (-14.24, -51.93),
    "PL": (51.92, 19.15), "CZ": (49.82, 15.47), "SK": (48.67, 19.70),
    "RU": (61.52, 105.32), "BY": (53.71, 27.95), "KZ": (48.02, 66.92),
    "UA": (48.38, 31.17), "BG": (42.73, 25.49), "RS": (44.02, 21.01),
    "BA": (43.92, 17.68), "HR": (45.10, 15.20), "SI": (46.15, 14.99),
    "MK": (41.61, 21.75), "ME": (42.71, 19.37), "AL": (41.15, 20.17),
    "XK": (42.60, 20.90), "GR": (39.07, 21.82), "CY": (35.13, 33.43),
    "TR": (38.96, 35.24), "GE": (42.32, 43.36), "AM": (40.07, 45.04),
    "AZ": (40.14, 47.58), "SE": (60.13, 18.64), "NO": (60.47, 8.47),
    "DK": (56.26, 9.50), "FI": (61.92, 25.75), "IS": (64.96, -19.02),
    "EE": (58.60, 25.01), "LV": (56.88, 24.60), "LT": (55.17, 23.88),
    "IE": (53.14, -7.69), "GB": (55.38, -3.44), "US": (37.09, -95.71),
    "CA": (56.13, -106.35), "MX": (23.63, -102.55), "GT": (15.78, -90.23),
    "CU": (21.52, -77.78), "JM": (18.11, -77.30), "DO": (18.74, -70.16),
    "HN": (15.20, -86.24), "NI": (12.87, -85.21), "CR": (9.75, -83.75),
    "PA": (8.54, -80.78), "CO": (4.57, -74.30), "VE": (6.42, -66.59),
    "EC": (-1.83, -78.18), "PE": (-9.19, -75.02), "BO": (-16.29, -63.59),
    "CL": (-35.68, -71.54), "AR": (-38.42, -63.62), "UY": (-32.52, -55.77),
    "PY": (-23.44, -58.44), "MA": (31.79, -7.09), "DZ": (28.03, 1.66),
    "TN": (33.89, 9.54), "LY": (26.34, 17.23), "EG": (26.82, 30.80),
    "SA": (23.89, 45.08), "AE": (23.42, 53.85), "KW": (29.31, 47.48),
    "QA": (25.35, 51.18), "OM": (21.51, 55.92), "LB": (33.85, 35.86),
    "SY": (34.80, 38.99), "JO": (30.59, 36.24), "IQ": (33.22, 43.68),
    "IL": (31.05, 34.85), "IR": (32.43, 53.69), "PK": (30.38, 69.35),
    "JP": (36.20, 138.25), "KR": (35.91, 127.77), "KP": (40.34, 127.51),
    "CN": (35.86, 104.20), "TW": (23.70, 120.96), "HK": (22.32, 114.17),
    "SG": (1.35, 103.82), "TH": (15.87, 100.99), "VN": (14.06, 108.28),
    "ID": (-0.79, 113.92), "MY": (4.21, 101.98), "BN": (4.54, 114.73),
    "PH": (12.88, 121.77), "TL": (-8.87, 125.73), "MM": (21.91, 95.96),
    "KH": (12.57, 104.99), "LA": (19.86, 102.50), "IN": (20.59, 78.96),
    "LK": (7.87, 80.77), "NP": (28.39, 84.12), "BT": (27.51, 90.43),
    "BD": (23.68, 90.36), "MN": (46.86, 103.85), "UZ": (41.38, 64.59),
    "KG": (41.20, 74.77), "TJ": (38.86, 71.28), "TM": (38.97, 59.56),
    "AF": (33.94, 67.71), "ZA": (-30.56, 22.94), "NA": (-22.96, 18.49),
    "ZW": (-19.02, 29.15), "BW": (-22.33, 24.68), "MZ": (-18.67, 35.53),
    "AO": (-11.20, 17.87), "ZM": (-13.13, 27.85), "MW": (-13.25, 34.30),
    "TZ": (-6.37, 34.89), "KE": (-0.02, 37.91), "UG": (1.37, 32.29),
    "RW": (-1.94, 29.87), "ET": (9.15, 40.49), "GH": (7.95, -1.02),
    "NG": (9.08, 8.68), "CI": (7.54, -5.55), "SN": (14.50, -14.45),
    "AU": (-25.27, 133.78), "NZ": (-40.90, 174.89), "FJ": (-17.71, 178.07),
    "PG": (-6.31, 143.96),
}
