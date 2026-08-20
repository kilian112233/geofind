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
