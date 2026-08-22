"""Places365 scene classifier module — classifies images into 365 scene categories
and maps them to geographic coordinates."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from geofind.core.candidate import ModuleHit
from geofind.core.config import PipelineConfig
from geofind.modules.base import BaseModule

logger = logging.getLogger(__name__)

# ImageNet normalization
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Geographic distributions for scene categories.
# Each entry: {scene_index: [(lat, lon, weight), ...]}
# Weight is the probability mass assigned to that coordinate for this scene.
_SCENE_GEOGRAPHY: dict[int, list[tuple[float, float, float]]] = {
    # ── Southeast Asia / Tropics ──────────────────────────────────────────
    24:   [(25, 105, 0.7), (15, 100, 0.8), (35, 135, 0.5)],   # bamboo_forest
    229:  [(0, 110, 0.8), (-5, -60, 0.7), (5, 30, 0.6)],      # rainforest
    236:  [(20, 105, 0.9), (10, 100, 0.8), (25, 120, 0.7)],   # rice_paddy
    275:  [(35, 135, 0.8), (37, 127, 0.7), (22, 114, 0.6)],   # teahouse
    257:  [(19, 73, 0.8), (-23, -43, 0.7), (-6, 107, 0.6)],   # slum
    213:  [(35, 136, 0.7), (37, 127, 0.6), (22, 114, 0.5)],   # phone_booth (yellow UK vs red JP vs HK)
    112:  [(0, 110, 0.7), (10, 77, 0.6), (-5, -60, 0.5)],     # field/wild
    178:  [(0, 110, 0.7), (-5, -60, 0.6), (5, 30, 0.5)],      # marsh
    270:  [(0, 110, 0.7), (-5, -60, 0.6), (5, 30, 0.5)],      # swamp
    194:  [(35, 135, 0.7), (37, 127, 0.5), (10, 100, 0.4)],   # nursery
    41:   [(0, 110, 0.6), (35, 135, 0.5), (-34, 18, 0.4)],    # botanical_garden

    # ── Temple / Religious ────────────────────────────────────────────────
    277:  [(35, 135, 0.9), (30, 100, 0.8), (15, 100, 0.7)],   # temple/east_asia
    278:  [(25, 80, 0.9), (15, 77, 0.8), (20, 96, 0.7)],      # temple/south_asia
    279:  [(32, 35, 0.8), (30, 45, 0.7), (25, 55, 0.8)],      # temple/mideast
    185:  [(24, 46, 0.8), (30, 31, 0.7), (33, 44, 0.8)],      # mosque/outdoor
    186:  [(24, 46, 0.8), (30, 31, 0.7), (33, 44, 0.8)],      # mosque/outdoor (duplicate index? keep)
    66:   [(48, 2, 0.6), (51, -1, 0.5), (40, -74, 0.5)],      # church/outdoor
    274:  [(40, -74, 0.7), (50, 14, 0.5), (32, 35, 0.6)],     # synagogue/outdoor
    217:  [(21.4, 39.8, 0.9), (32, 35, 0.8)],                 # pilgrimage
    159:  [(35, 135, 0.8), (30, 100, 0.7), (22, 114, 0.6)],   # joss_house
    311:  [(35, 135, 0.9), (37, 127, 0.6)],                    # zen_garden
    157:  [(35, 135, 0.9), (37, 127, 0.5)],                    # japanese_garden

    # ── Middle East / Desert ──────────────────────────────────────────────
    90:   [(25, 45, 0.8), (23, 55, 0.9), (30, 0, 0.5)],       # desert/sand
    92:   [(37, -117, 0.6), (25, 45, 0.7), (-23, 135, 0.5)],  # desert/rock
    91:   [(34, 40, 0.7), (33, -112, 0.6), (-25, 130, 0.5)],  # desert/vegetation
    182:  [(34, -4, 0.8), (30, 31, 0.7), (24, 46, 0.7)],      # medina
    183:  [(48, 2, 0.5), (30, 31, 0.5), (40, -74, 0.4)],      # memorial/monument
    61:   [(42, 12, 0.7), (32, 35, 0.6), (30, 31, 0.5)],      # catacomb
    55:   [(45, 12, 0.8), (50, 14, 0.6), (48, 2, 0.5)],       # canal/urban
    134:  [(40, -4, 0.7), (38, -5, 0.6), (44, 8, 0.5)],       # grotto
    154:  [(25, 45, 0.6), (33, -112, 0.5), (52, 0, 0.4)],     # industrial_area

    # ── European ──────────────────────────────────────────────────────────
    60:   [(47, 10, 0.7), (51, -1, 0.6), (40, -4, 0.5)],      # castle
    193:  [(40, -4, 0.7), (45, 12, 0.8), (50, 14, 0.6)],      # narrow_street
    45:   [(48, 2, 0.7), (41, -82, 0.6), (52, 13, 0.6)],      # building_facade
    202:  [(48, 2, 0.7), (40, -4, 0.6), (51, 0, 0.8)],        # palace
    296:  [(47, 10, 0.7), (50, 5, 0.6), (46, 15, 0.6)],       # village
    21:   [(48, 2, 0.6), (50, 14, 0.5), (40, -74, 0.4)],      # bakery/shop
    22:   [(48, 2, 0.6), (51, -1, 0.5), (45, 12, 0.5)],       # balcony/exterior
    175:  [(48, 2, 0.5), (40, -74, 0.5), (51, 0, 0.4)],       # mansion
    294:  [(48, 2, 0.5), (51, -1, 0.5), (45, 12, 0.4)],       # veranda
    36:   [(48, 2, 0.5), (34, -118, 0.4), (40, -74, 0.4)],    # boardwalk
    37:   [(48, 2, 0.5), (45, 12, 0.5), (34, 132, 0.4)],      # boat_deck
    82:   [(48, 2, 0.6), (40, -4, 0.6), (45, 12, 0.5)],       # courtyard
    125:  [(48, 2, 0.5), (40, -4, 0.4), (51, -1, 0.4)],       # fountain
    20:   [(48, 2, 0.5), (43, -110, 0.6), (35, -105, 0.5)],   # badlands
    54:   [(48, 2, 0.5), (52, 5, 0.5), (45, 12, 0.5)],        # canal/natural
    142:  [(48, 2, 0.6), (51, -1, 0.5), (45, 12, 0.5)],       # home_office

    # ── North American ────────────────────────────────────────────────────
    207:  [(38, -97, 0.6), (40, -74, 0.5), (34, -118, 0.5)],  # parking_lot
    99:   [(-34, 151, 0.5), (40, -74, 0.6), (41, -88, 0.5)],  # downtown
    130:  [(38, -97, 0.7), (40, -74, 0.6), (34, -118, 0.6)],  # gas_station
    256:  [(40, -74, 0.7), (41, -88, 0.6), (37, -122, 0.5)],  # skyscraper
    146:  [(38, -97, 0.6), (33, -84, 0.5), (29, -95, 0.5)],   # house
    107:  [(34, -118, 0.7), (40, -74, 0.5), (41, -88, 0.5)],  # freeway
    297:  [(38, -122, 0.9), (45, -123, 0.7), (-33, 19, 0.5)], # vineyard
    253:  [(46, 7, 0.7), (43, -114, 0.8), (65, 15, 0.5)],    # ski_resort
    254:  [(46, 7, 0.6), (43, -114, 0.7), (65, 15, 0.5)],    # ski_slope
    128:  [(38, -97, 0.6), (34, -118, 0.5), (40, -74, 0.5)],  # garage/outdoor
    129:  [(38, -97, 0.7), (40, -74, 0.6), (34, -118, 0.6)],  # gas_station (dup ok)
    81:   [(38, -97, 0.5), (40, -74, 0.5), (34, -118, 0.4)],  # cottage
    111:  [(38, -97, 0.6), (40, -74, 0.5), (34, -118, 0.5)],  # fast_food_restaurant
    168:  [(38, -97, 0.5), (40, -74, 0.5), (34, -118, 0.4)],  # laundromat
    225:  [(51, -1, 0.5), (48, 2, 0.5), (38, -97, 0.4)],     # pub/indoor
    206:  [(38, -97, 0.5), (34, -118, 0.5), (40, -74, 0.4)],  # parking_garage/outdoor
    79:   [(38, -97, 0.4), (34, -118, 0.4), (52, 13, 0.3)],   # construction_site

    # ── South American ────────────────────────────────────────────────────
    258:  [(-23, -43, 0.9), (-4, -79, 0.7), (-15, -47, 0.6)], # slum (favela)
    184:  [(-23, -43, 0.7), (19, -99, 0.6), (10, -67, 0.5)],  # mews
    177:  [(-23, -43, 0.7), (-4, -79, 0.6), (10, -67, 0.5)],  # market/outdoor
    234:  [(-23, -43, 0.6), (40, -74, 0.5), (48, 2, 0.5)],    # restaurant
    210:  [(-23, -43, 0.6), (-4, -79, 0.5), (10, -67, 0.4)],  # patio

    # ── Japanese ──────────────────────────────────────────────────────────
    159:  [(35, 135, 0.8), (30, 100, 0.7), (22, 114, 0.6)],   # joss_house
    282:  [(35, 135, 0.8), (37, 127, 0.7), (22, 114, 0.6)],   # toyshop

    # ── Ocean / Coastal ───────────────────────────────────────────────────
    31:   [(25, -80, 0.5), (-34, 18, 0.5), (35, 140, 0.4)],   # beach
    137:  [(40, -74, 0.5), (-34, 18, 0.4), (35, 139, 0.5)],   # harbor
    196:  [(35, 140, 0.4), (-34, 18, 0.3), (25, -80, 0.3)],   # ocean
    216:  [(37, -122, 0.7), (34, -118, 0.5), (51, 0, 0.4)],   # pier
    144:  [(35, 140, 0.5), (-34, 18, 0.5), (25, -80, 0.4)],   # hot_spring
    215:  [(45, 12, 0.7), (40, -4, 0.6), (48, 2, 0.5)],       # piazza
    164:  [(25, -80, 0.4), (-34, 18, 0.4), (35, 140, 0.3)],   # lagoon
    44:   [(51, -1, 0.5), (40, -74, 0.5), (37, -122, 0.4)],   # bridge
    83:   [(48, 2, 0.5), (51, -1, 0.5), (47, 10, 0.4)],       # covered_bridge

    # ── Mountain / Nature ─────────────────────────────────────────────────
    187:  [(27, 86, 0.7), (46, 10, 0.6), (-43, 170, 0.5)],    # mountain
    188:  [(27, 86, 0.6), (46, 10, 0.6), (-43, 170, 0.4)],    # mountain_path
    301:  [(-20, -44, 0.7), (6, -62, 0.6), (-34, 18, 0.5)],   # waterfall
    119:  [(35, 140, 0.6), (-34, 18, 0.5), (25, -80, 0.4)],   # water_park
    239:  [(48, 2, 0.5), (37, -117, 0.4), (27, 86, 0.4)],     # rock_arch
    292:  [(46, 10, 0.6), (43, -114, 0.5), (27, 86, 0.5)],    # valley
    298:  [(65, -17, 0.6), (38, 140, 0.6), (-43, -70, 0.5)],  # volcano
    70:   [(27, 86, 0.6), (46, 10, 0.5), (-43, 170, 0.4)],    # cliff
    141:  [(46, 10, 0.5), (43, -114, 0.5), (27, 86, 0.4)],    # hill
    289:  [(65, -17, 0.6), (68, 20, 0.5), (70, -40, 0.4)],    # tundra
    149:  [(75, -40, 0.6), (68, 20, 0.5), (65, -17, 0.4)],    # ice_floe
    150:  [(75, -40, 0.6), (68, 20, 0.5), (65, -17, 0.4)],    # ice_shelf
    153:  [(75, -40, 0.6), (68, -20, 0.5)],                    # igloo
    259:  [(46, 10, 0.6), (43, -114, 0.5), (65, 15, 0.5)],    # snowfield

    # ── Urban general ─────────────────────────────────────────────────────
    140:  [(34, -118, 0.6), (40, -74, 0.5), (41, -88, 0.5)],  # highway
    123:  [(34, -118, 0.6), (40, -74, 0.5), (41, -88, 0.5)],  # freeway (dup)
    85:   [(38, -97, 0.5), (40, -74, 0.5), (34, -118, 0.4)],  # crosswalk
    267:  [(40, -74, 0.6), (41, -88, 0.5), (37, -122, 0.5)],  # subway_station/platform
    15:   [(48, 2, 0.5), (40, -74, 0.5), (51, -1, 0.4)],      # art_studio
    14:   [(48, 2, 0.5), (40, -74, 0.5), (51, -1, 0.4)],      # art_gallery
    190:  [(48, 2, 0.5), (40, -74, 0.5), (51, -1, 0.4)],      # museum/indoor
    191:  [(48, 2, 0.5), (40, -74, 0.5), (51, -1, 0.4)],      # museum/outdoor
    39:   [(48, 2, 0.5), (40, -74, 0.5), (34, -118, 0.4)],    # bookstore
    124:  [(38, -97, 0.5), (34, -118, 0.5), (40, -74, 0.4)],  # front_yard
    131:  [(38, -97, 0.5), (40, -74, 0.5), (34, -118, 0.4)],  # golf_course
    204:  [(38, -97, 0.5), (40, -74, 0.5), (34, -118, 0.4)],  # park
    209:  [(48, 2, 0.5), (40, -74, 0.5), (51, -1, 0.4)],      # path/nature
    181:  [(48, 2, 0.5), (40, -74, 0.5), (51, -1, 0.4)],      # meadow
    293:  [(48, 2, 0.4), (38, -97, 0.4), (40, -74, 0.3)],     # vegetable_garden
    117:  [(40, -74, 0.6), (37, -122, 0.5), (-34, 18, 0.4)],  # fishing_pier
    251:  [(38, -97, 0.5), (34, -118, 0.5), (40, -74, 0.4)],  # shopping_mall/indoor
    268:  [(38, -97, 0.5), (34, -118, 0.5), (40, -74, 0.4)],  # supermarket
    228:  [(38, -97, 0.4), (34, -118, 0.4), (51, -1, 0.3)],   # railroad_track
    170:  [(48, 2, 0.5), (51, -1, 0.5), (40, -74, 0.4)],      # library/indoor
    219:  [(38, -97, 0.5), (34, -118, 0.4), (40, -74, 0.4)],  # playground
    100:  [(48, 2, 0.5), (40, -74, 0.5), (51, -1, 0.4)],      # dressing_room
    148:  [(38, -97, 0.5), (40, -74, 0.5), (34, -118, 0.4)],  # ice_cream_parlor
    38:   [(40, -74, 0.6), (51, -1, 0.5), (-34, 18, 0.4)],    # boathouse
    57:   [(38, -97, 0.5), (40, -74, 0.5), (34, -118, 0.4)],  # car_interior
    189:  [(38, -97, 0.6), (34, -118, 0.5), (40, -74, 0.4)],  # muscle_car
    180:  [(40, -74, 0.6), (48, 2, 0.5), (35, 135, 0.4)],     # mausoleum

    # ── Agricultural / Rural ──────────────────────────────────────────────
    26:   [(38, -97, 0.6), (47, 10, 0.5), (-33, 148, 0.4)],   # barn
    138:  [(38, -97, 0.5), (47, 10, 0.5), (-33, 148, 0.4)],   # hayfield
    305:  [(38, -97, 0.6), (47, 10, 0.5), (-33, 148, 0.4)],   # wheat_field
    286:  [(38, -97, 0.5), (47, 10, 0.5), (-33, 148, 0.4)],   # tree_farm
    132:  [(38, -97, 0.5), (34, -118, 0.5), (40, -74, 0.4)],  # greenhouse/indoor
    200:  [(38, -97, 0.5), (47, 10, 0.4), (-33, 148, 0.3)],   # orchard
    133:  [(38, -97, 0.5), (47, 10, 0.4), (-33, 148, 0.3)],   # greenhouse/outdoor
    143:  [(38, -97, 0.6), (34, -118, 0.4), (47, 10, 0.4)],   # horse_ranch
    120:  [(38, -97, 0.5), (47, 10, 0.5), (-33, 148, 0.4)],   # forest/broadleaf
    121:  [(38, -97, 0.5), (47, 10, 0.5), (-33, 148, 0.4)],   # forest_path
    122:  [(38, -97, 0.5), (47, 10, 0.5), (-33, 148, 0.4)],   # forest_road

    # ── Water features ────────────────────────────────────────────────────
    238:  [(48, 2, 0.5), (40, -74, 0.5), (51, -1, 0.4)],      # river
    165:  [(48, 2, 0.5), (40, -74, 0.4), (51, -1, 0.4)],      # lake/natural
    221:  [(48, 2, 0.5), (40, -74, 0.4), (51, -1, 0.4)],      # pond
    233:  [(48, 2, 0.5), (40, -74, 0.4)],                      # reflecting_pool
    214:  [(48, 2, 0.4), (40, -74, 0.4), (51, -1, 0.3)],      # physics_lab
    87:   [(48, 2, 0.5), (35, 135, 0.4), (40, -74, 0.4)],     # dam
    300:  [(25, -80, 0.4), (-34, 18, 0.4), (35, 140, 0.3)],   # water_park

    # ── Nordic / Arctic ───────────────────────────────────────────────────
    307:  [(52, 5, 0.7), (48, 2, 0.5), (55, 12, 0.6)],        # windmill
    283:  [(52, 5, 0.5), (40, -74, 0.4), (34, -118, 0.4)],    # track
    310:  [(48, 2, 0.5), (51, -1, 0.4), (47, 10, 0.4)],       # youth_hostel
    306:  [(52, 5, 0.6), (48, 2, 0.4), (-33, 148, 0.4)],      # wind_farm

    # ── India / South Asia ────────────────────────────────────────────────
    278:  [(25, 80, 0.9), (15, 77, 0.8), (20, 96, 0.7)],      # temple/south_asia
    257:  [(19, 73, 0.8), (-23, -43, 0.7), (-6, 107, 0.6)],   # slum
    25:   [(19, 73, 0.6), (28, 77, 0.5), (22, 114, 0.4)],     # bar
    75:   [(19, 73, 0.5), (28, 77, 0.5), (22, 114, 0.4)],     # coffee_shop

    # ── Misc / Universal ──────────────────────────────────────────────────
    0:    [(48, 2, 0.5), (51, -1, 0.5), (40, -74, 0.4)],      # abbey
    12:   [(48, 2, 0.5), (40, -74, 0.5), (51, -1, 0.4)],      # arena/performance
    11:   [(48, 2, 0.5), (40, -74, 0.5), (51, -1, 0.4)],      # archive
    281:  [(48, 2, 0.5), (40, -74, 0.5), (51, -1, 0.4)],      # tower
    244:  [(38, -97, 0.5), (34, -118, 0.4), (40, -74, 0.4)],  # runway
    139:  [(38, -97, 0.4), (34, -118, 0.4), (40, -74, 0.3)],  # heliport
    86:   [(48, 2, 0.4), (40, -74, 0.4), (34, -118, 0.3)],    # cubicle/office
    197:  [(48, 2, 0.4), (40, -74, 0.4), (51, -1, 0.3)],      # office
    198:  [(48, 2, 0.4), (40, -74, 0.4), (51, -1, 0.3)],      # office_cubicles
    163:  [(35, 135, 0.6), (37, 127, 0.5), (22, 114, 0.4)],   # labyrinth/outdoor
    64:   [(48, 2, 0.4), (40, -74, 0.4), (51, -1, 0.3)],      # chemistry_lab
    171:  [(48, 2, 0.5), (51, -1, 0.5), (40, -74, 0.4)],      # library/outdoor
    30:   [(38, -97, 0.5), (40, -74, 0.5), (34, -118, 0.4)],  # bathroom
    33:   [(38, -97, 0.5), (40, -74, 0.5), (34, -118, 0.4)],  # bedroom
    94:   [(48, 2, 0.5), (40, -74, 0.5), (34, -118, 0.4)],    # dining_room
    162:  [(38, -97, 0.5), (40, -74, 0.5), (34, -118, 0.4)],  # kitchen
    51:   [(38, -97, 0.5), (40, -74, 0.5), (34, -118, 0.4)],  # cafeteria
    118:  [(48, 2, 0.5), (40, -74, 0.5), (34, -118, 0.4)],    # florist_shop/indoor
    158:  [(48, 2, 0.5), (40, -74, 0.5), (51, -1, 0.4)],      # jewelry_shop
    280:  [(48, 2, 0.5), (40, -74, 0.5), (34, -118, 0.4)],    # terrace
    220:  [(48, 2, 0.5), (40, -74, 0.5), (34, -118, 0.4)],    # plaza
}

# Scene names for logging
_SCENE_NAMES: dict[int, str] = {
    0: 'abbey', 24: 'bamboo_forest', 31: 'beach', 36: 'boardwalk',
    41: 'botanical_garden', 44: 'bridge', 45: 'building_facade',
    60: 'castle', 66: 'church/outdoor', 70: 'cliff', 82: 'courtyard',
    85: 'crosswalk', 90: 'desert/sand', 91: 'desert/vegetation',
    92: 'desert/rock', 99: 'downtown', 107: 'freeway', 112: 'field/wild',
    120: 'forest/broadleaf', 125: 'fountain', 130: 'gas_station',
    134: 'grotto', 137: 'harbor', 140: 'highway', 141: 'hill',
    144: 'hot_spring', 146: 'house', 154: 'industrial_area',
    157: 'japanese_garden', 163: 'labyrinth/outdoor', 164: 'lagoon',
    165: 'lake/natural', 178: 'marsh', 182: 'medina', 185: 'mosque/outdoor',
    187: 'mountain', 188: 'mountain_path', 193: 'narrow_street',
    196: 'ocean', 202: 'palace', 204: 'park', 207: 'parking_lot',
    216: 'pier', 217: 'pilgrimage', 219: 'playground', 229: 'rainforest',
    236: 'rice_paddy', 238: 'river', 239: 'rock_arch', 253: 'ski_resort',
    256: 'skyscraper', 257: 'slum', 258: 'slum/favela', 267: 'subway',
    270: 'swamp', 274: 'synagogue/outdoor', 275: 'teahouse',
    277: 'temple/east_asia', 278: 'temple/south_asia', 279: 'temple/mideast',
    281: 'tower', 289: 'tundra', 292: 'valley', 296: 'village',
    297: 'vineyard', 298: 'volcano', 301: 'waterfall',
}


class Places365Module(BaseModule):
    """Classifies images into 365 scene categories and maps to geographic coordinates."""

    name = "places365"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)
        self._model: Any = None
        self._classes: list[str] | None = None

    def is_available(self) -> bool:
        try:
            import torchvision  # noqa: F401
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    _WEIGHTS_FILENAME = "resnet50_places365.pth.tar"
    _WEIGHTS_URLS = [
        "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar",
        "https://huggingface.co/CepiPerez/Places365/resolve/main/resnet50_places365.pth.tar",
    ]

    def prepare(self) -> None:
        """Load Places365 ResNet50 model."""
        import torch
        import torchvision.models as models

        try:
            model = models.resnet50(num_classes=365)
            weights_dir = self.config.models_dir / "places365"
            local_path = weights_dir / self._WEIGHTS_FILENAME
            weights_dir.mkdir(parents=True, exist_ok=True)

            state_dict = None
            if local_path.is_file():
                try:
                    state_dict = torch.load(local_path, map_location="cpu", weights_only=False)
                    self._log(f"loaded Places365 weights from {local_path}")
                except Exception as e:
                    self._log(f"local weights at {local_path} unreadable ({e}) — falling back to download", logging.WARNING)
                    state_dict = None

            if state_dict is None:
                for url in self._WEIGHTS_URLS:
                    try:
                        state_dict = torch.hub.load_state_dict_from_url(
                            url, map_location="cpu", weights_dir=str(weights_dir)
                        )
                        self._log(f"downloaded Places365 weights from {url.split('/')[2]}")
                        break
                    except Exception as e:
                        self._log(f"failed to download from {url.split('/')[2]}: {e}", logging.DEBUG)
                        continue

            if state_dict is None:
                self._log("could not download Places365 weights — module unavailable", logging.WARNING)
                return

            # Handle different checkpoint formats
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]

            # Strip "module." prefix if present (DataParallel wrapping)
            cleaned = {}
            for k, v in state_dict.items():
                cleaned[k.replace("module.", "")] = v

            model.load_state_dict(cleaned, strict=False)
            model.eval()
            self._model = model
            self._log("Places365 ResNet50 loaded successfully")
            super().prepare()

        except Exception as e:
            self._log(f"failed to load Places365 model: {e}", logging.ERROR)

    def detect(
        self,
        media_path: Path,
        *,
        frames: list[Any] | None = None,
        audio_path: Path | None = None,
    ) -> list[ModuleHit]:
        if not self._ready or self._model is None:
            return []

        import torch

        image = self._get_image(media_path, frames)
        if image is None:
            return []

        try:
            # Preprocess: resize to 224x224, to tensor, normalize
            image = image.resize((224, 224), Image.BILINEAR)
            img_array = np.array(image, dtype=np.float32) / 255.0
            img_array = (img_array - _MEAN) / _STD
            img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0)

            # Forward pass
            with torch.no_grad():
                output = self._model(img_tensor)
                probs = torch.nn.functional.softmax(output, dim=1).squeeze().numpy()

            # Find scenes above threshold
            hits: list[ModuleHit] = []
            seen_coords: list[tuple[float, float]] = []

            # Get top-10 scene indices
            top_indices = np.argsort(probs)[-10:][::-1]

            for idx in top_indices:
                scene_prob = float(probs[idx])
                if scene_prob < 0.03:  # 3% minimum
                    continue

                geo = _SCENE_GEOGRAPHY.get(int(idx))
                if not geo:
                    continue

                scene_name = _SCENE_NAMES.get(int(idx), f"scene_{idx}")

                for lat, lon, geo_weight in geo:
                    combined_conf = scene_prob * geo_weight

                    if combined_conf < 0.01:
                        continue

                    # Skip if we already have a hit very close to this
                    too_close = False
                    for slat, slon in seen_coords:
                        dlat = abs(lat - slat)
                        dlon = abs(lon - slon)
                        if dlat < 2.0 and dlon < 2.0:
                            too_close = True
                            break

                    if not too_close:
                        hits.append(self._make_hit(
                            lat, lon,
                            min(combined_conf, 0.9),
                            sigma_km=400.0,
                            scene=scene_name,
                            scene_prob=scene_prob,
                        ))
                        seen_coords.append((lat, lon))

            # Cap at 8 hits
            hits.sort(key=lambda h: h.confidence, reverse=True)
            return hits[:8]

        except Exception as e:
            self._log(f"detection failed: {e}", logging.ERROR)
            return []

    def _get_image(self, media_path: Path, frames: list[Any] | None) -> Image.Image | None:
        if frames:
            f = frames[0]
            if isinstance(f, Image.Image):
                return f.convert("RGB")
            try:
                return Image.fromarray(f).convert("RGB")
            except Exception:
                pass

        try:
            return Image.open(str(media_path)).convert("RGB")
        except Exception:
            return None
