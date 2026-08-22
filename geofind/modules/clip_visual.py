"""CLIP Visual Scene Classification module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from geofind.core.candidate import ModuleHit
from geofind.core.config import PipelineConfig
from geofind.modules.base import BaseModule

logger = logging.getLogger(__name__)

_BIOME_CENTROIDS: dict[str, tuple[float, float]] = {
    "tropical_rainforest": (0.0, 40.0),
    "desert": (25.0, 30.0),
    "temperate_forest": (42.0, 10.0),
    "boreal_forest": (60.0, 0.0),
    "tundra": (72.0, 0.0),
    "mediterranean": (37.0, 15.0),
    "grassland": (40.0, -40.0),
    "subtropical": (27.0, 60.0),
}

_CITY_PROMPTS: dict[str, list[str]] = {
    "tokyo": [
        "a busy Tokyo street with neon signs",
        "Japanese urban street scene with kanji",
        "Tokyo Shibuya crossing with crowds",
        "Japanese cityscape with Tokyo Tower",
        "neon-lit Japanese alley at night",
        "Akihabara electric town with anime billboards",
        "Harajuku Takeshita street fashion",
        "Tokyo Ginza upscale shopping district",
        "Japanese temple surrounded by modern buildings",
    ],
    "paris": [
        "a Parisian street with Haussmann architecture",
        "Parisian café with wrought-iron chairs",
        "Paris boulevard with classic limestone buildings",
        "a view of Paris rooftops with zinc grey tiles",
        "Champs-Élysées looking towards the Arc de Triomphe",
    ],
    "new_york": [
        "Manhattan skyline with skyscrapers",
        "New York City yellow taxi cab",
        "Brooklyn Bridge with Manhattan in the background",
        "Times Square billboards at night",
        "Central Park surrounded by NYC high-rises",
    ],
    "london": [
        "London red double-decker bus",
        "Big Ben clock tower at dusk",
        "London street with black cab and bobbies",
        "Thames riverbank with Tower Bridge",
        "London Underground sign at a station",
    ],
    "rome": [
        "Roman cobblestone street with ochre buildings",
        "Rome piazza with baroque fountain",
        "narrow Italian alley with laundry hanging",
        "Rome street with Vespa scooter",
        "ancient Roman ruins in a modern city",
    ],
    "bangkok": [
        "Thai street food market at night",
        "Bangkok tuk-tuk on a busy road",
        "golden Thai temple spires among city buildings",
        "busy Bangkok street with overhead wires",
        "Thai floating market with boats",
    ],
    "istanbul": [
        "Istanbul skyline with minarets and domes",
        "Turkish bazaar with spices and lanterns",
        "Istanbul street with tram and mosque",
        "Bosphorus view with ferries",
        "Turkish tea in a tulip glass with city view",
    ],
    "cairo": [
        "Cairo skyline with minarets and the Nile",
        "bustling Cairo street market",
        "Cairo traffic with pyramids visible in haze",
        "Egyptian street with Arabic signage",
        "Khan el-Khalili bazaar alley",
    ],
    "mumbai": [
        "Mumbai local train packed with commuters",
        "Indian street scene with auto-rickshaws",
        "Gateway of India with Taj Mahal Hotel",
        "Mumbai street vendor with colorful spices",
        "Dharavi neighborhood density",
    ],
    "sydney": [
        "Sydney Harbour with Opera House",
        "Sydney street with harbour bridge view",
        "Australian coastal cityscape",
        "Bondi Beach with surfers",
        "Sydney street café culture",
    ],
    "rio_de_janeiro": [
        "Rio de Janeiro beach with Christ statue on hill",
        "Copacabana beach promenade pattern",
        "Rio favela on hillside",
        "Sugarloaf Mountain cable car",
        "Rio carnival float or costumes",
    ],
    "beijing": [
        "Beijing hutong alleyway with courtyard gates",
        "Forbidden City red walls and golden roofs",
        "Beijing street with Chinese lanterns",
        "Tiananmen Gate with portrait",
        "Beijing modern skyline with CCTV building",
    ],
    "moscow": [
        "Moscow Red Square with colorful domes",
        "Moscow Metro ornate station interior",
        "Moscow street with Stalinist architecture",
        "Russian cityscape with onion domes",
        "Moscow winter street scene",
    ],
    "dubai": [
        "Dubai skyline with Burj Khalifa",
        "Dubai luxury shopping mall interior",
        "Dubai desert safari with dunes",
        "Dubai Palm Jumeirah aerial view",
        "Dubai Marina with glass skyscrapers",
    ],
    "singapore": [
        "Singapore Marina Bay Sands rooftop infinity pool",
        "Singapore hawker centre food stalls",
        "Gardens by the Bay supertrees at night",
        "Singapore Orchard Road shopping street",
        "Singapore shophouse street in Chinatown",
    ],
    "barcelona": [
        "Barcelona Sagrada Familia spires",
        "Barcelona Gothic Quarter narrow streets",
        "Las Ramblas pedestrian boulevard",
        "Gaudí mosaic benches at Park Güell",
        "Barcelona beach with W Hotel sail shape",
    ],
    "amsterdam": [
        "Amsterdam canal with narrow brick houses",
        "Amsterdam bicycle parked on a bridge",
        "Dutch canal houses with gabled facades",
        "Amsterdam canal boat tour",
        "Jordaan district with flower market",
    ],
    "prague": [
        "Prague Charles Bridge with statues at dawn",
        "Prague Old Town Square with astronomical clock",
        "Red-roofed Prague cityscape from hillside",
        "Prague narrow street with baroque facades",
    ],
    "lisbon": [
        "Lisbon tram 28 on steep cobblestone street",
        "Lisbon azulejo tile facade on building",
        "Lisbon Alfama district alleyway",
        "Pastéis de Belém bakery counter",
    ],
    "havana": [
        "Havana vintage American car on malecón",
        "Colorful colonial buildings in Old Havana",
        "Havana street with crumbling architecture",
        "Cuban cigar and rum bar scene",
    ],
    "seoul": [
        "Seoul Myeongdong shopping district with K-beauty signs",
        "Seoul Gyeongbokgung Palace with hanbok visitors",
        "Seoul Hongdae street performance area",
        "Seoul skyline with Namsan Tower",
        "Korean BBQ restaurant scene",
    ],
    "kyoto": [
        "Kyoto bamboo grove path",
        "Kyoto geisha district at dusk",
        "Fushimi Inari shrine vermillion torii gates",
        "Kyoto zen rock garden",
        "Kiyomizu-dera temple on hillside",
        "Kyoto traditional wooden machiya townhouse",
        "Kyoto cherry blossom along river",
    ],
    "venice_italy": [
        "Venice gondola on Grand Canal",
        "Venice St. Mark's Square with pigeons",
        "Venice narrow canal with bridges",
        "Venice Burano colorful houses",
    ],
    "san_francisco": [
        "San Francisco cable car on steep hill",
        "San Francisco painted Victorian houses",
        "Golden Gate Bridge through fog",
        "San Francisco Mission District murals",
    ],
    "chicago": [
        "Chicago deep-dish pizza",
        "Chicago bean sculpture Millennium Park",
        "Chicago L train elevated tracks",
        "Chicago skyline from lakefront",
    ],
    "melbourne": [
        "Melbourne laneway street art",
        "Melbourne coffee culture café",
        "Melbourne Federation Square modern architecture",
        "Great Ocean Road coastal drive",
    ],
    "buenos_aires": [
        "Buenos Aires La Boca colorful buildings",
        "Buenos Aires tango dancers on street",
        "Buenos Aires Recoleta cemetery",
        "Argentine asado steak restaurant",
    ],
    "cape_town": [
        "Cape Town Table Mountain cableway",
        "Cape Town Victoria waterfront",
        "Cape Town Bo-Kaap colorful houses",
        "Cape Town vineyard with mountain backdrop",
    ],
    "marrakech": [
        "Marrakech Jemaa el-Fnaa square at night",
        "Marrakech souk with colorful textiles",
        "Marrakech riad courtyard with tiles",
        "Moroccan tagine in traditional pot",
    ],
    "stockholm": [
        "Stockholm Gamla Stan old town buildings",
        "Stockholm archipelago islands",
        "Swedish fika coffee and cinnamon bun",
        "Stockholm subway art station",
    ],
    "helsinki": [
        "Helsinki Design District modern buildings",
        "Finnish sauna wooden interior",
        "Helsinki Market Hall food stalls",
        "Helsinki Cathedral white steps",
    ],
    "berlin": [
        "Berlin Wall East Side Gallery murals",
        "Berlin Alexanderplatz TV Tower",
        "Berlin street art in Kreuzberg",
        "Brandenburg Gate at night",
    ],
    "vienna": [
        "Vienna Schönbrunn Palace gardens",
        "Vienna café culture with Sachertorte",
        "Vienna Stephansdom cathedral spire",
        "Vienna horse carriage on Ringstraße",
    ],
    "warsaw": [
        "Warsaw Old Town colorful market square",
        "Warsaw Uprising Monument",
        "Polish pierogi restaurant scene",
    ],
    "athens": [
        "Athens Plaka district below Acropolis",
        "Athens street with street art and graffiti",
        "Greek taverna with blue shutters",
    ],
    "tel_aviv": [
        "Tel Aviv Bauhaus White City buildings",
        "Tel Aviv beach promenade",
        "Tel Aviv Carmel Market stalls",
    ],
    "hanoi": [
        "Hanoi Old Quarter narrow streets with signs",
        "Hanoi street phở vendor",
        "Hanoi train street with café tables",
        "Vietnamese motorbike traffic",
    ],
    "ho_chi_minh": [
        "Ho Chi Minh City Reunification Palace",
        "Ben Thanh Market interior",
        "Saigon café with motorbikes outside",
    ],
    "taipei": [
        "Taipei 101 tower skyline",
        "Taipei Shilin Night Market neon signs",
        "Taipei temple with dragon sculptures",
        "Taiwanese bubble tea shop",
    ],
    "kuala_lumpur": [
        "Kuala Lumpur Petronas Twin Towers",
        "Kuala Lumpur Batu Caves temple stairs",
        "Jalan Alor food street at night",
        "KL Tower observation deck view",
    ],
    "shanghai": [
        "Shanghai Bund colonial buildings at night",
        "Shanghai Pudong futuristic skyline",
        "Shanghai narrow lane with traditional shop",
        "Shanghai French Concession tree-lined street",
    ],
    "hong_kong": [
        "Hong Kong Victoria Peak skyline view",
        "Hong Kong Star Ferry on harbour",
        "Hong Kong Mongkok neon signs density",
        "Hong Kong temple incense coils",
    ],
    "zurich": [
        "Zurich lake with Alps backdrop",
        "Zurich Old Town narrow medieval streets",
        "Swiss chocolate shop display",
    ],
    "oslo": [
        "Oslo Opera House angular white marble",
        "Oslo Vigeland Sculpture Park",
        "Norwegian fjord cruise view",
    ],
    "reykjavik": [
        "Reykjavik Hallgrímskirkja church",
        "Iceland geothermal hot spring steam",
        "Northern lights over Icelandic landscape",
    ],
    # Africa
    "addis_ababa": [
        "Addis Ababa Ethiopian Orthodox church with round architecture",
        "Addis Ababa skyline with Entoto Hills backdrop",
        "Ethiopian coffee ceremony in traditional setting",
    ],
    "nairobi": [
        "Nairobi skyline with Kenyan high-rises",
        "Nairobi National Park with giraffes and city backdrop",
        "Nairobi Maasai Market with colorful crafts",
    ],
    "lagos": [
        "Lagos bustling street market with okada motorbikes",
        "Lagos Island skyline with lagoon views",
        "Lagos traffic jam of yellow danfo buses",
    ],
    "accra": [
        "Accra Makola Market bustling stalls",
        "AccraIndependence Square with Black Star arch",
        "Ghanaian kente cloth weaving scene",
    ],
    "casablanca": [
        "Casablanca Hassan II Mosque minaret on waterfront",
        "Casablanca Art Deco buildings on boulevard",
        "Moroccan white cityscape with blue sky",
    ],
    "tunis": [
        "Tunis Medina narrow alley with white walls and blue doors",
        "Sidi Bou Said blue and white cliffside village",
        "Tunis Cathedral Romanesque architecture",
    ],
    "dar_es_salaam": [
        "Dar es Salaam Kariakoo Market crowded stalls",
        "Dar es Salaam waterfront with dhow boats",
        "Tanzanian street with Bajaj auto-rickshaws",
    ],
    "kigali": [
        "Kigali hillside city with neat streets",
        "Kigali Genocide Memorial modern architecture",
        "Rwandan craft market with imigongo patterns",
    ],
    "lusaka": [
        "Lusaka market stalls with fresh produce",
        "Lusaka Great East Road with modern buildings",
        "Zambian shopping arcade with crafts",
    ],
    "windhoek": [
        "Windhoek Christuskirche sandstone church",
        "Windhoek German colonial architecture street",
        "Namibian desert landscape near city",
    ],
    "luanda": [
        "Luanda waterfront Marginal with skyscrapers",
        "Luanda Ilha do Cabo beach",
        "Angolan street with colorful pitstop vendors",
    ],
    "dakar": [
        "Dakar African Renaissance Monument statue",
        "Dakar fish market with colorful pirogues",
        "Senegalese street scene with mbolax",
    ],
    "kampala": [
        "Kampala Kasubi Tombs thatched architecture",
        "Kampala Owino Market crowded stalls",
        "Ugandan boda-boda motorbike taxi street",
    ],
    "khartoum": [
        "Khartoum confluence of Blue and White Nile",
        "Khartoum Sudanese mud-brick architecture",
        "Sudanese souk with spices and textiles",
    ],
    "tripoli": [
        "Tripoli old town with Ottoman architecture",
        "Tripoli Red Castle overlooking harbour",
        "Libyan Mediterranean waterfront boulevard",
    ],
    "algiers": [
        "Algiers Casbah white-washed hillside medina",
        "Algiers Grande Poste Moorish architecture",
        "Algiers waterfront with palm-lined boulevard",
    ],
    "abidjan": [
        "Abidjan Plateau skyscrapers and lagoon",
        "Abidjan Cocody neighborhood villas",
        "Ivorian street scene with attiéké vendor",
    ],
    # South America
    "medellin": [
        "Medellin cable car over colorful hillside barrios",
        "Medellin Botanical Garden tropical plants",
        "Colombian paisa street scene with flowers",
    ],
    "lima": [
        "Lima Miraflores cliffside park overlooking Pacific",
        "Lima historic center colonial architecture",
        "Peruvian ceviche street vendor",
    ],
    "santiago": [
        "Santiago skyline with Andes Mountains backdrop",
        "Santiago Plaza de Armas colonial cathedral",
        "Chilean street with vine-covered balconies",
    ],
    "bogota": [
        "Bogota La Candelaria colorful colonial district",
        "Bogota Monserrate hill funicular view",
        "Bogota street with graffiti art murals",
    ],
    "quito": [
        "Quito old town colonial churches on steep hills",
        "Quito Basilica Gothic spires",
        "Ecuadorian Andean market with indigenous textiles",
    ],
    "montevideo": [
        "Montevideo Plaza Independencia with equestrian statue",
        "Montevideo rambla waterfront promenade",
        "Uruguayan parrilla asado restaurant",
    ],
    "caracas": [
        "Caracas valley surrounded by green mountains",
        "Caracas Teresa Carreno performing arts center",
        "Venezuelan arepa street vendor",
    ],
    "asuncion": [
        "Asuncion Palacio de los Goernes balcony",
        "Asuncion Costanera riverfront promenade",
        "Paraguayan Mercado 4 bustling market",
    ],
    "la_paz": [
        "La Paz cable car system over steep cityscape",
        "La Paz Witches Market with dried llama fetuses",
        "Bolivian cityscape with snow-cimited Illimani",
    ],
    "cusco": [
        "Cusco Plaza de Armas with colonial arcades",
        "Cusco Inca stone walls under Spanish church",
        "Machu Picchu citadel in the clouds",
    ],
    "salvador": [
        "Salvador Pelourinho colorful colonial buildings",
        "Salvador Church of Sao Francisco gold interior",
        "Bahian street with acaraje vendor",
    ],
    "manaus": [
        "Manaus Teatro Amazonas opera house pink dome",
        "Manaus floating port on Rio Negro",
        "Amazonian rainforest meeting of waters",
    ],
    "brasilia": [
        "Brasilia Cathedral hyperboloid glass structure",
        "Brasilia National Congress twin towers",
        "Brazilian modernist Oscar Niemeyer architecture",
    ],
    "valparaiso": [
        "Valparaiso colorful hillside houses and funiculars",
        "Valparaiso harbor with ships and painted buildings",
        "Chilean street art murals on steep alley",
    ],
    # Southeast Asia
    "manila": [
        "Manila Intramuros Spanish colonial walls",
        "Manila skyline with bay sunset",
        "Jeepney colorful Filipino public transport",
    ],
    "yangon": [
        "Yangon Shwedagon Pagoda golden stupa at sunset",
        "Yangon colonial British buildings on Strand Road",
        "Burmese street vendor with thanaka face paint",
    ],
    "phnom_penh": [
        "Phnom Penh Royal Palace silver pagoda",
        "Phnom Penh riverside Sisowath Quay promenade",
        "Cambodian tuk-tuk near Angkorian-style buildings",
    ],
    "vientiane": [
        "Vientiane Patuxai victory arch avenue",
        "Vientiane Pha That Luang golden stupa",
        "Laotian temple with multi-tiered roof",
    ],
    "luang_prabang": [
        "Luang Prabang golden temples along Mekong River",
        "Luang Prabang morning alms-giving ceremony",
        "French colonial and Lao wooden architecture mix",
    ],
    "bali": [
        "Bali rice terraces with palm trees",
        "Balinese temple gate with carved stone",
        "Bali beach temple Uluwatu on cliff edge",
    ],
    "ubud": [
        "Ubud monkey forest stone temple entrance",
        "Ubud traditional dance performance in pavilion",
        "Balinese artisan wood carving workshop",
    ],
    "yogyakarta": [
        "Yogyakarta Sultan Palace kraton courtyard",
        "Borobudur temple stupa in morning mist",
        "Javanese batik fabric making workshop",
    ],
    "cebu": [
        "Cebu Magellan's Cross chapel",
        "Cebu jeepney on busy Philippine street",
        "Cebu Temple of Leeper white structure",
    ],
    "chiang_mai": [
        "Chiang Mai old city moat and temple walls",
        "Chiang Mai Doi Suthep temple on mountain",
        "Thai yi peng lantern festival sky full of lights",
    ],
    "pattaya": [
        "Pattaya Beach road with neon nightlife signs",
        "Pattaya coastal city skyline at dusk",
        "Thai temple Wat Phra Yai Big Buddha on hill",
    ],
    "krabi": [
        "Krabi Railay Beach limestone karst cliffs",
        "Krabi longtail boats on emerald water",
        "Thai island beach with karst formations",
    ],
    # Eastern Europe
    "budapest": [
        "Budapest Parliament Danube riverfront",
        "Budapest thermal bath art nouveau interior",
        "Buda Castle hilltop with Chain Bridge view",
    ],
    "bucharest": [
        "Bucharest Palace of Parliament massive facade",
        "Bucharest old town Lipscani street",
        "Romanian Athenaeum concert hall dome",
    ],
    "sofia": [
        "Sofia Alexander Nevsky Cathedral golden domes",
        "Sofia Vitosha Boulevard pedestrian street",
        "Bulgarian Orthodox church with fresco facade",
    ],
    "belgrade": [
        "Belgrade Kalemegdan fortress with Danube view",
        "Belgrade Skadarlija bohemian quarter cobblestone",
        "Serbian nightlife district street scene",
    ],
    "zagreb": [
        "Zagreb Upper Town stone gates and chapel",
        "Zagreb Ban Jelacic Square tram and fountains",
        "Croatian Museum of Broken Relationships building",
    ],
    "ljubljana": [
        "Ljubljana triple bridge and pink church",
        "Ljubljana castle hilltop overlooking city",
        "Slovenian willow-lined Ljubljanica river cafe",
    ],
    "bratislava": [
        "Bratislava Castle white tower above Danube",
        "Bratislava old town Michael's Gate tower",
        "Slovakian UFO Bridge observation deck",
    ],
    "tallinn": [
        "Tallinn medieval old town towers and walls",
        "Tallinn Town Hall Gothic spire",
        "Estonian Seaplane Harbour museum domes",
    ],
    "riga": [
        "Riga Art Nouveau facades on Alberta Street",
        "Riga Dome Cathedral and House of Blackheads",
        "Latvian Central Market in former zeppelin hangar",
    ],
    "vilnius": [
        "Vilnius Gediminas Tower hilltop view of old town",
        "Vilnius Baroque church spires and courtyards",
        "Lithuanian Uzupis bohemian neighborhood art",
    ],
    "tbilisi": [
        "Tbilisi Abanotubani colorful bathhouse domes",
        "Tbilisi old town wooden carved balconies",
        "Tbilisi Bridge of Peace glass structure over river",
    ],
    "yerevan": [
        "Yerevan Cascade pink tufa stone stairway",
        "Yerevan Republic Square fountains at night",
        "Armenian Mount Ararat view from city",
    ],
    "baku": [
        "Baku Flame Towers illuminated skyline",
        "Baku Old City Icherisheher medieval walls",
        "Azerbaijan Heydar Aliyev Center curved architecture",
    ],
    "kiev": [
        "Kiev golden-domed Saint Sophia Cathedral",
        "Kiev Maidan Nezalezhnosti Independence Square",
        "Ukrainian wooden church in Pirogovo open-air museum",
    ],
    "minsk": [
        "Minsk Independence Avenue Stalinist architecture",
        "Minsk Holy Spirit Cathedral twin towers",
        "Belarusian Trinity Suburb old town colorful houses",
    ],
    # Middle East
    "abu_dhabi": [
        "Abu Dhabi Sheikh Zayed Grand Mosque white marble",
        "Abu Dhabi skyline with waterfront corniche",
        "Emirati desert oasis with date palms",
    ],
    "doha": [
        "Doha Corniche skyline glass towers",
        "Doha Souq Waqif traditional market alley",
        "Qatar Museum of Islamic Art island building",
    ],
    "riyadh": [
        "Riyadh Kingdom Centre tower sky bridge",
        "Riyadh Edge of the World cliff desert view",
        "Saudi Arabian mud-brick Masmak Fortress",
    ],
    "jeddah": [
        "Jeddah Al-Balad coral stone tower houses",
        "Jeddah waterfront Corniche with fountain",
        "Saudi floating mosque on Red Sea",
    ],
    "muscat": [
        "Muscat Sultan Qaboos Grand Mosque crystal chandelier",
        "Muscat Muttrah Corniche with dhows",
        "Omani mountain wadi with turquoise water",
    ],
    "amman": [
        "Amman Citadel Roman temple columns",
        "Amman Rainbow Street colorful hillside houses",
        "Jordanian white limestone cityscape",
    ],
    "beirut": [
        "Beirut Downtown Solidere waterfront promenade",
        "Beirut Hamra Street cafes and shops",
        "Lebanese mountain village with stone houses",
    ],
    "baghdad": [
        "Baghdad Al-Mustansiriya University historic arches",
        "Baghdad Tigris riverfront with minarets",
        "Iraqi market with ornate tile work",
    ],
    "tehran": [
        "Tehran Milad Tower skyline",
        "Tehran Grand Bazaar vaulted ceiling corridor",
        "Iranian Azadi Tower white marble gateway",
    ],
    "kuwait_city": [
        "Kuwait City Kuwait Towers striped spheres",
        "Kuwait City Gulf Road skyline",
        "Kuwaiti souq with spices and perfumes",
    ],
    "bahrain": [
        "Bahrain World Trade Center wind turbines",
        "Bahrain Fort Portuguese-era desert fortress",
        "Manama Bahrain skyline at night",
    ],
    "aden": [
        "Aden Crater City volcano rim settlement",
        "Aden harbour with traditional boats",
        "Yemeni multi-story tower houses brick facade",
    ],
    # India / South Asia
    "delhi": [
        "Delhi Red Fort sandstone walls",
        "Delhi Chandni Chowk crowded market lane",
        "India Gate war memorial with evening lights",
    ],
    "kolkata": [
        "Kolkata Howrah Bridge over Hooghly River",
        "Kolkata Victoria Memorial white marble dome",
        "Kolkata Park Street with colonial buildings",
    ],
    "bangalore": [
        "Bangalore Vidhana Soudha grand legislative building",
        "Bangalore Cubbon Park greenery and heritage",
        "Bengaluru craft brewery street scene",
    ],
    "chennai": [
        "Chennai Kapaleeshwarar Temple Dravidian gopuram",
        "Chennai Marina Beach promenade",
        "Tamil Nadu street with kolam floor art",
    ],
    "hyderabad": [
        "Hyderabad Charminar four minarets monument",
        "Hyderabad Golconda Fort hilltop ramparts",
        "Hyderabad biryani street food stall",
    ],
    "jaipur": [
        "Jaipur Hawa Mahal pink sandstone lattice facade",
        "Jaipur City Palace ornate courtyards",
        "Amber Fort hilltop with elephant approach",
    ],
    "varanasi": [
        "Varanasi Ganges riverfront ghat steps at dawn",
        "Varanasi narrow alley with temples and shops",
        "Varanasi evening aarti ceremony with fire lamps",
    ],
    "lucknow": [
        "Lucknow Bara Imambara massive corridor",
        "Lucknow Rumi Darwaza Turkish gate",
        "Awadhi kebab restaurant scene",
    ],
    "ahmedabad": [
        "Ahmedabad Sabarmati Ashram riverside retreat",
        "Ahmedabad Pols narrow wooden house alley",
        "Indian stepwell Adalaj intricate carvings",
    ],
    "colombo": [
        "Colombo Galle Face Green oceanfront promenade",
        "Colombo Gangaramaya Temple eclectic architecture",
        "Sri Lankan Pettah market crowded street",
    ],
    "kathmandu": [
        "Kathmandu Durbar Square pagoda temples",
        "Kathmandu Boudhanath stupa prayer flags",
        "Nepali narrow street with marigold garlands",
    ],
    "dhaka": [
        "Dhaka Sadarghat riverfront with launches",
        "Dhaka National Assembly Jatiyo Sangsad Bhaban",
        "Bangladeshi rickshaw with colorful decorations",
    ],
    "islamabad": [
        "Islamabad Faisal Mosque white tent structure",
        "Islamabad Margalla Hills backdrop",
        "Pakistan Monument petal-shaped museum",
    ],
    "lahore": [
        "Lahore Badshahi Mosque red sandstone courtyard",
        "Lahore Fort Shish Mahal mirror palace",
        "Lahore Food Street with Mughal-era buildings",
    ],
    # Central Asia
    "almaty": [
        "Almaty Zenkov Cathedral wooden pink church",
        "Almaty Tian Shan mountain backdrop",
        "Kazakh bazaar with dried fruits and horse meat",
    ],
    "tashkent": [
        "Tashkent Chorsu Bazaar turquoise dome market",
        "Tashkent Khast Imam complex Islamic architecture",
        "Uzbek metro station ornate chandelier interior",
    ],
    "samarkand": [
        "Samarkand Registan madrasa turquoise mosaic tiles",
        "Samarkand Shah-i-Zinda avenue of mausoleums",
        "Silk Road caravanserai courtyard with arches",
    ],
    "bishkek": [
        "Bishkek Ala-Too Square Manas statue",
        "Bishkek Soviet-era wide boulevards",
        "Kyrgyz Osh Bazaar with naan bread stalls",
    ],
    "astana": [
        "Astana Bayterek Tower white trunk with golden sphere",
        "Astana Khan Shatyr futuristic tent palace",
        "Kazakh futuristic architecture white marble city",
    ],
    "urumqi": [
        "Urumqi Grand Bazaar Uyghur Islamic architecture",
        "Urumqi Xinjiang regional museum",
        "Silk Road night market with lamb skewers",
    ],
    # Japan / Korea
    "osaka": [
        "Osaka Dotonbori neon signs over canal",
        "Osaka Castle stone walls and moat",
        "Japanese street food takoyaki stall",
    ],
    "nagoya": [
        "Nagoya Castle golden shachihoko on top",
        "Nagoya Atsuta Shrine forested path",
        "Toyota factory tour modern industrial scene",
    ],
    "sapporo": [
        "Sapporo Snow Festival ice sculptures",
        "Sapporo beer garden red brick brewery",
        "Hokkaido ramen noodle shop steam",
    ],
    "fukuoka": [
        "Fukuoka Canal City Hakata modern complex",
        "Fukuoka yatai street food stalls along river",
        "Japanese temple Kushida ornate float display",
    ],
    "hiroshima": [
        "Hiroshima Peace Memorial Genbaku Dome skeletal ruin",
        "Hiroshima Peace Park memorial cenotaph",
        "Miyajima floating torii gate at high tide",
    ],
    "pyeongchang": [
        "Pyeongchang Winter Olympics ski resort",
        "Korean mountain temple in snowy forest",
        "Taebaek mountain village winter landscape",
    ],
    # China
    "chengdu": [
        "Chengdu Kuanzhai Alley teahouse street",
        "Chengdu Giant Panda base bamboo enclosure",
        "Sichuan hotpot restaurant with red chili broth",
    ],
    "guangzhou": [
        "Guangzhou Canton Tower futuristic twisted structure",
        "Guangzhou Chen Clan Ancestral Hall ornate roof",
        "Pearl River night cruise with skyline lights",
    ],
    "shenzhen": [
        "Shenzhen futuristic skyline with glass towers",
        "Shenzhen OCT Loft creative art district",
        "Shenzhen tech market electronics stalls",
    ],
    "wuhan": [
        "Wuhan Yellow Crane Tower pagoda on hill",
        "Wuhan Wuhan Yangtze River Bridge",
        "Wuhan hot dry noodle breakfast stall",
    ],
    "xian": [
        "Xian Terracotta Warriors underground pit",
        "Xian City Wall with bicycle ride",
        "Xian Muslim Quarter street food alley",
    ],
    "hangzhou": [
        "Hangzhou West Lake pagoda and causeway",
        "Hangzhou Lingyin Temple forested hillside",
        "Longjing tea plantation hillside terraces",
    ],
    "nanjing": [
        "Nanjing Sun Yat-sen Mausoleum on Purple Mountain",
        "Nanjing Confucius Temple Qinhuai River lanterns",
        "Nanjing Ming City Wall with autumn leaves",
    ],
    "chongqing": [
        "Chongqing Hongya Cave stilted buildings at night",
        "Chongqing Yangtze cable car over river gorge",
        "Chongqing hotpot restaurant spicy red broth",
    ],
    "dunhuang": [
        "Dunhuang Mogao Caves Buddhist painted grottoes",
        "Dunhuang Crescent Moon Spring desert oasis",
        "Mingsha Dunes sand dunes with camel caravan",
    ],
    "lhasa": [
        "Lhasa Potala Palace white and red hilltop fortress",
        "Lhasa Jokhang Temple prayer wheels and monks",
        "Tibetan Barkhor Street pilgrims and prayer flags",
    ],
    # Oceania
    "perth": [
        "Perth skyline with Swan River reflections",
        "Perth Kings Park with city and river view",
        "Western Australia Rottnest Island quokka",
    ],
    "brisbane": [
        "Brisbane Story Bridge illuminated at night",
        "Brisbane South Bank Parklands city beach",
        "Queensland subtropical river cityscape",
    ],
    "auckland": [
        "Auckland Sky Tower needle over harbour",
        "Auckland Viaduct Harbour waterfront restaurants",
        "New Zealand volcanic island Rangitoto from ferry",
    ],
    "wellington": [
        "Wellington Cable Car to Botanic Garden",
        "Wellington waterfront Te Papa museum",
        "Windy Wellington harbour with ferry terminal",
    ],
    "queenstown": [
        "Queenstown Remarkables mountains reflected in lake",
        "Queenstown Skyline gondola luge track",
        "New Zealand adventure bungee bridge canyon",
    ],
    "adelaide": [
        "Adelaide St Peter's Cathedral twin spires",
        "Adelaide Central Market food hall",
        "South Australia vineyard Barossa Valley hills",
    ],
    "hobart": [
        "Hobart MONA museum modern art underground",
        "Hobart waterfront Salamanca Place sandstone warehouses",
        "Tasmanian harbor with Mount Wellington backdrop",
    ],
    # Caribbean / Islands
    "santo_domingo": [
        "Santo Domingo Zona Colonial stone fortress walls",
        "Santo Domingo Alcázar de Colón courtyard",
        "Dominican Republic beach with palm trees",
    ],
    "kingston": [
        "Kingston Jamaica Boba Marley museum",
        "Kingston waterfront with Blue Mountains backdrop",
        "Jamaican street jerk chicken vendor smoke",
    ],
    "nassau": [
        "Nassau pastel colonial buildings on Bay Street",
        "Nassau Atlantis Resort underwater aquarium",
        "Bahamas pink sand beach with turquoise water",
    ],
    "san_juan": [
        "San Juan El Morro fortress overlooking Atlantic",
        "San Juan cobblestone Old San Juan blue accents",
        "Puerto Rican colorful La Perla houses by sea",
    ],
    "barbados": [
        "Barbados Bridgetown Independence Square colonial",
        "Barbados beach with turquoise Caribbean water",
        "Rum distillery with sugar cane field backdrop",
    ],
    "fiji": [
        "Fiji white sand beach with palm trees and blue lagoon",
        "Fijian bure thatched hut overwater resort",
        "Fiji coral reef snorkeling clear water",
    ],
    "mauritius": [
        "Mauritius underwater waterfall aerial illusion",
        "Mauritius Le Morne mountain peninsula",
        "Mauritian beach with sugarcane plantation backdrop",
    ],
    "seychelles": [
        "Seychelles Anse Source d'Argent granite boulders beach",
        "Seychelles Vallée de Mai palm forest",
        "Seychellois beach with pink granite rocks",
    ],
    "maldives": [
        "Maldives overwater bungalow villa on turquoise lagoon",
        "Maldives white sand island with palm trees aerial",
        "Maldivian coral reef with tropical fish",
    ],
}

_CITY_CENTROIDS: dict[str, tuple[float, float]] = {
    "tokyo": (35.6762, 139.6503),
    "paris": (48.8566, 2.3522),
    "new_york": (40.7128, -74.0060),
    "london": (51.5074, -0.1278),
    "rome": (41.9028, 12.4964),
    "bangkok": (13.7563, 100.5018),
    "istanbul": (41.0082, 28.9784),
    "cairo": (30.0444, 31.2357),
    "mumbai": (19.0760, 72.8777),
    "sydney": (-33.8688, 151.2093),
    "rio_de_janeiro": (-22.9068, -43.1729),
    "beijing": (39.9042, 116.4074),
    "moscow": (55.7558, 37.6173),
    "dubai": (25.2048, 55.2708),
    "singapore": (1.3521, 103.8198),
    "barcelona": (41.3851, 2.1734),
    "amsterdam": (52.3676, 4.9041),
    "prague": (50.0755, 14.4378),
    "lisbon": (38.7223, -9.1393),
    "havana": (23.1136, -82.3666),
    "seoul": (37.5665, 126.9780),
    "kyoto": (35.0116, 135.7681),
    "venice_italy": (45.4408, 12.3155),
    "san_francisco": (37.7749, -122.4194),
    "chicago": (41.8781, -87.6298),
    "melbourne": (-37.8136, 144.9631),
    "buenos_aires": (-34.6037, -58.3816),
    "cape_town": (-33.9249, 18.4241),
    "marrakech": (31.6295, -7.9811),
    "stockholm": (59.3293, 18.0686),
    "helsinki": (60.1699, 24.9384),
    "berlin": (52.5200, 13.4050),
    "vienna": (48.2082, 16.3738),
    "warsaw": (52.2297, 21.0122),
    "athens": (37.9838, 23.7275),
    "tel_aviv": (32.0853, 34.7818),
    "hanoi": (21.0278, 105.8342),
    "ho_chi_minh": (10.8231, 106.6297),
    "taipei": (25.0330, 121.5654),
    "kuala_lumpur": (3.1390, 101.6869),
    "shanghai": (31.2304, 121.4737),
    "hong_kong": (22.3193, 114.1694),
    "zurich": (47.3769, 8.5417),
    "oslo": (59.9139, 10.7522),
    "reykjavik": (64.1466, -21.9426),
    # Africa
    "addis_ababa": (9.0192, 38.7525),
    "nairobi": (-1.2921, 36.8219),
    "lagos": (6.5244, 3.3792),
    "accra": (5.6037, -0.1870),
    "casablanca": (33.5731, -7.5898),
    "tunis": (36.8065, 10.1815),
    "dar_es_salaam": (-6.7924, 39.2083),
    "kigali": (-1.9403, 29.8739),
    "lusaka": (-15.3875, 28.3228),
    "windhoek": (-22.5609, 17.0658),
    "luanda": (-8.8399, 13.2894),
    "dakar": (14.7167, -17.4677),
    "kampala": (0.3476, 32.5825),
    "khartoum": (15.5007, 32.5599),
    "tripoli": (32.9022, 13.1800),
    "algiers": (36.7538, 3.0588),
    "abidjan": (5.3600, -4.0083),
    # South America
    "medellin": (6.2442, -75.5812),
    "lima": (-12.0464, -77.0428),
    "santiago": (-33.4489, -70.6693),
    "bogota": (4.7110, -74.0721),
    "quito": (-0.1807, -78.4678),
    "montevideo": (-34.9011, -56.1645),
    "caracas": (10.4806, -66.9036),
    "asuncion": (-25.2637, -57.5759),
    "la_paz": (-16.5000, -68.1500),
    "cusco": (-13.5320, -71.9675),
    "salvador": (-12.9714, -38.5124),
    "manaus": (-3.1190, -60.0217),
    "brasilia": (-15.7975, -47.8919),
    "valparaiso": (-33.0472, -71.6127),
    # Southeast Asia
    "manila": (14.5995, 120.9842),
    "yangon": (16.8661, 96.1951),
    "phnom_penh": (11.5564, 104.9282),
    "vientiane": (17.9757, 102.6331),
    "luang_prabang": (19.8856, 102.1347),
    "bali": (-8.3405, 115.0920),
    "ubud": (-8.5069, 115.2625),
    "yogyakarta": (-7.7956, 110.3695),
    "cebu": (10.3157, 123.8854),
    "chiang_mai": (18.7883, 98.9853),
    "pattaya": (12.9236, 100.8825),
    "krabi": (8.0863, 98.9063),
    # Eastern Europe
    "budapest": (47.4979, 19.0402),
    "bucharest": (44.4268, 26.1025),
    "sofia": (42.6977, 23.3219),
    "belgrade": (44.7866, 20.4489),
    "zagreb": (45.8150, 15.9819),
    "ljubljana": (46.0569, 14.5058),
    "bratislava": (48.1486, 17.1077),
    "tallinn": (59.4370, 24.7536),
    "riga": (56.9496, 24.1052),
    "vilnius": (54.6872, 25.2797),
    "tbilisi": (41.7151, 44.8271),
    "yerevan": (40.1792, 44.4991),
    "baku": (40.4093, 49.8671),
    "kiev": (50.4501, 30.5234),
    "minsk": (53.9045, 27.5615),
    # Middle East
    "abu_dhabi": (24.4539, 54.3773),
    "doha": (25.2854, 51.5310),
    "riyadh": (24.7136, 46.6753),
    "jeddah": (21.4858, 39.1925),
    "muscat": (23.5859, 58.4059),
    "amman": (31.9454, 35.9284),
    "beirut": (33.8938, 35.5018),
    "baghdad": (33.3128, 44.3615),
    "tehran": (35.6892, 51.3890),
    "kuwait_city": (29.3759, 47.9774),
    "bahrain": (26.2285, 50.5860),
    "aden": (12.7855, 45.0187),
    # India / South Asia
    "delhi": (28.7041, 77.1025),
    "kolkata": (22.5726, 88.3639),
    "bangalore": (12.9716, 77.5946),
    "chennai": (13.0827, 80.2707),
    "hyderabad": (17.3850, 78.4867),
    "jaipur": (26.9124, 75.7873),
    "varanasi": (25.3176, 82.9739),
    "lucknow": (26.8467, 80.9462),
    "ahmedabad": (23.0225, 72.5714),
    "colombo": (6.9271, 79.8612),
    "kathmandu": (27.7172, 85.3240),
    "dhaka": (23.8103, 90.4125),
    "islamabad": (33.6844, 73.0479),
    "lahore": (31.5204, 74.3587),
    # Central Asia
    "almaty": (43.2220, 76.8512),
    "tashkent": (41.2995, 69.2401),
    "samarkand": (39.6542, 66.9597),
    "bishkek": (42.8746, 74.5698),
    "astana": (51.1605, 71.4704),
    "urumqi": (43.8256, 87.6168),
    # Japan / Korea
    "osaka": (34.6937, 135.5023),
    "nagoya": (35.1815, 136.9066),
    "sapporo": (43.0618, 141.3545),
    "fukuoka": (33.5902, 130.4017),
    "hiroshima": (34.3853, 132.4553),
    "pyeongchang": (37.3700, 128.3900),
    # China
    "chengdu": (30.5728, 104.0668),
    "guangzhou": (23.1291, 113.2644),
    "shenzhen": (22.5431, 114.0579),
    "wuhan": (30.5928, 114.3055),
    "xian": (34.3416, 108.9398),
    "hangzhou": (30.2741, 120.1551),
    "nanjing": (32.0603, 118.7969),
    "chongqing": (29.5630, 106.5516),
    "dunhuang": (40.1421, 94.6623),
    "lhasa": (29.6500, 91.1000),
    # Oceania
    "perth": (-31.9505, 115.8605),
    "brisbane": (-27.4698, 153.0251),
    "auckland": (-36.8485, 174.7633),
    "wellington": (-41.2866, 174.7756),
    "queenstown": (-45.0312, 168.6626),
    "adelaide": (-34.9285, 138.6007),
    "hobart": (-42.8821, 147.3257),
    # Caribbean / Islands
    "santo_domingo": (18.4861, -69.9312),
    "kingston": (18.0179, -76.8099),
    "nassau": (25.0480, -77.3554),
    "san_juan": (18.4655, -66.1057),
    "barbados": (13.1939, -59.5432),
    "fiji": (-18.1416, 178.4419),
    "mauritius": (-20.1609, 57.5012),
    "seychelles": (-4.6796, 55.4920),
    "maldives": (4.1755, 73.5093),
}

# ── Country centroids (manually curated — capital coordinates) ──────
_COUNTRY_CENTROIDS: dict[str, tuple[float, float]] = {
    "US": (38.9, -77.0), "CN": (39.9, 116.4), "IN": (28.6, 77.2),
    "BR": (-15.8, -47.9), "JP": (35.7, 139.7), "DE": (52.5, 13.4),
    "GB": (51.5, -0.1), "FR": (48.9, 2.3), "IT": (41.9, 12.5),
    "ES": (40.4, -3.7), "MX": (19.4, -99.1), "KR": (37.6, 127.0),
    "RU": (55.8, 37.6), "AU": (-35.3, 149.1), "CA": (45.4, -75.7),
    "TH": (13.8, 100.5), "TR": (39.9, 32.9),
    "SA": (24.7, 46.7), "EG": (30.0, 31.2), "NG": (9.1, 7.5),
    "ZA": (-33.9, 18.4), "KE": (-1.3, 36.8), "PH": (14.6, 121.0),
    "ID": (-6.2, 106.8), "PK": (33.7, 73.1), "BD": (23.8, 90.4),
    "VN": (21.0, 105.8), "MY": (3.1, 101.7), "SG": (1.3, 103.8),
    "CO": (4.6, -74.1), "AR": (-34.6, -58.4), "CL": (-33.4, -70.7),
    "PE": (-12.0, -77.0), "PL": (52.2, 21.0), "NL": (52.4, 4.9),
    "SE": (59.3, 18.1), "NO": (59.9, 10.8), "FI": (60.2, 24.9),
    "DK": (55.7, 12.6), "AT": (48.2, 16.4), "CH": (46.9, 7.4),
    "BE": (50.9, 4.4), "PT": (38.7, -9.1), "GR": (38.0, 23.7),
    "CZ": (50.1, 14.4), "RO": (44.4, 26.1), "HU": (47.5, 19.1),
    "UA": (50.4, 30.5), "IL": (31.8, 35.2), "AE": (25.3, 55.3),
    "MA": (33.6, -7.6), "TZ": (-6.8, 37.3), "ET": (9.0, 38.7),
    "GH": (5.6, -0.2), "EC": (-0.2, -78.5), "BO": (-16.5, -68.1),
}

_COUNTRY_NAMES: dict[str, str] = {
    "US": "the United States", "CN": "China", "IN": "India",
    "BR": "Brazil", "JP": "Japan", "DE": "Germany",
    "GB": "the United Kingdom", "FR": "France", "IT": "Italy",
    "ES": "Spain", "MX": "Mexico", "KR": "South Korea",
    "RU": "Russia", "AU": "Australia", "CA": "Canada",
    "TH": "Thailand", "TR": "Turkey",
    "SA": "Saudi Arabia", "EG": "Egypt", "NG": "Nigeria",
    "ZA": "South Africa", "KE": "Kenya", "PH": "the Philippines",
    "ID": "Indonesia", "PK": "Pakistan", "BD": "Bangladesh",
    "VN": "Vietnam", "MY": "Malaysia", "SG": "Singapore",
    "CO": "Colombia", "AR": "Argentina", "CL": "Chile",
    "PE": "Peru", "PL": "Poland", "NL": "the Netherlands",
    "SE": "Sweden", "NO": "Norway", "FI": "Finland",
    "DK": "Denmark", "AT": "Austria", "CH": "Switzerland",
    "BE": "Belgium", "PT": "Portugal", "GR": "Greece",
    "CZ": "the Czech Republic", "RO": "Romania", "HU": "Hungary",
    "UA": "Ukraine", "IL": "Israel", "AE": "the United Arab Emirates",
    "MA": "Morocco", "TZ": "Tanzania", "ET": "Ethiopia",
    "GH": "Ghana", "EC": "Ecuador", "BO": "Bolivia",
}

_COUNTRY_PROMPTS: dict[str, list[str]] = {}
for _code, _name in _COUNTRY_NAMES.items():
    _COUNTRY_PROMPTS[_code] = [
        f"a street scene in {_name}",
        f"an outdoor photo taken in {_name}",
        f"road and buildings in {_name}",
    ]


class ClipVisualModule(BaseModule):
    """Zero-shot CLIP scene classification mapped to geographic biome priors."""

    name = "clip_visual"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)

    def is_available(self) -> bool:
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    def prepare(self) -> None:
        from geofind.utils.models import get_clip_shared
        from geofind.utils.constants import BIOME_PROMPTS

        self._model, self._processor = get_clip_shared()
        self._biome_prompts = BIOME_PROMPTS
        self._biome_centroids = _BIOME_CENTROIDS
        self._country_prompts = _COUNTRY_PROMPTS
        self._country_centroids = _COUNTRY_CENTROIDS
        self._city_prompts = _CITY_PROMPTS
        self._city_centroids = _CITY_CENTROIDS
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

        from geofind.utils.models import clip_softmax_scores

        image = self._get_image(media_path, frames)
        if image is None:
            return []

        # ── Biome classification ─────────────────────────────────────────
        all_prompts: list[str] = []
        prompt_to_biome: dict[int, str] = {}
        for biome, prompts in self._biome_prompts.items():
            for p in prompts:
                idx = len(all_prompts)
                all_prompts.append(p)
                prompt_to_biome[idx] = biome

        probs = clip_softmax_scores(image, "clipvis_biome", all_prompts)

        biome_scores: dict[str, float] = {}
        for idx, prob in enumerate(probs):
            biome = prompt_to_biome[idx]
            biome_scores[biome] = biome_scores.get(biome, 0.0) + prob

        hits: list[ModuleHit] = []
        for biome, score in sorted(biome_scores.items(), key=lambda x: -x[1]):
            if score < 0.05:
                continue
            lat, lon = self._biome_centroids.get(biome, (0.0, 0.0))
            hits.append(self._make_hit(
                lat, lon, min(score, 1.0),
                sigma_km=500.0,  # Biome-level — wide spread
                biome=biome,
                raw_score=score,
                hint_level="biome",
            ))

        # ── Country scene classification ─────────────────────────────────
        country_prompts: list[str] = []
        prompt_to_country: dict[int, str] = {}
        for country, prompts in self._country_prompts.items():
            for p in prompts:
                idx = len(country_prompts)
                country_prompts.append(p)
                prompt_to_country[idx] = country

        country_probs = clip_softmax_scores(
            image, "clipvis_country", country_prompts
        )

        country_scores: dict[str, float] = {}
        for idx, prob in enumerate(country_probs):
            country = prompt_to_country[idx]
            country_scores[country] = country_scores.get(country, 0.0) + prob

        top_countries = sorted(country_scores.items(), key=lambda x: -x[1])[:5]
        for country, score in top_countries:
            if score < 0.03:
                continue
            lat, lon = self._country_centroids.get(country, (0.0, 0.0))
            hits.append(self._make_hit(
                lat, lon, min(score, 1.0),
                sigma_km=800.0,  # Country-level — wide spread
                country=country,
                raw_score=score,
                hint_level="country",
            ))

        # ── City scene classification ────────────────────────────────────
        city_prompts: list[str] = []
        prompt_to_city: dict[int, str] = {}
        for city, prompts in self._city_prompts.items():
            for p in prompts:
                idx = len(city_prompts)
                city_prompts.append(p)
                prompt_to_city[idx] = city

        city_probs = clip_softmax_scores(image, "clipvis_city", city_prompts)

        city_scores: dict[str, float] = {}
        for idx, prob in enumerate(city_probs):
            city = prompt_to_city[idx]
            city_scores[city] = city_scores.get(city, 0.0) + prob

        top_cities = sorted(city_scores.items(), key=lambda x: -x[1])[:3]
        for city, score in top_cities:
            if score < 0.06:
                continue
            lat, lon = self._city_centroids.get(city, (0.0, 0.0))
            hits.append(self._make_hit(
                lat, lon, min(score, 1.0),
                city=city,
                raw_score=score,
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
