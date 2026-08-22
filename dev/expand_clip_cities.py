#!/usr/bin/env python3
"""Generate expanded CLIP city data for clip_visual.py."""

# New cities to add: (key, lat, lon, [prompts])
# Each city gets 3-5 descriptive CLIP prompts

NEW_CITIES = [
    # === USA ===
    ("phoenix", 33.4484, -112.0740, [
        "Phoenix Arizona desert cityscape with saguaro cactus",
        "Phoenix downtown skyline with Camelback Mountain",
        "Arizona suburban sprawl with palm trees and stucco houses",
    ]),
    ("san_diego", 32.7157, -117.1611, [
        "San Diego skyline with Coronado Bridge",
        "San Diego Balboa Park Spanish Colonial architecture",
        "San Diego beach with La Jolla cliffs",
    ]),
    ("dallas", 32.7767, -96.7970, [
        "Dallas skyline with Reunion Tower ball",
        "Dallas downtown with Margaret Hunt Hill Bridge",
        "Texas suburban strip mall with pickup trucks",
    ]),
    ("houston", 29.7604, -95.3698, [
        "Houston skyline with highway overpasses",
        "Houston Texas Medical Center buildings",
        "Houston Space Center rocket display",
    ]),
    ("austin", 30.2672, -97.7431, [
        "Austin Texas downtown skyline at sunset",
        "Austin Sixth Street neon signs and bars",
        "Austin Congress Avenue Bridge with bats at dusk",
    ]),
    ("san_antonio", 29.4241, -98.4936, [
        "San Antonio River Walk with restaurant patios",
        "San Antonio Alamo Mission facade",
        "San Antonio Tex-Mex cantilevered bridge architecture",
    ]),
    ("denver", 39.7392, -104.9903, [
        "Denver skyline with Rocky Mountains backdrop",
        "Denver Union Station train hall interior",
        "Colorado red rock landscape near Denver",
    ]),
    ("seattle", 47.6062, -122.3321, [
        "Seattle Space Needle with Mount Rainier",
        "Seattle Pike Place Market neon sign and fish throwing",
        "Seattle Capitol Hill neighborhood with coffee shops",
    ]),
    ("portland", 45.5152, -122.6784, [
        "Portland Oregon food truck pod on street",
        "Portland Vaux Brewery district with bridges",
        "Portland Powell's Books interior with stacked shelves",
    ]),
    ("nashville", 36.1627, -86.7816, [
        "Nashville Broadway honky tonk bars with neon",
        "Nashville Parthenon replica in Centennial Park",
        "Tennessee rural landscape with rolling hills",
    ]),
    ("charlotte", 35.2271, -80.8431, [
        "Charlotte North Carolina downtown skyline",
        "Charlotte NASCAR Hall of Fame building",
        "North Carolina pine forest suburban neighborhood",
    ]),
    ("atlanta", 33.7490, -84.3880, [
        "Atlanta Georgia skyline with Mercedes-Benz dome",
        "Atlanta BeltLine trail with street art murals",
        "Atlanta Peachtree Street downtown corridor",
    ]),
    ("new_orleans", 29.9511, -90.0715, [
        "New Orleans French Quarter wrought iron balconies",
        "New Orleans Bourbon Street jazz club neon",
        "New Orleans St. Louis Cathedral in Jackson Square",
    ]),
    ("pittsburgh", 40.4406, -79.9959, [
        "Pittsburgh three rivers with yellow bridges",
        "Pittsburgh Strip District market stalls",
        "Pittsburgh incline railway on steep hillside",
    ]),
    ("cincinnati", 39.1031, -84.5120, [
        "Cincinnati Roebling Suspension Bridge",
        "Cincinnati Findlay Market interior",
        "Ohio River cityscape at twilight",
    ]),
    ("minneapolis", 44.9778, -93.2650, [
        "Minneapolis Stone Arch Bridge over Mississippi River",
        "Minneapolis Mill District stone mills and skyline",
        "Minneapolis Mall of America interior",
    ]),
    ("tampa", 27.9506, -82.4572, [
        "Tampa Bay waterfront with Convention Center",
        "Tampa Ybor City cigar factory district",
        "Florida Gulf Coast sunset over mangroves",
    ]),
    ("orlando", 28.5383, -81.3792, [
        "Orlando International Drive tourist strip",
        "Orlando Lake Eola fountain downtown",
        "Florida theme park entrance with crowds",
    ]),
    ("las_vegas", 36.1699, -115.1398, [
        "Las Vegas Strip casino hotels neon lights at night",
        "Las Vegas Fremont Street Experience canopy light show",
        "Nevada desert highway approaching Las Vegas",
    ]),
    ("honolulu", 21.3069, -157.8583, [
        "Honolulu Waikiki Beach with Diamond Head crater",
        "Honoluluolulu Chinatown market with lei sellers",
        "Hawaiian tropical urban street with plumeria trees",
    ]),
    ("salt_lake_city", 40.7608, -111.8910, [
        "Salt Lake City Temple Square with white granite temple",
        "Salt Lake City downtown with Wasatch Mountains",
        "Utah red rock desert landscape",
    ]),
    ("savannah", 32.0809, -81.0912, [
        "Savannah Georgia oak trees with Spanish moss",
        "Savannah historic district cobblestone squares",
        "Savannah Forsyth Park fountain",
    ]),
    ("charleston_sc", 32.7765, -79.9311, [
        "Charleston South Carolina pastel row houses",
        "Charleston Waterfront Battery promenade",
        "Charleston cobblestone street with horse carriage",
    ]),
    ("santa_fe", 35.6870, -105.9378, [
        "Santa Fe adobe buildings with turquoise doors",
        "Santa Fe Plaza with Native American market stalls",
        "New Mexico high desert landscape with sagebrush",
    ]),
    ("key_west", 24.5551, -81.7800, [
        "Key West Duval Street with tropical bars and roosters",
        "Key West pastel conch houses with gingerbread trim",
        "Key West sunset celebration at Mallory Square",
    ]),
    ("asheville", 35.5951, -82.5515, [
        "Asheville Blue Ridge Parkway mountain overlook",
        "Asheville downtown street art and brewery district",
        "Biltmore Estate chateau in mountain setting",
    ]),
    ("annapolis", 38.9784, -76.4922, [
        "Annapolis Maryland harbor with sailboats",
        "Annapolis historic district colonial brick buildings",
        "US Naval Academy chapel dome",
    ]),
    ("boston", 42.3601, -71.0589, [
        "Boston Freedom Trail red brick walking path",
        "Boston Fenway Park Green Monster wall",
        "Boston Harbor with historic tall ships",
    ]),
    ("miami", 25.7617, -80.1918, [
        "Miami South Beach Art Deco pastel buildings on Ocean Drive",
        "Miami skyline with Biscayne Bay reflections",
        "Miami Wynwood Walls colorful street art murals",
    ]),
    ("philadelphia", 39.9526, -75.1652, [
        "Philadelphia Independence Hall and Liberty Bell",
        "Philadelphia Reading Terminal Market food stalls",
        "Philadelphia Rocky steps at Art Museum",
    ]),
    ("detroit", 42.3314, -83.0458, [
        "Detroit Renaissance Center glass towers",
        "Detroit Michigan Central Station abandoned grand lobby",
        "Detroit Motown Hitsville USA house museum",
    ]),
    ("new_haven", 41.3083, -72.9279, [
        "New Haven Yale University gothic campus buildings",
        "New Haven pizza restaurant with coal-fired oven",
        "Connecticut colonial green town square",
    ]),
    ("saint_louis", 38.6270, -90.1994, [
        "St. Louis Gateway Arch stainless steel curve",
        "St. Louis Forest Park art museum exterior",
        "Mississippi River barge traffic near St. Louis",
    ]),
    ("chicago", 41.8781, -87.6298, [
        "Chicago Cloud Gate bean sculpture in Millennium Park",
        "Chicago L train elevated tracks over street",
        "Chicago deep dish pizza at a crowded restaurant",
    ]),
    ("san_francisco", 37.7749, -122.4194, [
        "San Francisco Golden Gate Bridge through morning fog",
        "San Francisco cable car climbing steep hill",
        "San Francisco painted Victorian houses on steep street",
    ]),
    ("new_york_city", 40.7128, -74.0060, [
        "New York Manhattan skyline from Brooklyn",
        "New York yellow taxi cab on busy avenue",
        "New York Times Square billboards at night",
    ]),
    # === EUROPE ===
    ("edinburgh", 55.9533, -3.1883, [
        "Edinburgh Castle perched on volcanic rock above the city",
        "Edinburgh Royal Mile cobblestone street with historic closes",
        "Edinburgh colorful Georgian New Town terraces",
    ]),
    ("dublin", 53.3498, -6.2603, [
        "Dublin Temple Bar cobblestone street with pubs",
        "Dublin Ha'penny Bridge over River Liffey",
        "Dublin Georgian doorways with colorful painted doors",
    ]),
    ("copenhagen", 55.6761, 12.5683, [
        "Copenhagen Nyhavn colorful waterfront houses",
        "Copenhagen Tivoli Gardens amusement park lights",
        "Copenhagen bicycle commuter lane on canal bridge",
    ]),
    ("helsinki", 60.1699, 24.9384, [
        "Helsinki Senate Square white cathedral and market",
        "Helsinki Design District modern Scandinavian buildings",
        "Finnish lakeside sauna wooden cottage",
    ]),
    ("gothenburg", 57.7089, 11.9746, [
        "Gothenburg Haga district wooden houses and fika café",
        "Gothenburg canal with old harbor cranes",
        "Swedish west coast rocky archipelago islands",
    ]),
    ("tallinn", 59.4370, 24.7536, [
        "Tallinn Old Town medieval towers and red roofs",
        "Tallinn Town Hall square with Christmas market",
        "Tallinn Alexander Nevsky Cathedral onion domes",
    ]),
    ("riga", 56.9496, 24.1052, [
        "Riga Old Town Art Nouveau architecture facades",
        "Riga Central Market halls in former Zeppelin hangars",
        "Riga Daugava River embankment with bridge",
    ]),
    ("vilnius", 54.6872, 25.2797, [
        "Vilnius Old Town baroque church towers",
        "Vilnius Uzupis bohemian district with art installations",
        "Vilnius Gediminas Tower hilltop panorama",
    ]),
    ("bratislava", 48.1486, 17.1077, [
        "Bratislava Castle white rectangular fortress on hill",
        "Bratislava Old Town narrow pedestrian streets",
        "Danube River bridge with UFO observation tower",
    ]),
    ("zagreb", 45.8150, 15.9819, [
        "Zagreb Upper Town stone gate and St. Mark's colorful roof",
        "Zagreb Dolac market with red umbrellas",
        "Croatian Plitvice Lakes turquoise waterfalls",
    ]),
    ("belgrade", 44.7866, 20.4489, [
        "Belgrade Kalemegdan Fortress overlooking rivers",
        "Belgrade Skadarlija bohemian street with musicians",
        "Serbian Orthodox church dome and fresco interior",
    ]),
    ("sofia", 42.6977, 23.3219, [
        "Sofia Alexander Nevsky Cathedral golden domes",
        "Sofia Vitosha Boulevard pedestrian shopping street",
        "Bulgarian Rila Monastery mountain backdrop",
    ]),
    ("bucharest", 44.4268, 26.1025, [
        "Bucharest Palace of Parliament massive concrete facade",
        "Bucharest Old Town lipscani street with cafes",
        "Romanian painted monasteries of Bucovina",
    ]),
    ("dubrovnik", 42.6507, 18.0944, [
        "Dubrovnik Old Town red rooftops from above",
        "Dubrovnik city walls overlooking Adriatic Sea",
        "Dubrovnik Stradun marble-pedestrian street",
    ]),
    ("split", 43.5081, 16.4402, [
        "Split Diocletian's Palace basement columns",
        "Split Riva waterfront promenade with palm trees",
        "Croatian Dalmatian coast island hopping view",
    ]),
    ("santorini", 36.3932, 25.4615, [
        "Santorini Oia blue dome churches and white buildings",
        "Santorini caldera sunset over volcanic cliffs",
        "Santorini narrow cobblestone alley with bougainvillea",
    ]),
    ("florence", 43.7696, 11.2558, [
        "Florence Duomo Brunelleschi red-tiled dome",
        "Florence Ponte Vecchio bridge with gold shops",
        "Florence Arno River reflection at golden hour",
    ]),
    ("naples", 40.8518, 14.2681, [
        "Naples Spanish Steps with Mount Vesuvius in background",
        "Naples Spaccanapoli narrow street with laundry overhead",
        "Naples pizza restaurant with wood-fired oven",
    ]),
    ("milan", 45.4642, 9.1900, [
        "Milan Duomo marble Gothic cathedral facade",
        "Milan Galleria Vittorio Emanuele glass ceiling arcade",
        "Milan fashion district Via Montenapoleone shops",
    ]),
    ("bologna", 44.4949, 11.3426, [
        "Bologna porticoed streets with arched walkways",
        "Bologna Two Towers medieval leaning towers",
        "Bologna food market with Parmigiano wheels",
    ]),
    ("verona", 45.4384, 10.9916, [
        "Verona Arena Roman amphitheater with piazza",
        "Verona Ponte Pietra stone bridge over Adige River",
        "Verona Juliet's balcony with rose petals",
    ]),
    ("marseille", 43.2965, 5.3698, [
        "Marseille Vieux Port fishing boats and Notre-Dame de la Garde",
        "Marseille MuCEM glass cube museum by the sea",
        "Marseille bouillabaisse seafood restaurant scene",
    ]),
    ("lyon", 45.7640, 4.8357, [
        "Lyon traboules covered passageways between buildings",
        "Lyon Presqu'île peninsula with Saône River",
        "Lyon bouchon Lyonnais traditional restaurant",
    ]),
    ("nice", 43.7102, 7.2620, [
        "Nice Promenade des Anglais pebble beach and blue chairs",
        "Nice Vieux Nice colorful market streets",
        "Nice Cimiez monastery hilltop gardens",
    ]),
    ("bordeaux", 44.8378, -0.5792, [
        "Bordeaux Place de la Bourse reflected water mirror",
        "Bordeaux wine cellar barrel room underground",
        "Bordeaux limestone Georgian architecture facades",
    ]),
    ("cologne", 50.9375, 6.9603, [
        "Cologne Cathedral twin Gothic spires above rooftops",
        "Cologne Hohenzollern Bridge love locks over Rhine",
        "Cologne Altstadt beer hall with Kölsch glasses",
    ]),
    ("hamburg", 53.5511, 9.9937, [
        "Hamburg Elbphilharmonie glass concert hall on harbor",
        "Hamburg Speicherstadt red-brick warehouse district",
        "Hamburg fish market Sunday morning stalls",
    ]),
    ("munich", 48.1351, 11.5820, [
        "Munich Marienplatz Glockenspiel clock tower",
        "Munich Hofbräuhaus beer hall with communal tables",
        "English Garden Munich river surfers in urban park",
    ]),
    ("dresden", 51.0504, 13.7373, [
        "Dresden Frauenkirche sandstone dome rebuilt baroque church",
        "Dresden Zwinger Palace courtyard with fountain",
        "Dresden Elbe River embankment with paddle steamer",
    ]),
    ("salzburg", 47.8095, 13.0550, [
        "Salzburg Hohens Fortress on hill above baroque city",
        "Salzburg Getreidegasse narrow shopping street with signs",
        "Austrian Alpine meadow with edelweiss flowers",
    ]),
    ("innsbruck", 47.2692, 11.4041, [
        "Innsbruck Golden Roof gothic balcony with copper tiles",
        "Innsbruck Nordkette cable car from city to mountain",
        "Tyrolean valley with painted chalet houses",
    ]),
    ("lucerne", 47.0502, 8.3093, [
        "Lucerne Chapel Bridge painted ceiling panels over river",
        "Lucerne Old Town painted medieval tower facades",
        "Swiss Lake Lucerne with Mount Pilatus backdrop",
    ]),
    ("interlaken", 46.6863, 7.8632, [
        "Interlaken view between two turquoise lakes",
        "Interlaken Jungfrau region mountain panorama",
        "Swiss paragliding over green valley with chalets",
    ]),
    ("zurich", 47.3769, 8.5417, [
        "Zurich Grossmünster church twin towers on Limmat River",
        "Zurich Bahnhofstrasse luxury shopping street",
        "Zurich Old Town Lindenhof hill terrace view",
    ]),
    ("geneva", 46.2044, 6.1432, [
        "Geneva Jet d'Eau water fountain on Lake Geneva",
        "Geneva St. Pierre Cathedral in Old Town",
        "Swiss Geneva lakefront with Mont Blanc in distance",
    ]),
    ("bilbao", 43.2630, -2.9350, [
        "Bilbao Guggenheim Museum titanium curved exterior",
        "Bilbao Nervión River reflecting glass buildings",
        "Basque Country pintxos bar with skewered snacks",
    ]),
    ("seville", 37.3891, -5.9845, [
        "Seville Plaza de España curved tile alcoves",
        "Seville Alcázar Moorish palace courtyard with fountain",
        "Seville flamenco dancer on cobblestone street",
    ]),
    ("granada", 37.1773, -3.5986, [
        "Granada Alhambra palace red fortress walls and gardens",
        "Granada Albayzín white-washed hillside neighborhood",
        "Granada Sacromonte cave houses with flamenco shows",
    ]),
    ("malaga", 36.7213, -4.4214, [
        "Malaga Picasso Museum courtyard in old fortress",
        "Malaga port Paseo del Parque palm-lined promenade",
        "Costa del Sol beach resort with white apartment buildings",
    ]),
    ("porto", 41.1579, -8.6291, [
        "Porto Dom Luís I Bridge double-deck over Douro River",
        "Porto Ribeira colorful waterfront houses stacked on hill",
        "Porto Livraria Lello ornate bookshop interior staircase",
    ]),
    ("farol", 37.0194, -7.9304, [
        "Algarve Faro old town walls and marina",
        "Algarve Benagil sea cave with sunlight hole",
        "Portugal cork oak landscape in Alentejo region",
    ]),
    ("budapest", 47.4979, 19.0402, [
        "Budapest Parliament building illuminated along Danube",
        "Budapest Széchenyi thermal bath outdoor pool",
        "Budapest Fisherman's Bastion white turrets overlooking city",
    ]),
    ("krakow", 50.0647, 19.9450, [
        "Kraków Main Market Square cloth hall and St. Mary trumpet",
        "Kraków Wawel Castle dragon den entrance",
        "Kraków Kazimierz Jewish quarter courtyard",
    ]),
    ("wroclaw", 51.1079, 17.0385, [
        "Wrocław dwarf statues scattered on sidewalks",
        "Wrocław Market Hall colorful stalls under iron roof",
        "Wrocław Ostrów Tumski island cathedral at dusk",
    ]),
    # === ASIA ===
    ("chiang_mai", 18.7883, 98.9853, [
        "Chiang Mai old city temple moat and walls",
        "Chiang Mai Sunday night market walking street stalls",
        "Thai mountain temple with golden chedi surrounded by jungle",
    ]),
    ("phnom_penh", 11.5564, 104.9282, [
        "Phnom Penh Royal Palace silver pagoda golden spires",
        "Phnom Penh riverside promenade at sunset",
        "Cambodian floating village houses on Tonlé Sap",
    ]),
    ("siem_reap", 13.3633, 103.8600, [
        "Angkor Wat temple reflection in morning moat",
        "Angkor Thom Bayon stone faces tower",
        "Siem Reap Pub Street night market with lanterns",
    ]),
    ("hoi_an", 15.8801, 108.3380, [
        "Hội An Ancient Town yellow walls and lantern-lit river",
        "Hội An Japanese Covered Bridge over canal",
        "Hội An tailor shop with colorful silk fabrics",
    ]),
    ("kathmandu", 27.7172, 85.3240, [
        "Kathmandu Durbar Square pagoda temples and market",
        "Kathmandu Swayambhunath monkey temple stupa eyes",
        "Nepalese Himalayan prayer flags on mountain pass",
    ]),
    ("colombo", 6.9271, 79.8612, [
        "Colombo Galle Face Green oceanfront promenade",
        "Colombo Pettah market bazaar crowded streets",
        "Sri Lankan tea plantation hillside with pickers",
    ]),
    ("chengdu", 30.5728, 104.0668, [
        "Chengdu Research Base giant panda enclosure",
        "Chengdu Jinli ancient street with red lanterns",
        "Sichuan hot pot restaurant with boiling red broth",
    ]),
    ("xi_an", 34.3416, 108.9398, [
        "Xi'an Terracotta Warriors army rows underground",
        "Xi'an City Wall cycling on ancient ramparts",
        "Xi'an Muslim Quarter food street with lamb skewers",
    ]),
    ("hangzhou", 30.2741, 120.1551, [
        "Hangzhou West Lake pagoda and willow-lined causeway",
        "Hangzhou Longjing tea plantation terraced hills",
        "Hangzhou Hefang Street traditional medicine shops",
    ]),
    ("nanjing", 32.0603, 118.7969, [
        "Nanjing Confucius Temple Qinhuai River lantern-lit boats",
        "Nanjing Sun Yat-sen Mausoleum blue-tiled grand staircase",
        "Nanjing city wall with Purple Mountain in background",
    ]),
    ("shenzhen", 22.5431, 114.0579, [
        "Shenzhen futuristic skyline with glass skyscrapers",
        "Shenzhen Huaqiangbei electronics market towers",
        "Shenzhen window of the world miniature landmarks park",
    ]),
    ("guangzhou", 23.1291, 113.2644, [
        "Guangzhou Canton Tower LED spiral at night",
        "Guangzhou Shamian Island colonial European buildings",
        "Canton dim sum bamboo steamer baskets restaurant",
    ]),
    ("xiamen", 24.4798, 118.0894, [
        "Xiamen Gulangyu Island piano museum Victorian villas",
        "Xiamen South Putuo Temple entrance gate",
        "Fujian Tulou circular earthen building communal",
    ]),
    ("qingdao", 36.0671, 120.3826, [
        "Qingdao Tsingtao Brewery German colonial red-roof buildings",
        "Qingdao Zhanqiao Pier octagonal pavilion over sea",
        "Qingdao May Fourth Square red sculpture by coastline",
    ]),
    ("osaka", 34.6937, 135.5023, [
        "Osaka Dotonbori canal with giant crab and Glico man neon",
        "Osaka Castle white walls and gold fish-hawk ornaments",
        "Osaka Shinsekai Tsutenkaku Tower retro district",
    ]),
    ("kyoto", 35.0116, 135.7681, [
        "Kyoto Fushimi Inari shrine vermillion torii gate tunnel",
        "Kyoto Arashiyama bamboo grove towering green stalks",
        "Kyoto Kinkaku-ji golden pavilion reflected in mirror pond",
    ]),
    ("hiroshima", 34.3853, 132.4553, [
        "Hiroshima Peace Memorial skeletal dome building",
        "Hiroshima Miyajima floating torii gate at high tide",
        "Japanese okonomiyaki pancake on griddle restaurant",
    ]),
    ("nara", 34.6851, 135.8048, [
        "Nara deer park bowing deer with temple backdrop",
        "Nara Todai-ji wooden temple world's largest Buddha",
        "Nara Kasuga Taisha stone lantern path through forest",
    ]),
    ("tokyo", 35.6762, 139.6503, [
        "Tokyo Shibuya crossing crowds under giant neon screens",
        "Tokyo Akihabara electric town with anime billboards",
        "Tokyo Senso-ji temple Kaminarimon thunder gate red lantern",
        "Tokyo Shinjuku golden gai narrow bar alley",
        "Tokyo Meiji Shrine wooden gate in forest",
    ]),
    ("hakone", 35.2324, 139.1069, [
        "Hakone open-air museum sculpture park with mountains",
        "Hakone Lake Ashi pirate ship with Mount Fuji",
        "Hakone onsen hot spring ryokan traditional inn",
    ]),
    ("kanazawa", 36.5613, 136.6562, [
        "Kanazawa Kenroku-en garden with snow-covered lanterns",
        "Kanazawa Higashi Chaya geisha district wooden lattice",
        "Kanazawa Omicho market fresh seafood stalls",
    ]),
    ("busan", 35.1796, 129.0756, [
        "Busan Haeundae Beach high-rise resort skyline",
        "Busan Gamcheon Culture Village pastel hillside houses",
        "Busan Jagalchi fish market with fresh catch displays",
    ]),
    ("jeju", 33.4890, 126.4983, [
        "Jeju Hallasan volcanic peak above tangerine orchards",
        "Jeju Manjanggul lava tube cave interior",
        "Jeju Haenyeo women divers on rocky shore",
    ]),
    ("hanoi", 21.0278, 105.8342, [
        "Hanoi Old Quarter narrow streets with tube houses",
        "Hanoi Hoàn Kiếm Lake red bridge to Jade Island temple",
        "Hanoi street phở vendor with steaming bowls",
    ]),
    ("taipei", 25.0330, 121.5654, [
        "Taipei 101 tower rising above city skyline",
        "Taipei Shilin Night Market neon signs and food stalls",
        "Taipei Longshan Temple ornate dragon roof decorations",
    ]),
    ("kuala_lumpur", 3.1390, 101.6869, [
        "Kuala Lumpur Petronas Twin Towers silver spires at night",
        "Kuala Lumpur Batu Caves Hindu temple golden statue stairs",
        "Jalan Alor night market food stalls with wok flames",
    ]),
    ("singapore", 1.3521, 103.8198, [
        "Singapore Marina Bay Sands infinity pool rooftop",
        "Singapore Gardens by the Bay supertrees light show",
        "Singapore Chinatown shophouses with red lanterns",
    ]),
    ("jakarta", -6.2088, 106.8456, [
        "Jakarta Bundaran HI fountain roundabout skyscrapers",
        "Jakarta Kota Tua Dutch colonial old town square",
        "Indonesian nasi goreng street food cart with wok",
    ]),
    ("yogyakarta", -7.7956, 110.3695, [
        "Yogyakarta Borobudur temple stupa dome sunrise",
        "Yogyakarta Prambanan Hindu temple tall spires",
        "Yogyakarta Malioboro street with batik shops and rickshaws",
    ]),
    ("manila", 14.5995, 120.9842, [
        "Manila Intramuros walled city Spanish colonial gate",
        "Manila skyline Makati CBD glass towers at dusk",
        "Manila jeepney colorful decorated public transport",
    ]),
    ("hong_kong", 22.3193, 114.1694, [
        "Hong Kong Victoria Peak skyline view with harbor",
        "Hong Kong Mongkok neon signs over crowded streets",
        "Hong Kong Star Ferry crossing Victoria Harbour at night",
    ]),
    ("macau", 22.1987, 113.5439, [
        "Macau Ruins of St. Paul's stone facade steps",
        "Macau Cotai Strip casino resort buildings",
        "Macau Portuguese egg tart bakery counter",
    ]),
    ("shanghai", 31.2304, 121.4737, [
        "Shanghai Pudong Lujiazui futuristic skyline Oriental Pearl Tower",
        "Shanghai Bund colonial buildings illuminated at night",
        "Shanghai Yu Garden traditional Chinese pavilion and pond",
    ]),
    ("beijing", 39.9042, 116.4074, [
        "Beijing Forbidden City red walls and golden roofs gate",
        "Beijing Great Wall winding along mountain ridge",
        "Beijing hutong alleyway with grey brick courtyard gates",
    ]),
    ("ulaanbaatar", 47.8864, 106.9057, [
        "Ulaanbaatar Sükhbaatar Square with blue sky Ger district",
        "Mongolian steppe with felt Ger tent and horses",
        "Ulaanbaatar Gandantegchinlen monastery golden Buddha",
    ]),
    # === MIDDLE EAST ===
    ("amman", 31.9454, 35.9284, [
        "Amman Citadel ancient Roman pillars above city",
        "Amman rainbow street pastel buildings and cafes",
        "Jordanian Petra Treasury carved into rose-red cliff",
    ]),
    ("muscat", 23.5880, 58.3829, [
        "Muscat Sultan Qaboos mosque white marble and golden dome",
        "Muscat Muttrah souk spice and frankincense market",
        "Omani wadi turquoise pools in desert canyon",
    ]),
    ("doha", 25.2854, 51.5310, [
        "Doha skyline futuristic towers along Corniche",
        "Doha Souq Waqif traditional market with falcon shops",
        "Qatar desert sand dunes with camel caravan",
    ]),
    ("beirut", 33.8938, 35.5018, [
        "Beirut downtown reconstructed Ottoman and French buildings",
        "Beirut Corniche seaside promenade at sunset",
        "Lebanese mezze spread of small dishes on table",
    ]),
    ("jerusalem", 31.7683, 35.2137, [
        "Jerusalem Old City golden Dome of the Rock",
        "Jerusalem Western Wall prayer plaza with worshippers",
        "Jerusalem Via Dolorosa narrow stone market street",
    ]),
    ("antalya", 36.8969, 30.7133, [
        "Antalya old town Kaleiçi Ottoman houses and harbor",
        "Turkish Riviera turquoise Mediterranean water and cliffs",
        "Antalya Aspendos ancient Roman theater marble seats",
    ]),
    ("cappadocia", 38.6431, 34.8293, [
        "Cappadocia fairy chimney rock formations and cave hotels",
        "Cappadocia hot air balloons over sunrise valley",
        "Cappadocia underground city carved rock chambers",
    ]),
    ("tbilisi", 41.7151, 44.8271, [
        "Tbilisi Old Town colorful wooden balconies on sulfur bath quarter",
        "Tbilisi Trinity Cathedral gold dome on hilltop",
        "Georgian wine qvevri clay pot buried in ground cellar",
    ]),
    ("yerevan", 40.1792, 44.4991, [
        "Yerevan Republic Square pink tuff stone buildings",
        "Yerevan Cascade white limestone staircase with fountains",
        "Ararat mountain view over Yerevan city panorama",
    ]),
    ("baku", 40.4093, 49.8671, [
        "Baku Flame Towers glass skyscrapers shaped like flames",
        "Baku Old City Icherisheher stone walls and Maiden Tower",
        "Baku waterfront promenade along Caspian Sea",
    ]),
    ("mashhad", 36.2972, 59.5736, [
        "Mashhad Imam Reza shrine golden dome and blue tiles",
        "Iranian bazaar corridor with spice and carpet vendors",
        "Persian garden with reflecting pool and cypress trees",
    ]),
    ("isfahan", 32.6546, 51.6680, [
        "Isfahan Naqsh-e Jahan Square blue tile mosque and palace",
        "Isfahan Khaju Bridge arched reflection in Zayandeh River",
        "Isfahan bazaar muqarnas ceiling with turquoise tiles",
    ]),
    # === AFRICA ===
    ("lagos", 6.5244, 3.3792, [
        "Lagos Makoko floating village wooden stilts on lagoon",
        "Lagos Victoria Island modern skyline and traffic",
        "Nigerian market stall with colorful wax print fabrics",
    ]),
    ("nairobi", -1.2921, 36.8219, [
        "Nairobi skyline with Nairobi National Park giraffes in foreground",
        "Nairobi Maasai market colorful beadwork and crafts",
        "Kenyan savanna acacia tree silhouette at sunset",
    ]),
    ("addis_ababa", 9.0250, 38.7469, [
        "Addis Ababa Holy Trinity Cathedral copper domes",
        "Ethiopian coffee ceremony ceremony with incense",
        "Ethiopian Rift Valley lake flamingo flocks",
    ]),
    ("cape_town", -33.9249, 18.4241, [
        "Cape Town Table Mountain flat top with cableway",
        "Cape Town Bo-Kaap brightly colored houses on slope",
        "Cape Town V&A Waterfront harbor with seal colony",
    ]),
    ("marrakech", 31.6295, -7.9811, [
        "Marrakech Jemaa el-Fnaa square with snake charmers and stalls",
        "Marrakech souk narrow alley with hanging leather goods",
        "Marrakech riad courtyard with plunge pool and orange trees",
    ]),
    ("zanzibar", -6.1659, 39.2026, [
        "Zanzibar Stone Town carved wooden doors and narrow alleys",
        "Zanzibar dhow sailing boat on turquoise Indian Ocean",
        "Zanzibar spice plantation tour with tropical fruits",
    ]),
    ("kigali", -1.9403, 29.8739, [
        "Kigali hillside city with red-roof houses and green hills",
        "Rwanda mountain gorilla in misty bamboo forest",
        "Kigali modern market building with local vendors",
    ]),
    ("accra", 5.6037, -0.1870, [
        "Accra Makola market crowded stalls with textiles",
        "Accra Jamestown lighthouse and fishing boats",
        "Ghanaian kente cloth weaving loom colorful patterns",
    ]),
    ("dakar", 14.7167, -17.4677, [
        "Dakar African Renaissance Monument tall bronze statue",
        "Dakar Île de Gorée colonial buildings and slave house",
        "Senegalese fishing pirogue boats painted colorful on beach",
    ]),
    ("tunis", 36.8065, 10.1815, [
        "Tunis medina alleyway with blue and white walls",
        "Tunis Bardo Museum Roman mosaic collection",
        "Tunisian harissa chili paste and couscous food scene",
    ]),
    ("fez", 34.0331, -5.0003, [
        "Fez medina world's largest car-free urban area",
        "Fez Chouara tannery colorful dye vats from above",
        "Fez Bou Inania madrasa carved stucco and cedar ceiling",
    ]),
    ("luxor", 25.6872, 32.6396, [
        "Luxor Temple columns illuminated at night",
        "Luxor Valley of the Kings painted tomb entrance",
        "Egyptian feluca sailboat on Nile River at sunset",
    ]),
    ("windhoek", -22.5609, 17.0658, [
        "Windhoek German colonial Christuskirche white stone church",
        "Namibian desert dune landscape at golden hour",
        "Etosha salt pan with wildlife at waterhole",
    ]),
    ("kilimanjaro", -3.0674, 37.3556, [
        "Mount Kilimanjaro snow-capped peak above savanna",
        "Tanzanian coffee plantation green hills with Mount in background",
        "Kilimanjaro Machame route hiking through cloud forest",
    ]),
    # === SOUTH AMERICA ===
    ("lima", -12.0464, -77.0428, [
        "Lima Miraflores cliffs with Pacific Ocean and paragliders",
        "Lima historic center Plaza Mayor colonial yellow buildings",
        "Lima ceviche restaurant with fresh fish display",
    ]),
    ("bogota", 4.7110, -74.0721, [
        "Bogotá La Candelaria colorful colonial houses",
        "Bogotá Monserrate funicular railway to mountain top",
        "Colombian emerald market with jeweler's loupe",
    ]),
    ("medellin", 6.2476, -75.5658, [
        "Medellín Communa 13 colorful painted hillside houses",
        "Medellín Metrocable gondola over green valley",
        "Medellín Plaza Botero oversized bronze sculptures",
    ]),
    ("quito", -0.1807, -78.4678, [
        "Quito Old Town baroque churches and narrow streets",
        "Quito Mitad del Mundo equator monument line",
        "Ecuadorian cloud forest orchids and hummingbirds",
    ]),
    ("santiago", -33.4489, -70.6693, [
        "Santiago skyline with Andes snow-capped mountains behind",
        "Santiago Mercado Central seafood market stalls",
        "Chilean vineyard rows with Andes backdrop",
    ]),
    ("la_paz", -16.5000, -68.1193, [
        "La Paz El Alto cable car system over hillside city",
        "La Paz Witches Market llama fetuses and herbal stalls",
        "Bolivian altiplano landscape with vicuñas grazing",
    ]),
    ("montevideo", -34.9011, -56.1645, [
        "Montevideo Ciudad Vieja Plaza Independencia gateway arch",
        "Montevideo Rambla waterfront promenade along River Plate",
        "Uruguayan asado barbecue with chimichurri on ranch",
    ]),
    ("cartagena", 10.3910, -75.5144, [
        "Cartagena Old Town colorful balconied colonial houses",
        "Cartagena Walled City clock tower plaza",
        "Cartagena Getsemaní street art murals and nightlife",
    ]),
    ("cusco", -13.5320, -71.9675, [
        "Cusco Plaza de Armas Inca stone walls under colonial church",
        "Cusco San Pedro market fruit juice stalls",
        "Machu Picchu stone terraces in misty mountain setting",
    ]),
    ("valparaiso", -33.0472, -71.6127, [
        "Valparaíso hillside houses painted rainbow colors",
        "Valparaíso ascensor funicular wooden car on steep tracks",
        "Chilean street art mural on corrugated metal wall",
    ]),
    ("manaus", -3.1190, -60.0217, [
        "Manaus Teatro Amazonas pink opera house with green dome",
        "Meeting of Waters dark Rio Negro meets sandy Amazon",
        "Amazon rainforest canopy walkway bridge",
    ]),
    ("recife", -8.0476, -34.8770, [
        "Recife old town bridge over waterway with colorful buildings",
        "Recife Marco Zero waterfront square compass rose mosaic",
        "Brazilian frevo dancer with colorful umbrella carnival",
    ]),
    ("buenos_aires", -34.6037, -58.3816, [
        "Buenos Aires La Boca Caminito colorful tin houses",
        "Buenos Aires Recoleta Cemetery ornate mausoleum rows",
        "Buenos Aires San Telmo antique market with tango dancers",
    ]),
    ("rio_de_janeiro", -22.9068, -43.1729, [
        "Rio de Janeiro Christ the Redeemer statue with city below",
        "Rio Copacabana beach black and white wave sidewalk",
        "Rio Sugarloaf Mountain cable car over Guanabara Bay",
    ]),
    ("ushuaia", -54.8019, -68.3030, [
        "Ushuaia Beagle Channel with lighthouse and snow mountains",
        "Ushuaia end of world sign with colorful buildings",
        "Tierra del Fuego national park lenga forest path",
    ]),
    # === CENTRAL ASIA ===
    ("almaty", 43.2220, 76.8512, [
        "Almaty Central State Market green pyramid glass roof",
        "Almaty mountain backdrop with Soviet-era apartment blocks",
        "Kazakh beshbarmak horse meat dish traditional feast",
    ]),
    ("tashkent", 41.2995, 69.2401, [
        "Tashkent Chorsu bazaar giant green dome market hall",
        "Tashkent metro ornate Soviet-era station chandeliers",
        "Uzbek plov rice dish in giant kazan cauldron",
    ]),
    ("samarkand", 39.6542, 66.9597, [
        "Samarkand Registan madrasa blue tile mosaic facade",
        "Samarkand Shah-i-Zinda tiled mausoleum street",
        "Uzbek silk road caravanserai courtyard with arches",
    ]),
    ("bukhara", 39.7747, 64.4286, [
        "Bukhara old town covered bazaar trading domes",
        "Bukhara Ark fortress wall and entrance gate",
        "Bukhara Kalyan minaret and mosque against blue sky",
    ]),
    # === OCEANIA ===
    ("christchurch", -43.5321, 172.6362, [
        "Christchurch cardboard cathedral modern architecture",
        "Christchurch Botanic Gardens rose garden in park",
        "New Zealand Canterbury Plains flat farmland with mountains",
    ]),
    ("cairns", -16.9186, 145.7781, [
        "Cairns Esplanade lagoon pool with mountain backdrop",
        "Great Barrier Reef coral reef snorkeling underwater",
        "Tropical North Queensland Daintree rainforest boardwalk",
    ]),
    ("fiji", -17.7134, 178.0650, [
        "Fiji white sand beach with overwater bure and palm trees",
        "Fiji Sabeto mud pool hot springs volcanic landscape",
        "Fijian village bure traditional thatched roof house",
    ]),
    # === CARIBBEAN ===
    ("jamaica", 18.1096, -77.2975, [
        "Jamaica Dunn's River Falls cascading into tropical pool",
        "Jamaica Kingston street scene with reggae culture",
        "Jamaican Blue Mountains coffee plantation hillside",
    ]),
    ("cuba", 23.1136, -82.3666, [
        "Cuba Havana Malecón waterfront with vintage American cars",
        "Cuba Trinidad colonial town pastel buildings cobblestone",
        "Cuban tobacco farm drying barn with cured leaves",
    ]),
    ("barbados", 13.1939, -59.5432, [
        "Barbados Crane Beach pink sand cliff and turquoise water",
        "Bridgetown historic garrison colonial buildings",
        "Barbadian rum distillery copper still and sugarcane",
    ]),
    ("trinidad", 10.6918, -61.2225, [
        "Trinidad Carnival costume dancers on street",
        "Port of Spain Brian Lara skyline with Aripo Towers",
        "Tobago Nylon Pool crystal clear shallow Caribbean water",
    ]),
]


def build_prompts_dict():
    """Build the _CITY_PROMPTS entries."""
    lines = []
    for key, lat, lon, prompts in NEW_CITIES:
        lines.append(f'    "{key}": [')
        for p in prompts:
            lines.append(f'        "{p}",')
        lines.append('    ],')
    return '\n'.join(lines)


def build_centroids_dict():
    """Build the _CITY_CENTROIDS entries."""
    lines = []
    for key, lat, lon, prompts in NEW_CITIES:
        lines.append(f'    "{key}": ({lat:.4f}, {lon:.4f}),')
    return '\n'.join(lines)


if __name__ == "__main__":
    prompts = build_prompts_dict()
    centroids = build_centroids_dict()
    
    with open("W:/geofind/dev/new_city_prompts.txt", "w") as f:
        f.write(prompts)
    with open("W:/geofind/dev/new_city_centroids.txt", "w") as f:
        f.write(centroids)
    
    print(f"Generated {len(NEW_CITIES)} new city entries")
    print(f"Prompts: {len(prompts)} chars")
    print(f"Centroids: {len(centroids)} chars")
    print(f"Total cities after merge: {167 + len(NEW_CITIES)}")
