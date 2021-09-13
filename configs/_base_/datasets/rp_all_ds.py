data_root = '/home/ubuntu/data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ],
    )
]

logos_ds_classes = ['1002', '102', '103', '1052', '1102', '1152', '1252', '12521', '1302', '1352', '1452', '1454', '1504', '1506', '152', '1556', '1606', '1656', '1706', '1756', '1844', '1845', '1846', '1847', '1848', '1849', '1850', '1851', '19088', '2', '202', '203', '204', '21761', '21762', '21763', '21764', '21765', '21766', '21767', '21768', '21769', '21770', '21771', '21772', '21773', '21774', '21775', '21776', '21777', '21778', '21779', '21780', '21781', '21782', '21783', '21784', '21785', '21786', '21787', '21788', '21789', '21790', '21791', '21792', '21793', '21794', '21795', '21796', '21797', '21798', '21799', '21800', '21801', '21802', '21803', '21804', '21805', '21806', '21807', '21808', '21809', '21810', '21811', '21812', '21813', '21814', '21815', '21816', '21817', '21818', '21819', '21820', '21821', '21822', '21823', '21824', '21825', '21826', '21827', '21828', '21829', '21830', '21831', '21832', '21833', '21834', '21835', '21836', '21837', '21838', '21839', '21840', '21841', '21842', '21843', '21844', '21845', '21846', '21847', '21848', '21849', '21850', '21851', '21852', '21853', '21854', '21855', '21856', '21857', '21858', '21859', '21860', '21861', '21862', '21864', '21865', '21866', '21867', '21868', '21869', '21870', '21871', '21872', '21873', '21874', '21875', '21876', '21877', '21878', '21879', '21880', '21881', '21882', '21883', '21884', '21885', '21886', '21887', '21888', '21889', '21890', '21891', '21892', '21893', '21894', '21895', '21896', '21897', '21898', '21899', '21900', '21901', '21902', '21903', '21904', '21905', '21906', '21907', '21908', '21909', '21910', '21911', '21912', '21913', '21914', '21915', '21916', '21917', '21918', '21919', '21921', '21922', '21923', '21924', '21925', '21926', '21927', '21928', '21929', '21930', '21931', '21932', '21933', '21934', '21935', '21936', '21937', '21938', '21939', '21940', '21941', '21942', '21943', '21944', '21945', '21946', '21947', '21948', '21949', '21950', '21951', '21952', '21953', '21954', '21956', '21957', '21958', '252', '253', '302', '303', '352', '353', '354', '402', '4248', '4310', '4311', '4313', '4314', '4315', '4316', '4317', '4318', '4325', '4331', '4332', '4333', '4334', '4335', '4346', '4347', '4350', '4351', '4361', '4365', '4366', '4370', '4371', '4388', '4414', '452', '4832', '503', '52', '53', '5320', '552', '5524', '602', '652', '702', '752', '802', '852', '902', '952']
logodet_classes = ["2xist", "3D-GOLD", "4Skins", "66 North", "7-PE", "76", "A. Favre & Fils", "A.P.C", "AFuBeiBei", "AIMER", "AKOO", "AND1", "ANNICK GOUTAL", "ANTA", "AOKANG", "ASPINAL OF LONDON", "Abercrombie", "Abercrombie & Fitch", "Acne Studios", "Acqua Limone", "Addicted", "Adidas SB", "Admiral", "Adriatica", "Aerosoles", "Agacio", "Agnes b", "Ahnu", "Aigle", "Airness", "Akris", "Alain Figaret", "Alchemist", "Alfani", "Alfred Dunhill", "Allen Edmonds", "Allrounder", "Alpina Watches", "Altra", "American Apparel", "American Eagle", "Andrew Christian", "Andrew Marc", "Anna Sui", "Anne Klein", "Antony Morato", "Aquascutum", "Aravon", "Arc'teryx", "Arcopedico", "Ariat", "Aristoc", "Aritzia", "Armani", "Armani Exchange", "Armani Junior", "Arri", "Athletic DNA", "Atletica", "Atticus", "Audemars Piguet", "Ayilian", "BALENO", "BOSE", "BUCO", "Baci Lingerie", "Badgley Mischka", "Baggallini", "Balenciaga", "Bally Shoe", "Balmain", "Barbie", "Barcode Berlin", "Bare Necessities", "Barneys New York", "Baume et Mercier-1", "Baume et Mercier-2", "Beach Bunny", "BeautiFeel", "BeiJiRong", "Bella Vita", "Ben Davis", "Berghaus", "Bestseller", "Betsey Johnson", "Betty Crocker", "Bill Blass", "Birkenstock", "Birki's", "Bjorn Borg", "Black Label Skateboards", "Black-eyed Pea", "Blondo", "Bloomingdale's", "Blundstone", "Bob Evans Restaurants", "Boboli", "Boconi", "Bomber", "Bonds", "Bonia", "Bootights", "Boscov's", "Bosideng", "Bostonian", "Bottega Veneta", "Boxfresh", "Breitling", "British Knights", "Bulgari", "Bulova", "Bunker", "Burlington Coat Factory", "C-IN2", "CDELALA", "Cacharel", "CaiBai", "Calvin Klein", "Calvin Klein Underwear", "Campmor", "Canada Goose", "Canterbury", "Capezio", "Captain Planet", "Carbrini", "Care Bears", "Carhartt", "Carlo Ferrara", "Casio", "Caslon", "Ceceba", "Celine", "Cesare Paciotti", "Chaco", "Champion", "Chipotle Mexican Grill-1", "Chipotle Mexican Grill-2", "Chippewa", "Chiquita", "Chloe", "Chooka", "Chopard", "Christian Lacroix", "Christian Louboutin", "Chrome Hearts", "Church's", "Churchkey Can", "Circa", "Ck Calvin Klein-1", "Ck Calvin Klein-2", "Clarks", "Cocksox", "Coldwater Creek", "Columbia", "Converse", "Covert", "Crumpler", "EKCO", "ERAL", "ERKE-1", "ERKE-2", "Earthies", "Easy Spirit", "Ecco", "Ecko Unltd", "Ed Hardy", "Eddie Bauer", "Eileen Fisher", "Elder-Beerman", "Ella Moss", "Ellesse", "Emerica", "Emilio", "Emilio Pucci", "Era", "Ergowear", "Ermanno Scervino", "Ermenegildo Zegna", "Errea", "Etnies", "Etro", "Everlast", "Excelsior", "Exofficio", "FUBU", "FUCT", "Faconnable", "Frank Dandy", "Frankie Morello", "Freegun", "French Connection", "French Sole", "Freya", "Frye", "Gabor", "Galvin Green", "Gap", "Geographics", "Georg Jensen", "Georgia Boot", "Gibson", "Giorgio", "Givova", "Glock-1", "Glock-2", "Golite", "Goya", "Grundies", "Guess", "Gul", "Guy Cotten", "Guy Laroche", "HBC", "HOM", "Haglofs", "Hammary", "Hanes", "Hanro", "Happy Socks", "Harry Winston-1", "Harry Winston-2", "Head", "Heat-1", "Heat-2", "Heavy Eco", "Heckler & Koch", "Helsport", "Henri Wintermans", "Herschel Supply", "Hilleberg", "Hoka One One", "Hoya", "Hugo Boss", "Hummel-1", "Hummel-2", "Hurley International", "Hurley-1", "Hurley-2", "Hush Puppies", "Imperial", "Independent Truck", "Injinji", "InterCall", "Intymen", "Invicta", "Isabel Maran-1", "Isabel Maran-2", "Izod", "JOULES", "Joe Fresh", "John Galliano", "John Richmond", "John Varvatos", "Joma-1", "Joma-2", "Josef Seibel", "KAIER", "KATE SPADE-1", "KATE SPADE-2", "KELA", "KaZhuMi", "Karhu", "Karl Kani-1", "Karl Kani-2", "Karl Lagerfeld", "Kathmandu", "Kavu", "Kelty", "Kenneth Cole", "Kenzo", "Kiton", "Knomo", "KooGa", "Kookai", "L.K.Bennett", "La Perla", "Los Angeles", "MARC BY MARC JACOBS", "MBT", "Man O' War", "Manhattan Portage", "adyson", "alpinestars-1", "alpinestars-2", "amnesia", "amy butler", "antonio", "atari 2600-1", "atari 2600-2", "aussieBum", "bobble", "brace yourself", "brine", "bukta", "cartelo", "chanel", "chaps", "charles owo", "charlotte olympia", "charlotte russe", "charming charlie", "chaya", "chilewich", "clement", "conlia", "corneliani", "eBags", "fox river mills", "garanimals", "gigo", "gumby", "gunnar", "habixiong", "he-man", "hubba wheels", "itg", "jianjiang", "jlindeberg-1", "jlindeberg-2", "kasumi", "kensie", "kookaburra", "krink", "l.l.bean", "la miu", "lacoste", "lang sha", "lanidor", "lesportsac", "levis", "loewe", "lonsdale", "looptworks", "loro piana", "louis vuitton-1", "louis vuitton-2", "luciano soprani", "lucky brand", "luminox", "luscious pink", "lyle", "maidenform", "majestic", "mandarina duck", "mansmiling", "marc o'polo", "marchesa", "marina rinaldi", "marni", "maui jim", "max mara", "mephisto", "mesa boogie", "meters bonwe", "mexx", "meyba-1", "meyba-2", "mido", "minnetonka", "miss sixty", "missoni", "mitre", "miu miu", "moncler-1", "moncler-2", "montagu", "montane", "moonbasa", "mossimo", "mountain hardwear", "movado", "moving comfort-1", "moving comfort-2", "muchachomalo", "muzak-1", "muzak-2", "nanjiren", "naot", "napapijri", "nasty pig-1", "nasty pig-2", "naturalizer-1", "naturalizer-2", "new balance-1", "new balance-2", "new era", "nicole lee-1", "nicole lee-2", "nina ricci", "norrona", "obey", "old navy", "orient", "oriocx", "oris", "oroton", "orvis", "oryany", "osgoode marley", "osprey", "otto versand", "otz shoes", "ow lee", "oxxford", "pacsafe", "paddy pallin", "pajar", "panerai", "panoz", "patagonia", "patek philippe", "paul green", "pausa", "peace bird", "peak performance", "peebles", "pejo", "penalty", "penfield", "pepe jeans", "pequignet", "perlina", "petronor", "petzl", "pf flyers", "phat farm", "piado", "pierre cardin", "pikante", "ping", "pirma", "playboy", "polo ralph lauren", "pony", "prada", "propet", "pvh", "pythia", "qeelin", "qipai", "r & r collections", "r. m. williams", "rab", "rado", "rapha", "rare", "rax", "rebecca minkoff", "regatta", "regina", "relic", "reusch", "rieker", "rioni", "roamer", "robert allen", "roberto cavalli", "rochas", "rockport", "rocky", "rodania", "rolex", "royce leather", "ruans", "rudsak", "rvca", "ryb", "ryka", "sakroots", "saks fifth avenue", "salomon", "salvatore ferragamo", "samsonite", "sanita", "sansabelt", "sanuk", "satya paul", "saxx", "scarpa", "schiesser", "scotch and soda", "scottevest", "seafolly", "seagull", "seavees", "seiko", "septwolves", "sergio rossi", "sergio tacchini", "sergio valente", "seven fathoms", "shg", "shure", "simon", "simond", "sinaina", "sistema", "six deuce", "skins", "skiny", "skora", "smartwool", "smythson", "soffe", "sofft", "softspots", "sonia rykiel", "sophie paris", "sorel", "spanx", "sparco", "speedplay", "spenco", "sperry top-sider", "spiewak", "spring step", "sram", "stacy adams-1", "stacy adams-2", "star in the hood", "starbury-1", "starbury-2", "starter", "steve madden", "stowa", "strellson", "stussy", "sumdex", "supawear", "superfeet", "superga-1", "superga-2", "suunto", "swatch", "tarocash", "technomarine", "the flexx", "the north face", "the original muck boot company", "the timberland", "tiffany", "tiger of sweden", "tilted kilt", "timberland", "timbuk2-1", "timbuk2-2", "timex", "tissot", "titan industries", "titoni", "tom ford", "tom tailor", "tommy bahama", "tommy hilfiger", "topman", "trangoworld", "triumph-1", "triumph-2", "tudor", "tyr", "u.s. polo assn", "ugg australia", "under armour", "uniqlo", "valentino's", "vaneli", "vera bradley", "versace", "vibram", "viennois", "vince camuto-1", "vince camuto-2", "vionic", "vision street wear", "viso", "vivobarefoot-1", "vivobarefoot-2", "voit", "wesc", "whataburger-1", "whataburger-2", "wildcountry", "williams-sonoma", "wolford", "wolky", "wonderbra", "woolrich", "wrangler", "xoxo", "xtep", "yinshidai", "yishion", "yoko", "youngor-1", "youngor-2", "youngor-3", "yuzhaolin", "zara", "zaya", "zero halliburton", "zingara", "zino", "zodiac", "zoke", "3nod", "65amps", "AKG", "ASUS", "Abacus 5", "Admiral Oversea Corporation", "Alcatel-1", "Alcatel-2", "Amkette", "Amoi", "Amstrad", "Andrea Electronics", "Angenieux", "Ansco", "Aopen", "Apple", "Arcam", "Arecont Vision", "Audiosonic", "Audiovox", "BBK", "BEATS BY DRE", "BELKIN-1", "BELKIN-2", "Behringer", "BenQ", "Black Box Distribution", "BlackBerry", "Blancpain", "Blaupunkt", "Bovet", "Breguet", "BridgePort", "Brionvega", "Bron Elektronik", "Bronica", "Bush Radio", "Canon", "Carlsbro", "Celkon", "Cerwin-Vega", "Chronoswiss-1", "Chronoswiss-2", "Cirrus", "Citizen", "Clarion", "Clatronic", "Compact", "Connector", "Consort", "Contax", "Contec", "Currys", "Curtis", "Daewoo Electronics", "Dynatron", "EPSON", "Ebony cameras", "Egnater", "Electrohome", "Empyrean", "Ericsson", "Everex", "FangZheng", "Fuji Electric", "Fujinon", "Fujitsu Siemens", "Funai", "GPM", "Gateway", "Gigabyte", "Gitzo", "Ground Round", "HARMAN KARDON", "HEC", "HTC", "Haceb", "Hama", "Hanimex", "Hannspree", "Hanseatic", "Hasselblad", "High Endurance", "Hisense", "Hiwatt", "Hoffmann", "Holga", "Hua Yi Elec Apparatus Group", "Hublot", "Hunt's", "IBM", "IFR", "IQinVision", "Iiyama", "Ikegami", "Ilford", "Ilford Photo", "Ingelen", "Inno Hit", "Inventec", "Itautec", "KYOCERA", "Karbonn", "Kathrein", "Konka", "Logitech", "Lowepro", "Mamiya", "Maple", "Mavic", "MediaTek", "Medion", "airmate", "andor technology", "anoto", "auxx-1", "auxx-2", "changhong", "chimei", "chinon", "continental edison", "core 2 duo", "cosina", "fiyta", "franck muller", "frederique constant", "fuyou", "game boy", "girard-perregaux", "glashutte original", "gome-1", "gome-2", "gome-3", "gree-1", "gree-2", "gree-3", "greubel forsey", "haier", "hanvon", "hd", "hertz", "hualu", "huari appliance", "iBall", "iPhone-1", "iPhone-2", "iPod", "ingersoll", "jacob jensen", "jaeger-lecoultre", "jagex", "jetbeam", "jowissa", "junghans", "k'nex", "kingdee", "labtec", "laney", "lensbaby", "lightolier", "lishen", "littmann", "magnatone", "malata-1", "malata-2", "manfrotto", "me too", "meizu", "minox", "misfit", "murco-1", "murco-2", "nimslo", "nixon watches-1", "nixon watches-2", "onida", "peavey", "pentacon", "pentax", "phicomm", "piano", "pignose", "pisen", "planex", "polk audio", "prestigio", "qmobile", "radiola", "rediffusion", "refinor", "renesas", "revox", "rickenbacker", "rockstar", "roland", "rollei", "sachs", "sanyo", "scosche", "sii", "silicon power", "simmtronics", "siragon", "skyworth", "soldano", "soleus", "sonitron", "sonos", "speedlink", "supor", "univox", "vax", "vertex standard", "viewsonic", "vir-sec", "voltas", "walkman", "westinghouse", "xiaomi", "yaesu", "yhd.com", "zanussi", "zipp", "zopo", "10 Cane", "1519 Tequila", "241 Pizza", "42 Below", "7-Up", "85C Bakery Cafe", "A. Marinelli", "AMT Coffee", "AN JI WHITE TEA", "APEROL", "Act II", "Adnams", "Adrian's", "AktaVite", "Alete", "Allagash", "Amora", "Amy's Ice Creams", "Angela Mia", "Angelo Brocato's", "Angie's Kettle", "Anglo Bubbly", "Angry Orchard", "Anthony's Pizza", "Aoki's Pizza", "Apple Cinnamon Chex", "Apple Jacks", "Apple Jacks Gliders", "Apple Zings", "Applebee's", "Appletiser", "Appleton Estate", "Aqua Carpatica", "Arcaffe", "Arehucas", "Arette", "Argus Cider", "Aroma Cafe", "Aroma Espresso Bar", "Arrowroot biscuits", "Arthur Treacher's Fish & Chips", "Ashridge Cider", "Atlanta Bread Company", "Au Bon Pain", "Auntie Anne's-1", "Auntie Anne's-2", "Aurelio's Pizza", "BARSOL", "BaMa", "Bacardi", "Back Yard Burgers", "Bacon soda", "Baja Fresh", "Baker's", "Baker's Dozen Donuts", "Bakers Square", "Ballast Point", "Batchelors", "Bawls", "Bazooka", "Beacon Drive In", "Bear Naked", "Bear Republic", "Bearno's", "Beaulieu", "Becel", "Beers of Stone", "Bellywashers", "Belvita", "Benedetti's Pizza", "Beneful", "Benihana", "Bennigan's", "Berry Berry Kix", "Berry Bones Scooby-Doo", "Berry Burst Cheerios", "Berry Krispies", "Berry Lucky Charms", "Berthillon", "Bertie Beetle", "Best Foods", "Better Cheddars", "Bewley's", "BiFi", "Bicerin", "Big Mama Sausage", "Big Turk", "BigBabol", "Bigg Mixx", "Biggby Coffee", "Bird's Custard", "Bisquick", "Black Angus Steakhouse", "Blackjack Pizza", "Blavod", "Blenz Coffee-1", "Blenz Coffee-2", "Blimpie", "Blue Bottle Coffee Company", "Blue Riband", "Blue State Coffee", "Blueberry Morning", "Boca Burger", "Bojangles' Famous Chicken 'n Biscuits", "Bold Rock Hard Cider", "Bon Pari", "Boo Berry", "Boston Market", "Boston Pizza", "Bournvita", "Bovril", "Bran Flakes", "Braum's", "Bravo!,  Cucina Italiana", "Breakfast with Barbie", "Breakstone's", "Breath Savers", "Breyers", "Bridgehead Coffee", "Brigham's Ice Cream", "Brothers Cider", "Bruegger's", "Brugal", "Brummel & Brown", "Bruster's Ice Cream", "Bubba Gump Shrimp Company", "Bubbaloo", "Bubblicious", "Buc Wheats", "Buca di Beppo", "Bucanero", "Buddy's Pizza", "Budget Gourmet", "Bulls-Eye Barbecue", "Bullwinkle's Restaurant", "Bully Boy Distillers", "Bundaberg", "Buondi", "Burger Street", "Burgerville", "Burrow Hill Cider", "Bushells", "Buzz Blasts", "CHIVAS", "CHIVAS REGAL", "Cabo Wabo", "Cachantun", "Cadwalader's Ice Cream", "Cafe A Brasileira", "Cafe Coffee Day", "Cafe HAG", "Cafe Hillel", "Cafe Rio", "Cafe du Monde", "Caffe Bene", "Caffe Cova", "Caffe Luxxe", "Caffe Nero", "Caffe Pascucci", "Caffe Ritazza", "Caffe Trieste", "Caffe Umbria", "Cailler", "California Free Former", "California Pizza Kitchen", "California Tortilla", "Calistoga", "Cameron's", "Camille's Sidewalk Cafe", "Campbells", "Canada Dry", "Candy Land", "Candyman", "Cap'n Crunch", "Cap'n Crunch Crunch Berries", "Cape Cod Potato Chips", "Capri Sun", "Captain D's", "Captain Morgan", "Caramac", "Caribou Coffee", "Carino's Italian Grill", "Carl's Ice Cream", "Carl's Jr", "Carling", "Carling Black Label", "Carling Cider", "Carlos V", "Carnation", "Carrabba's Italian Grill", "Carrows", "Carte Noire", "Carupano", "Carvel Ice Cream", "Casa Dragones", "Casa Noble", "Cascadian Farm", "Cassano's Pizza King", "Cat Chow", "Cavalier", "Celestial Seasonings", "Cerelac", "Certs", "Cha Dao", "Charmin", "Cheader's", "Cheerios", "Cheese Nips", "Cheez Whiz", "Chef Boyardee", "Cherry 7Up", "Chex", "Chick-fil-A", "Chiclets", "Chips Ahoy!", "Chipsmore", "Choc-Ola", "Chocapic", "Chocolate Cheerios", "Chocolate Chex", "Chocolate D'Onofrio", "Chocolate Flakes", "Chocolate Lucky Charms", "Chocolate Surpresa", "Chocolate Toast Crunch", "Chocolate liqueur", "Chocomel", "Chocos", "Chronic Tacos-1", "Chronic Tacos-2", "Chuck-A-Rama", "CiCi's Pizza", "Ciao Bella Gelato Company", "Cibo Espresso", "Ciego Montero", "Cigar City", "Cili", "Cini Minis", "Cinna-Crunch Pebbles", "Cinnabon", "Cinnamon Burst Cheerios", "Cinnamon Chex", "Cinnamon Grahams", "Cinnamon Jacks", "Cinnamon Mini-Buns", "Cinnamon Toast Crunch", "Cinnamon Toasters", "Cinnzeo", "Claussen", "Club Social", "CoCo Wheats", "Coca-Cola Zero", "Cocio", "Coco Pops", "Coco Roos", "Coco's Bakery", "Cocoa Krispies", "Cocoa Pebbles", "Cocoa Puffs", "Cocoa Puffs Combos", "Cocosette", "Coffee Beanery", "Coffee Crisp", "Coffee Republic", "Coffee Time", "Coffee-Mate", "Cola Cao", "Cold Stone Creamery", "Colectivo Coffee Roasters", "Colman's", "Colossal Crunch", "ConAgra Foods", "Conimex", "Cool Whip", "Coors", "Corn Flakes", "Corn Pops", "Corona", "Costa Coffee", "Cote d'Or", "Count Chocula", "Country Crock", "CoverGirl", "Cows", "Cracker Jack", "Cream of Wheat", "Creemore", "Crispin Cider", "Crunch", "Crunchy Corn Bran", "Crunchy Nut", "Crush", "Cruz Tequila", "East of Chicago Pizza", "Eat'n Park", "EatZi's", "Eco de los Andes", "Eden Cheese", "Eegee's", "Egg Beaters", "Eggo", "El Chico", "El Dorado", "El Jimador", "El Paso", "El Taco Tote", "Erikli", "Eristoff", "Espolon", "Espresso Vivace", "Eukanuba", "Four Seas Ice Cream", "Four Star Pizza", "Fox's Biscuits", "Fox's Pizza Den", "Franconia", "Frank Pepe Pizzeria Napoletana", "Freddy's Frozen Custard & Steakburgers", "Freia", "Freihofers", "French Toast Crunch", "Friendly's", "Frosted Cheerios", "Frosted Mini Spooners", "Frosted Shredded Wheat", "Frosty Jack Cider", "Frosty O's", "FruChocs", "Fruco", "Fruit Brute", "Fruit Selection", "Fruit Selection Yogurt", "Fruit2O", "Fruity Cheerios", "Fudgee-O", "Fudgsicle", "Furst Bismarck", "Futurelife SmartFood", "GUINNESS", "Gabriel Pizza", "Galaxy Counters", "Gale's", "Gardetto's", "Gatti's Pizza", "General Mills", "Gerber", "Gevalia", "Gimme! Coffee", "Ginger Ale", "Gino's Pizza and Spaghetti", "Ginsters", "Giolitti", "Giordano's Pizzeria", "Glee Gum", "Glider Cider", "Gloria Jean's Coffees", "Gobstoppers", "Golden Grahams", "Good Humor", "Graeter's", "Grapette", "Gritty McDuff's", "Growers Direct", "GuanShengYuan", "GuiFaXiangShiBaJie", "Guigoz", "H. J. Heinz", "H. P. Bulmer-1", "H. P. Bulmer-2", "HP Sauce", "Hale's Ales", "Half Acre", "Half Pints", "Hamburger Helper", "Handel's Homemade Ice Cream & Yogurt", "Handi-Snacks", "Hard Rock Cafe", "Hardee's", "Harvest Crunch", "Hav-A-Tampa", "Hebrew National", "Heineken", "Heinz Baked Beans", "Heinz Tomato Ketchup", "Hell Pizza", "Hellmann's", "Henniez", "Henri Lloyd", "Herrell's Ice Cream", "Hershey's Cookies 'n' Creme", "Highland Spring", "Highland Toffee", "Highlands Coffee", "HobNobs", "Holgate", "Hollys Coffee", "Homer's Cinnamon Donut", "Homey", "Honey Bunches of Oats", "Honey Graham", "Honey Kix", "Honey Maid", "Honey Nut Cheerios", "Honey Nut Clusters", "Honey Nut Corn Flakes", "Honey Nut Shredded Wheat", "Honey Nut Toasty O's", "Honey Smacks", "Honey Stars", "Horlicks", "Horniman's", "Hornitos", "Hot Pockets", "Hot chocolate", "Hubba Bubba", "Huddle House", "Hudsons Coffee-1", "Hudsons Coffee-2", "HuiShan", "Huiyuan", "Hula Hoops", "Hungry Howie's Pizza", "Hungry Hungry Hippos", "Hunt Brothers Pizza", "Hunt's Snack Pack", "I Can't Believe It's Not Butter", "IHOP", "Iams", "Ice Breakers", "Ice Mountain", "Ijml", "Imo's Pizza", "Imperial Margarine", "In-N-Out Burger", "Indomie Mi goreng", "Infacare", "Ingman Ice Cream", "Insomnia Coffee Company", "Intelligentsia Coffee & Tea", "Irn Bru Bar", "It's a Grind Coffee House", "Ito En", "Joe's Crab Shack", "John's Incredible Pizza", "Johnny Rockets", "Juan Valdez Cafe", "Juicy Fruit", "Jumbo Krispies", "Jus-Rol", "Keglevich", "Ketel One", "Kibon", "Kissan", "Kleiner Feigling", "Kola Shaler", "Koskenkorva", "Kotipizza", "Kraft Ramek", "Krave", "Krupnik", "Kwality Wall's", "LAVAZZA", "La Choy", "La Laitiere", "La Lechera", "Lacta", "Lady's Choice", "Laffy Taffy", "Lagunitas", "Lamb Weston", "Loacker", "Long John Silver's", "LongHorn Steakhouse", "Lucky Charms", "Lucozade", "Lunchables", "Maestro Dobel", "Maggi", "Maggi Masala noodles", "MaggieMoo's Ice Cream and Treatery", "Magners Irish", "Maille", "Maizena", "Maker's Mark", "Maltesers", "Marble Slab Creamery", "Margherita", "Marie Callender's", "Mario's Pizzeria", "Market Pantry", "Marmite", "Marshmallow", "Maxibon", "Maxwell House", "Maynards", "Mazzio's", "McCoys Crisps", "McDonald's", "Mecca Cola", "Poulain", "biostime-1", "biostime-2", "calfee design", "champps americana", "chamyto", "chandelle", "charley's grilled subs", "charlie brown's steakhouse", "chatime", "cheddar's casual cafe", "cheddarie", "cheeburger cheeburger", "cheese flavoured moments", "cheeseburger in paradise", "cheesecake factory", "cheesybite", "cheezels", "chef-mate", "chenkeming-1", "chenkeming-2", "chevys fresh mex", "chewits", "chewy louie", "chex mix", "chicago town", "chicken express", "chicken in a biskit", "chicken out rotisserie", "chicken tonight", "chicza", "chiffon", "chiffon margarine", "ching's secret", "coca cola", "coffeeheaven", "contrex", "cookie crisp", "cookie crisp brownie", "cookie crisp sprinkles", "copella", "cornnuts", "frigor", "galak", "ganningpai", "glad wrap", "gladiator cycle", "great lakes", "haagen-dazs", "hengshui laobaigan", "holy cow casino", "humdinger", "huntkey", "jack in the box", "jittery joe's", "jixiangju", "joe muggs", "kichesippi-1", "kichesippi-2", "la barberie", "la cocinera", "la porchetta", "la saltena", "laimao", "lan-choo", "lappert's", "larosa's pizzeria", "laura secord chocolates", "le viennois", "ledo pizza", "lee's famous recipe chicken", "lender's", "lightlife", "lindt", "lion cereal", "liwayway", "lizano", "lobethal bierhaus", "logan's roadhouse", "lollicup coffee & tea", "lollo", "lolly gobble bliss", "lost coast", "luby's", "luhua", "lunazul", "lupicia", "luzhoulaojiao-1", "luzhoulaojiao-2", "luzhoulaojiao-3", "maarud", "mactarnahan's", "maggi noodles", "maggiano's little italy", "mallow oats", "manwich", "marco's pizza", "margie's candies", "mars muffin", "marshmallow mateys", "matusalem", "mauds ice cream", "mcalister's deli", "mccafe", "mcdonalds", "mellow bird's", "mellow mushroom", "merrydown", "michel's patisserie", "mikado", "mikel coffee company", "milk-bone", "miller's ale house", "milo's hamburgers", "minties", "miracoli", "moe's southwest grill", "molson brador", "molson exel", "mondaine", "montana mike's", "montecruz", "montesierpe", "mother's pride", "mountain mike's pizza", "moutai", "mr. noodles", "mr. pizza", "mueslix", "mulata", "mutong", "nabob", "nairobi java house", "nandos", "national coney island", "nemiroff", "nestle", "nestle corn flakes", "nestle milk chocolate", "nesvita", "new zealand natural-1", "new zealand natural-2", "new zealand natural-3", "newport creamery-1", "newport creamery-2", "noble roman's", "nong shim ltd", "nut 'n honey", "nutren", "oatmeal crisp", "oberweis dairy-1", "oberweis dairy-2", "oh henry", "old el paso", "old style pilsner", "ommegang", "on the border mexican grill & cantina", "pacific coffee company", "panarotti's", "papa gino's", "papa guiseppi", "papa murphy's take 'n' bake pizza", "pei wei asian diner", "penguin mints", "penn station (restaurant)", "penpont", "peppes pizza", "pepsi jazz", "pepsi lime", "peptamen", "perkins restaurant and bakery", "peter piper pizza", "philadelphia", "phileas fogg", "pianyifang", "pick up stix", "piltti", "pipsqueak", "pisco capel", "pisco horcon quemado", "pisco porton", "pizza 73", "pizza corner", "pizza my heart", "pizza pizza", "pizza ranch", "pizza schmizza", "pizzeria venti", "plancoet", "planet cola", "plochman's", "pollo campero", "pomegreat", "pop mie", "pop secret", "pop weavei", "pop-tarts crunch", "popeyes chicken & biscuits", "poppycock", "port city java", "port of subs", "portillo's", "potbelly sandwich works", "powdered donutz", "prezzo", "pronutro", "pucko", "puffa puffa rice", "puffs", "punch crunch", "pusser's", "qdoba mexican grill", "qingmei", "quaker steak & lube", "quality street", "queirolo", "quickchek", "quorn", "r.whites", "ragu", "raisin bran crunch", "raisin nut bran", "raisin wheats", "raisinets", "raising cane's chicken fingers", "ray's pizza", "razzle dazzle rice krispies", "ready brek", "real ale", "red lobster", "red robin", "red rose tea", "reddi-wip", "regina pizzeria", "rhum barbancourt", "ribena", "rice bubbles", "rice chex", "rice honeys nabisco", "rickard's dark", "rickard's red", "rickard's white", "ricoffy", "ristretto", "rj rockers", "ro-tel", "robeks", "robin's donuts", "robust", "rock bottom", "rogue ales", "rohrbach", "rolo", "romano's macaroni grill", "ron zacapa", "rooibee", "rosarita", "rosati's", "round table pizza", "rowntree", "roy rogers restaurants", "royal crown cola", "royal farms", "royco", "rubio's fresh mexican grill", "ruby tuesday", "ruffles", "rum-bar", "runts", "russian river", "russo's new york pizzeria", "rutter's", "sabra liqueur", "sahne nuss", "saimaza", "san pellegrino", "sandford orchards", "sanka", "sanquan", "sariwangi", "sarris candies", "sasini", "scottish blend", "screaming yellow zonkers", "second cup", "secret recipe", "sheetz", "shengfeng", "shmaltz", "showbiz pizza place", "showmars", "shuanghui", "shuijingfang", "silver gulch", "sixpoint", "skyline chili", "skyy", "slim fast", "slotts", "smart bran", "smint", "smirnoff", "smokey bones", "smokin' joes", "smuttynose", "snot shots", "spizzico", "sprinkle spangles", "square one organic", "st arnou", "stadium mustard", "steak 'n shake", "stephen's gourmet", "steve's ice cream", "stolichnaya", "straw hat pizza", "strawberry rice krispies", "stroh", "stumptown coffee roasters", "suerdieck", "sugar wafers", "supligen", "svedka", "sveltesse", "svelty", "sweet tomatoes", "sweetened wheat-fuls", "swensons", "swiss miss", "tacama demonio", "taco bueno", "taco cabana", "taco john's", "taco time", "taixiang", "taoranju", "tart n tinys", "tassimo", "taza chocolate", "tealuxe", "teavana", "teekampagne", "tenwowfood", "tequila don weber", "texan bar", "the capital grille", "the coffee bean & tea leaf", "the melting pot", "the old spaghetti factory", "the original pancake house", "thomy", "three olives", "tignanello", "timothy's world coffee", "tired hands", "tito's", "toasted cinnamon squares", "toasties", "toblerone", "tony roma's", "tooty footies", "toppers pizza", "toscanini's", "toxic waste", "treets", "tres agaves", "tres generaciones", "true coffee", "trung nguyen", "tudor crisps", "tully's coffee", "turun sinappi", "ucc ueshima coffee co", "upslope", "valiojaatelo", "van houtte", "vapiano", "vascolet", "vico", "vikingfjord", "viladrau", "violet crumble", "viru valge", "vladivar", "waffelos", "waffle crisp", "waffle house", "walkers lites", "wall's", "wanchai ferry", "wandashan", "wangs", "wangzhihe", "warburtons", "wayne's coffee", "weetos", "weight watchers", "weston's cider", "wheat chex", "wheat stax", "which wich", "whitey's ice cream", "widmer brothers", "wienerschnitzel", "wiesmann", "wild berry skittles", "williams fresh cafe", "wilton", "winiary", "wolf brand chili", "wolverine", "wrigley's", "wudeli", "wufangzhai", "wuyutai", "wyder's", "xiangeqing", "xiao nan guo", "xifeng", "xinghualou", "yesmywine", "yili", "yonho-1", "yonho-2", "yorkshire tea", "youyou", "yurun", "zatarain's", "zephyrhills", "zhoujunji", "zjs express", "zoegas", "zoladkowa gorzka", "4Kids", "Artillery", "Audison", "B-Daman", "BCP", "Baron von Redberry", "Big Boy", "Cabbage Patch Kids", "Cinemax", "Element", "Elta", "GameWorks", "Ghostbusters", "Gold Rush", "H. Upmann", "HMV", "Harley-Davidson", "Harvest Moon", "High School Musical", "Hitachi", "Hooters", "Hoover", "Hot Wheels", "Hovis", "Klei", "Krusty-O's", "Kung Fu Panda Crunchers", "Lares Ice Cream Parlor", "Mattel", "adika", "bahama breeze", "barrel o' monkeys", "batman returns", "battleship", "c-3po's", "comet balls", "frontier developments", "g.i. joe", "gamelab", "lego mindstorms", "lego mindstorms nxt", "lincoln logs", "lite-brite", "little golden books", "little green men", "magic 8-ball", "magic hat", "marklin", "matryoshka", "mcintosh", "meccano", "mega bloks", "mickey mouse", "micro machines", "mohawk-1", "mohawk-2", "morel", "mtx audio", "music man-1", "music man-2", "nerf", "nestle wonder ball", "no fear", "oggi", "onepiece", "oso", "ouija board", "pink panther flakes", "pixy stix", "play-doh", "playmobil", "pound puppies", "prs guitars", "punch entertainment", "q8", "qq", "qunar.com", "qzone", "rasti", "scrabble", "silly putty", "soultone", "sound city", "standard fireworks", "starworld", "stein world-1", "stein world-2", "suhr", "sun 'n' sand", "superman stars", "tecent weibo", "testor's", "the fullbright", "tinkertoy", "tonka toys", "tortex", "tum yeto", "turtles", "ultravox", "unisonic", "vibrations", "war horse", "webkinz", "weebles", "weibo", "wild animal crunch", "winsor", "yoplait", "zhenai.com", "zonda", "zoob", "999", "Apiretal", "Calpol", "Efferalgan", "aeknil", "alvedon", "aspirin", "atamel", "band-aid", "bauschlomb-1", "bauschlomb-2", "benuron", "biogesic", "biotene", "camlin", "jarrow", "jiang zhong", "jimin", "jindan", "kangmei", "lekadol", "longmu-1", "longmu-2", "luoxin", "mayinglong", "melatonin", "paralen", "perdolan", "ringtons", "rubophen", "shi hui da", "shuangyan", "tachipirina", "tafirol", "tapsin", "termalgin", "thomapyrin", "ultra brite", "uphamol", "xiangxue", "yangshengtang", "yunnanbaiyao", "yuyue-1", "yuyue-2", "yuyue-3", "yuyue-4", "zendium", "A. Turrent", "ACQUA DI PARMA", "ADCO", "ADOX", "ALFAPARF", "ARCHIPELAGO", "ART Furniture", "AUPRES", "ActiPlus", "After Dinner", "AiTeFu", "Alba Botanica", "Alterna Haircare", "American Crew", "American Leather", "American Standard", "Ammonite", "Andrelon", "Aqua Flow", "Aroma de San Andres", "Arturo Fuente", "Aubrey Organics", "Avalon Organic Botanicals", "Aveda", "Aveeno", "Aviance Cosmetics", "Avid", "Avlon", "BEELY", "BODUM", "BW Filterfabrik", "Beko", "Belk", "Ben Franklin Stores", "Bergdorf Goodman", "Better Botanicals", "Big Lots", "Billy Jealousy", "Binatone", "Biolage", "Biosilk", "Biotex", "Biotherm", "Blaser", "Blue Stratos", "Bols", "Bossner", "Boulevard", "Braddington Young", "Bruno Banani", "Brylcreem", "Burt's Bees", "CHRISTOFLE", "Camay", "Candle", "Canson", "Caress", "Carlos Torano", "Carolina Herrera", "Carslan", "Carter's Ink", "Casa Magna", "Caswell Massey", "Cerruti 1881", "Chromcraft", "Cif", "Ciroc", "Clear", "Clear Clean", "Coccolino", "Colonial", "Crayola", "Cretacolor", "E-CLOTH", "Elf", "Escada", "Eskimo", "EskinoL", "Esselte", "Estee Lauder", "Estrella", "FSL", "Fabricut", "Fair and Lovely", "Faith Hill Parfums", "Frische Brise", "Frizz-Ease", "Frommer's", "Fundacion Ancestral", "GF", "GUZZINI", "Gain", "Galp", "Gel", "Gianfranco Ferre", "Gillette", "Gispert", "Gorenje", "Guantanamera", "Hancock-1", "Hancock-2", "Havana Club", "Heartland", "HengYuXiang", "Herbal Essences", "Herborist", "Hermes", "Hickory White", "Homme by David", "Honeywell", "Howard Miller", "Hulk", "Hypson", "IITTALA", "INOHERB-1", "INOHERB-2", "IVORY", "Ifb", "Infusium 23", "Intermarche", "Intex", "Ipana", "J Fuego", "JUICY COUTURE", "John Frieda", "Joico", "Jonathan Charles", "KAISHENG", "KANS", "KITCHEN AID", "Kalco", "Kalodont", "Kiehl's", "Kieninger", "King Edward the Seventh", "Kolynos", "Kravet", "L'ARTISAN PARFUMEUR", "L'OCCITANE", "L'Occitane en Provence", "LOTOS-1", "LOTOS-2", "La Aroma de Cuba", "La Aurora", "La Flor Dominicana", "La Gloria Cubana", "La Palina", "Lacto", "Lane Furniture", "Las Cabrillas", "Longines", "Lucky", "Lux", "Macy's", "Makro", "Maple Leaf", "Marsh Wheeling", "Marshalls", "Marvel", "Marvis", "Massmart", "Max Factor", "alessi", "amish", "anerle", "anglepoise", "anolon", "apica", "apothia-1", "apothia-2", "artline", "asd", "attack", "babool", "baby bullet", "bain de terre-1", "bain de terre-2", "ballograf", "bermex", "bernzomatic", "bic cristal", "bigtime", "binaca", "bio ionic", "bisque", "bluemoon", "brita", "brown jordan", "brownstone", "burton james", "buscapina", "california house", "calligraphy pen", "canadel", "canbo", "captain stag", "caran d'ache", "chahua", "chando", "chantelle", "charatan", "chateau real", "clarins", "corioliss", "corzo", "country garden", "faceshop", "freakies", "ghd", "glorix", "good grips", "grohe", "haers", "hekman", "hershesons", "hotata", "huayang", "huida", "joyoung", "jurlique", "kettler", "kilner", "kimani", "kleenex", "kokuyo", "la escepicon", "la flor de caney", "la roche-posay", "ladycare", "lakme", "lancome", "lantianliubizhi", "lanvin", "le creuset", "levenger", "lianle", "liquid assets", "lite source", "loblaw", "longhu", "longrich", "lonkey", "loreal", "los statos de luxe", "lovera", "lumiquest", "luvs", "luxottica", "lysoform", "m&g", "macanudo", "macro", "magicamah", "magimix", "maglite", "mak lubricants", "mamonde", "manjiamei", "marimekko", "markor furnishings", "marlboro", "marshal", "maskfamily", "mathmos", "mauviel", "maydos", "meifubao", "meiling", "melon bicycles", "meng jie-1", "meng jie-2", "mentadent", "mentadent sr", "miele", "miquelrius-1", "miquelrius-2", "mitchell", "mobil", "mokingran", "molton brown", "mona lisa-1", "mona lisa-2", "mona lisa-3", "monbento", "montegrappa", "morphy richards", "moser baer", "musco", "nakamichi-1", "nakamichi-2", "nallin", "navneet", "neutrogena", "nexxus", "nioxin", "nivea", "nondo", "nongfu spring-1", "nongfu spring-2", "nongfu spring-3", "norton-1", "norton-2", "norton-3", "noxzema", "nutribullet", "opinel", "osm", "palecek", "pantene", "paris hilton", "parkay", "paul garmirian", "paul mitchell", "peet dryer", "penhaligon's", "pentel", "perdomo", "perry ellis", "pianor", "pigeon", "pilot parallel", "pipidog", "playtex", "ponds", "por larranaga", "powerdekor", "powervolt", "ppu", "prandina", "pride", "prismacolor", "pritt stick", "proya", "puros indios", "quai d'orsay", "quanwudi", "quesada", "ralph lauren home", "rama", "ramon allones", "redken", "revlon", "revol", "rinso", "rizla", "robijn", "rocky patel", "rotomac", "royal albert", "royal talens", "royalstar", "safeguard", "safilo", "sam moore", "saran wrap", "scanpan", "schwarzkopf", "seiyu", "sellotape", "sennelier", "sephora", "seventh generation", "sexy hair", "shangdedianli", "shiseido", "shiyou", "shuangye", "sibao", "siliconezone", "simple human", "simply amish", "sk-ii", "skin food", "sleemon", "softto", "sofy", "south bedding", "speedball", "staedtler", "stakmore", "stelton", "stime", "stomatol", "streamlight", "summer classics", "sunset west", "tabak especial", "taiyangshen", "tanmujiang", "tefal", "teflon", "tempur-pedic", "the laundress", "therapedic", "thermos", "tiangxiaxiu", "tjoy", "tombow", "toscanello", "tungsram", "two brothers", "unifon", "uzero", "vaseline", "vegafina", "vegas robaina", "verbatim", "villa zamorano", "villiger", "vipshop", "vo5", "voluspa", "wahl", "weida", "wenger", "wetherm", "weyerbacher", "whisper", "wildkin", "winalot", "windex", "x-acto", "xellent swiss", "xinxiangyin", "yalijie", "ykk", "yongli blueleopard", "yu mei jing", "yuhangren-1", "yuhangren-2", "yukon", "zwitsal", "7Days Inn", "A T CROSS", "ACE Team", "ACN", "AIST", "AKPET", "AMAT", "ANLI", "AP Industries", "APCO", "Absolut", "Absolwent", "Acratech", "Adobe Systems", "Aethra", "Afriquia", "AirNow", "Alpha Male", "Altus", "Amoco", "Ampeg", "Ampol", "Angelina", "Apex", "Aqua Guard", "Aral AG", "Argos Energies", "Armco", "Ashland Inc", "Attock Petroleum", "Augusta", "Auri", "Aussie", "Azura", "BELOMO", "BEST EXPRESS-1", "BEST EXPRESS-2", "BEST EXPRESS-3", "BJ's Restaurant", "BOH Plantations", "BOSCH", "BOSS", "BP", "BT Conferencing", "BWOC", "BaDuYoua", "Baihe", "Bapco", "Barex-1", "Barex-2", "Barkerville", "Beeline", "Beretta", "Bergner's", "Big League", "Big Red", "Bikinis Sports Bar & Grill", "Black Jack", "Black Rapid", "Blue Avocado", "Blue Hills", "Blue Point", "Bona", "Bonefish Grill", "Boston Store", "Botany", "Bradford Exchange", "Bravo", "Bushmaster", "Buxton", "CARL ZEISS", "CBCI Telecom", "CBS", "CHOW TAI SENG-1", "CHOW TAI SENG-2", "CHS", "CMC", "CMD", "Cabaiguan", "Caltex", "Calumet", "Carbolite", "Carbontech", "Cargill", "Carola", "Carolina-1", "Carolina-2", "Carpathia", "Carrefour", "Cartenz", "Cathay", "Cenex", "Cesars", "Che", "Chicago", "Choice Hotels", "Citra", "Clark", "Classmate", "Colgate", "Colt", "Connaught", "Copeland's", "Cosan", "Costco Wholesale", "Cresta", "Cybertron", "Edeka", "Edison", "Elfin", "Elsa", "Excel", "FSS", "Garcia Y Vega", "General Electric", "Greenergy", "Gurkha-1", "Gurkha-2", "HY Markets", "Harpoon", "Havells", "Hedex", "Herberger's", "Hexacta", "Hilton Worldwide-1", "Hilton Worldwide-2", "Hilton Worldwide-3", "Home Inn", "HongYan", "Houlihan's", "Howard Johnson's", "HuaLian", "HuaNeng", "Humber-1", "Humber-2", "Hunor", "Hurtu", "IHG", "Ibis", "Ice Harbor", "Ideal", "Illinois", "Indiana", "Ipso", "Isla-1", "Isla-2", "Island Fantasy", "Isosource", "Ito-Yokado", "JX Holdings", "John Lewis", "Johnson Outdoors", "Kauffman", "Killer Queen", "LLoyd's", "Lanka", "MAD DOG", "Mack & Dave's Department Store", "Mad River", "Marmara", "Marriott International", "Mars", "Maverik Inc", "Maxum", "abus", "anschutz", "apollo", "avery dennison-1", "avery dennison-2", "baird", "basilisk", "bealls", "bermudez", "berol", "brm buggy", "burley design", "by nord", "calpak", "caltex woolworths", "capote", "caravelas", "cenpin", "ceravision", "chica rica", "china post group corporation", "china railway group limited", "chinese petroleum", "citic", "cofco", "coles", "commuter", "corgi", "goko", "guolv", "happigo", "huarun", "huasheng", "huaweimei", "hurom", "icbc-1", "icbc-2", "icbc-3", "icsoncom", "imperial tea court", "impulse", "isTone", "isaura", "jd", "jump rope", "ka-bar", "kaixinwang", "kangshu", "knudsen", "kronan", "la corona", "lefeng", "linum", "lionel", "liu gong", "longping", "los libertadores", "matamp", "maxol", "mecca espresso", "meituan", "metcash", "michelin", "millet-1", "millet-2", "millonario", "minsheng", "mmmuffins", "monte", "mossberg", "mugg & bean", "murphyusa", "networker", "new holland-1", "new holland-2", "new holland-3", "newegg.com", "nos (nitrous oxide systems)", "old country buffet", "old vienna", "ollie's bargain outlet", "ono", "oovoo", "orkan bensin", "pacon", "paltalk", "photoshop", "picc", "pidgin", "pipeworks", "planta", "posiflex", "primus", "prince", "private moon", "profex", "prudenial", "ps", "ptt", "pulaski", "pulser", "qalam", "qixing", "quanfeng express", "quick", "quiktrip", "r. e. dietz", "rated m", "red herring", "repsol ypf", "rfeng", "rough draft", "sam's club", "sf express", "shenghong", "shimano", "shopko", "skeena", "skype", "sld", "sma", "sneaky pete's", "socar-1", "socar-2", "sontec", "southbeauty-1", "southbeauty-2", "southbeauty-3", "southworth", "sowden", "spira", "stage stores inc", "stein mart", "stewart-warner", "summit", "susy", "t.j. maxx", "taihao", "tanduay", "taobao.com", "taser-1", "taser-2", "teamviewer", "teva", "thatchers", "the bon-ton", "the forbidden city", "theodora", "thomasville", "thomson reuters", "thorn lighting", "tnt", "tokyobay", "town pump", "transitions", "travel smart by conair", "trinidad", "trinium", "trueconf", "tubao-1", "tubao-2", "tubao-3", "tuckerman", "tumi", "twisted pixel", "uber-1", "uber-2", "uber-3", "unionpay", "unisys", "universiti teknologi petronas", "ursus", "vaude-1", "vaude-2", "veken-1", "veken-2", "veken-3", "vertu", "vidyo-1", "vidyo-2", "vigo", "vogue", "von maur", "vostok", "vox", "vpl", "waltham", "webster's dictionary", "weg industries", "weixing", "whistle", "wight trash", "wigwam", "winnebago", "wistron", "womai.com", "woolworths (south africa)", "xdf.cn", "xero", "xgma", "xierdun", "xinruncheng", "xunxing", "yards", "younkers", "yucheng ceramics", "yx energi", "2XU", "3t cycling", "Asics", "Brooks", "CCM", "California Skateparks", "Callaway", "Chocolate Skateboards", "Edelrid", "Gazelle", "HCL", "Hanwag", "Kelme", "Komperdell", "Kryptonics", "Kukri", "agv", "alcan", "la sportiva", "lafuma", "le coq", "legea-1", "legea-2", "leki", "lescon", "lowa", "lowe alpine", "lululemon athletica-1", "lululemon athletica-2", "macron", "mammut", "molten", "mountainsmith", "neil pryde-1", "neil pryde-2", "oakley-1", "oakley-2", "rawlings", "razor", "riddell", "santa cruz", "santa cruz bikes", "sea to summit", "sea-doo", "sherrin", "shua", "sierra designs", "skis rossignol", "slazenger", "snow peak-1", "snow peak-2", "spalding", "sprinter", "steeden", "strida", "surly bikes", "swix", "taylormade", "tecnica", "titleist", "uhlsport", "vango", "wiffle bat and ball", "wilson", "yonex", "yong master", "ANCAP-1", "ANCAP-2", "Air Choice One Airlines", "Ameriflight", "Amerijet International", "Ameristar Air Cargo", "Aprilia", "Atala", "Autovox", "BMC", "BMW", "BUGATTI", "Benelli", "Bentley", "Bertone", "Bharat Petroleum", "Bi-Mart", "Bilisten", "Bridgestone", "Brilliance", "Brooks Saddle", "Caloi", "Cannondale", "Caparo", "Chery", "Chevrolet", "Chevron", "Cimc-1", "Cimc-2", "Citgo", "Citroen", "Colnago", "Comfort", "Condor", "Conoco", "Continental", "Copec", "Dacia", "Daewoo", "EMS", "Ecopetrol", "Embraer", "Era Alaska", "Esso", "Executive Airlines", "ExxonMobil", "GMC", "Genesis", "Gulf Oil", "Hascol Petroleum", "Heinkel", "Huntsville International Airport", "Husqvarna-1", "Husqvarna-2", "Hyundai-1", "Hyundai-2", "ITS-1", "ITS-2", "Ideal Bikes", "Infiniti", "Isuzu-1", "Isuzu-2", "KTM", "Kaipan", "Kuota", "Kuwahara", "Lada", "Lamborghini", "Lancia", "Landwind-1", "Landwind-2", "Landwind-3", "Lapierre", "Los Angeles International Airport", "Lotus Cars", "Luxgen", "MDX", "Magna-1", "Magna-2", "Mahindra-1", "Mahindra-2", "Marussia", "Maserati", "Mazda", "ac propulsion", "akurra", "arrinera", "artega", "atlas air", "batavus", "bhpetrol", "bianchi", "bike friday", "bilenky", "boardman bikes", "bobcat", "bolwell", "bottecchia", "british eagle", "brodie bicycles", "brompton bicycle", "campion cycle", "canyon bicycles", "cape air", "cervelo", "chang'an-1", "chang'an-2", "chang'an-3", "chautauqua airlines", "gary fisher", "gendron bicycles", "ginetta", "gitane", "gleagle", "gnome et rhone", "gocycle", "grand sport", "guerciotti", "gumpert", "haro bikes", "hase bikes", "hetchins", "hongqi", "huffy", "hutchinson tires", "ideagro", "ikco", "independent fabrication", "intercity", "iron horse bicycles", "isdera", "islabikes", "jamis bicycles", "jinbei", "kia da", "lambretta", "land rover", "levante", "lexus-1", "lexus-2", "lifan-1", "lifan-2", "malvern star", "marin bikes", "maruishi", "masi bicycles", "maxxis", "maybach-1", "maybach-2", "mclaren", "merckx", "merida bikes", "misc berhad", "mitsuoka-1", "mitsuoka-2", "moots cycles", "muddy fox", "munro", "mylch", "nataraj", "opel", "pacific pride", "pakistan oilfields", "pakistan state", "pakistan state oil", "panaracer", "petro-canada", "petron", "phillips cycles", "phillips petroleum", "pinarello", "pogliaghi", "pontiac", "quintana roo", "r-line", "racetrac", "ram", "ranch hand", "rft", "ridley", "riese und muller", "rimac", "roadmaster", "rockshox", "roewe", "rowbike", "s-oil", "saab", "saipa", "saracen cycles", "scalextric", "scania", "sdlg", "sealink", "seaoil philippines", "seat", "sentra", "shell v-power", "sisu", "ski-doo", "speedway llc", "supertest petroleum", "tempra", "terpel", "toyota", "upland", "venko", "viper", "vw", "yutong-1", "yutong-2", "yutong-3", "zamboni"]
openbrand_classes = ["LAMY", "tumi", "warrior", "sandisk", "belle", "ThinkPad", "rolex", "balabala", "vlone", "nanfu", "KTM", "VW", "libai", "snoopy", "Budweiser", "armani", "gree", "GOON", "KielJamesPatrick", "uniqlo", "peppapig", "valentino", "GUND", "christianlouboutin", "toyota", "moutai", "semir", "marcjacobs", "esteelauder", "chaoneng", "goldsgym", "airjordan", "bally", "fsa", "jaegerlecoultre", "dior", "samsung", "fila", "hellokitty", "Jansport", "barbie", "VDL", "manchesterunited", "coach", "PopSockets", "haier", "banbao", "omron", "fendi", "erke", "lachapelle", "chromehearts", "leader", "pantene", "motorhead", "girdear", "fresh", "katespade", "pandora", "Aape", "edwin", "yonghui", "Levistag", "kboxing", "yili", "ugg", "CommedesGarcons", "Bosch", "palmangels", "razer", "guerlain", "balenciaga", "anta", "Duke", "kingston", "nestle", "FGN", "vrbox", "toryburch", "teenagemutantninjaturtles", "converse", "nanjiren", "Josiny", "kappa", "nanoblock", "lincoln", "michael_kors", "skyworth", "olay", "cocacola", "swarovski", "joeone", "lining", "joyong", "tudor", "YEARCON", "hyundai", "OPPO", "ralphlauren", "keds", "amass", "thenorthface", "qingyang", "mujosh", "baishiwul", "dissona", "honda", "newera", "brabus", "hera", "titoni", "decathlon", "DanielWellington", "moony", "etam", "liquidpalisade", "zippo", "mistine", "eland", "wodemeiliriji", "ecco", "xtep", "piaget", "gloria", "hp", "loewe", "Levis_AE", "Anna_sui", "MURATA", "durex", "zebra", "kanahei", "ihengima", "basichouse", "hla", "ochirly", "chloe", "miumiu", "aokang", "SUPERME", "simon", "bosideng", "brioni", "moschino", "jimmychoo", "adidas", "lanyueliang", "aux", "furla", "parker", "wechat", "emiliopucci", "bmw", "monsterenergy", "Montblanc", "castrol", "HUGGIES", "bull", "zhoudafu", "leaders", "tata", "oldnavy", "OTC", "levis", "veromoda", "Jmsolution", "triangle", "Specialized", "tries", "pinarello", "Aquabeads", "deli", "mentholatum", "molsion", "tiffany", "moco", "SANDVIK", "franckmuller", "oakley", "bulgari", "montblanc", "beaba", "nba", "shelian", "puma", "PawPatrol", "offwhite", "baishiwuliu", "lexus", "cainiaoguoguo", "hugoboss", "FivePlus", "shiseido", "abercrombiefitch", "rejoice", "mac", "chigo", "pepsicola", "versacetag", "nikon", "TOUS", "huawei", "chowtaiseng", "Amii", "jnby", "jackjones", "THINKINGPUTTY", "bose", "xiaomi", "moussy", "Miss_sixty", "Stussy", "stanley", "loreal", "dhc", "sulwhasoo", "gentlemonster", "midea", "beijingweishi", "mlb", "cree", "dove", "PJmasks", "reddragonfly", "emerson", "lovemoschino", "suzuki", "erdos", "seiko", "cpb", "royalstar", "thehistoryofwhoo", "otterbox", "disney", "lindafarrow", "PATAGONIA", "seven7", "ford", "bandai", "newbalance", "alibaba", "sergiorossi", "lacoste", "bear", "opple", "walmart", "clinique", "asus", "ThomasFriends", "wanda", "lenovo", "metallica", "stuartweitzman", "karenwalker", "celine", "miui", "montagut", "pampers", "darlie", "toray", "bobdog", "ck", "flyco", "alexandermcqueen", "shaxuan", "prada", "miiow", "inman", "3t", "gap", "Yamaha", "fjallraven", "vancleefarpels", "acne", "audi", "hunanweishi", "henkel", "mg", "sony", "CHAMPION", "iwc", "lv", "dolcegabbana", "avene", "longchamp", "anessa", "satchi", "hotwheels", "nike", "hermes", "jiaodan", "siemens", "Goodbaby", "innisfree", "Thrasher", "kans", "kenzo", "juicycouture", "evisu", "volcom", "CanadaGoose", "Dickies", "angrybirds", "eddrac", "asics", "doraemon", "hisense", "juzui", "samsonite", "hikvision", "naturerepublic", "Herschel", "MANGO", "diesel", "hotwind", "intel", "arsenal", "rayban", "tommyhilfiger", "ELLE", "stdupont", "ports", "KOHLER", "thombrowne", "mobil", "Belif", "anello", "zhoushengsheng", "d_wolves", "FridaKahlo", "citizen", "fortnite", "beautyBlender", "alexanderwang", "charles_keith", "panerai", "lux", "beats", "Y-3", "mansurgavriel", "goyard", "eral", "OralB", "markfairwhale", "burberry", "uno", "okamoto", "only", "bvlgari", "heronpreston", "jimmythebull", "dyson", "kipling", "jeanrichard", "PXG", "pinkfong", "Versace", "CCTV", "paulfrank", "lanvin", "vans", "cdgplay", "baojianshipin", "rapha", "tissot", "casio", "patekphilippe", "tsingtao", "guess", "Lululemon", "hollister", "dell", "supor", "MaxMara", "metersbonwe", "jeanswest", "lancome", "lee", "omega", "lets_slim", "snp", "PINKFLOYD", "cartier", "zenith", "LG", "monchichi", "hublot", "benz", "apple", "blackberry", "wuliangye", "porsche", "bottegaveneta", "instantlyageless", "christopher_kane", "bolon", "tencent", "dkny", "aptamil", "makeupforever", "kobelco", "meizu", "vivo", "buick", "tesla", "septwolves", "samanthathavasa", "tomford", "jeep", "canon", "nfl", "kiehls", "pigeon", "zhejiangweishi", "snidel", "hengyuanxiang", "linshimuye", "toread", "esprit", "BASF", "gillette", "361du", "bioderma", "UnderArmour", "TommyHilfiger", "ysl", "onitsukatiger", "house_of_hello", "baidu", "robam", "konka", "jack_wolfskin", "office", "goldlion", "tiantainwuliu", "wonderflower", "arcteryx", "threesquirrels", "lego", "mindbridge", "emblem", "grumpycat", "bejirog", "ccdd", "3concepteyes", "ferragamo", "thermos", "Auby", "ahc", "panasonic", "vanguard", "FESTO", "MCM", "lamborghini", "laneige", "ny", "givenchy", "zara", "jiangshuweishi", "daphne", "longines", "camel", "philips", "nxp", "skf", "perfect", "toshiba", "wodemeilirizhi", "Mexican", "VANCLEEFARPELS", "HARRYPOTTER", "mcm", "nipponpaint", "chenguang", "jissbon", "versace", "girardperregaux", "chaumet", "columbia", "nissan", "3M", "yuantong", "sk2", "liangpinpuzi", "headshoulder", "youngor", "teenieweenie", "tagheuer", "starbucks", "pierrecardin", "vacheronconstantin", "peskoe", "playboy", "chanel", "HarleyDavidson_AE", "volvo", "be_cheery", "mulberry", "musenlin", "miffy", "peacebird", "tcl", "ironmaiden", "skechers", "moncler", "rimowa", "safeguard", "baleno", "sum37", "holikaholika", "gucci", "theexpendables", "dazzle", "vatti", "nintendo"]

train_openbrand = dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/train_20210409_1_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_1/',
                pipeline=train_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/train_20210409_2_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_2/',
                pipeline=train_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/train_20210409_3_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_3/',
                pipeline=train_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/train_20210409_4_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_4/',
                pipeline=train_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/train_20210409_5_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_5/',
                pipeline=train_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/train_20210409_6_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_6/',
                pipeline=train_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/train_20210409_7_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_7/',
                pipeline=train_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/train_20210409_8_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_8/',
                pipeline=train_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/train_20210409_9_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_9/',
                pipeline=train_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/train_20210409_10_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_10/',
                pipeline=train_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/train_20210409_11_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_11/',
                pipeline=train_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/train_20210409_12_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_12/',
                pipeline=train_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/train_20210409_13_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_13/',
                pipeline=train_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/train_20210409_14_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_14/',
                pipeline=train_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/train_20210409_15_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_15/',
                pipeline=train_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/train_20210409_16_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_16/',
                pipeline=train_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/train_20210409_17_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_17/',
                pipeline=train_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/train_20210409_18_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_18/',
                pipeline=train_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/train_20210409_19_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_19/',
                pipeline=train_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/train_20210409_20_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_20/',
                pipeline=train_pipeline,
                force_one_class=True
            ),
        ],
        separate_eval=False,
    )

validation_openbrand=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/validation_20210409_1_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_1/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/validation_20210409_2_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_2/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/validation_20210409_3_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_3/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/validation_20210409_4_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_4/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/validation_20210409_5_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_5/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/validation_20210409_6_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_6/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/validation_20210409_7_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_7/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/validation_20210409_8_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_8/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/validation_20210409_9_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_9/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/validation_20210409_10_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_10/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/validation_20210409_11_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_11/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/validation_20210409_12_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_12/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/validation_20210409_13_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_13/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/validation_20210409_14_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_14/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/validation_20210409_15_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_15/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/validation_20210409_16_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_16/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/validation_20210409_17_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_17/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/validation_20210409_18_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_18/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/validation_20210409_19_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_19/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/validation_20210409_20_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_20/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
        ],
        separate_eval=False,
    )

test_openbrand = dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/test_20210409_1_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_1/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/test_20210409_2_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_2/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/test_20210409_3_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_3/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/test_20210409_4_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_4/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/test_20210409_5_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_5/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/test_20210409_6_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_6/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/test_20210409_7_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_7/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/test_20210409_8_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_8/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/test_20210409_9_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_9/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/test_20210409_10_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_10/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/test_20210409_11_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_11/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/test_20210409_12_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_12/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/test_20210409_13_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_13/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/test_20210409_14_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_14/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/test_20210409_15_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_15/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/test_20210409_16_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_16/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/test_20210409_17_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_17/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/test_20210409_18_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_18/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/test_20210409_19_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_19/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='OpenBrandDataset',
                classes=openbrand_classes,
                data_root=data_root + 'OpenBrands/',
                ann_file='annotations/test_20210409_20_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_20/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
        ],
        separate_eval=False,
        force_one_class=True
    )

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    train=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='XMLDataset',
                classes=logodet_classes,
                data_root=data_root + 'LogoDet-3K',
                ann_file='train_reduced.txt',
                ann_subdir='',
                img_prefix='',
                img_subdir='',
                pipeline=train_pipeline,
                force_one_class=True
            ),
            dict(
                type='XMLDataset',
                classes=logos_ds_classes,
                data_root=data_root + 'logo_dataset',
                ann_file='ImageSets/Main/train.txt',
                ann_subdir='Annotations',
                img_prefix='',
                img_subdir='JPEGImages',
                pipeline=train_pipeline,
                force_one_class=True
            ),
            train_openbrand,
        ]
    ),
    val=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='XMLDataset',
                classes=logodet_classes,
                data_root=data_root + 'LogoDet-3K',
                ann_file='val_reduced.txt',
                ann_subdir='',
                img_prefix='',
                img_subdir='',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='XMLDataset',
                data_root=data_root + 'logo_dataset',
                ann_file='ImageSets/Main/validation.txt',
                img_prefix='',
                classes=logos_ds_classes,
                pipeline=test_pipeline,
                force_one_class=True
            ),
            validation_openbrand,
        ]
    ),
    test=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='XMLDataset',
                classes=logodet_classes,
                data_root=data_root + 'LogoDet-3K',
                ann_file='test_reduced.txt',
                ann_subdir='',
                img_prefix='',
                img_subdir='',
                pipeline=test_pipeline,
                force_one_class=True
            ),
            dict(
                type='XMLDataset',
                data_root=data_root + 'logo_dataset',
                ann_file='ImageSets/Main/test.txt',
                img_prefix='',
                classes=logos_ds_classes,
                pipeline=test_pipeline,
                force_one_class=True
            ),
            test_openbrand,
        ]
    )
)
