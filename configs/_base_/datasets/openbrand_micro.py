dataset_type = 'OpenBrandDataset'
data_root = '/home/ubuntu/data/OpenBrands/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='Resize', img_scale=[(1333,800), (1333,880), (1333,960), (1333,1024), (1333,1100)], keep_ratio=True,multiscale_mode='value'),
    dict(type='Resize', img_scale=[(1333,800), (1333,896), (1333,1024)], keep_ratio=True,multiscale_mode='value'),
    dict(type='Pad', size_divisor=32),
    #dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 896),
        #img_scale=[(1333,800), (1333,880), (1333,960), (1333,1024), (1333,1100)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

openbrand_classes = ["LAMY", "tumi", "warrior", "sandisk", "belle", "ThinkPad", "rolex", "balabala", "vlone", "nanfu", "KTM", "VW", "libai", "snoopy", "Budweiser", "armani", "gree", "GOON", "KielJamesPatrick", "uniqlo", "peppapig", "valentino", "GUND", "christianlouboutin", "toyota", "moutai", "semir", "marcjacobs", "esteelauder", "chaoneng", "goldsgym", "airjordan", "bally", "fsa", "jaegerlecoultre", "dior", "samsung", "fila", "hellokitty", "Jansport", "barbie", "VDL", "manchesterunited", "coach", "PopSockets", "haier", "banbao", "omron", "fendi", "erke", "lachapelle", "chromehearts", "leader", "pantene", "motorhead", "girdear", "fresh", "katespade", "pandora", "Aape", "edwin", "yonghui", "Levistag", "kboxing", "yili", "ugg", "CommedesGarcons", "Bosch", "palmangels", "razer", "guerlain", "balenciaga", "anta", "Duke", "kingston", "nestle", "FGN", "vrbox", "toryburch", "teenagemutantninjaturtles", "converse", "nanjiren", "Josiny", "kappa", "nanoblock", "lincoln", "michael_kors", "skyworth", "olay", "cocacola", "swarovski", "joeone", "lining", "joyong", "tudor", "YEARCON", "hyundai", "OPPO", "ralphlauren", "keds", "amass", "thenorthface", "qingyang", "mujosh", "baishiwul", "dissona", "honda", "newera", "brabus", "hera", "titoni", "decathlon", "DanielWellington", "moony", "etam", "liquidpalisade", "zippo", "mistine", "eland", "wodemeiliriji", "ecco", "xtep", "piaget", "gloria", "hp", "loewe", "Levis_AE", "Anna_sui", "MURATA", "durex", "zebra", "kanahei", "ihengima", "basichouse", "hla", "ochirly", "chloe", "miumiu", "aokang", "SUPERME", "simon", "bosideng", "brioni", "moschino", "jimmychoo", "adidas", "lanyueliang", "aux", "furla", "parker", "wechat", "emiliopucci", "bmw", "monsterenergy", "Montblanc", "castrol", "HUGGIES", "bull", "zhoudafu", "leaders", "tata", "oldnavy", "OTC", "levis", "veromoda", "Jmsolution", "triangle", "Specialized", "tries", "pinarello", "Aquabeads", "deli", "mentholatum", "molsion", "tiffany", "moco", "SANDVIK", "franckmuller", "oakley", "bulgari", "montblanc", "beaba", "nba", "shelian", "puma", "PawPatrol", "offwhite", "baishiwuliu", "lexus", "cainiaoguoguo", "hugoboss", "FivePlus", "shiseido", "abercrombiefitch", "rejoice", "mac", "chigo", "pepsicola", "versacetag", "nikon", "TOUS", "huawei", "chowtaiseng", "Amii", "jnby", "jackjones", "THINKINGPUTTY", "bose", "xiaomi", "moussy", "Miss_sixty", "Stussy", "stanley", "loreal", "dhc", "sulwhasoo", "gentlemonster", "midea", "beijingweishi", "mlb", "cree", "dove", "PJmasks", "reddragonfly", "emerson", "lovemoschino", "suzuki", "erdos", "seiko", "cpb", "royalstar", "thehistoryofwhoo", "otterbox", "disney", "lindafarrow", "PATAGONIA", "seven7", "ford", "bandai", "newbalance", "alibaba", "sergiorossi", "lacoste", "bear", "opple", "walmart", "clinique", "asus", "ThomasFriends", "wanda", "lenovo", "metallica", "stuartweitzman", "karenwalker", "celine", "miui", "montagut", "pampers", "darlie", "toray", "bobdog", "ck", "flyco", "alexandermcqueen", "shaxuan", "prada", "miiow", "inman", "3t", "gap", "Yamaha", "fjallraven", "vancleefarpels", "acne", "audi", "hunanweishi", "henkel", "mg", "sony", "CHAMPION", "iwc", "lv", "dolcegabbana", "avene", "longchamp", "anessa", "satchi", "hotwheels", "nike", "hermes", "jiaodan", "siemens", "Goodbaby", "innisfree", "Thrasher", "kans", "kenzo", "juicycouture", "evisu", "volcom", "CanadaGoose", "Dickies", "angrybirds", "eddrac", "asics", "doraemon", "hisense", "juzui", "samsonite", "hikvision", "naturerepublic", "Herschel", "MANGO", "diesel", "hotwind", "intel", "arsenal", "rayban", "tommyhilfiger", "ELLE", "stdupont", "ports", "KOHLER", "thombrowne", "mobil", "Belif", "anello", "zhoushengsheng", "d_wolves", "FridaKahlo", "citizen", "fortnite", "beautyBlender", "alexanderwang", "charles_keith", "panerai", "lux", "beats", "Y-3", "mansurgavriel", "goyard", "eral", "OralB", "markfairwhale", "burberry", "uno", "okamoto", "only", "bvlgari", "heronpreston", "jimmythebull", "dyson", "kipling", "jeanrichard", "PXG", "pinkfong", "Versace", "CCTV", "paulfrank", "lanvin", "vans", "cdgplay", "baojianshipin", "rapha", "tissot", "casio", "patekphilippe", "tsingtao", "guess", "Lululemon", "hollister", "dell", "supor", "MaxMara", "metersbonwe", "jeanswest", "lancome", "lee", "omega", "lets_slim", "snp", "PINKFLOYD", "cartier", "zenith", "LG", "monchichi", "hublot", "benz", "apple", "blackberry", "wuliangye", "porsche", "bottegaveneta", "instantlyageless", "christopher_kane", "bolon", "tencent", "dkny", "aptamil", "makeupforever", "kobelco", "meizu", "vivo", "buick", "tesla", "septwolves", "samanthathavasa", "tomford", "jeep", "canon", "nfl", "kiehls", "pigeon", "zhejiangweishi", "snidel", "hengyuanxiang", "linshimuye", "toread", "esprit", "BASF", "gillette", "361du", "bioderma", "UnderArmour", "TommyHilfiger", "ysl", "onitsukatiger", "house_of_hello", "baidu", "robam", "konka", "jack_wolfskin", "office", "goldlion", "tiantainwuliu", "wonderflower", "arcteryx", "threesquirrels", "lego", "mindbridge", "emblem", "grumpycat", "bejirog", "ccdd", "3concepteyes", "ferragamo", "thermos", "Auby", "ahc", "panasonic", "vanguard", "FESTO", "MCM", "lamborghini", "laneige", "ny", "givenchy", "zara", "jiangshuweishi", "daphne", "longines", "camel", "philips", "nxp", "skf", "perfect", "toshiba", "wodemeilirizhi", "Mexican", "VANCLEEFARPELS", "HARRYPOTTER", "mcm", "nipponpaint", "chenguang", "jissbon", "versace", "girardperregaux", "chaumet", "columbia", "nissan", "3M", "yuantong", "sk2", "liangpinpuzi", "headshoulder", "youngor", "teenieweenie", "tagheuer", "starbucks", "pierrecardin", "vacheronconstantin", "peskoe", "playboy", "chanel", "HarleyDavidson_AE", "volvo", "be_cheery", "mulberry", "musenlin", "miffy", "peacebird", "tcl", "ironmaiden", "skechers", "moncler", "rimowa", "safeguard", "baleno", "sum37", "holikaholika", "gucci", "theexpendables", "dazzle", "vatti", "nintendo"]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                data_root=data_root,
                ann_file='annotations/train_20210409_1_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_1/',
                pipeline=train_pipeline,
                force_one_class=True
            ),
        ],
        separate_eval=False,
    ),
    val=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                data_root=data_root,
                ann_file='annotations/validation_20210409_1_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_1/',
                pipeline=test_pipeline,
                force_one_class=True
            ),
        ],
        separate_eval=False,
    ),
    test=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type,
                classes=openbrand_classes,
                data_root=data_root,
                ann_file='annotations/test_20210409_1_reduced.json',
                img_prefix='电商标识检测大赛_train_20210409_1/',
                pipeline=test_pipeline,
                force_one_class=True,
            ),
        ],
        separate_eval=False,
    ),
)
evaluation = dict(interval=1, metric='bbox')
