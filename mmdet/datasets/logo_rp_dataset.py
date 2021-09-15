import os.path as osp
import xml.etree.ElementTree as ET


import numpy as np
from .builder import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module()
class LogoRPDataset(XMLDataset):

    def __init__(self,
                 min_size=None,
                 img_subdir='JPEGImages',
                 ann_subdir='Annotations',
                 force_one_class=False,
                 **kwargs):
        if not force_one_class:
            self.CLASSES = (
            'northweek', 'apple', 'fashion nova', 'color street', 'lelo',
            'wahl', "o'neal", 'thrasher', 'reuzel', 'gelish', 'fat freezer',
            '100% moto', 'northwave', 'mr beast', 'gucci', 'team secret',
            'cambro', 'ove glove', 'louis vuitton', 'alpha industries', 'dakar',
            'buck knives', 'boppy', 'borealis', 'poker stars',
            'cry babies magic tears', 'rammstein', 'match madness', 'steyr',
            'sophie la giraffe', 'kaws', 'asics', 'foreo', 'fridababy', 'oyo',
            'schitts creek', 'les nereides', 'fat brain toys',
            'case construction', 'iveco', 'new holland', 'case ih', 'spiraflex',
            'brastemp', 'phonesoap', 'squigz', 'consul', 'one-step', 'dimpl',
            'kreg', 'billionaire boys club', 'ice cream', 'princess polly',
            'c.p. company', 'fleshlight', 'petal and pup', 'n2', 'rebdolls',
            'slip', 'vinturi', 'whites', 'wahl - 5 star', 'wahl - magic clip',
            'wahl - super taper', 'moser', 'tenga', 'alientech',
            'the tour de france', 'calorstat', 'gcds', 'ktag', 'kess v2',
            'yonanans', 'gmt', "white's goldmaster", 'duncan',
            'gmt micro tireur 140mm', 'distroller', 'neonato', 'felco', 'brady',
            'tac-force', 'mtech usa', 'hyperdrive', 'hyper', 'elk ridge',
            'cloud9', 'the mountain', 'tree hut', 'curaprox', 'baby magic',
            'coravin', 'fashionnova', 'heelys', 'gooby', 'current', 'aqua',
            'ergo baby', 'hyperthin', 'ooze', 'cleto reyes', 'cookies', 'seat',
            'cummins', 'cupra', 'seat - fr', 'fleetguard', 'ganni',
            'lyle and scott', 'troy lee', 'designs for health', 'fff', 'mvmt',
            'dxracer', 'doulton', 'cheveux', 'mavic', 'extrema ratio',
            'british berkefeld', 'tyme', 'chemex', 'lovense', 'johnny cupcakes',
            'i heart revolution', 'fx chocolate', 'glad', "burt's bees",
            'maclaren', 'revolution beauty', 'varo', 'emsculpt', 'kimber kable',
            'wattgate', 'gluedots', 'arm & hammer', 'ica', 'btl', 'solo stove',
            'keen', 'milwaukee tool', 'm18', 'true religion brand jeans',
            'jacto', 'hart', 'schiek', 'hoover', 'oreck', 'one+', 'vax',
            'kyte baby', 'volcano', 'catan studios', 'hobart', 'vulcan',
            'xenvo', 'spinagain', 'little big shot', 'pjh', 'monkey hook',
            'surfmaster', 'catan (chinese)', 'new era caps', 'snow joe',
            'ag - angelo galasso', 'sunjoe', 'aqua joe', 'ryzon', 'einhell',
            'cinelli', 'colnago', 'the comfy', 'young living essential oils',
            'hogue', 'onvif', 'bad dragon', 'lokai', 'vanquish fitness',
            'bebes llorones', 'vq', 'lord nermal', "tito's", 'frankies bikinis',
            'cookies sf', 'cookies logo', 'jelly', 'mvmt logo', 'moroccan oil',
            'chevron phillips', 'five minute journal', 'marlex',
            'productivity planner', 'goldbug', '999', 'r logo', 'rohl',
            'wounded warrior project', 'psycho bunny word', 'tsume', 'riobel',
            'paul valentine', '3doddler', 'griots garage', 'saxx underwear',
            'creature', 'dmc', 'figs', 'solaray', 'nifty', 'fretwraps',
            'alo yoga', 'ultimaker', 'zhou', 'fidget cube', 'bunch o balloons',
            'lister petter', 'robo fish', 'dewalt', 'black&decker',
            'gilded reverie', 'rider waite', 'aerogarden',
            '3doodler - owl item', 'platinum tools', 'adopt me', 'cpchem',
            'clear tv', '3doodler - tour eiffel item', 'honest amish', 'nike',
            'incoco', 'adidas', 'copper chef', 'grunt style', 'titos', 'hp',
            'sugar bear hair', 'puma', '3m', 'onitsuka tiger',
            'hewlett packard enterprise', 'arai helmet limited', 'goorin bros.',
            'goorin brothers inc.', 'foreo inc.', 'grunt style - gs',
            'wubbanub', 'cry babies', 'whitmor', 'comotomo', 'screaming hand',
            'scratch map', 'lindberg', 'ripndip', 'frida kahlo', 'ofra', 'omp',
            'keith haring foundation', 'by far', 'beauty blender', 'prada',
            'hoonigan', 'tulip', 'psycho bunny', 'matco tools',
            'goorin brothers', 'arai helmet', 'illesteva', 'rummikub', 's-bag',
            'under armour', 'hawkers', 'fox knives', 'keen footwear',
            'freds swim academy', 'fédération française de football',
            'santa cruz', 'spartan race', 'garrett')

            self.class_conversion_dict = {
                2: 'Sophie la giraffe',
                52: 'Arai Helmet',
                53: 'illesteva',
                102: 'Apple',
                103: 'Fashion Nova',
                152: 'GUCCI',
                202: 'Kaws',
                203: 'Asics',
                204: 'Foreo',
                252: 'Nike',
                253: 'Incoco',
                302: 'Adidas',
                303: 'Copper Chef',
                352: 'Grunt Style',
                353: 'Titos',
                354: 'HP',
                402: 'Sugar Bear Hair',
                452: 'Psycho Bunny',
                503: 'Goorin Brothers',
                552: 'S-bag',
                602: 'Hawkers',
                652: 'FOX Knives',
                702: 'Keen Footwear',
                752: 'Freds Swim Academy',
                802: 'FÉDÉRATION FRANÇAISE DE FOOTBALL',
                852: 'Santa Cruz',
                902: 'Spartan Race',
                952: 'Garrett',
                1002: 'Northweek',
                1052: 'Color Street',
                1102: 'Lelo',
                1152: 'Wahl',
                1252: "O'Neal",
                1302: 'Reuzel',
                1352: 'Gelish',
                1452: 'Fat Freezer',
                1454: '100% Moto',
                1504: 'Northwave',
                1506: 'Mr Beast',
                1556: 'Team Secret',
                1606: 'Cambro',
                1656: 'Ove glove',
                1706: 'Louis Vuitton',
                1756: 'Alpha Industries',
                1844: 'Dakar',
                1845: 'Buck Knives',
                1846: 'Boppy',
                1847: 'Borealis',
                1848: 'Poker Stars',
                1849: 'Cry Babies Magic Tears',
                1850: 'Rammstein',
                1851: 'Match Madness',
                4248: 'Puma',
                4310: '3M',
                4311: 'Onitsuka Tiger',
                4313: 'Hewlett Packard Enterprise',
                4314: 'Arai Helmet Limited',
                4315: 'Goorin Bros.',
                4316: 'Goorin Brothers Inc.',
                4317: 'Foreo Inc.',
                4318: 'Grunt Style - GS',
                4325: 'Wubbanub',
                4331: 'Cry Babies',
                4332: 'Whitmor',
                4333: 'Comotomo',
                4334: 'Screaming Hand',
                4335: 'Scratch Map',
                4346: 'Lindberg',
                4347: 'RIPNDIP',
                4350: 'Frida Kahlo',
                4351: 'OFRA',
                4361: 'OMP',
                4365: 'Keith Haring Foundation',
                4366: 'By Far',
                4370: 'Beauty blender',
                4371: 'Prada',
                4388: 'Hoonigan',
                4414: 'Tulip',
                4832: 'Matco Tools',
                5320: 'Rummikub',
                5524: 'Under Armour',
                12521: 'Thrasher',
                19088: 'Steyr',
                21761: 'Fridababy',
                21762: 'OYO',
                21763: 'Schitts Creek',
                21764: 'Les Nereides',
                21765: 'Fat Brain Toys',
                21766: 'CASE Construction',
                21767: 'Iveco',
                21768: 'New Holland',
                21769: 'Case IH',
                21770: 'Spiraflex',
                21771: 'Brastemp',
                21772: 'PhoneSoap',
                21773: 'SQUIGZ',
                21774: 'Consul',
                21775: 'One-Step',
                21776: 'DIMPL',
                21777: 'Kreg',
                21778: 'Billionaire boys club',
                21779: 'Ice cream',
                21780: 'Princess Polly',
                21781: 'C.P. Company',
                21782: 'Fleshlight',
                21783: 'Petal and Pup',
                21784: 'N2',
                21785: 'Rebdolls',
                21786: 'Slip',
                21787: 'Vinturi',
                21788: 'Whites',
                21789: 'Wahl - 5 Star',
                21790: 'Wahl - Magic Clip',
                21791: 'Wahl - Super Taper',
                21792: 'Moser',
                21793: 'Tenga',
                21794: 'Alientech',
                21795: 'The Tour de France',
                21796: 'Calorstat',
                21797: 'GCDS',
                21798: 'KTAG',
                21799: 'Kess V2',
                21800: 'Yonanans',
                21801: 'GMT',
                21802: "White's goldmaster",
                21803: 'Duncan',
                21804: 'GMT Micro Tireur 140mm',
                21805: 'Distroller',
                21806: 'Neonato',
                21807: 'Felco',
                21808: 'Brady',
                21809: 'Tac-Force',
                21810: 'Mtech USA',
                21811: 'Hyperdrive',
                21812: 'Hyper',
                21813: 'Elk Ridge',
                21814: 'Cloud9',
                21815: 'The Mountain',
                21816: 'Tree Hut',
                21817: 'Curaprox',
                21818: 'Baby Magic',
                21819: 'Coravin',
                21820: 'FashionNova',
                21821: 'Heelys',
                21822: 'Gooby',
                21823: 'Current',
                21824: 'Aqua',
                21825: 'Ergo Baby',
                21826: 'Hyperthin',
                21827: 'Ooze',
                21828: 'Cleto Reyes',
                21829: 'Cookies',
                21830: 'Seat',
                21831: 'Cummins',
                21832: 'Cupra',
                21833: 'Seat - FR',
                21834: 'Fleetguard',
                21835: 'Ganni',
                21836: 'Lyle and Scott',
                21837: 'Troy Lee',
                21838: 'Designs for Health',
                21839: 'fff',
                21840: 'MVMT',
                21841: 'DXRacer',
                21842: 'Doulton',
                21843: 'Cheveux',
                21844: 'Mavic',
                21845: 'Extrema Ratio',
                21846: 'British Berkefeld',
                21847: 'Tyme',
                21848: 'Chemex',
                21849: 'Lovense',
                21850: 'Johnny Cupcakes',
                21851: 'I heart revolution',
                21852: 'FX Chocolate',
                21853: 'Glad',
                21854: "Burt's Bees",
                21855: 'Maclaren',
                21856: 'Revolution Beauty',
                21857: 'Varo',
                21858: 'Emsculpt',
                21859: 'Kimber Kable',
                21860: 'Wattgate',
                21861: 'Gluedots',
                21862: 'Arm & Hammer',
                21864: 'ICA',
                21865: 'BTL',
                21866: 'Solo Stove',
                21867: 'Keen',
                21868: 'Milwaukee Tool',
                21869: 'M18',
                21870: 'True Religion Brand Jeans',
                21871: 'Jacto',
                21872: 'Hart',
                21873: 'Schiek',
                21874: 'Hoover',
                21875: 'Oreck',
                21876: 'One+',
                21877: 'Vax',
                21878: 'Kyte Baby',
                21879: 'Volcano',
                21880: 'Catan Studios',
                21881: 'Hobart',
                21882: 'Vulcan',
                21883: 'Xenvo',
                21884: 'Spinagain',
                21885: 'Little Big Shot',
                21886: 'PJH',
                21887: 'Monkey Hook',
                21888: 'Surfmaster',
                21889: 'Catan (Chinese)',
                21890: 'New Era Caps',
                21891: 'Snow Joe',
                21892: 'AG - Angelo Galasso',
                21893: 'Sunjoe',
                21894: 'Aqua Joe',
                21895: 'Ryzon',
                21896: 'Einhell',
                21897: 'Cinelli',
                21898: 'Colnago',
                21899: 'The Comfy',
                21900: 'Young Living Essential Oils',
                21901: 'Hogue',
                21902: 'Onvif',
                21903: 'Bad Dragon',
                21904: 'Lokai',
                21905: 'Vanquish Fitness',
                21906: 'Bebes Llorones',
                21907: 'VQ',
                21908: 'Lord Nermal',
                21909: "Tito's",
                21910: 'Frankies Bikinis',
                21911: 'Cookies SF',
                21912: 'Cookies Logo',
                21913: 'Jelly',
                21914: 'MVMT Logo',
                21915: 'Moroccan Oil',
                21916: 'Chevron Phillips',
                21917: 'Five Minute Journal',
                21918: 'Marlex',
                21919: 'Productivity Planner',
                21921: 'Goldbug',
                21922: '999',
                21923: 'R Logo',
                21924: 'Rohl',
                21925: 'Wounded Warrior Project',
                21926: 'Psycho Bunny Word',
                21927: 'Tsume',
                21928: 'Riobel',
                21929: 'Paul Valentine',
                21930: '3Doddler',
                21931: 'Griots Garage',
                21932: 'Saxx Underwear',
                21933: 'Creature',
                21934: 'DMC',
                21935: 'Figs',
                21936: 'Solaray',
                21937: 'Nifty',
                21938: 'Fretwraps',
                21939: 'Alo Yoga',
                21940: 'Ultimaker',
                21941: 'Zhou',
                21942: 'Fidget Cube',
                21943: 'Bunch O Balloons',
                21944: 'Lister Petter',
                21945: 'Robo Fish',
                21946: 'DeWalt',
                21947: 'Black&Decker',
                21948: 'Gilded Reverie',
                21949: 'Rider Waite',
                21950: 'AeroGarden',
                21951: '3Doodler - Owl Item',
                21952: 'Platinum Tools',
                21953: 'Adopt me',
                21954: 'cpchem',
                21956: 'Clear TV',
                21957: '3Doodler - Tour Eiffel Item',
                21958: 'Honest Amish'
            }

            kwargs['classes'] = self.CLASSES
        super(LogoRPDataset, self).__init__(
            min_size, img_subdir, ann_subdir, force_one_class, **kwargs)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without annotation."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) < min_size:
                continue
            if self.filter_empty_gt:
                img_id = img_info['id']
                xml_path = osp.join(self.img_prefix, self.ann_subdir,
                                    f'{img_id}.xml')
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    if self.force_one_class:
                        name = self.default_class_name
                    else:
                        name = self.class_conversion_dict[int(name)].lower()
                    if name in self.CLASSES:
                        valid_inds.append(i)
                        break
            else:
                valid_inds.append(i)
        return valid_inds

    def get_ann_info(self, idx):
        """Get annotation from XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, self.ann_subdir, f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text

            if self.force_one_class:
                name = self.default_class_name
            else:
                name = self.class_conversion_dict[int(name)].lower()
            if name in self.CLASSES:
                continue
            label = self.cat2label[name]
            difficult = obj.find('difficult')
            difficult = 0 if difficult is None else int(difficult.text)
            bnd_box = obj.find('bndbox')
            # TODO: check whether it is necessary to use int
            # Coordinates may be float type
            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]
            ignore = False
            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0,))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0,))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann

    def get_cat_ids(self, idx):
        """Get category ids in XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        cat_ids = []
        img_id = self.data_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, self.ann_subdir, f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            if self.force_one_class:
                name = self.default_class_name
            else:
                name = self.class_conversion_dict[int(name)].lower()
            if name in self.CLASSES:
                continue
            label = self.cat2label[name]
            cat_ids.append(label)

        return cat_ids