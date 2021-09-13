dataset_type = 'XMLDataset'
data_root = '/home/ubuntu/data/logo_dataset/'


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

logos_ds_classes = [
    '1002', '102', '103', '1052', '1102', '1152', '1252', '12521', '1302',
    '1352', '1452', '1454', '1504', '1506', '152', '1556', '1606', '1656',
    '1706', '1756', '1844', '1845', '1846', '1847', '1848', '1849', '1850',
    '1851', '19088', '2', '202', '203', '204', '21761', '21762', '21763',
    '21764', '21765', '21766', '21767', '21768', '21769', '21770', '21771',
    '21772', '21773', '21774', '21775', '21776', '21777', '21778', '21779',
    '21780', '21781', '21782', '21783', '21784', '21785', '21786', '21787',
    '21788', '21789', '21790', '21791', '21792', '21793', '21794', '21795',
    '21796', '21797', '21798', '21799', '21800', '21801', '21802', '21803',
    '21804', '21805', '21806', '21807', '21808', '21809', '21810', '21811',
    '21812', '21813', '21814', '21815', '21816', '21817', '21818', '21819',
    '21820', '21821', '21822', '21823', '21824', '21825', '21826', '21827',
    '21828', '21829', '21830', '21831', '21832', '21833', '21834', '21835',
    '21836', '21837', '21838', '21839', '21840', '21841', '21842', '21843',
    '21844', '21845', '21846', '21847', '21848', '21849', '21850', '21851',
    '21852', '21853', '21854', '21855', '21856', '21857', '21858', '21859',
    '21860', '21861', '21862', '21864', '21865', '21866', '21867', '21868',
    '21869', '21870', '21871', '21872', '21873', '21874', '21875', '21876',
    '21877', '21878', '21879', '21880', '21881', '21882', '21883', '21884',
    '21885', '21886', '21887', '21888', '21889', '21890', '21891', '21892',
    '21893', '21894', '21895', '21896', '21897', '21898', '21899', '21900',
    '21901', '21902', '21903', '21904', '21905', '21906', '21907', '21908',
    '21909', '21910', '21911', '21912', '21913', '21914', '21915', '21916',
    '21917', '21918', '21919', '21921', '21922', '21923', '21924', '21925',
    '21926', '21927', '21928', '21929', '21930', '21931', '21932', '21933',
    '21934', '21935', '21936', '21937', '21938', '21939', '21940', '21941',
    '21942', '21943', '21944', '21945', '21946', '21947', '21948', '21949',
    '21950', '21951', '21952', '21953', '21954', '21956', '21957', '21958',
    '252', '253', '302', '303', '352', '353', '354', '402', '4248', '4310',
    '4311', '4313', '4314', '4315', '4316', '4317', '4318', '4325', '4331',
    '4332', '4333', '4334', '4335', '4346', '4347', '4350', '4351', '4361',
    '4365', '4366', '4370', '4371', '4388', '4414', '452', '4832', '503',
    '52', '53', '5320', '552', '5524', '602', '652', '702', '752', '802',
    '852', '902', '952'
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        classes=logos_ds_classes,
        data_root=data_root,
        ann_file='ImageSets/Main/train.txt',
        img_prefix='',
        pipeline=train_pipeline,
        force_one_class=True
    ),
    val=dict(
        type=dataset_type,
        classes=logos_ds_classes,
        data_root=data_root,
        ann_file='ImageSets/Main/validation.txt',
        img_prefix='',
        force_one_class=True,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        classes=logos_ds_classes,
        data_root=data_root,
        ann_file='ImageSets/Main/test.txt',
        img_prefix='',
        pipeline=test_pipeline,
        force_one_class=True,
    ),
)

evaluation = dict(interval=1, metric='mAP')
