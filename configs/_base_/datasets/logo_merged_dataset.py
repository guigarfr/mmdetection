dataset_type = 'LogosDataset'
data_root = '../data/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
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
        img_scale=(512, 512),
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

logo_train = dict(
        type=dataset_type,
        data_root=data_root + 'logo_dataset/',
        ann_file='ImageSets/Main/train_100.txt',
        img_prefix='',
        pipeline=train_pipeline,
    )

logo_test = dict(
        type=dataset_type,
        data_root=data_root + 'logo_dataset/',
        ann_file='ImageSets/Main/test.txt',
        img_prefix='',
        pipeline=test_pipeline,
    )

logo_val = dict(
        type=dataset_type,
        data_root=data_root + 'logo_dataset/',
        ann_file='ImageSets/Main/validation.small.txt',
        img_prefix='',
        pipeline=test_pipeline,
    )


open_train = dict(
        type=dataset_type,
        data_root=data_root + 'openbrand_dataset/',
        ann_file='ImageSets/Main/train_100.txt',
        img_prefix='',
        pipeline=train_pipeline,
    )


open_test = dict(
        type=dataset_type,
        data_root=data_root + 'openbrand_dataset/',
        ann_file='ImageSets/Main/test.txt',
        img_prefix='',
        pipeline=test_pipeline,
    )

open_val = dict(
        type=dataset_type,
        data_root=data_root + 'openbrand_dataset/',
        ann_file='ImageSets/Main/validation.txt',
        img_prefix='',
        pipeline=test_pipeline,
    )


data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    train=dict(
        type='ConcatDataset',
        datasets=[logo_train, open_train],
        separate_eval=False
    ),
    val=dict(
        type='ConcatDataset',
        datasets=[logo_val, open_val],
        separate_eval=False
    ),
    test=dict(
        type='ConcatDataset',
        datasets=[logo_test, open_test],
        separate_eval=False
    ),
)
# evaluation = dict(interval=1, metric='bbox')
