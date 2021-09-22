_base_ = './rp_logos_dataset.py'

data = dict(
    train=dict(
        ann_file='ImageSets/Main/train_micro.txt',
    ),
    val=dict(
        ann_file='ImageSets/Main/validation_micro.txt',
    ),
    test=dict(
        ann_file='ImageSets/Main/test_micro.txt',
    ),
)
