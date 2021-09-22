_base_ = [
    '../_base_/models/robust_logo_r50_rfp.py',
    '../_base_/datasets/logodet_3k.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

data = dict(
    test=dict(
        force_one_class=True,
        collapse_multiclass=True,
    )
)

# Modify grad clip
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=5, norm_type=2))

# Change learning policy step
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 12, 16])

# Set max epochs to 18
runner = dict(type='EpochBasedRunner', max_epochs=18)

work_dir = "/home/ubuntu/logos_dataset/outputEnd_detectors_r50"
load_from = "/home/ubuntu/epoch_21.pth"
data_root = '/home/ubuntu/data/logo_dataset/'

