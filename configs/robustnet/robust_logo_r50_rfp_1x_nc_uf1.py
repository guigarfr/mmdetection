_base_ = [
    '../_base_/models/robust_logo_r50_rfp_nc_uf1.py',
    '../_base_/datasets/rp_all_ds.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

data = dict(
    samples_per_gpu=2,
)

# optimizer
optimizer = dict(
    lr=0.0025, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
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

work_dir = "/home/ubuntu/train_checkpoints/robustnet_no_class"
load_from = "/home/ubuntu/epoch_21.pth"
data_root = '/home/ubuntu/data/logo_dataset/'
