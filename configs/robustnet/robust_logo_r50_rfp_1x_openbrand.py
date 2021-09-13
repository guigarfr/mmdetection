_base_ = [
    '../_base_/models/robust_logo_r50_rfp.py',
    '../_base_/datasets/openBrand_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# Modify grad clip
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=5, norm_type=2))

# Change learning policy step
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8,12,16])

# Set max epochs to 18
runner = dict(type='EpochBasedRunner', max_epochs=18)

work_dir = '/apdcephfs/share_1290939/jiaxiaojun/OpenBrandData/outputEnd_detectors_r50'
load_from="/apdcephfs/share_1290939/jiaxiaojun/OpenBrandData/output_detectors_ratio-multinode/epoch_14.pth"
