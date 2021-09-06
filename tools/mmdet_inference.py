import os

import torch.cuda
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import matplotlib.pyplot as plt
import rpimage

import time

config_file = 'mmdetection/configs/logo/cascade_detectors.py'
config_file = '/home/cgarriga/projects/machinelearning-logo-library/custom' \
              '/mmdetection/configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_logo.py'
"""
config_file = '/home/cgarriga/projects/machinelearning-logo-library/custom' \


              '/mmdetection/configs/fcos/fcos_r50_caffe_fpn_gn_1x_4gpu.py'
config_file = '/home/cgarriga/projects/machinelearning-logo-library/custom' \
              '/mmdetection/configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_logo.py'

config_file = '/home/cgarriga/projects/machinelearning-logo-library/custom' \
              '/mmdetection/configs/centermask/centermask_r50_caffe_logo.py'

config_file = '/home/cgarriga/projects/machinelearning-logo-library/custom' \
              '/mmdetection/configs/yolact/yolact_r50_1x8_coco.py'
"""


# download the checkpoint from model zoo and put it in `checkpoints/` url:
# https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn
# /faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118
# .pth
checkpoint_file = 'data/sample_test/epoch_21.pth'
checkpoint_file = '/home/cgarriga/projects/machinelearning-logo-library' \
                  '/custom/mmdetection/epoch_18.pth'
checkpoint_file = '/home/cgarriga/projects/machinelearning-logo-library/custom' \
              '/mmdetection/epoch_26.pth'
"""
checkpoint_file = '/home/cgarriga/projects/machinelearning-logo-library/custom' \
              '/mmdetection/configs/yolact/yolact_r50_1x8_coco_20200908-f38d58df.pth'
"""

device = 'cuda:0'
# init a detector

model = init_detector(config_file, checkpoint_file, device=device)
torch.cuda.empty_cache()
# inference the demo image


count = 0
#for category in os.listdir('data/litw/voc_format'):
category = 'voc_format'
ROOT_DIR = '../demo/test_images'
# for brand in sorted(os.listdir(ROOT_DIR + category)):
    #CAT_DIR = ROOT_DIR + category + '/'


files = [
    ROOT_DIR + '/' + IMG_FILE
    for IMG_FILE in sorted(os.listdir(ROOT_DIR))
    if '.jpg' in IMG_FILE
]

BATCH_SIZE = 2
pre = time.time()
results = inference_detector(model, files[:BATCH_SIZE])
post = time.time()

print(f"time: {post - pre}, batch size {min(BATCH_SIZE, len(files))}")

for result, file in zip(results, files):
    show_result_pyplot(
        model,
        file,
        result,
        title=file,
        score_thr=0.1
    )
