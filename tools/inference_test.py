import os

import torch.cuda
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import matplotlib.pyplot as plt
import rpimage

import time

config_file = 'configs/robustlogodet/robust_logo_r50_rfp_1x_micro.py'

checkpoint_file = '/home/ubuntu/epoch_21.pth'

device = 'cuda:0'
# init a detector

model = init_detector(config_file, checkpoint_file, device=device)
torch.cuda.empty_cache()
# inference the demo image

count = 0
category = 'voc_format'
ROOT_DIR = '/home/ubuntu/data/kitti_tiny/training/image_2/'


files = [
    ROOT_DIR + '/' + IMG_FILE
    for IMG_FILE in sorted(os.listdir(ROOT_DIR))
    if '.jpeg' in IMG_FILE
]

BATCH_SIZE = 8
pre = time.time()
results = inference_detector(model, files[:BATCH_SIZE])
post = time.time()
model.CLASSES = list(map(str, range(515)))

true_bs = min(BATCH_SIZE, len(files))
elapsed_time = post - pre
print(f"time: {elapsed_time}, batch size {true_bs}, "
      f"time per sample {elapsed_time / true_bs}")

for result, file in zip(results, files):
    show_result_pyplot(
        model,
        file,
        result,
        title=file,
        score_thr=0.1
    )
