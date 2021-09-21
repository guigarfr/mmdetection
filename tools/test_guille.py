# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import time
import warnings

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from mmcv import Config, DictAction
from mmcv.visualization import image as view_img
from mmcv.cnn import fuse_conv_bn
from mmcv.image import tensor2imgs
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.parallel import collate, scatter
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector

from mmdet.datasets.pipelines import Compose

import faiss

def replace_pipeline(my_cfg, eval=True, samples_per_gpu=None):
    if isinstance(my_cfg, dict):
        if hasattr(my_cfg, 'datasets') and isinstance(my_cfg.datasets, list):
            samples_per_gpu = replace_pipeline(
                my_cfg.datasets,
                eval=eval,
                samples_per_gpu=samples_per_gpu)
        else:
            if not eval:
                my_cfg.test_mode = True
            defined_samples_gpu = my_cfg.pop('samples_per_gpu', None)
            if defined_samples_gpu:
                if samples_per_gpu is not None:
                    samples_per_gpu = min(defined_samples_gpu, samples_per_gpu)
                else:
                    samples_per_gpu = defined_samples_gpu
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            my_cfg.pipeline = replace_ImageToTensor(my_cfg.pipeline)
    else:
        for c in my_cfg:
            samples_per_gpu = replace_pipeline(
                c,
                eval=eval,
                samples_per_gpu=samples_per_gpu)

    return samples_per_gpu


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')


    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # in case the test dataset is concatenated
    samples_per_gpu = replace_pipeline(
        cfg.data.test,
        eval=args.eval,
        samples_per_gpu=cfg.data.pop('samples_per_gpu', None))

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)

    rank, world_size = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    model.eval()

    feature_cfg = mmcv.Config.fromfile(
        '/home/cgarriga/sources/mmclassification/configs/resnet'
        '/resnet50_b32x8_imagenet.py'
    )
    from mmcls.apis import init_model as feat_init_model

    feature_model = feat_init_model(
        feature_cfg,
        '/home/cgarriga/sources/mmclassification'
        '/resnet50_batch256_imagenet_20200708-cfb998bf.pth',
        'cpu'
    )

    from mmcls.datasets.pipelines import Compose as feat_compose

    feature_cfg.data.test.pipeline.pop(0)
    feature_test_pipeline = feat_compose(feature_cfg.data.test.pipeline)

    results = []
    prog_bar = mmcv.ProgressBar(len(dataset)) if rank == 0 else None
    time.sleep(2)  # This line can prevent deadlock problem in some cases.

    import pickle
    with open('class_labels', 'rb') as f:
        class_labels = pickle.load(f)
        print(class_labels)
    index = faiss.read_index('index_100')
    assert index.is_trained

    for i, data in enumerate(data_loader):
        cropped_imgs = []
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if prog_bar is not None:
            for _ in range(len(result)):
                prog_bar.update()

        batch_size = len(result)

        if batch_size == 1 and isinstance(result['img'][0], torch.Tensor):
            img_tensor = data['img'][0]
        else:
            img_tensor = data['img'][0].data[0]
        img_metas = data['img_metas'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        assert len(imgs) == len(img_metas)

        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None

        from mmdet.apis import show_result_pyplot

        for i, (img, img_meta, bboxes) in enumerate(
                zip(imgs, img_metas, bbox_result)):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]

            ori_h, ori_w = img_meta['ori_shape'][:-1]
            img_show = mmcv.imresize(img_show, (ori_w, ori_h))

            show_result_pyplot(model, img_show, bboxes, score_thr=0.01)

            inner_cropped_imgs = []
            for class_idx, class_bboxes in enumerate(bboxes):
                for i_bbox, bbox in enumerate(class_bboxes):
                    if bbox[4] < 0.1:
                        continue
                    bbox_int = bbox.astype(np.int32)

                    x1, x2 = sorted([bbox_int[0], bbox_int[2]])
                    y1, y2 = sorted([bbox_int[1], bbox_int[3]])
                    crop = img_show[y1:y2, x1:x2]
                    inner_cropped_imgs.append(crop)
            cropped_imgs.append(inner_cropped_imgs)

        logging.info("EXTRACTING FEATURES")

        datas = []
        raw_imgs = []
        for img_meta, inner_cropped_imgs in zip(img_metas, cropped_imgs):
            for cropid, img in enumerate(inner_cropped_imgs):
                assert img is not None
                # prepare data
                dats = dict(img=img)
                # build the data pipeline
                dats = feature_test_pipeline(dats)
                datas.append(dats)

        dats = collate(datas, samples_per_gpu=1)
        # just get the actual data from DataContainer

        if next(feature_model.parameters()).is_cuda:
            # scatter to specified GPU. [0] if single GPU
            dats = scatter(dats, ['cuda:0'])[0]

        with torch.inference_mode():
            for crop_batch in dats['img'].data:
                feats = feature_model.extract_feat(crop_batch.unsqueeze(0))

                for feat in feats.numpy():
                    ds, ids = index.search(np.array([feat]), 5)
                    chosen_one = sorted(zip(ds, ids), key=lambda x: x[0])[0]
                    print(
                        "chosen id: ",
                        chosen_one,
                        list(class_labels.keys())[
                            list(class_labels.values()).index(chosen_one[1][0])]
                    )

                    plt.imshow(tensor2imgs(crop_batch.unsqueeze(0),
                                           **feature_cfg.img_norm_cfg)[0])
                    plt.show()
        break


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
