# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import time
import warnings

from collections.abc import Sequence
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
                            replace_ImageToTensor, ConcatDataset)
from mmdet.models import build_detector

from mmdet.datasets.pipelines import Compose


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


import itertools

def get_dataset_config(cfg):
    if isinstance(cfg, Sequence):
        return itertools.chain.from_iterable([get_dataset_config(c) for c in
                                              cfg])
    elif isinstance(cfg, dict) and \
         hasattr(cfg, 'datasets') and \
         isinstance(cfg.datasets, list):
        return itertools.chain.from_iterable([get_dataset_config(c) for c in
                                              cfg['datasets']])
    else:
        return [cfg]


def retrieve_data_cfg(config_path, skip_type, cfg_options):

    def skip_pipeline_steps(config):
        config['pipeline'] = [
            x for x in config.pipeline if x['type'] not in skip_type
        ]

    cfg = Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    train_data_cfg = cfg.data.train
    while 'dataset' in train_data_cfg and train_data_cfg[
            'type'] != 'MultiImageMixDataset':
        train_data_cfg = train_data_cfg['dataset']

    for c in get_dataset_config(train_data_cfg):
        skip_pipeline_steps(c)

    return cfg


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

    cfg = retrieve_data_cfg(
        args.config,
        ['DefaultFormatBundle', 'Normalize', 'Collect'],
        args.cfg_options
    )

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # build the dataloader
    dataset = build_dataset(cfg.data.train)

    class_dict = {}
    for d in dataset.datasets:
        if isinstance(d, ConcatDataset):
            for ds in d.datasets:
                class_dict.update(ds.cat2label)
        else:
            class_dict.update(d.cat2label)

    import pickle
    with open('class_labels', 'wb+') as f:
        pickle.dump(class_dict, f)
    # Get train samples

    feature_cfg = mmcv.Config.fromfile(
        '/home/cgarriga/sources/mmclassification/configs/resnet/resnet50_b32x8_imagenet.py')
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

    import faiss

    quantizer = faiss.IndexFlatL2(2048)
    index = faiss.IndexIVFFlat(quantizer, 2048, 2, faiss.METRIC_L2)

    MAX_SAMPLES = 1000
    train = []
    for i, samples in enumerate(dataset):
        if i > MAX_SAMPLES:
            break

        img = samples['img']
        bboxes = samples['gt_bboxes']
        labels = samples['gt_labels']
        for bbox, label in zip(bboxes, labels):
            dats = dict(
                img=img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])],
            )
            # build the data pipeline
            dats = feature_test_pipeline(dats)

            with torch.no_grad():
                feats = feature_model.extract_feat(
                    torch.as_tensor([dats['img'].data.numpy()])
                )
                train.append((feats.numpy().astype('float32'), label))
        print(i)
    feats, labels = zip(*train)
    print(len(feats))
    feats = np.vstack(feats)
    print(feats.shape)
    print(feats[0].shape)
    index.train(feats)
    index.add_with_ids(feats, np.array(labels))
    faiss.write_index(index, 'index_100')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
