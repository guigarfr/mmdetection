from mmdet.models.builder import build_loss
from mmdet.models.builder import HEADS
from mmcv.cnn import Linear, ConvModule
from mmcv.runner import force_fp32
import torch


@HEADS.register_module()
class OpenBrandFeatureRepresentation(object):

    def __init__(
            self,
            in_channels=512,
            out_channels=4096,
            num_classes=1,
            loss_cls=None,
            init_cfg=None
    ):
        self.conv5 = ConvModule(
                in_channels,
                in_channels,
                1,
                stride=1,
                padding=0
        )
        self.pool5 = torch.max_pool2d
        self.fc1 = Linear(in_channels * 7 * 7, out_channels)
        self.fc2 = Linear(out_channels, num_classes+1)
        self.loss_cls = build_loss(loss_cls)

    def forward_train(self, feats, labels):
        pred = self.simple_test(feats)
        return pred, self.loss()

    def simple_test(self, x):
        x = self.conv5(x)
        x = self.pool5(x, 2).flatten(1)
        x = self.fc1(x)
        x = self.fc2(x).softmax(-1)
        return x

    #@force_fp32(apply_to=('preds', 'gt_labels'))
    def loss(self, preds, gt_labels):
        loss_feat = self.loss_cls(preds, gt_labels)
        return dict(loss_feature=loss_feat)


if __name__ == '__main__':
    import numpy as np
    import torch

    test = np.random.random((1, 512, 14, 14))
    a = OpenBrandFeatureRepresentation(
        num_classes=1,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    )
    output = a.simple_test(torch.Tensor(test))
    a.loss(output, torch.Tensor([0]))
    print(output.shape)
