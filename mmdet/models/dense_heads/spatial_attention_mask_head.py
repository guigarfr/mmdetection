from mmcv.cnn import ConvModule, ConvTranspose2d, kaiming_init

from ..builder import HEADS
import torch
from torch import nn
from torch.nn import functional as F


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = ConvModule(2, 1, kernel_size, padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.shape[0] == 0:
            return x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv(scale)
        return x * self.sigmoid(scale)


@HEADS.register_module()
class SpatialAttentionMaskHead(nn.Module):
    """
    A mask head with several conv layers and spatial attention module 
    in CenterMask paper, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(
        self,
        in_channels,
        num_convs,
        out_channels,
        num_classes=1,
        init_cfg=None
    ):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(SpatialAttentionMaskHead, self).__init__()

        # fmt: off
        conv_dims = out_channels
        self.norm = None
        num_conv = num_convs
        input_channels = in_channels
        cls_agnostic_mask = num_classes == 1
        # fmt: on

        self.conv_norm_relus = []

        for k in range(num_conv):
            conv = ConvModule(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)

        self.spatialAtt = SpatialAttention()

        self.deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0),
        )

        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = ConvModule(
            conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)

    def _init_weights(self):
        for layer in self.conv_norm_relus:
            layer.init_weights()
        kaiming_init(self.deconv)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def simple_test(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        x = self.spatialAtt(x)
        x = F.relu(self.deconv(x))
        return self.predictor(x)

"""
if __name__ == '__main__':
    import numpy as np
    tensor = torch.Tensor(np.random.random((1, 256, 14, 14)))
    head = SpatialAttentionMaskHead(
        256,
        4,
        256,
        1
    )
    result = head.simple_test(tensor)
    print(result.shape)
"""