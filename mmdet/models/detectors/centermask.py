from ..builder import build_head
from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class CenterMask(SingleStageDetector):
    """Implementation of `CenterMask <https://arxiv.org/pdf/1911.06667.pdf>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 mask_head,
                 feature_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(CenterMask, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)

        self.mask_head = build_head(mask_head)
        self.feature_head = build_head(feature_head)

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        raise NotImplementedError

    def forward_train(
        self,
        img,
        img_metas,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        gt_masks=None
    ):
        super(SingleStageDetector, self).forward_train(img, img_metas)

        # Backbone (ResNet50) + FPN
        x = self.extract_feat(img)

        # Bbox, class and centerness prediction (FCOSHead)
        cls_score, bbox_pred, centerness = self.bbox_head(x)

        # Compute bbox, class and centerness loss (FCOSHead loss)
        bbox_head_loss_inputs = (cls_score, bbox_pred) + (gt_bboxes, gt_labels,
                                                          img_metas)
        losses, sampling_results = self.bbox_head.loss(
            *bbox_head_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        # Compute RoiAlign -> Output 512 Maps of 7x7 + SpatialMaskAttention
        # return mask based on classes
        masks = self.mask_head(x, (cls_score, bbox_pred), gt_bboxes,
                               num_classes=self.num)

        # Add the masks with the image.
        masks, patches

        # Compute CNN for feature extraction
        classes = self.feature_head(masks, patches)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results