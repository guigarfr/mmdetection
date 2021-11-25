import random

import numpy as np

from .builder import DATASETS
from .openBrand import OpenBrandDataset
from .xml_style import XMLDataset


class DatasetClassificationConverter(object):
    """Basic classification converter structure for detection datasets."""

    def __init__(self):
        """Init for XMLDatasetClassification."""
        self.expanded_annotations = []
        for idx in range(len(self.data_infos)):
            ann_info = self.get_ann_info(idx)
            for adx in range(len(ann_info['bboxes'])):
                self.expanded_annotations.append(
                    dict(img_info=self.data_infos[idx],
                         ann_info=dict(
                             bboxes=ann_info['bboxes'][adx].reshape(1, -1),
                             labels=ann_info['labels'][adx].reshape(1, -1),
                             bboxes_ignore=np.empty((0, 4)),
                             labels_ignore=np.empty(0),
                         ))
                )

    def __len__(self):
        """Length of the dataset."""
        if self.test_mode:
            return len(self.data_infos)
        else:
            return len(self.expanded_annotations)

    def _rand_another(self, idx):
        new_idx = random.randint(0, len(self) - 1)
        while new_idx == idx:
            new_idx = random.randint(0, len(self) - 1)

        return new_idx

    def prepare_train_img(self, idx):
        """Get one item of the dataset."""
        results = self.expanded_annotations[idx]
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)


@DATASETS.register_module()
class XMLDatasetClassification(XMLDataset, DatasetClassificationConverter):
    """XML dataset for classification using detection structures."""

    def __init__(self, **kwargs):
        """Init for XMLDatasetClassification."""
        kwargs['force_one_class'] = False
        super(XMLDatasetClassification, self).__init__(**kwargs)


@DATASETS.register_module()
class OpenBrandDatasetClassification(
    OpenBrandDataset, DatasetClassificationConverter):
    """XML dataset for classification using detection structures."""

    def __init__(self, **kwargs):
        """Init for OpenBrandDatasetClassification."""
        kwargs['force_one_class'] = False
        super(OpenBrandDatasetClassification, self).__init__(**kwargs)

