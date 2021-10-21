"""Class for bbox detector"""
import numpy as np

import rpimage

from rpmlbaselib import models as base_models

import typing
import torch.cuda

from ..models import result


class BBoxDetector(base_models.NNModel):
    """Bounding box detector model based on mmdetection"""

    def __init__(self, **kwargs):
        self.is_trained = False
        """
        self.config_file = kwargs.get('config_file', None)
        self.checkpoint_file = kwargs.get('checkpoint_file', None)
            
        model = init_detector(config_file, checkpoint_file, device=device)
        torch.cuda.empty_cache()
        """

        super(BBoxDetector, self).__init__(**kwargs)

    def __getstate__(self):
        state = super(BBoxDetector, self).__getstate__()
        state['model'] = self.model
        state['is_trained'] = self.is_trained
        return state

    def __setstate__(self, state):
        super(BBoxDetector, self).__setstate__(state)
        self.model = state['model']
        self.is_trained = state['is_trained']

    def create(self):
        """Create model structure by default. Could be used as point to load
        the model."""
        pass

    def evaluate(self, data_generator, **kwargs):
        """Evaluate with already trained model. Not used in bbox detector."""
        raise NotImplementedError

    def train(self, samples, **kwargs):
        """Train bbox model. Not used in bbox detector."""
        raise NotImplementedError

    def update(self, data_generator, **kwargs):
        """Update bbox model. Not used in bbox detector."""
        raise NotImplementedError

    def predict(self, samples: typing.List[rpimage.RPImage], **kwargs) -> \
            typing.List[result.BBoxExtractorResult]:
        """Predict with bbox model.

        :param samples: List of RPimages.
        :return: List of lists. Predicted bboxes for each result for each
        image."""
        pass
