
import typing

import rpimage

from rpmlbaselib import models as base_models

import numpy as np

from ..models import result


class FeatureExtractor(base_models.NNModel):
    """Feature Extractor model based on mmclassification"""

    def __init__(self, **kwargs):
        self.is_trained = False
        """
        self.config_file = kwargs.get('config_file', None)
        self.checkpoint_file = kwargs.get('checkpoint_file', None)

        model = init_detector(config_file, checkpoint_file, device=device)
        torch.cuda.empty_cache()
        """

        super(FeatureExtractor, self).__init__(**kwargs)

    def __getstate__(self):
        state = super(FeatureExtractor, self).__getstate__()
        state['model'] = self.model
        state['is_trained'] = self.is_trained
        return state

    def __setstate__(self, state):
        super(FeatureExtractor, self).__setstate__(state)
        self.model = state['model']
        self.is_trained = state['is_trained']

    def create(self):
        """Create model structure by default. Could be used as point to load
                the model."""
        pass

    def evaluate(self, data_generator, **kwargs):
        """Evaluate with already trained model. Not used in feature
        extractor."""
        raise NotImplementedError

    def train(self, samples, **kwargs):
        """Train feature extractor model. Not used in feature extractor."""
        raise NotImplementedError

    def update(self, data_generator, **kwargs):
        """Update feature extractor model. Not used in feature extractor."""
        raise NotImplementedError

    def predict(self, samples: typing.List[rpimage.RPImage], **kwargs) -> \
            typing.List[result.FeatureExtractorResult]:
        """Predict with feature extraction.

        :param samples: List of already cropped RPImages.
        :return: List of features. Predicted features for each cropped image."""
        pass
