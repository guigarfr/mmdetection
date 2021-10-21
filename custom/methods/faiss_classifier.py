"""Faiss classifier class."""

from rpmlbaselib import models as base_models

from ..models import result

import numpy as np

import typing


class FaissClassifier(base_models.NNModel):
    """Faiss classifier class"""

    def __init__(self, **kwargs):
        self.is_trained = False

        super(FaissClassifier, self).__init__(**kwargs)

    def __getstate__(self):
        state = super(FaissClassifier, self).__getstate__()
        state['model'] = self.model
        state['is_trained'] = self.is_trained
        return state

    def __setstate__(self, state):
        super(FaissClassifier, self).__setstate__(state)
        self.model = state['model']
        self.is_trained = state['is_trained']

    def create(self):
        pass

    def evaluate(self, data_generator, **kwargs):
        """Update faiss model. Not used at the moment"""

        raise NotImplementedError

    def train(self, samples, **kwargs):
        """Train faiss classifier.
        :param samples:
        :return: trained index"""

        pass

    def update(self, data_generator, **kwargs):
        """Update faiss classifier ."""

        pass

    def predict(self, samples: typing.List[np.array], **kwargs) -> \
            typing.List[result.FaissClassifierResult]:
        """Search with Faiss classifier.

        :param samples: List of already cropped image features.
        :return: List of results. Predicted indexes and distances for each
        cropped image features."""
        pass
