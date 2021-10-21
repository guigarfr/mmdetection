"""Results classes file."""

import typing

import numpy as np

import rpimage


class BBoxExtractorResult(object):
    """Result class from bbox extractor class."""

    def __init__(
        self,
        sample: rpimage.RPImage,
        bboxes: typing.List[np.array],
        scores: typing.List[np.array]
    ):
        self.sample = sample
        self.bboxes = bboxes
        self.scores = scores


class FeatureExtractorResult(object):
    """Result class from bbox extractor class."""

    def __init__(
            self,
            sample: rpimage.RPImage,
            features: typing.List[np.array],
    ):
        self.sample = sample
        self.features = features


class FaissClassifierResult(object):
    """Result class from bbox extractor class."""

    def __init__(
        self,
        features: np.array,
        idx: typing.List[np.array],
        distance: typing.List[np.array]
    ):
        self.features = features
        self.idx = idx
        self.distance = distance
