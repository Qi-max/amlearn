from abc import ABCMeta, abstractmethod
from sklearn.externals import six
from sklearn.base import BaseEstimator, TransformerMixin


class BasePreprocessor(BaseEstimator, TransformerMixin):
    """Base class for all preprocess in amlearn."""

    pass


class AmBaseLearn(six.with_metaclass(ABCMeta, BaseEstimator)):
    """Base class for machine learning, setup backend, decimals... environment.

    """
    def __init__(self, backend, decimals=None, seed=1):
        self.backend = backend
        self.decimals = decimals
        self.seed = seed

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def valid_components(self):
        pass
