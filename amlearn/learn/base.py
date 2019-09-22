import os
import time
from abc import ABCMeta, abstractmethod
from functools import lru_cache

from amlearn.utils.backend import BackendContext, MLBackend
from sklearn.externals import six
from sklearn.base import BaseEstimator, TransformerMixin

@lru_cache(maxsize=5)
def create_ml_backend(output_path='tmp'):
    """Create default ml backend.

    Returns:
        featurizer_backend (object): Featurizer Backend.
    """
    output_path = 'tmp' if output_path is None or output_path == 'tmp' \
        else output_path
    tmp_path = 'tmp' if output_path is None or output_path == 'tmp' \
        else os.path.join(output_path, 'tmp_{}'.format(int(time.time())))
    backend_context = BackendContext(merge_path=True,
                                     output_path=output_path,
                                     tmp_path=tmp_path)
    featurizer_backend = MLBackend(backend_context)
    return featurizer_backend


class BasePreprocessor(BaseEstimator, TransformerMixin):
    """Base class for all preprocess in amlearn."""

    pass


class AmBaseLearn(six.with_metaclass(ABCMeta, BaseEstimator)):
    """Base class for machine learning, setup backend, decimals... environment.

    """
    def __init__(self, backend, output_path=None, decimals=None, seed=1):
        self.backend = backend if backend is not None \
            else create_ml_backend(output_path=output_path)
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
