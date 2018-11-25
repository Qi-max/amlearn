import six
from amlearn.utils.backend import BackendContext, FeatureBackend
from sklearn.base import BaseEstimator, TransformerMixin
from abc import ABCMeta, abstractmethod


def create_featurizer_backend():
    backend_context = BackendContext()
    featurizer_backend = FeatureBackend(backend_context)
    return featurizer_backend


class BaseFeaturize(six.with_metaclass(ABCMeta,
                                       BaseEstimator, TransformerMixin)):
    def __init__(self, atoms_df=None, tmp_save=True, context=None):
        self.atoms_df = atoms_df
        self.tmp_save = tmp_save
        self.context = context if context is not None \
            else create_featurizer_backend()
        self._dependency = None

    def transform(self, X):
        pass

    @abstractmethod
    def get_feature_names(self):
        pass

    @property
    def dependency(self):
        return self._dependency