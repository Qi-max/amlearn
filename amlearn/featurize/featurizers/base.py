from abc import ABCMeta, abstractmethod

import six
from sklearn.base import BaseEstimator, TransformerMixin


class BaseFeaturize(six.with_metaclass(ABCMeta,
                                       BaseEstimator, TransformerMixin)):
    def transform(self, X):
        pass

    @abstractmethod
    def get_feature_names(self):
        pass
