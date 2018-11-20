from sklearn.externals import six
from sklearn.base import BaseEstimator, TransformerMixin
from abc import ABCMeta, abstractmethod


class BasePreprocess(BaseEstimator, TransformerMixin):
    """Base class for all preprocess in amlearn."""

    pass
