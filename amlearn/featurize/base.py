import os
import six
import json
import numpy as np
from amlearn.utils.backend import BackendContext, FeatureBackend
from sklearn.base import BaseEstimator, TransformerMixin
from abc import ABCMeta, abstractmethod


try:
    from amlearn.featurize.featurizers.src \
        import voronoi_stats
except Exception:
    print("import fortran file voronoi_stats error!\n")

module_dir = os.path.dirname(os.path.abspath( __file__))


def load_radii():
    """Get Periodic Table of Elements dict.

    Returns:
        PTE_dict_ (dict): The Periodic Table of Elements dict, key is atomic id,
            value is dict which contains 'symbol', 'traditional_radius' and
            'miracle_radius'.

    """
    with open(os.path.join(module_dir, '..', 'data',
                           'PTE.json'), 'r') as rf:
        PTE_dict_ = json.load(rf)
    return PTE_dict_


def create_featurizer_backend():
    """Create default featurizer backend.

    Returns:
        featurizer_backend (object): Featurizer Backend.
    """
    backend_context = BackendContext(merge_path=True, output_path='tmp',
                                     tmp_path='tmp')
    featurizer_backend = FeatureBackend(backend_context)
    return featurizer_backend


def line_percent(value_list):
    percent_list = np.zeros(value_list.shape)

    percent_list = \
        voronoi_stats.line_percent(percent_list, value_list)
    return percent_list


class BaseFeaturize(six.with_metaclass(ABCMeta,
                                       BaseEstimator, TransformerMixin)):
    """Base featurize class for amlearn.

    Args:
        save (Boolean): save file or not.
        backend (object): Amlearn Backend object, to prepare amlearn needed
            paths and define the common amlearn's load/save method.
    """

    def __init__(self, save=True, backend=None):
        self.save = save
        self.backend = backend if backend is not None \
            else create_featurizer_backend()
        self.dependency_class_ = None
        self.dependency_cols_ = None

    def transform(self, X):
        pass

    @abstractmethod
    def get_feature_names(self):
        pass

    @property
    def dependency_class(self):
        return self.dependency_class_

    @property
    def category(self):
        return 'sro'

    def check_dependency(self, X):
        if self.dependency_class_ is None:
            dependency_class = None
        elif set(self.dependency_cols_).issubset(set(X.columns)):
            dependency_class = None
        else:
            dependency_class = self.dependency_class_
        return dependency_class

