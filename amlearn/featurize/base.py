import os
import six
import json
import numpy as np
import pandas as pd
from amlearn.featurize.featurizers.nearest_neighbor import VoroNN, DistanceNN
from amlearn.utils.backend import BackendContext, FeatureBackend
from amlearn.utils.check import check_dependency
from amlearn.utils.data import read_imd
from sklearn.base import BaseEstimator, TransformerMixin
from abc import ABCMeta, abstractmethod


try:
    from amlearn_beta.amlearn_beta.featurize.featurizers.sro_mro \
        import voronoi_stats
except Exception:
    print("import fortran file voronoi_stats error!\n")

module_dir = os.path.dirname(os.path.abspath( __file__))


def load_radii():
    with open(os.path.join(module_dir, '..', 'data',
                           'PTE.json'), 'r') as rf:
        PTE_dict_ = json.load(rf)
    return PTE_dict_


def create_featurizer_backend():
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
    def __init__(self, tmp_save=True, context=None):
        self.tmp_save = tmp_save
        self.context = context if context is not None \
            else create_featurizer_backend()
        self._dependency = None
        self.voro_depend_cols = None
        self.dist_denpend_cols = None

    def transform(self, X):
        pass

    @abstractmethod
    def get_feature_names(self):
        pass

    @property
    def dependency(self):
        return self._dependency

    @property
    def category(self):
        return 'sro'

    def check_dependency(self, X):
        if self._dependency is None:
            depend = None
        elif check_dependency(depend_cls=self._dependency,
                              df_cols=X.columns,
                              voro_depend_cols=self.voro_depend_cols,
                              dist_denpend_cols=self.dist_denpend_cols):
            depend = None
        else:
            depend = self._dependency
        return depend


class BaseSRO(six.with_metaclass(ABCMeta, BaseFeaturize)):
    def __init__(self, tmp_save=True, context=None,
                 dependency=None, remain_stat="none", **nn_kwargs):
        """

        Args:
            tmp_save:
            context:
            dependency: only accept "voro"/"voronoi" or "dist"/"distance"
            remain_stat: (boolean) default: False
                whether remain the source dataframe cols to result dataframe.
            **nn_kwargs:
        """
        super(BaseSRO, self).__init__(tmp_save=tmp_save,
                                      context=context)
        if dependency is None:
            self._dependency = None
            self.dependency_name = 'voro'
        elif isinstance(dependency, type):
            self.dependency_name = dependency.__class__.__name__.lower()[:-2]
            self._dependency = dependency
        elif isinstance(dependency, str):
            self.dependency_name = dependency[:4]
            if dependency == "voro" or dependency == "voronoi":
                self._dependency = VoroNN(context=context, **nn_kwargs)
            elif dependency == "dist" or dependency == "distance":
                self._dependency = DistanceNN(context=context, **nn_kwargs)
            else:
                raise ValueError('dependency {} if unknown, Possible values '
                                 'are {}'.format(dependency,
                                                 '[voro, voronoi, '
                                                 'dist, distance]'))
        else:
            raise ValueError('dependency {} if unknown, Possible values '
                             'are {} or voro/dist object.'.format(
                              dependency, '[voro, voronoi, dist, distance]'))
        self.remain_stat = remain_stat

    def fit(self, X=None, dependence_df=None):
        dependence_df = dependence_df if dependence_df is not None else X
        self._dependency = self.check_dependency(dependence_df)
        if self._dependency:
            self.dependence_df = self._dependency.fit_transform(dependence_df)
        return self

    @property
    def category(self):
        return 'sro'

