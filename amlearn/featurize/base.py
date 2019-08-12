import os
import six
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from abc import ABCMeta, abstractmethod
from amlearn.utils.backend import BackendContext, FeatureBackend
from amlearn.utils.check import check_featurizer_X, check_dependency
from amlearn.utils.data import read_imd


def create_featurizer_backend():
    backend_context = BackendContext(merge_path=True)
    featurizer_backend = FeatureBackend(backend_context)
    return featurizer_backend


def remain_df_calc(remain_df, result_df, source_df,
                   n_neighbor_col='n_neighbors_voro'):
    if remain_df:
        result_df = source_df.join(result_df)
    else:
        remain_cols = [n_neighbor_col] + [col for col in source_df.columns
                                          if col.startswith('neighbor_id_')]
        result_df = source_df[remain_cols].join(result_df)
    return result_df


class BaseFeaturize(six.with_metaclass(ABCMeta,
                                       BaseEstimator, TransformerMixin)):
    def __init__(self, atoms_df=None, tmp_save=True, context=None):
        self.atoms_df = atoms_df
        self.tmp_save = tmp_save
        self.context = context if context is not None \
            else create_featurizer_backend()
        self._dependency = None
        self.voro_depend_cols = None
        self.dist_denpend_cols = None

    def transform(self, X):
        pass

    @abstractmethod
    def get_feature_cols(self):
        pass

    @property
    def dependency(self):
        return self._dependency

    @property
    def double_dependency(self):
        return False

    @property
    def category(self):
        return 'voro_and_dist'

    def check_dependency(self, X):
        self.atoms_df = check_featurizer_X(X=X, atoms_df=self.atoms_df)
        if self._dependency is None:
            depend = None
        elif check_dependency(depend_cls=self._dependency,
                              df_cols=self.atoms_df.columns,
                              voro_depend_cols=self.voro_depend_cols,
                              dist_denpend_cols=self.dist_denpend_cols):
            depend = None
        else:
            depend = self._dependency
        return depend


class BaseNN(six.with_metaclass(ABCMeta,
                                BaseEstimator, TransformerMixin)):
    def __init__(self, cutoff=5, allow_neighbor_limit=300, n_neighbor_limit=80,
                 pbc=None, Bds=None, atoms_df=None, context=None,
                 tmp_save=True):
        self.cutoff = cutoff
        self.allow_neighbor_limit = allow_neighbor_limit
        self.n_neighbor_limit = n_neighbor_limit
        self.pbc = pbc if pbc else [1, 1, 1]
        self.Bds = Bds if Bds else [[-35.5040474, 35.5040474],
                                    [-35.5040474, 35.5040474],
                                    [-35.5040474, 35.5040474]]
        self.atoms_df = atoms_df
        self.tmp_save = tmp_save
        self.context = context if context is not None \
            else create_featurizer_backend()
        self._dependency = None
        self.voro_depend_cols = None
        self.dist_denpend_cols = None

    @classmethod
    def from_file(cls, data_path_file, cutoff, allow_neighbor_limit,
                  n_neighbor_limit, pbc, **kwargs):
        if os.path.exists(data_path_file):
            _, atom_type, atom_coords, Bds = read_imd(data_path_file)
        else:
            raise FileNotFoundError("File {} not found".format(data_path_file))

        atoms_df = pd.DataFrame(atom_coords, columns=['x', 'y', 'z'],
                                index=range(len(atom_coords)))
        atoms_df['type'] = pd.Series(atom_type, index=atoms_df.index)
        return cls(cutoff=cutoff, atoms_df=atoms_df,
                   allow_neighbor_limit=allow_neighbor_limit,
                   n_neighbor_limit=n_neighbor_limit, pbc=pbc, Bds=Bds,
                   **kwargs)

    def transform(self, X):
        pass

    def fit_transform(self, X=None, y=None, **fit_params):
        return self.transform(X)

    @abstractmethod
    def get_nn_cols(self):
        pass

    @property
    def dependency(self):
        return self._dependency

    @property
    def double_dependency(self):
        return False

    @property
    def category(self):
        return 'voro_and_dist'

    def check_dependency(self, X):
        self.atoms_df = check_featurizer_X(X=X, atoms_df=self.atoms_df)
        if self._dependency is None:
            depend = None
        elif check_dependency(depend_cls=self._dependency,
                              df_cols=self.atoms_df.columns,
                              voro_depend_cols=self.voro_depend_cols,
                              dist_denpend_cols=self.dist_denpend_cols):
            depend = None
        else:
            depend = self._dependency
        return depend
