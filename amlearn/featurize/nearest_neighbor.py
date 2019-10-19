import six
import numpy as np
import pandas as pd
from abc import ABCMeta
from amlearn.featurize.base import create_featurizer_backend
from amlearn.utils.data import get_valid_lists
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from amlearn.featurize.src import voronoi_nn, distance_nn
except ImportError:
    print("import fortran file voronoi_nn, distance_nn error!\n")

__author__ = "Qi Wang"
__email__ = "qiwang.mse@gmail.com"


class BaseNN(six.with_metaclass(ABCMeta, BaseEstimator, TransformerMixin)):
    """Base Nearest Neighbor class.

    Args:
        cutoff (float): Cut off radius.
        allow_neighbor_limit (int): If after cut off atom's neighbor number
            still more than allow_neighbor_limit, then just choose
            allow_neighbor_limit neighbors to calculate voronoi.
        n_neighbor_limit (int):
        pbc (list like): Periodic bond chains.
        bds (list like): X, y, z boundaries.
        save (Boolean): Save file or not.
        backend (Backend): Amlearn Backend object, to prepare amlearn needed
            paths and define the common amlearn's load/save method.
    """

    def __init__(self, cutoff=5, allow_neighbor_limit=300, n_neighbor_limit=80,
                 type_col='type', coords_cols=None, pbc=None, bds=None,
                 save=True, backend=None, output_path=None,
                 output_file_prefix=None, print_freq=1000):
        self.cutoff = cutoff
        self.allow_neighbor_limit = allow_neighbor_limit
        self.n_neighbor_limit = n_neighbor_limit
        self.type_col = type_col
        self.coords_cols = coords_cols \
            if coords_cols is not None else ['x', 'y', 'z']
        self.pbc = pbc if pbc else [1, 1, 1]
        self.bds = bds if bds else [[-36, 36], [-36, 36], [-36, 36]]
        self.save = save
        if self.save:
            self.backend = backend if backend is not None \
                else create_featurizer_backend(output_path=output_path)
        self.output_file_prefix = output_file_prefix
        self.print_freq = print_freq

    def fit_transform(self, X=None, y=None, **fit_params):
        return self.transform(X)

    def transform(self, X=None):
        pass

    @property
    def category(self):
        return 'nearest_neighbor'


class VoroNN(BaseNN):
    def __init__(self, small_face_thres=0.05, cutoff=5,
                 allow_neighbor_limit=300, n_neighbor_limit=80,
                 type_col='type', coords_cols=None, pbc=None,
                 bds=None, save=True, backend=None, output_path=None,
                 output_file_prefix='voro_nn', print_freq=1000):
        super(VoroNN, self).__init__(
            cutoff=cutoff,  allow_neighbor_limit=allow_neighbor_limit,
            n_neighbor_limit=n_neighbor_limit, type_col=type_col,
            coords_cols=coords_cols, pbc=pbc, bds=bds, save=save,
            backend=backend, output_path=output_path,
            output_file_prefix=output_file_prefix, print_freq=print_freq)
        self.small_face_thres = small_face_thres

    def transform(self, X=None):
        """
        Args:
            X (DataFrame): Should contains ['type', 'x', 'y', 'z'...] columns

        Returns:
            prop_df (DataFrame): Every atom's neighbor number, neighbor id list,
                neighbor distance list, neighbor area list, neighbor volume
                list and neighbor edge list, which calculated based Voronoi.
        """
        n_atoms = len(X)
        neighbor_num_list = \
            np.zeros(n_atoms, dtype=np.longdouble)
        neighbor_id_lists = \
            np.zeros((n_atoms, self.n_neighbor_limit), dtype=np.longdouble)
        neighbor_edge_lists = \
            np.zeros((n_atoms, self.n_neighbor_limit), dtype=np.longdouble)
        neighbor_area_lists = \
            np.zeros((n_atoms, self.n_neighbor_limit), dtype=np.longdouble)
        neighbor_vol_lists = \
            np.zeros((n_atoms, self.n_neighbor_limit), dtype=np.longdouble)
        neighbor_dist_lists = \
            np.zeros((n_atoms, self.n_neighbor_limit), dtype=np.longdouble)

        (neighbor_num_list, neighbor_id_lists,  neighbor_area_lists,
         neighbor_vol_lists, neighbor_dist_lists, neighbor_edge_lists)= \
            voronoi_nn.voronoi(X[self.type_col].values,
                               X[self.coords_cols].values,
                               self.cutoff, self.allow_neighbor_limit,
                               self.small_face_thres, self.pbc, self.bds,
                               neighbor_num_list, neighbor_id_lists,
                               neighbor_area_lists, neighbor_vol_lists,
                               neighbor_dist_lists, neighbor_edge_lists,
                               n_atoms=n_atoms,
                               n_neighbor_limit=self.n_neighbor_limit,
                               print_freq=self.print_freq)

        voro_props = list()
        for neighbor_num, neighbor_id_list, \
            neighbor_dist_list, neighbor_area_list, \
            neighbor_vol_list, neighbor_edge_list in \
                zip(neighbor_num_list, neighbor_id_lists,
                    neighbor_dist_lists, neighbor_area_lists,
                    neighbor_vol_lists, neighbor_edge_lists):
            voro_props.append(
                get_valid_lists([neighbor_id_list, neighbor_dist_list,
                                 neighbor_area_list, neighbor_vol_list,
                                 neighbor_edge_list], valid_num=neighbor_num))

        prop_cols = self.get_feature_names()
        prop_df = pd.DataFrame(voro_props, index=X.index, columns=prop_cols)

        if self.save:
            self.backend.save_featurizer_as_dataframe(
                output_df=prop_df, name=self.output_file_prefix)
        return prop_df

    def get_feature_names(self):
        prop_columns = ['neighbor_num_voro', 'neighbor_ids_voro',
                        'neighbor_dists_voro', 'neighbor_areas_voro',
                        'neighbor_vols_voro', 'neighbor_edges_voro']
        return prop_columns


class DistanceNN(BaseNN):
    def __init__(self, cutoff=4, allow_neighbor_limit=300,
                 n_neighbor_limit=80, type_col='type',
                 coords_cols=None, pbc=None, bds=None,
                 backend=None, save=True, output_path=None,
                 output_file_prefix='dist_nn', print_freq=1000):
        super(DistanceNN, self).__init__(
            cutoff=cutoff, allow_neighbor_limit=allow_neighbor_limit,
            n_neighbor_limit=n_neighbor_limit, type_col=type_col,
            coords_cols=coords_cols, pbc=pbc, bds=bds, save=save,
            backend=backend, output_path=output_path,
            output_file_prefix=output_file_prefix, print_freq=print_freq)

    def transform(self, X=None):
        """
        Args:
            X (DataFrame): Should contains ['type', 'x', 'y', 'z'...] columns

        Returns:
            prop_df (DataFrame): Every atom's neighbor number, neighbor id list,
                neighbor distance list, which calculated based Distance.
        """

        n_atoms = len(X)
        neighbor_num_list = \
            np.zeros(n_atoms, dtype=np.longdouble)
        neighbor_id_lists = \
            np.zeros((n_atoms, self.n_neighbor_limit), dtype=np.longdouble)
        neighbor_dist_lists = \
            np.zeros((n_atoms, self.n_neighbor_limit), dtype=np.longdouble)

        (neighbor_num_list, neighbor_id_lists, neighbor_dist_lists) = \
            distance_nn.distance_neighbor(
                X[self.type_col].values, X[self.coords_cols].values,
                self.cutoff, self.allow_neighbor_limit, self.pbc,
                self.bds, neighbor_num_list,
                neighbor_id_lists, neighbor_dist_lists,
                n_atoms=n_atoms, n_neighbor_limit=self.n_neighbor_limit,
                print_freq=self.print_freq)

        dist_props = list()
        for neighbor_num, neighbor_id_list, neighbor_dist_list in \
                zip(neighbor_num_list, neighbor_id_lists, neighbor_dist_lists):
            dist_props.append(get_valid_lists(
                [neighbor_id_list, neighbor_dist_list], valid_num=neighbor_num))

        prop_df = pd.DataFrame(dist_props, index=X.index,
                               columns=self.get_feature_names())

        if self.save:
            self.backend.save_featurizer_as_dataframe(
                output_df=prop_df, name=self.output_file_prefix)

        return prop_df

    def get_feature_names(self):
        prop_cols = ['neighbor_num_dist', 'neighbor_ids_dist',
                     'neighbor_dists_dist']
        return prop_cols

