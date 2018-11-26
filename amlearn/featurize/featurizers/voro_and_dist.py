import os

import numpy as np
import pandas as pd
import six
from abc import ABCMeta
from amlearn.featurize.featurizers.base import BaseFeaturize
from amlearn.utils.check import check_featurizer_X

try:
    from amlearn.featurize.featurizers.sro_mro import voronoi_nn, \
        distance_nn
except Exception:
    print("import fortran file voronoi_nn, distance_nn error!\n")


class BaseNN(six.with_metaclass(ABCMeta, BaseFeaturize)):

    def __init__(self, cutoff=5, allow_neighbor_limit=300, n_neighbor_limit=80,
                 pbc=None, Bds=None, atoms_df=None, context=None,
                 tmp_save=True):
        super(BaseNN, self).__init__(tmp_save=tmp_save,
                                     context=context,
                                     atoms_df=atoms_df)
        self.cutoff = cutoff
        self.allow_neighbor_limit = allow_neighbor_limit
        self.n_neighbor_limit = n_neighbor_limit
        self.pbc = pbc if pbc else [1, 1, 1]
        self.Bds = Bds if Bds else [[-35.5040474, 35.5040474],
                                    [-35.5040474, 35.5040474],
                                    [-35.5040474, 35.5040474]]

    def fit_transform(self, X=None, y=None, **fit_params):
        return self.transform(X)

    @property
    def category(self):
        return 'voro_and_dist'


class VoroNN(BaseNN):

    def __init__(self, cutoff=5, allow_neighbor_limit=300, n_neighbor_limit=80,
                 pbc=None, Bds=None, atoms_df=None, small_face_thres=0.05,
                 context=None, tmp_save=True):
        super(VoroNN, self).__init__(tmp_save=tmp_save,
                                     context=context,
                                     atoms_df=atoms_df,
                                     cutoff=cutoff,
                                     allow_neighbor_limit=allow_neighbor_limit,
                                     n_neighbor_limit=n_neighbor_limit,
                                     pbc=pbc, Bds=Bds)
        self.small_face_thres = small_face_thres

    def transform(self, X=None):
        """

        Args:
            X: dataframe(should contains ['type', 'x', 'y', 'z'...] columns)

        Returns:

        """
        X = check_featurizer_X(X=X, atoms_df=self.atoms_df)
        n_atoms = len(X)
        n_neighbor_list = np.zeros(n_atoms, dtype=np.float128)
        neighbor_lists = np.zeros((n_atoms, self.n_neighbor_limit),
                                  dtype=np.float128)
        neighbor_edge_lists = np.zeros((n_atoms, self.n_neighbor_limit),
                                       dtype=np.float128)
        neighbor_area_lists = np.zeros((n_atoms, self.n_neighbor_limit),
                                       dtype=np.float128)
        neighbor_vol_lists = np.zeros((n_atoms, self.n_neighbor_limit),
                                      dtype=np.float128)
        neighbor_distance_lists = np.zeros(
            (n_atoms, self.n_neighbor_limit), dtype=np.float128)

        n_neighbor_max = 0
        n_edge_max = 0

        n_neighbor_list, neighbor_lists, \
        neighbor_area_lists, neighbor_vol_lists, neighbor_distance_lists, \
        neighbor_edge_lists, n_neighbor_max, n_edge_max = \
            voronoi_nn.voronoi(X['type'].values, X[['x', 'y', 'z']].values,
                               self.cutoff, self.allow_neighbor_limit,
                               self.small_face_thres,
                               self.pbc, self.Bds, n_neighbor_list,
                               neighbor_lists, neighbor_area_lists,
                               neighbor_vol_lists, neighbor_distance_lists,
                               neighbor_edge_lists, n_neighbor_max, n_edge_max,
                               n_atoms=n_atoms,
                               n_neighbor_limit=self.n_neighbor_limit)

        voro_nn = np.append(np.array([n_neighbor_list]).T,
                            neighbor_lists, axis=1)
        voro_nn = np.append(voro_nn, neighbor_area_lists, axis=1)
        voro_nn = np.append(voro_nn, neighbor_vol_lists, axis=1)
        voro_nn = np.append(voro_nn, neighbor_distance_lists, axis=1)
        voro_nn = np.append(voro_nn, neighbor_edge_lists, axis=1)
        result_df = pd.DataFrame(voro_nn, index=range(n_atoms),
                                 columns=self.get_feature_names())

        if self.tmp_save:
            self.context.save_featurizer_as_dataframe(output_df=result_df,
                                                      name='voro_nn')
        return result_df

    def get_feature_names(self):
        columns = ['n_neighbors_voro'] + \
                  ['neighbor_id_{}_voro'.format(i)
                   for i in range(self.n_neighbor_limit)] +\
                  ['neighbor_area_{}_voro'.format(i)
                   for i in range(self.n_neighbor_limit)] + \
                  ['neighbor_vol_{}_voro'.format(i)
                   for i in range(self.n_neighbor_limit)] + \
                  ['neighbor_distance_{}_voro'.format(i)
                   for i in range(self.n_neighbor_limit)] + \
                  ['neighbor_edge_{}_voro'.format(i)
                   for i in range(self.n_neighbor_limit)]
        return columns


class DistanceNN(BaseNN):

    def __init__(self, cutoff=5, allow_neighbor_limit=300,
                 n_neighbor_limit=80, pbc=None, Bds=None,
                 atoms_df=None, context=None, tmp_save=True):
        super(DistanceNN, self).__init__(
            tmp_save=tmp_save, context=context, atoms_df=atoms_df,
            cutoff=cutoff, allow_neighbor_limit=allow_neighbor_limit,
            n_neighbor_limit=n_neighbor_limit, pbc=pbc, Bds=Bds)

    def transform(self, X=None):
        X = check_featurizer_X(X=X, atoms_df=self.atoms_df)
        n_atoms = len(X)
        n_neighbor_list = np.zeros(n_atoms, dtype=np.float128)
        neighbor_lists = np.zeros((n_atoms, self.n_neighbor_limit),
                                  dtype=np.float128)
        neighbor_distance_lists = np.zeros(
            (n_atoms, self.n_neighbor_limit), dtype=np.float128)

        n_neighbor_max = 0

        print(distance_nn.distance_nn.distance_neighbor.__doc__)
        (n_neighbor_max, n_neighbor_list, neighbor_lists,
         neighbor_distance_lists) = \
            distance_nn.distance_nn.distance_neighbor(
                X['type'].values, X[['x', 'y', 'z']].values,
                self.cutoff, self.allow_neighbor_limit, self.pbc,
                self.Bds, n_neighbor_max, n_neighbor_list,
                neighbor_lists, neighbor_distance_lists,
                n_atoms=n_atoms, n_neighbor_limit=self.n_neighbor_limit)

        dist_nn = np.append(np.array([n_neighbor_list]).T,
                                 neighbor_lists, axis=1)
        dist_nn = np.append(dist_nn,
                                 neighbor_distance_lists, axis=1)
        result_df = pd.DataFrame(dist_nn, index=range(n_atoms),
                                 columns=self.get_feature_names())

        if self.tmp_save:
            self.context.save_featurizer_as_dataframe(output_df=result_df,
                                                      name='dist_nn')

        return result_df

    def get_feature_names(self):
        columns = ['n_neighbors_dist'] + \
                  ['neighbor_id_{}_dist'.format(i) for i in
                   range(self.n_neighbor_limit)] + \
                  ['neighbor_distance_{}_dist'.format(i) for i in
                   range(self.n_neighbor_limit)]
        return columns

