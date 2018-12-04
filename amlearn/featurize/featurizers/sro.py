import os
import numpy as np
import pandas as pd
import six
from abc import ABCMeta
from amlearn.featurize.featurizers.base import BaseFeaturize, remain_df_calc
from amlearn.featurize.featurizers.voro_and_dist import VoroNN, DistanceNN
from amlearn.utils.check import check_featurizer_X, check_dependency
from amlearn.utils.data import read_imd

try:
    from amlearn.featurize.featurizers.sro_mro import voronoi_stats, boop
except Exception:
    print("import fortran file voronoi_stats error!\n")


class BaseSRO(six.with_metaclass(ABCMeta, BaseFeaturize)):
    def __init__(self, atoms_df=None, tmp_save=True, context=None,
                 dependency=None, remain_df=False, **nn_kwargs):
        """

        Args:
            atoms_df:
            tmp_save:
            context:
            dependency: only accept "voro"/"voronoi" or "dist"/"distance"
            remain_df: (boolean) default: False
                whether remain the source dataframe cols to result dataframe.
            **nn_kwargs:
        """
        super(BaseSRO, self).__init__(tmp_save=tmp_save,
                                      context=context,
                                      atoms_df=atoms_df)
        if dependency is None:
            self._dependency = None
            self.dependency_name = 'voro'
        elif isinstance(dependency, type):
            self.dependency_name = dependency.__class__.__name__.lower()[:-2]
            self._dependency = dependency
        elif isinstance(dependency, str):
            self.dependency_name = dependency[:4]
            if dependency == "voro" or dependency == "voronoi":
                self._dependency = VoroNN(**nn_kwargs)
            elif dependency == "dist" or dependency == "distance":
                self._dependency = DistanceNN(**nn_kwargs)
            else:
                raise ValueError('dependency {} if unknown, Possible values '
                                 'are {}'.format(dependency,
                                                 '[voro, voronoi, '
                                                 'dist, distance]'))
        else:
            raise ValueError('dependency {} if unknown, Possible values '
                             'are {} or voro/dist object.'.format(
                              dependency, '[voro, voronoi, dist, distance]'))
        self.remain_df = remain_df

    def fit(self, X=None):
        self._dependency = self.check_dependency(X)
        if self._dependency:
            self.atoms_df = self._dependency.fit_transform(self.atoms_df)
        return self

    @property
    def category(self):
        return 'sro'


class CN(BaseSRO):
    def __init__(self, atoms_df=None, dependency="voro", tmp_save=True,
                 context=None, remain_df=False, **nn_kwargs):
        """

        Args:
            dependency: (object or string) default: "voro"
                if object, it can be "VoroNN()" or "DistanceNN()",
                if string, it can be "voro" or "distance"
        """
        super(CN, self).__init__(tmp_save=tmp_save,
                                 context=context,
                                 dependency=dependency,
                                 atoms_df=atoms_df,
                                 remain_df=remain_df,
                                 **nn_kwargs)
        self.voro_depend_cols = ['n_neighbors_voro']
        self.dist_denpend_cols = ['n_neighbors_dist']

    def transform(self, X=None):
        X = check_featurizer_X(X=X, atoms_df=self.atoms_df)
        cn_list = np.zeros(len(X))
        neighbor_col = 'n_neighbors_{}'.format(self.dependency_name)
        cn_list = \
            voronoi_stats.cn_voro(cn_list, X[neighbor_col].values,
                                  n_atoms=len(X))
        cn_list_df = pd.DataFrame(cn_list,
                                  index=range(len(X)),
                                  columns=self.get_feature_names())

        cn_list_df = \
            remain_df_calc(remain_df=self.remain_df, result_df=cn_list_df,
                           source_df=X, n_neighbor_col=neighbor_col)

        if self.tmp_save:
            name = '{}_cn'.format(self.dependency_name)
            self.context.save_featurizer_as_dataframe(output_df=cn_list_df,
                                                      name=name)

        return cn_list_df

    def get_feature_names(self):
        feature_names = ['CN {}'.format(self.dependency_name)]
        return feature_names

    @property
    def double_dependency(self):
        return True


class VoroIndex(BaseSRO):
    def __init__(self, n_neighbor_limit=80,
                 include_beyond_edge_max=False,
                 atoms_df=None, dependency="voro",
                 tmp_save=True, context=None, remain_df=False,
                 edge_min=3, edge_max=8, **nn_kwargs):
        """

        Args:
            dependency: (object or string) default: "voro"
                if object, it can be "VoroNN()" or "DistanceNN()",
                if string, it can be "voro" or "distance"
        """
        super(VoroIndex, self).__init__(tmp_save=tmp_save,
                                        context=context,
                                        dependency=dependency,
                                        atoms_df=atoms_df,
                                        remain_df=remain_df,
                                        **nn_kwargs)
        self.n_neighbor_limit = n_neighbor_limit
        self.include_beyond_edge_max = include_beyond_edge_max
        self.edge_min = edge_min
        self.edge_max = edge_max
        self.voro_depend_cols = ['n_neighbors_voro'] + \
                                ['neighbor_edge_{}_voro'.format(edge)
                                 for edge in range(edge_min, edge_max + 1)]
        self.dist_denpend_cols = None

    def transform(self, X=None):
        X = check_featurizer_X(X=X, atoms_df=self.atoms_df)
        n_atoms = len(X)
        columns = X.columns
        edge_cols = [col for col in columns if col.startswith('neighbor_edge_')]
        edge_num = self.edge_max - self.edge_min + 1

        voronoi_index_list = np.zeros((n_atoms, edge_num))

        voro_index_list = \
            voronoi_stats.voronoi_index(voronoi_index_list,
                                        X['n_neighbors_voro'].values,
                                        X[edge_cols].values,
                                        self.edge_min, self.edge_max,
                                        self.include_beyond_edge_max,
                                        n_atoms=n_atoms,
                                        n_neighbor_limit=self.n_neighbor_limit)

        voro_index_df = pd.DataFrame(voro_index_list,
                                     index=range(n_atoms),
                                     columns=self.get_feature_names())
        voro_index_df = \
            remain_df_calc(remain_df=self.remain_df, result_df=voro_index_df,
                           source_df=X, n_neighbor_col='n_neighbors_voro')
        if self.tmp_save:
            self.context.save_featurizer_as_dataframe(output_df=voro_index_df,
                                                      name='voro_index')

        return voro_index_df

    def get_feature_names(self):
        feature_names = ['Voronoi idx{} voro'.format(edge)
                         for edge in range(self.edge_min,
                                           self.edge_max + 1)]
        return feature_names


class CharacterMotif(BaseSRO):
    def __init__(self, n_neighbor_limit=80,
                 include_beyond_edge_max=False,
                 atoms_df=None, dependency="voro",
                 edge_min=3, target_voro_idx=None, frank_kasper=1,
                 tmp_save=True, context=None, **nn_kwargs):
        super(CharacterMotif, self).__init__(tmp_save=tmp_save,
                                             context=context,
                                             dependency=dependency,
                                             atoms_df=atoms_df,
                                             **nn_kwargs)
        self.n_neighbor_limit = n_neighbor_limit
        self.include_beyond_edge_max = include_beyond_edge_max
        if target_voro_idx is None:
            self.target_voro_idx = np.array([[0, 0, 12, 0, 0],
                                             [0, 0, 12, 4, 0]],
                                            dtype=np.float128)
        self.frank_kasper = frank_kasper
        self.voro_depend_cols = ['n_neighbors_voro', 'neighbor_edge_5_voro']
        self.dist_denpend_cols = None
        self.edge_min = edge_min

    def fit(self, X=None):
        self._dependency = self.check_dependency(X)

        # This class is only dependent on 'Voronoi idx*' col, so if dataframe
        # has this col, this class don't need calculate it again.
        if self._dependency is None:
            self.voro_depend_cols = ['Voronoi idx5 voro']
            self._dependency = self.check_dependency(X)
            if self._dependency is None:
                return self

        self.atoms_df = self._dependency.fit_transform(self.atoms_df)
        return self

    def transform(self, X=None):
        X = check_featurizer_X(X=X, atoms_df=self.atoms_df)
        n_atoms = len(X)
        columns = X.columns
        if "Voronoi idx5 voro" not in columns:
            voro_index = \
                VoroIndex(n_neighbor_limit=self.n_neighbor_limit,
                          include_beyond_edge_max=self.include_beyond_edge_max,
                          atoms_df=X, dependency=self.dependency,
                          tmp_save=False, context=self.context)
            X = voro_index.fit_transform(X)

        columns = X.columns
        voro_index_cols = [col for col in columns
                           if col.startswith("Voronoi idx")]

        motif_one_hot = np.zeros((n_atoms,
                                  len(self.target_voro_idx) + self.frank_kasper))

        motif_one_hot = \
            voronoi_stats.character_motif(motif_one_hot,
                                          X[voro_index_cols].values,
                                          self.edge_min, self.target_voro_idx,
                                          self.frank_kasper, n_atoms=n_atoms)
        motif_one_hot_array = np.array(motif_one_hot)
        is_120_124 = motif_one_hot_array[:, 0] | motif_one_hot_array[:, 1]
        print(motif_one_hot_array.shape)
        print(is_120_124.shape)
        motif_one_hot_array = np.append(motif_one_hot_array,
                                        np.array([is_120_124]).T, axis=1)
        character_motif_df = pd.DataFrame(motif_one_hot_array,
                                          index=range(n_atoms),
                                          columns=self.get_feature_names())
        character_motif_df = \
            remain_df_calc(remain_df=self.remain_df,
                           result_df=character_motif_df,
                           source_df=X, n_neighbor_col='n_neighbors_voro')

        if self.tmp_save:
            self.context.save_featurizer_as_dataframe(
                output_df=character_motif_df, name='character_motif')

        return character_motif_df

    def get_feature_names(self):
        feature_names = ['is <0,0,12,0,0> voro', 'is <0,0,12,4,0> voro'] + \
                        ["_".join(map(str, v)) + " voro"
                         for v in self.target_voro_idx[2:]] + \
                        ['is polytetrahedral voro', 'is <0,0,12,0/4,0> voro']
        return feature_names


class IFoldSymmetry(BaseSRO):
    def __init__(self, n_neighbor_limit=80,
                 include_beyond_edge_max=False,
                 atoms_df=None, dependency="voro",
                 tmp_save=True, context=None, remain_df=False,
                 edge_min=3, edge_max=8, **nn_kwargs):
        super(IFoldSymmetry, self).__init__(tmp_save=tmp_save,
                                            context=context,
                                            dependency=dependency,
                                            atoms_df=atoms_df,
                                            remain_df=remain_df,
                                            **nn_kwargs)
        self.n_neighbor_limit = n_neighbor_limit
        self.include_beyond_edge_max = include_beyond_edge_max
        self.edge_min = edge_min
        self.edge_max = edge_max
        self.voro_depend_cols = ['n_neighbors_voro'] + \
                                ['neighbor_edge_{}_voro'.format(edge)
                                 for edge in range(edge_min, edge_max + 1)]
        self.dist_denpend_cols = None

    def transform(self, X=None):
        X = check_featurizer_X(X=X, atoms_df=self.atoms_df)
        n_atoms = len(X)
        columns = X.columns
        edge_cols = [col for col in columns if col.startswith('neighbor_edge_')]
        edge_num = self.edge_max - self.edge_min + 1
        i_symm_list = np.zeros((n_atoms, edge_num))

        i_symm_list = \
            voronoi_stats.i_fold_symmetry(i_symm_list,
                                          X['n_neighbors_voro'].values,
                                          X[edge_cols].values,
                                          self.edge_min, self.edge_max,
                                          self.include_beyond_edge_max,
                                          n_atoms=n_atoms,
                                          n_neighbor_limit=
                                          self.n_neighbor_limit)

        i_symm_df = pd.DataFrame(i_symm_list,
                                 index=range(n_atoms),
                                 columns=self.get_feature_names())
        i_symm_df = \
            remain_df_calc(remain_df=self.remain_df, result_df=i_symm_df,
                           source_df=X, n_neighbor_col='n_neighbors_voro')
        if self.tmp_save:
            self.context.save_featurizer_as_dataframe(output_df=i_symm_df,
                                                      name='i_fold_symmetry')

        return i_symm_df

    def get_feature_names(self):
        feature_names = ['{}-fold symm idx voro'.format(edge)
                         for edge in range(self.edge_min, self.edge_max+1)]
        return feature_names


class AreaWtIFoldSymmetry(BaseSRO):
    def __init__(self, n_neighbor_limit=80,
                 include_beyond_edge_max=False,
                 atoms_df=None, dependency="voro",
                 tmp_save=True, context=None, remain_df=False,
                 edge_min=3, edge_max=8, **nn_kwargs):
        super(AreaWtIFoldSymmetry, self).__init__(tmp_save=tmp_save,
                                                  context=context,
                                                  dependency=dependency,
                                                  atoms_df=atoms_df,
                                                  remain_df=remain_df,
                                                  **nn_kwargs)
        self.n_neighbor_limit = n_neighbor_limit
        self.include_beyond_edge_max = include_beyond_edge_max
        self.edge_min = edge_min
        self.edge_max = edge_max
        self.voro_depend_cols = ['n_neighbors_voro'] + \
                                ['neighbor_edge_{}_voro'.format(edge)
                                 for edge in range(edge_min, edge_max + 1)] + \
                                ['neighbor_area_{}_voro'.format(edge)
                                 for edge in range(edge_min, edge_max + 1)]
        self.dist_denpend_cols = None

    def transform(self, X=None):
        X = check_featurizer_X(X=X, atoms_df=self.atoms_df)
        n_atoms = len(X)
        columns = X.columns
        edge_cols = [col for col in columns if
                     col.startswith('neighbor_edge_')]
        area_cols = [col for col in columns if
                     col.startswith('neighbor_area_')]
        edge_num = self.edge_max - self.edge_min + 1
        area_wt_i_symm_list = np.zeros((n_atoms, edge_num))

        area_wt_i_symm_list = \
            voronoi_stats.area_wt_i_fold_symmetry(area_wt_i_symm_list,
                                                  X['n_neighbors_voro'].values,
                                                  X[edge_cols].values,
                                                  X[area_cols].values.astype(
                                                      np.float128),
                                                  self.edge_min,
                                                  self.edge_max,
                                                  self.include_beyond_edge_max,
                                                  n_atoms=n_atoms,
                                                  n_neighbor_limit=
                                                  self.n_neighbor_limit)

        area_wt_i_symm_df = pd.DataFrame(area_wt_i_symm_list,
                                         index=range(n_atoms),
                                         columns=self.get_feature_names())
        area_wt_i_symm_df = \
            remain_df_calc(remain_df=self.remain_df, source_df=X,
                           result_df=area_wt_i_symm_df,
                           n_neighbor_col='n_neighbors_voro')
        if self.tmp_save:
            self.context.save_featurizer_as_dataframe(
                output_df=area_wt_i_symm_df, name='area_wt_i_fold_symmetry')

        return area_wt_i_symm_df

    def get_feature_names(self):
        feature_names = ['Area_wt {}-fold symm idx voro'.format(edge)
                         for edge in range(self.edge_min, self.edge_max + 1)]
        return feature_names


class VolWtIFoldSymmetry(BaseSRO):
    def __init__(self, n_neighbor_limit=80,
                 include_beyond_edge_max=False,
                 atoms_df=None, dependency="voro",
                 tmp_save=True, context=None, remain_df=False,
                 edge_min=3, edge_max=8, **nn_kwargs):
        super(VolWtIFoldSymmetry, self).__init__(tmp_save=tmp_save,
                                            context=context,
                                            dependency=dependency,
                                            atoms_df=atoms_df,
                                            remain_df=remain_df,
                                            **nn_kwargs)
        self.n_neighbor_limit = n_neighbor_limit
        self.include_beyond_edge_max = include_beyond_edge_max
        self.edge_min = edge_min
        self.edge_max = edge_max
        self.voro_depend_cols = ['n_neighbors_voro'] + \
                                ['neighbor_edge_{}_voro'.format(edge)
                                 for edge in range(edge_min, edge_max + 1)] + \
                                ['neighbor_vol_{}_voro'.format(edge)
                                 for edge in range(edge_min, edge_max + 1)]
        self.dist_denpend_cols = None

    def transform(self, X=None):
        X = check_featurizer_X(X=X, atoms_df=self.atoms_df)
        n_atoms = len(X)
        columns = X.columns
        edge_cols = [col for col in columns if
                     col.startswith('neighbor_edge_')]
        vol_cols = [col for col in columns if
                     col.startswith('neighbor_vol_')]
        edge_num = self.edge_max - self.edge_min + 1
        vol_wt_i_symm_list = np.zeros((n_atoms, edge_num))
        vol_wt_i_symm_list = \
            voronoi_stats.vol_wt_i_fold_symmetry(vol_wt_i_symm_list,
                                                 X['n_neighbors_voro'].values,
                                                 X[edge_cols].values,
                                                 X[vol_cols].values.astype(
                                                     np.float128),
                                                 self.edge_min,
                                                 self.edge_max,
                                                 self.include_beyond_edge_max,
                                                 n_atoms=n_atoms,
                                                 n_neighbor_limit=
                                                 self.n_neighbor_limit)

        vol_wt_i_symm_df = pd.DataFrame(vol_wt_i_symm_list,
                                         index=range(n_atoms),
                                         columns=self.get_feature_names())
        vol_wt_i_symm_df = \
            remain_df_calc(remain_df=self.remain_df, source_df=X,
                           result_df=vol_wt_i_symm_df,
                           n_neighbor_col='n_neighbors_voro')
        if self.tmp_save:
            self.context.save_featurizer_as_dataframe(
                output_df=vol_wt_i_symm_df, name='vol_wt_i_fold_symmetry')

        return vol_wt_i_symm_df

    def get_feature_names(self):
        feature_names = ['Vol_wt {}-fold symm idx voro'.format(edge)
                         for edge in range(self.edge_min, self.edge_max + 1)]
        return feature_names


class VoroAreaStats(BaseSRO):
    def __init__(self, n_neighbor_limit=80,
                 atoms_df=None, dependency="voro",
                 tmp_save=True, context=None, remain_df=False, **nn_kwargs):
        super(VoroAreaStats, self).__init__(tmp_save=tmp_save,
                                            context=context,
                                            dependency=dependency,
                                            atoms_df=atoms_df,
                                            remain_df=remain_df,
                                            **nn_kwargs)
        self.n_neighbor_limit = n_neighbor_limit
        self.voro_depend_cols = ['n_neighbors_voro'] + \
                                ['neighbor_area_5_voro']
        self.stats = ['mean', 'std', 'min', 'max']
        self.dist_denpend_cols = None

    def transform(self, X=None):
        X = check_featurizer_X(X=X, atoms_df=self.atoms_df)
        n_atoms = len(X)
        columns = X.columns
        area_cols = [col for col in columns if
                     col.startswith('neighbor_area_')]
        area_stats = np.zeros((n_atoms, len(self.stats) + 1))

        area_stats = \
            voronoi_stats.voronoi_area_stats(area_stats,
                                             X['n_neighbors_voro'].values,
                                             X[area_cols].values.astype(
                                                 np.float128),
                                             n_atoms=n_atoms,
                                             n_neighbor_limit=
                                             self.n_neighbor_limit)

        area_stats_df = pd.DataFrame(area_stats, index=range(n_atoms),
                                     columns=self.get_feature_names())
        area_stats_df = \
            remain_df_calc(remain_df=self.remain_df, source_df=X,
                           result_df=area_stats_df,
                           n_neighbor_col='n_neighbors_voro')
        if self.tmp_save:
            self.context.save_featurizer_as_dataframe(
                output_df=area_stats_df, name='voronoi_area_stats')

        return area_stats_df

    def get_feature_names(self):
        feature_names = ['Voronoi area voro'] + \
                        ['Facet area {} voro'.format(stat)
                         for stat in self.stats]
        return feature_names


class VoroAreaStatsSeparate(BaseSRO):
    def __init__(self, n_neighbor_limit=80, include_beyond_edge_max=False,
                 atoms_df=None, dependency="voro", edge_min=3, edge_max=8,
                 tmp_save=True, context=None, remain_df=False, **nn_kwargs):
        super(VoroAreaStatsSeparate, self).__init__(tmp_save=tmp_save,
                                                    context=context,
                                                    dependency=dependency,
                                                    atoms_df=atoms_df,
                                                    remain_df=remain_df,
                                                    **nn_kwargs)
        self.n_neighbor_limit = n_neighbor_limit
        self.voro_depend_cols = ['n_neighbors_voro'] + \
                                ['neighbor_edge_{}_voro'.format(edge)
                                 for edge in range(edge_min, edge_max + 1)] + \
                                ['neighbor_area_{}_voro'.format(edge)
                                 for edge in range(edge_min, edge_max + 1)]

        self.edge_min = edge_min
        self.edge_max = edge_max
        self.edge_num = edge_max - edge_min + 1
        self.include_beyond_edge_max = include_beyond_edge_max
        self.stats = ['sum', 'mean', 'std', 'min', 'max']
        self.dist_denpend_cols = None

    def transform(self, X=None):
        X = check_featurizer_X(X=X, atoms_df=self.atoms_df)
        n_atoms = len(X)
        columns = X.columns
        edge_cols = [col for col in columns if
                     col.startswith('neighbor_edge_')]
        area_cols = [col for col in columns if
                     col.startswith('neighbor_area_')]
        area_stats_separate = np.zeros((n_atoms,
                                        self.edge_num * len(self.stats)))

        area_stats_separate = \
            voronoi_stats.voronoi_area_stats_separate(
                area_stats_separate, X['n_neighbors_voro'].values,
                X[edge_cols].values, X[area_cols].values.astype(np.float128),
                self.edge_min, self.edge_max,
                self.include_beyond_edge_max,
                n_atoms=n_atoms,
                n_neighbor_limit=self.n_neighbor_limit)

        area_stats_separate_df = pd.DataFrame(area_stats_separate, index=range(n_atoms),
                                     columns=self.get_feature_names())
        area_stats_separate_df = \
            remain_df_calc(remain_df=self.remain_df, source_df=X,
                           result_df=area_stats_separate_df,
                           n_neighbor_col='n_neighbors_voro')
        if self.tmp_save:
            self.context.save_featurizer_as_dataframe(
                output_df=area_stats_separate_df,
                name='voro_area_stats_separate')

        return area_stats_separate_df

    def get_feature_names(self):
        feature_names = ['{}-edged area {} voro'.format(edge, stat)
                         for edge in range(self.edge_min, self.edge_max + 1)
                         for stat in self.stats]
        return feature_names


class VoroVolStats(BaseSRO):
    def __init__(self, n_neighbor_limit=80,
                 atoms_df=None, dependency="voro",
                 tmp_save=True, context=None, remain_df=False, **nn_kwargs):
        super(VoroVolStats, self).__init__(tmp_save=tmp_save,
                                            context=context,
                                            dependency=dependency,
                                            atoms_df=atoms_df,
                                            remain_df=remain_df,
                                            **nn_kwargs)
        self.n_neighbor_limit = n_neighbor_limit
        self.voro_depend_cols = ['n_neighbors_voro'] + \
                                ['neighbor_vol_5_voro']
        self.stats = ['mean', 'std', 'min', 'max']
        self.dist_denpend_cols = None

    def transform(self, X=None):
        X = check_featurizer_X(X=X, atoms_df=self.atoms_df)
        n_atoms = len(X)
        columns = X.columns
        vol_cols = [col for col in columns if
                     col.startswith('neighbor_vol_')]
        vol_stats = np.zeros((n_atoms, len(self.stats) + 1))

        vol_stats = \
            voronoi_stats.voronoi_vol_stats(vol_stats,
                                            X['n_neighbors_voro'].values,
                                            X[vol_cols].values.astype(
                                                np.float128),
                                            n_atoms=n_atoms,
                                            n_neighbor_limit=
                                            self.n_neighbor_limit)

        vol_stats_df = pd.DataFrame(vol_stats, index=range(n_atoms),
                                    columns=self.get_feature_names())
        vol_stats_df = \
            remain_df_calc(remain_df=self.remain_df, source_df=X,
                           result_df=vol_stats_df,
                           n_neighbor_col='n_neighbors_voro')
        if self.tmp_save:
            self.context.save_featurizer_as_dataframe(
                output_df=vol_stats_df, name='voronoi_vol_stats')

        return vol_stats_df

    def get_feature_names(self):
        feature_names = ['Voronoi vol voro'] + \
                        ['Sub-polyhedra vol {} voro'.format(stat)
                         for stat in self.stats]
        return feature_names


class VoroVolStatsSeparate(BaseSRO):
    def __init__(self, n_neighbor_limit=80, include_beyond_edge_max=False,
                 atoms_df=None, dependency="voro", edge_min=3, edge_max=8,
                 tmp_save=True, context=None, remain_df=False, **nn_kwargs):
        super(VoroVolStatsSeparate, self).__init__(tmp_save=tmp_save,
                                            context=context,
                                            dependency=dependency,
                                            atoms_df=atoms_df,
                                            remain_df=remain_df,
                                            **nn_kwargs)
        self.n_neighbor_limit = n_neighbor_limit
        self.voro_depend_cols = ['n_neighbors_voro'] + \
                                ['neighbor_edge_{}_voro'.format(edge)
                                 for edge in range(edge_min, edge_max + 1)] + \
                                ['neighbor_vol_{}_voro'.format(edge)
                                 for edge in range(edge_min, edge_max + 1)]

        self.edge_min = edge_min
        self.edge_max = edge_max
        self.edge_num = edge_max - edge_min + 1
        self.include_beyond_edge_max = include_beyond_edge_max
        self.stats = ['sum', 'mean', 'std', 'min', 'max']
        self.dist_denpend_cols = None

    def transform(self, X=None):
        X = check_featurizer_X(X=X, atoms_df=self.atoms_df)
        n_atoms = len(X)
        columns = X.columns
        edge_cols = [col for col in columns if col.startswith('neighbor_edge_')]
        vol_cols = [col for col in columns if col.startswith('neighbor_vol_')]
        vol_stats_separate = np.zeros((n_atoms,
                                       self.edge_num * len(self.stats)))

        vol_stats_separate = \
            voronoi_stats.voronoi_vol_stats_separate(
                vol_stats_separate, X['n_neighbors_voro'].values,
                X[edge_cols].values, X[vol_cols].values.astype(np.float128),
                self.edge_min, self.edge_max,
                self.include_beyond_edge_max,
                n_atoms=n_atoms,
                n_neighbor_limit=self.n_neighbor_limit)

        vol_stats_separate_df = pd.DataFrame(vol_stats_separate,
                                             index=range(n_atoms),
                                             columns=self.get_feature_names())
        vol_stats_separate_df = \
            remain_df_calc(remain_df=self.remain_df, source_df=X,
                           result_df=vol_stats_separate_df,
                           n_neighbor_col='n_neighbors_voro')
        if self.tmp_save:
            self.context.save_featurizer_as_dataframe(
                output_df=vol_stats_separate_df, name='voro_vol_stats_separate')

        return vol_stats_separate_df

    def get_feature_names(self):
        feature_names = ['{}-edged vol {} voro'.format(edge, stat)
                         for edge in range(self.edge_min, self.edge_max + 1)
                         for stat in self.stats]
        return feature_names


class DistStats(BaseSRO):
    def __init__(self, n_neighbor_limit=80,
                 atoms_df=None, dependency="voro",
                 tmp_save=True, context=None, remain_df=False, **nn_kwargs):
        super(DistStats, self).__init__(tmp_save=tmp_save,
                                        context=context,
                                        dependency=dependency,
                                        atoms_df=atoms_df,
                                        remain_df=remain_df,
                                        **nn_kwargs)
        self.n_neighbor_limit = n_neighbor_limit
        self.voro_depend_cols = ['n_neighbors_voro'] + \
                                ['neighbor_distance_5_voro']
        self.stats = ['sum', 'mean', 'std', 'min', 'max']
        self.dist_denpend_cols = ['n_neighbors_dist'] + \
                                 ['neighbor_distance_5_dist']

    def transform(self, X=None):
        X = check_featurizer_X(X=X, atoms_df=self.atoms_df)
        n_atoms = len(X)
        columns = X.columns
        dist_cols = [col for col in columns if
                     col.startswith('neighbor_distance_')]
        dist_stats = np.zeros((n_atoms, len(self.stats) + 1))

        dist_stats = \
            voronoi_stats.voronoi_area_stats(dist_stats,
                                             X['n_neighbors_{}'.format(
                                                 self.dependency_name)].values,
                                             X[dist_cols].values,
                                             n_atoms=n_atoms,
                                             n_neighbor_limit=
                                             self.n_neighbor_limit)

        dist_stats_df = pd.DataFrame(dist_stats, index=range(n_atoms),
                                    columns=self.get_feature_names())
        dist_stats_df = \
            remain_df_calc(remain_df=self.remain_df, source_df=X,
                           result_df=dist_stats_df,
                           n_neighbor_col='n_neighbors_{}'.format(
                               self.dependency_name))
        if self.tmp_save:
            self.context.save_featurizer_as_dataframe(
                output_df=dist_stats_df,
                name='{}_distance_stats'.format(self.dependency_name))

        return dist_stats_df

    def get_feature_names(self):
        feature_names = ['Distance {} {}'.format(stat, self.dependency_name)
                         for stat in self.stats]
        return feature_names

    @property
    def double_dependency(self):
        return False


class BOOP(BaseSRO):
    def __init__(self, coords_path=None, atom_coords=None, Bds=None, pbc=None,
                 low_order=1, higher_order=1, coarse_lower_order=1,
                 coarse_higher_order=1, n_neighbor_limit=80, atoms_df=None,
                 dependency="voro", tmp_save=True, context=None,
                 remain_df=False, **nn_kwargs):
        super(BOOP, self).__init__(tmp_save=tmp_save,
                                   context=context,
                                   dependency=dependency,
                                   atoms_df=atoms_df,
                                   remain_df=remain_df,
                                   **nn_kwargs)
        self.low_order = low_order
        self.higher_order = higher_order
        self.coarse_lower_order = coarse_lower_order
        self.coarse_higher_order = coarse_higher_order
        if coords_path is not None and os.path.exists(coords_path):
            _, _, self.atom_coords, self.Bds = read_imd(coords_path)
        else:
            self.atom_coords = atom_coords
            self.Bds = Bds
        if self.atom_coords is None or self.Bds is None:
            raise ValueError("Please make sure atom_coords and Bds are not None"
                             " or coords_path is not None")
        self.pbc = pbc if pbc else [1, 1, 1]
        self.n_neighbor_limit = n_neighbor_limit
        self.voro_depend_cols = ['n_neighbors_voro'] + \
                                ['neighbor_id_{}_voro'.format(idx)
                                 for idx in range(n_neighbor_limit)]
        self.dist_denpend_cols = ['n_neighbors_dist'] + \
                                 ['neighbor_id_{}_dist'.format(idx)
                                  for idx in range(n_neighbor_limit)]
        self.bq_tags = ['4', '6', '8', '10']

    def fit(self, X=None):
        self._dependency = self.check_dependency(X)
        if self._dependency:
            self.atoms_df = self._dependency.fit_transform(self.atoms_df)
        return self

    def transform(self, X=None):
        X = check_featurizer_X(X=X, atoms_df=self.atoms_df)
        n_atoms = len(X)
        neighbor_col = ['n_neighbors_{}'.format(self.dependency_name)]
        id_cols = ['neighbor_id_{}_{}'.format(idx, self.dependency_name)
                   for idx in range(self.n_neighbor_limit)]

        Ql = np.zeros((n_atoms, 4))
        Wlbar = np.zeros((n_atoms, 4))
        coarse_Ql = np.zeros((n_atoms, 4))
        coarse_Wlbar = np.zeros((n_atoms, 4))
        Ql, Wlbar, coarse_Ql, coarse_Wlbar = \
            boop.calculate_boop(
                self.atom_coords,
                self.pbc, self.Bds, X[neighbor_col].values, X[id_cols].values,
                self.low_order, self.higher_order, self.coarse_lower_order,
                self.coarse_higher_order, Ql, Wlbar, coarse_Ql, coarse_Wlbar,
                n_atoms=n_atoms, n_neighbor_limit=self.n_neighbor_limit)
        concat_array = np.append(Ql, Wlbar, axis=1)
        concat_array = np.append(concat_array, coarse_Ql, axis=1)
        concat_array = np.append(concat_array, coarse_Wlbar, axis=1)

        boop_df = pd.DataFrame(concat_array, index=range(n_atoms),
                               columns=self.get_feature_names())
        boop_df = \
            remain_df_calc(remain_df=self.remain_df, source_df=X,
                           result_df=boop_df,
                           n_neighbor_col=
                           'n_neighbors_{}'.format(self.dependency_name))
        if self.tmp_save:
            self.context.save_featurizer_as_dataframe(
                output_df=boop_df, name='boop_{}'.format(self.dependency_name))

        return boop_df

    def get_feature_names(self):
        feature_names = ['q_{} {}'.format(num, self.dependency_name)
                         for num in self.bq_tags] + \
                        ['w_{} {}'.format(num, self.dependency_name)
                         for num in self.bq_tags] + \
                        ['Coarse-grained q_{} {}'.format(num,
                                                         self.dependency_name)
                         for num in self.bq_tags] + \
                        ['Coarse-grained w_{} {}'.format(num,
                                                         self.dependency_name)
                         for num in self.bq_tags]
        return feature_names

    @property
    def double_dependency(self):
        return False
