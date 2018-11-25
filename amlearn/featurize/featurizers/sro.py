import numpy as np
import pandas as pd
from amlearn.featurize.featurizers.base import BaseFeaturize
from amlearn.featurize.featurizers.voro_and_distance import VoroNN, DistanceNN
from amlearn.utils.check import check_featurizer_X

try:
    from amlearn.featurize.featurizers.sro_mro import voronoi_stats, boop
except Exception:
    print("import fortran file voronoi_stats error!\n")


class BaseSro(BaseFeaturize):
    def __init__(self, atoms_df=None, tmp_save=True, context=None,
                 dependency=None, nn_kwargs=None):
        super(self.__class__, self).__init__(tmp_save=tmp_save,
                                             context=context,
                                             atoms_df=atoms_df)
        nn_kwargs = nn_kwargs if nn_kwargs else dict()
        if dependency is None:
            self._dependency = None
        elif isinstance(dependency, type):
            self.dependency_name = dependency.__class__.__name__.lower()[:-2]
            self._dependency = dependency
        elif isinstance(dependency, str):
            self.dependency_name = dependency
            if dependency == "voro" or dependency == "voronoi" :
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

    def get_feature_names(self):
        pass


class CNVoro(BaseSro):
    def __init__(self, atoms_df=None, dependency="voro", nn_kwargs=None,
                 tmp_save=True, context=None):
        """

        Args:
            dependency: (object or string) default: "voro"
                if object, it can be "VoroNN()" or "DistanceNN()",
                if string, it can be "voro" or "distance"
        """
        super(self.__class__, self).__init__(tmp_save=tmp_save,
                                             context=context,
                                             dependency=dependency,
                                             nn_kwargs=nn_kwargs,
                                             atoms_df=atoms_df)

    def fit(self, X=None):
        X = check_featurizer_X(X=X, atoms_df=self.atoms_df)
        columns = X.columns
        if self._dependency.__class__.__name__ == "VoroNN" and \
                        'n_neighbors_voro' in columns:
            return self
        elif self._dependency.__class__.__name__ == "DistanceNN" and \
                        'n_neighbors_dist' in columns:
            return self
        else:
            self.atoms_df = self._dependency.fit_transform(X)
            return self

    def transform(self, X=None):
        X = check_featurizer_X(X=X, atoms_df=self.atoms_df)
        cn_list = np.zeros(len(X))

        n_neighbor_col = 'n_neighbors_dist' \
            if self._dependency.__class__.__name__ == "DistanceNN" \
            else 'n_neighbors_voro'
        cn_list = \
            voronoi_stats.cn_voro(cn_list, X[n_neighbor_col].values(),
                                  n_atoms=len(X))
        cn_list_df = pd.DataFrame(cn_list,
                                  index=range(len(X)),
                                  columns=self.get_feature_names())

        if self.tmp_save:
            self.context.save_featurizer_as_dataframe(output_df=cn_list_df,
                                                      name='cn_voro')

        return cn_list_df

    def get_feature_names(self):
        feature_names = ['CN_Dist'] \
            if self.dependency_name == 'dist' \
               or self.dependency_name == 'distance' \
            else ['CN_Voro']
        return feature_names


class VoroIndex(BaseSro):
    def __init__(self, n_neighbor_limit, include_beyond_edge_max=False,
                 atoms_df=None, dependency="voro", nn_kwargs=None,
                 tmp_save=True, context=None):
        """

        Args:
            dependency: (object or string) default: "voro"
                if object, it can be "VoroNN()" or "DistanceNN()",
                if string, it can be "voro" or "distance"
        """
        super(self.__class__, self).__init__(tmp_save=tmp_save,
                                             context=context,
                                             dependency=dependency,
                                             nn_kwargs=nn_kwargs,
                                             atoms_df=atoms_df)
        self.n_neighbor_limit = n_neighbor_limit
        self.include_beyond_edge_max = include_beyond_edge_max

    def fit(self, X=None):
        X = check_featurizer_X(X=X, atoms_df=self.atoms_df)
        columns = X.columns

        if self._dependency.__class__.__name__ == "VoroNN" and \
                        'n_neighbors_voro' in columns:
            return self
        elif self._dependency.__class__.__name__ == "DistanceNN" and \
                        'n_neighbors_dist' in columns:
            return self
        else:
            self.atoms_df = self._dependency.fit_transform(X)
            return self

    def transform(self, X=None):
        X = check_featurizer_X(X=X, atoms_df=self.atoms_df)
        n_atoms = len(X)
        columns = X.columns
        edge_cols = [col for col in columns if col.startswith('neighbor_edge_')]
        edge_list = [int(col.split('_')[-1]) for col in edge_cols]
        edge_num = len(edge_list)
        self.edge_min = min(edge_list)
        self.edge_max = max(edge_list)

        voronoi_index_list = np.zeros((n_atoms, edge_num))

        print(voronoi_stats.voronoi_index.__doc__)
        voro_index_list = \
            voronoi_stats.voronoi_index(voronoi_index_list,
                                        X['n_neighbors_voro'].values(),
                                        X[edge_cols].values(),
                                        self.edge_min, self.edge_max,
                                        self.include_beyond_edge_max,
                                        n_atoms=n_atoms,
                                        n_neighbor_limit=self.n_neighbor_limit)

        voro_index_df = pd.DataFrame(voro_index_list,
                                     index=range(n_atoms),
                                     columns=self.get_feature_names())

        if self.tmp_save:
            self.context.save_featurizer_as_dataframe(output_df=voro_index_df,
                                                      name='cn_voro')

        return voro_index_df

    def get_feature_names(self):
        feature_names = ['Voronoi idx_{}'.format(edge)
                         for edge in range(self.edge_min,
                                           self.edge_max + 1)]
        return feature_names


