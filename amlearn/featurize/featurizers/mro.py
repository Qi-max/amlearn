import os
import numpy as np
import pandas as pd
from amlearn.featurize.featurizers.base import BaseFeaturize, line_percent
from amlearn.utils.check import check_featurizer_X, check_neighbor_col
from amlearn.utils.directory import create_path

try:
    from amlearn.featurize.featurizers.sro_mro import mro_stats, voronoi_stats
except Exception:
    print("import fortran file voronoi_stats error!")


class MRO(BaseFeaturize):
    def __init__(self, n_neighbor_limit=80, write_path="default",
                 stats_types="all", stats_names=None,
                 calc_features="all", neighbor_cols="all",
                 calc_neighbor_cols=False,
                 atoms_df=None, tmp_save=True, context=None):
        """

        Args:
            dependency: (object or string) default: "voro"
                if object, it can be "VoroNN()" or "DistanceNN()",
                if string, it can be "voro" or "distance"

            write_path: (string) default: "default"
                if None, the transform method don't save dataframe,
                         just return it.
                if string, the transform method save dataframe
                           to the write_path and return it.
                    if "default": set write_path as self.context.output_path.

            calced_sysmm: (boolean)
                default: False and it changes in transform method
                if calculated the Voronoi stats, then set the flag to True, and
                then calculate the Avg i-fold symm idx, this flag talls
                get_feature_names() function that Avg i-fold symm idx
                was calculated.
        """
        super(MRO, self).__init__(tmp_save=tmp_save,
                                  context=context,
                                  atoms_df=atoms_df)
        self.stats_types = stats_types if stats_types != "all" \
            else [1, 1, 1, 1, 1, 1]
        self.n_neighbor_limit = n_neighbor_limit
        self.neighbor_cols = check_neighbor_col(neighbor_cols)
        self.calc_features = calc_features
        self.calc_neighbor_cols = calc_neighbor_cols
        self.stats_names = ['sum_NN', 'mean_NN', 'std_NN',
                            'min_NN', 'max_NN', 'diff_NN']
        self.write_path = self.context.output_path if write_path == "default" \
            else write_path
        self.calced_sysmm = False

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def transform(self, X=None):
        X = check_featurizer_X(X=X, atoms_df=self.atoms_df)

        self.calc_features = list(X.columns) if self.calc_features == "all" \
            else self.calc_features
        if not self.calc_neighbor_cols:
            self.calc_features = [col for col in self.calc_features
                                  if not col.startswith('n_neighbors_')
                                  and not col.startswith('neighbor_id_')]
        if not set(self.calc_features).issubset(set(list(X.columns))):
            raise ValueError("calc_features {} is unkown. "
                             "Possible values are: {}".format(self.calc_features,
                                                              list(X.columns)))

        self.calced_neighbor_cols = list()
        concat_features = None
        n_atoms = len(X)

        for neighbor_col in self.neighbor_cols:
            if neighbor_col not in list(X.columns):
                self.context.logger.warning(
                    "neighbor_col {} is not in atoms_df. So ignore this "
                    "neighbor_col and continue to next neighbor_col "
                    "calculate".format(neighbor_col))
                continue
            self.calced_neighbor_cols.append(neighbor_col)

            n_neighbor_list = X[neighbor_col].values
            neighbor_tag = neighbor_col.split('_')[-1]
            neighbor_lists = \
                X[['neighbor_id_{}_{}'.format(num, neighbor_tag)
                   for num in range(self.n_neighbor_limit)]].values

            for feature in self.calc_features:
                if neighbor_tag not in feature:
                    continue
                mro_feature = np.zeros((n_atoms, sum(self.stats_types)))
                mro_feature = \
                    mro_stats.sro_to_mro(X[feature].values,
                                         n_neighbor_list, neighbor_lists,
                                         self.stats_types, mro_feature,
                                         n_atoms=n_atoms,
                                         n_neighbor_limit=self.n_neighbor_limit)
                concat_features = mro_feature if concat_features is None \
                    else np.append(concat_features, mro_feature, axis=1)

        result_df = pd.DataFrame(concat_features, index=range(n_atoms),
                                 columns=self.get_common_features())

        voro_mean_cols = [col for col in result_df.columns
                          if col.startswith('Voronoi idx_')
                          and col.endswith(' mean_NN')]

        if voro_mean_cols:
            self.calced_sysmm = True
            self.idx_list = [col.split('_')[-1].split(' ')[0]
                             for col in voro_mean_cols]
            percent_list = \
                line_percent(value_list=result_df[voro_mean_cols].values)
            percent_df = pd.DataFrame(percent_list, index=range(n_atoms),
                                      columns=self.get_symm_percent_features())
            result_df = result_df.join(percent_df)

        if self.tmp_save:
            self.context.save_featurizer_as_dataframe(output_df=result_df,
                                                      name='mro')
        if self.write_path:
            create_path(self.write_path, merge=True)
            featurizer_file = os.path.join(self.write_path,
                                           'featurizer_mro.csv')
            result_df.to_csv(featurizer_file)

        return result_df

    def get_common_features(self):
        feature_names = list()
        for neighbor_col in self.calced_neighbor_cols:
            neighbor_tag = neighbor_col.split('_')[-1]
            feature_names += \
                ["{} {}".format(feature, stats_name)
                 for feature in self.calc_features if neighbor_tag in feature
                 for stats_name, stats_type in zip(self.stats_names,
                                                   self.stats_types)
                 if stats_type == 1]
        return feature_names

    def get_symm_percent_features(self):
        feature_names = ['Avg {}-fold symm idx'.format(edge)
                         for edge in self.idx_list]
        return feature_names

    def get_feature_names(self):
        feature_names = self.get_common_features()
        if self.calced_sysmm:
            feature_names += self.get_symm_percent_features()
        return feature_names

    @property
    def category(self):
        return 'mro'
