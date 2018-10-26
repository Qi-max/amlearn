import os
import copy
import pandas as pd
import numpy as np
from amlearn.utils.check import check_output_path

try:
    from amlearn.amlearn.featurize.featurizers.sro_mro import \
        mro_stats, voronoi_stats
except Exception:
    print("import fortran file voronoi_stats error!")


class MROFeaturizers():
    def __init__(self, df, neighbor_df, n_neighbor_limit,
                 feature_list="all", feature_sets="all", fortran=True):
        """

        Args:
            feature_sets: "Voronoi", "BOOP_lower_voro", "BOOP_higher_voro",
                          or "all" (a total of 18 candidate feature sets)
        """
        self.df = df
        self.neighbor_df = neighbor_df
        self.n_neighbor_limit = n_neighbor_limit
        self.feature_list = feature_list
        return

    @classmethod
    def from_file(cls, data_path_file, neighbor_path_file, n_neighbor_limit,
                  feature_list="all", feature_sets="all", fortran=True):
        df = pd.read_csv(data_path_file, index_col=0)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        neighbor_df = pd.read_csv(neighbor_path_file, index_col=0)
        return cls(df, n_neighbor_limit=n_neighbor_limit,
                   feature_list=feature_list, feature_sets=feature_sets,
                   neighbor_df=neighbor_df, fortran=fortran)

    def sro_to_mro(self, stats_types, stats_names=None, write_path=None):
        concat_features = None
        n_atoms = len(self.df)
        n_neighbor_list = self.neighbor_df[['n_neighbors']].values
        neighbor_lists = \
            self.neighbor_df[['neighbor_id_{}'.format(num)
                              for num in range(self.n_neighbor_limit)]].values

        print(mro_stats.sro_to_mro.__doc__)
        feature_list = list(self.df.columns) \
            if self.feature_list == "all" else self.feature_list

        stats_names = stats_names if stats_names is not None else range(sum(stats_types))
        feature_names = ["{} {}".format(feature, stats_name)
                         for feature in feature_list
                         for stats_name in stats_names]
        for feature in feature_list:
            mro_feature = np.zeros((n_atoms, sum(stats_types)))
            mro_feature = \
                mro_stats.sro_to_mro(self.df[feature].values,
                                     n_neighbor_list, neighbor_lists,
                                     stats_types, mro_feature, n_atoms=n_atoms,
                                     n_neighbor_limit=self.n_neighbor_limit)
            concat_features = mro_feature if concat_features is None \
                else np.append(concat_features, mro_feature, axis=1)
        result_df = pd.DataFrame(concat_features, index=range(n_atoms),
                                 columns=feature_names)
        if write_path is not None:
            check_output_path(write_path)
            result_df.to_csv(os.path.join(write_path, 'sro_to_mro.csv'))
        return result_df

    def csro(self, type_col, raw_comp, type_names=None,
             tag_suffix='Voro', write_path=None):
        type_names = ["Atom{}".format(num+1) for num in range(len(raw_comp))] \
            if type_names is None else type_names
        if len(type_names) != len(raw_comp):
            raise ValueError('Length of type_names must equal length of raw_comp.')
        type_set = set(list(self.df[type_col]))
        print(type_set)
        if len(type_set) != len(raw_comp):
            raise ValueError("Length of raw_comp must equal to length of raw_comp.")
        n_atoms = len(self.df)
        stats_types = [1, 0, 0, 0, 0, 0]

        concat_features = None
        n_neighbor_list = self.neighbor_df[['n_neighbors']].values
        neighbor_lists = \
            self.neighbor_df[['neighbor_id_{}'.format(num)
                              for num in range(self.n_neighbor_limit)]].values

        for type in sorted(list(type_set)):
            type_df = copy.deepcopy(self.df[type_col].apply(lambda x: 1 if x == type else 0))

            print(mro_stats.sro_to_mro.__doc__)
            mro_feature = np.zeros((n_atoms, sum(stats_types)))
            mro_feature = \
                mro_stats.sro_to_mro(type_df.values,
                                     n_neighbor_list, neighbor_lists,
                                     stats_types, mro_feature, n_atoms=n_atoms,
                                     n_neighbor_limit=self.n_neighbor_limit)
            concat_features = mro_feature if concat_features is None \
                else np.append(concat_features, mro_feature, axis=1)

        percent_list = np.zeros(concat_features.shape)

        print(voronoi_stats.line_percent.__doc__)
        percent_list = \
            voronoi_stats.line_percent(percent_list, concat_features)

        feature_names = ["{} number_{}".format(type_name, tag_suffix)
                         for type_name in type_names] + \
                        ["{} comp_{}".format(type_name, tag_suffix)
                         for type_name in type_names]

        result_df = \
            pd.DataFrame(np.append(concat_features, percent_list, axis=1),
                         index=range(n_atoms), columns=feature_names)

        for comp, type_name in zip(raw_comp, type_names):
            result_df["{} CSRO_{}".format(type_name, tag_suffix)] = \
                (result_df["{} comp_{}".format(type_name, tag_suffix)]
                 - comp) / comp if comp != 0 else 1

        if write_path is not None:
            check_output_path(write_path)
            result_df.to_csv(os.path.join(write_path, 'csro.csv'))

        return result_df


if __name__ == "__main__":
    # nn = NearestNeighbor.from_file(data_path_file='/Users/Qi/Downloads/0.txt',
    #                             cutoff=4.2, allow_neighbor_limit=300,
    #                             n_neighbor_limit=40, pbc=[1, 1, 1])
    # nn.voro_nn(small_face_thres=0.05, write_path='/Users/Qi/Downloads/')

    # mro = MROFeaturizers.from_file(
    #     data_path_file='/Users/Qi/Downloads/voronoi_distance_stats.csv',
    #     neighbor_path_file='/Users/Qi/Downloads/neighbor_list_merge_voro_nn_edit.csv',
    #     n_neighbor_limit=40,
    #     feature_list="all", feature_sets="all", fortran=True)
    # stats_types = [1, 1, 1, 1, 1, 1]
    # mro.sro_to_mro(stats_types=stats_types, write_path='/Users/Qi/Downloads/')
    #

    data_path = r'/Users/Qi/Downloads/0_edit.txt'
    neighbor_data_path = r'/Users/Qi/Downloads/neighbor_list_merge_voro_nn_edit.csv'

    df = pd.read_csv(data_path, index_col=0, sep=' ')
    print(df.head(10))
    neighbor_df = pd.read_csv(neighbor_data_path, index_col=0)
    mro = MROFeaturizers(
        df, neighbor_df, n_neighbor_limit=40,
        feature_list="all", feature_sets="all", fortran=True)

    mro.csro(type_col='type', raw_comp=[.80, .20], type_names=['Ni', 'P'],
             write_path='/Users/Qi/Downloads/')
