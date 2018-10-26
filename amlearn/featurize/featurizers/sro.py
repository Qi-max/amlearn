import os
import numpy as np
import pandas as pd
from amlearn.utils.check import check_output_path


try:
    from amlearn.featurize.featurizers.sro_mro import voronoi_stats, boop
except Exception:
    print("import fortran file voronoi_stats error!")


def line_percent(value_list, feature_names=None,
                 write_path=None, write_file='line_percent.csv'):
    percent_list = np.zeros(value_list.shape)

    print(voronoi_stats.line_percent.__doc__)
    percent_list = \
        voronoi_stats.line_percent(percent_list, value_list)
    feature_names = ['percent_{}'.format(num)
                     for num in range(value_list.shape[1])] \
        if feature_names is None else feature_names
    percent_df = pd.DataFrame(percent_list,
                              index=range(len(percent_list)),
                              columns=feature_names)
    if write_path is not None:
        check_output_path(write_path)
        percent_df.to_csv(os.path.join(write_path, write_file))
    return percent_df


class SROFeaturizers():
    def __init__(self, n_atoms, n_neighbor_limit, n_neighbor_list,
                 edge_min=None, edge_max=None, include_beyond_edge_max=False,
                 neighbor_id_lists=None,
                 neighbor_distance_lists=None, neighbor_edge_lists=None,
                 neighbor_area_lists=None, neighbor_vol_lists=None,
                 feature_sets="all", fortran=True):
        """

        Args:
            feature_sets: "Voronoi", "BOOP_lower_voro", "BOOP_higher_voro",
                          or "all" (a total of 18 candidate feature sets)
        """

        self.n_atoms = n_atoms
        self.n_neighbor_limit = n_neighbor_limit
        self.n_neighbor_list = n_neighbor_list
        self.edge_min = edge_min
        self.edge_max = edge_max
        self.edge_num = self.edge_max - self.edge_min + 1
        self.include_beyond_edge_max = include_beyond_edge_max
        self.neighbor_id_lists = neighbor_id_lists
        self.neighbor_edge_lists = neighbor_edge_lists
        self.neighbor_distance_lists = neighbor_distance_lists
        self.neighbor_area_lists = neighbor_area_lists
        self.neighbor_vol_lists = neighbor_vol_lists
        self.feature_sets = feature_sets
        self.fortran = fortran
        self.stats = ['sum', 'mean', 'std', 'min', 'max']
        self.bq_tags = ['4', '6', '8', '10']

    @classmethod
    def from_voro_file(cls, data_path_file, n_neighbor_limit,
                       edge_min=None, edge_max=None,
                       include_beyond_edge_max=False,
                       feature_sets="all", fortran=True):
        if os.path.exists(data_path_file):
            df = pd.read_csv(data_path_file, index_col=0)
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(0)
            n_atoms = len(df)
            n_neighbor_list = df[['n_neighbors']].values
            neighbor_id_lists = df[['neighbor_id_{}'.format(num) for num in range(n_neighbor_limit)]].values
            neighbor_area_lists = df[['neighbor_area_{}'.format(num) for num in range(n_neighbor_limit)]].values
            neighbor_vol_lists = df[['neighbor_vol_{}'.format(num) for num in range(n_neighbor_limit)]].values
            neighbor_distance_lists = df[['neighbor_distance_{}'.format(num) for num in range(n_neighbor_limit)]].values
            neighbor_edge_lists = df[['neighbor_edge_{}'.format(num) for num in range(n_neighbor_limit)]].values
        else:
            raise FileNotFoundError("File {} not found".format(data_path_file))
        return cls(n_atoms, n_neighbor_limit, n_neighbor_list,
                   edge_min=edge_min, edge_max=edge_max,
                   include_beyond_edge_max=include_beyond_edge_max,
                   feature_sets=feature_sets, fortran=fortran,
                   neighbor_id_lists=neighbor_id_lists,
                   neighbor_area_lists=neighbor_area_lists,
                   neighbor_vol_lists=neighbor_vol_lists,
                   neighbor_distance_lists=neighbor_distance_lists,
                   neighbor_edge_lists=neighbor_edge_lists)

    @classmethod
    def from_dist_file(cls, data_path_file, n_neighbor_limit,
                       feature_sets="all", fortran=True):

        if os.path.exists(data_path_file):
            df = pd.read_csv(data_path_file, index_col=0)
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(0)

            n_atoms = len(df)
            n_neighbor_list = df[['n_neighbors']].values
            neighbor_id_lists = df[['neighbor_id_{}'.format(num) for num in range(n_neighbor_limit)]].values
            neighbor_distance_lists = df[['neighbor_distance_{}'.format(num) for num in range(n_neighbor_limit)]].values
        else:
            raise FileNotFoundError("File {} not found".format(data_path_file))
        return cls(n_atoms, n_neighbor_limit, n_neighbor_list,
                   edge_min=0, edge_max=0,
                   feature_sets=feature_sets, fortran=fortran,
                   neighbor_id_lists=neighbor_id_lists,
                   neighbor_distance_lists=neighbor_distance_lists)

    def get_func_list(self):
        return

    def cn_voro(self, voro_or_distance='voro', write_path=None):
        cn_list = np.zeros(self.n_atoms)

        print(voronoi_stats.cn_voro.__doc__)
        cn_list = \
            voronoi_stats.cn_voro(cn_list, self.n_neighbor_list,
                                  n_atoms=self.n_atoms)
        feature_names = ['CN_Voro'] if voro_or_distance == 'voro' else ['CN_Dist']

        if write_path is not None:
            check_output_path(write_path)
            cn_list_df = pd.DataFrame(cn_list,
                                         index=range(self.n_atoms),
                                         columns=feature_names)
            cn_list_df.to_csv(os.path.join(write_path, 'cn_voro.csv'))

        return feature_names, cn_list

    def voronoi_index(self, write_path=None):
        voronoi_index_list = np.zeros((self.n_atoms, self.edge_num))

        print(voronoi_stats.voronoi_index.__doc__)
        voro_index_list = \
            voronoi_stats.voronoi_index(voronoi_index_list,
                                        self.n_neighbor_list,
                                        self.neighbor_edge_lists,
                                        self.edge_min, self.edge_max,
                                        self.include_beyond_edge_max,
                                        n_atoms=self.n_atoms,
                                        n_neighbor_limit=self.n_neighbor_limit)
        feature_names = ['Voronoi idx_{}'.format(edge)
                         for edge in range(self.edge_min, self.edge_max+1)]
        if write_path is not None:
            check_output_path(write_path)
            voro_index_df = pd.DataFrame(voro_index_list,
                                         index=range(self.n_atoms),
                                         columns=feature_names)
            voro_index_df.to_csv(os.path.join(write_path, 'voronoi_index.csv'))
        return feature_names, voro_index_list

    def character_motif(self, voronoi_index_list, target_voro_idx=None,
                        frank_kasper=1, write_path=None):
        if target_voro_idx is None:
            target_voro_idx = np.array([[0, 0, 12, 0, 0], [0, 0, 12, 4, 0]],
                                       dtype=np.float128)
        print(target_voro_idx)
        motif_one_hot = np.zeros((self.n_atoms,
                                  len(target_voro_idx) + frank_kasper))

        print(voronoi_stats.character_motif.__doc__)

        motif_one_hot = \
            voronoi_stats.character_motif(motif_one_hot,
                                          voronoi_index_list,
                                          self.edge_min, target_voro_idx,
                                          frank_kasper,
                                          n_atoms=self.n_atoms)
        print(motif_one_hot)
        feature_names = ['is <0,0,12,0,0>', 'is <0,0,12,4,0>',
                         'is polytetrahedral', 'is <0,0,12,0/4,0>']
        motif_one_hot_array = np.array(motif_one_hot)
        is_120_124 = motif_one_hot_array[:, 0] | motif_one_hot_array[:, 1]
        print(motif_one_hot_array.shape)
        print(is_120_124.shape)
        motif_one_hot_array = np.append(motif_one_hot_array, np.array([is_120_124]).T, axis=1)
        if write_path is not None:
            check_output_path(write_path)
            voro_index_df = pd.DataFrame(motif_one_hot_array,
                                         index=range(self.n_atoms),
                                         columns=feature_names)
            voro_index_df.to_csv(os.path.join(write_path, 'character_motif.csv'))
        return feature_names, motif_one_hot_array

    def i_fold_symmetry(self, write_path=None):
        i_symm_list = np.zeros((self.n_atoms, self.edge_num))

        print(voronoi_stats.i_fold_symmetry.__doc__)
        i_symm_list = \
            voronoi_stats.i_fold_symmetry(i_symm_list,
                                          self.n_neighbor_list,
                                          self.neighbor_edge_lists,
                                          self.edge_min, self.edge_max,
                                          self.include_beyond_edge_max,
                                          n_atoms=self.n_atoms,
                                          n_neighbor_limit=self.n_neighbor_limit)
        feature_names = ['{}-fold symm idx'.format(edge)
                         for edge in range(self.edge_min, self.edge_max+1)]
        if write_path is not None:
            check_output_path(write_path)
            i_symm_list_df = pd.DataFrame(i_symm_list,
                                         index=range(self.n_atoms),
                                         columns=feature_names)
            i_symm_list_df.to_csv(os.path.join(write_path, 'i_symm_list.csv'))
        return feature_names, i_symm_list

    def area_wt_i_fold_symmetry(self, write_path=None):
        area_wt_i_symm_list = np.zeros((self.n_atoms, self.edge_num))

        print(voronoi_stats.area_wt_i_fold_symmetry.__doc__)
        area_wt_i_symm_list = \
            voronoi_stats.area_wt_i_fold_symmetry(area_wt_i_symm_list,
                                                  self.n_neighbor_list,
                                                  self.neighbor_edge_lists,
                                                  self.neighbor_area_lists,
                                                  self.edge_min, self.edge_max,
                                                  self.include_beyond_edge_max,
                                                  n_atoms=self.n_atoms,
                                                  n_neighbor_limit=
                                                  self.n_neighbor_limit)
        feature_names = ['Area_wt {}-fold symm idx'.format(edge)
                         for edge in range(self.edge_min, self.edge_max+1)]
        if write_path is not None:
            check_output_path(write_path)
            area_wt_i_symm_list_df = pd.DataFrame(area_wt_i_symm_list,
                                         index=range(self.n_atoms),
                                         columns=feature_names)
            area_wt_i_symm_list_df.to_csv(os.path.join(write_path, 'area_wt_i_fold_symmetry.csv'))
        return feature_names, area_wt_i_symm_list

    def vol_wt_i_fold_symmetry(self, write_path=None):
        vol_wt_i_symm_list = np.zeros((self.n_atoms, self.edge_num))

        print(voronoi_stats.vol_wt_i_fold_symmetry.__doc__)
        vol_wt_i_symm_list = \
            voronoi_stats.vol_wt_i_fold_symmetry(vol_wt_i_symm_list,
                                                  self.n_neighbor_list,
                                                  self.neighbor_edge_lists,
                                                  self.neighbor_vol_lists,
                                                  self.edge_min, self.edge_max,
                                                  self.include_beyond_edge_max,
                                                  n_atoms=self.n_atoms,
                                                  n_neighbor_limit=
                                                  self.n_neighbor_limit)
        feature_names = ['Vol_wt {}-fold symm idx'.format(edge)
                         for edge in range(self.edge_min, self.edge_max+1)]
        if write_path is not None:
            check_output_path(write_path)
            vol_wt_i_symm_list_df = pd.DataFrame(vol_wt_i_symm_list,
                                         index=range(self.n_atoms),
                                         columns=feature_names)
            vol_wt_i_symm_list_df.to_csv(os.path.join(write_path, 'vol_wt_i_fold_symmetry.csv'))
        return feature_names, vol_wt_i_symm_list

    def voronoi_area_stats(self, write_path=None):
        area_stats = np.zeros((self.n_atoms, self.edge_num))

        print(voronoi_stats.voronoi_area_stats.__doc__)
        area_stats = \
            voronoi_stats.voronoi_area_stats(
                area_stats, self.n_neighbor_list, self.neighbor_area_lists,
                n_atoms=self.n_atoms,
                n_neighbor_limit=self.n_neighbor_limit)
        feature_names = ['Voronoi area'] + ['Facet area {}'.format(stat)
                         for stat in self.stats[1:]]
        if write_path is not None:
            check_output_path(write_path)
            area_stats_df = pd.DataFrame(area_stats,
                                         index=range(self.n_atoms),
                                         columns=feature_names)
            area_stats_df.to_csv(os.path.join(write_path, 'voronoi_area_stats.csv'))
        return feature_names, area_stats

    def voronoi_area_stats_separate(self, write_path=None):
        area_stats_separate = np.zeros((self.n_atoms, self.edge_num * 5))

        print(voronoi_stats.voronoi_area_stats_separate.__doc__)
        area_stats_separate = \
            voronoi_stats.voronoi_area_stats_separate(
                area_stats_separate, self.n_neighbor_list,
                self.neighbor_edge_lists, self.neighbor_area_lists,
                self.edge_min, self.edge_max,
                self.include_beyond_edge_max,
                n_atoms=self.n_atoms,
                n_neighbor_limit=self.n_neighbor_limit)
        feature_names = ['{}-edged area {}'.format(edge, stat)
                         for edge in range(self.edge_min, self.edge_max+1)
                         for stat in self.stats]
        if write_path is not None:
            check_output_path(write_path)
            area_stats_separate_df = pd.DataFrame(area_stats_separate,
                                         index=range(self.n_atoms),
                                         columns=feature_names)
            area_stats_separate_df.to_csv(os.path.join(write_path, 'voronoi_area_stats_separate.csv'))
        return feature_names, area_stats_separate

    def voronoi_vol_stats(self, write_path=None):
        vol_stats = np.zeros((self.n_atoms, self.edge_num))

        print(voronoi_stats.voronoi_vol_stats.__doc__)
        vol_stats = \
            voronoi_stats.voronoi_vol_stats(
                vol_stats, self.n_neighbor_list, self.neighbor_vol_lists,
                n_atoms=self.n_atoms,
                n_neighbor_limit=self.n_neighbor_limit)
        feature_names = ['Voronoi vol']  + ['Sub-polyhedra vol {}'.format(stat)
                         for stat in self.stats[1:]]
        if write_path is not None:
            check_output_path(write_path)
            vol_stats_df = pd.DataFrame(vol_stats,
                                         index=range(self.n_atoms),
                                         columns=feature_names)
            vol_stats_df.to_csv(os.path.join(write_path, 'voronoi_vol_stats.csv'))
        return feature_names, vol_stats

    def voronoi_vol_stats_separate(self, write_path=None):
        vol_stats_separate = np.zeros((self.n_atoms, self.edge_num * 5))

        print(voronoi_stats.voronoi_vol_stats_separate.__doc__)
        vol_stats_separate = \
            voronoi_stats.voronoi_vol_stats_separate(
                vol_stats_separate, self.n_neighbor_list,
                self.neighbor_edge_lists, self.neighbor_vol_lists,
                self.edge_min, self.edge_max,
                self.include_beyond_edge_max,
                n_atoms=self.n_atoms,
                n_neighbor_limit=self.n_neighbor_limit)
        feature_names = ['{}-edged vol {}'.format(edge, stat)
                         for edge in range(self.edge_min, self.edge_max+1)
                         for stat in self.stats]
        if write_path is not None:
            check_output_path(write_path)
            vol_stats_separate_df = pd.DataFrame(vol_stats_separate,
                                         index=range(self.n_atoms),
                                         columns=feature_names)
            vol_stats_separate_df.to_csv(os.path.join(write_path, 'voronoi_vol_stats_separate.csv'))
        return feature_names, vol_stats_separate

    def voronoi_distance_stats(self, voro_or_distance='voro', write_path=None):
        distance_stats = np.zeros((self.n_atoms, 5))

        print(voronoi_stats.voronoi_distance_stats.__doc__)
        distance_stats = \
            voronoi_stats.voronoi_distance_stats(
                distance_stats, self.n_neighbor_list,
                self.neighbor_distance_lists,
                n_atoms=self.n_atoms, n_neighbor_limit=self.n_neighbor_limit)
        feature_names = \
            ['Distance_Voro {}'.format(stat) for stat in self.stats] \
                if voro_or_distance=='voro' else \
                ['Distance_Dist {}'.format(stat) for stat in self.stats]
        if write_path is not None:
            check_output_path(write_path)
            distance_stats_df = pd.DataFrame(distance_stats,
                                         index=range(self.n_atoms),
                                         columns=feature_names)
            distance_stats_df.to_csv(os.path.join(write_path, 'voronoi_distance_stats.csv'))
        return feature_names, distance_stats

    def boop(self, coords_path_file, pbc, low_order, higher_order,
             coarse_lower_order, coarse_higher_order,
             voro_or_distance='voro', write_path=None):
        Ql = np.zeros((self.n_atoms, 4))
        Wlbar = np.zeros((self.n_atoms, 4))
        coarse_Ql = np.zeros((self.n_atoms, 4))
        coarse_Wlbar = np.zeros((self.n_atoms, 4))

        if os.path.exists(coords_path_file):
            with open(coords_path_file, 'r') as rf:
                lines = rf.readlines()
                atom_coords = list()
                Bds = [list(map(float, lines[5].strip().split())),
                       list(map(float, lines[6].strip().split())),
                       list(map(float, lines[7].strip().split()))]
                i = 0
                for line in lines:
                    if i > 8:
                        column_values = line.strip().split()
                        atom_coords.append([np.float128(column_values[2]),
                                            np.float128(column_values[3]),
                                            np.float128(column_values[4])])
                    i += 1
        else:
            raise FileNotFoundError("File {} not found".format(coords_path_file))

        print(boop.calculate_boop.__doc__)
        Ql, Wlbar, coarse_Ql, coarse_Wlbar = \
            boop.calculate_boop(
                atom_coords, pbc, Bds,
                self.n_neighbor_list, self.neighbor_id_lists,
                low_order, higher_order,
                coarse_lower_order, coarse_higher_order,
                Ql, Wlbar, coarse_Ql, coarse_Wlbar,
                n_atoms=self.n_atoms, n_neighbor_limit=self.n_neighbor_limit)
        feature_names = \
            ['q_{}-Voro'.format(num) for num in self.bq_tags] + \
            ['w_{}-Voro'.format(num) for num in self.bq_tags] + \
            ['Coarse-grained q_{}-Voro'.format(num) for num in self.bq_tags] + \
            ['Coarse-grained w_{}-Voro'.format(num) for num in self.bq_tags] \
                if voro_or_distance=='voro' else \
            ['q_{}-Dist'.format(num) for num in self.bq_tags] + \
            ['w_{}-Dist'.format(num) for num in self.bq_tags] + \
            ['Coarse-grained q_{}-Dist'.format(num) for num in self.bq_tags] + \
            ['Coarse-grained w_{}-Dist'.format(num) for num in self.bq_tags]

        if write_path is not None:
            check_output_path(write_path)
            concat_array = np.append(Ql, Wlbar, axis=1)
            concat_array = np.append(concat_array, coarse_Ql, axis=1)
            concat_array = np.append(concat_array, coarse_Wlbar, axis=1)
            result_df = pd.DataFrame(concat_array, index=range(self.n_atoms),
                                     columns=feature_names)
            result_df.to_csv(os.path.join(write_path, 'boop.csv'))
        return feature_names, Ql, Wlbar, coarse_Ql, coarse_Wlbar
