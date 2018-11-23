import os
import numpy as np
import pandas as pd
from amlearn.featurize.featurizers.base import BaseFeaturize
try:
    from amlearn.featurize.featurizers.sro_mro import voronoi_nn, \
        distance_nn, boop
except Exception:
    print("import fortran file voronoi_nn, distance_nn error!")


class Voro(BaseFeaturize):

    def __init__(self, n_atoms, atom_type, cutoff, atom_coords,
                 allow_neighbor_limit, n_neighbor_limit, pbc, Bds,
                 small_face_thres):
        self.n_atoms = n_atoms
        self.atom_type = atom_type
        self.cutoff = cutoff
        self.atom_coords = atom_coords
        self.allow_neighbor_limit = allow_neighbor_limit
        self.n_neighbor_limit = n_neighbor_limit
        self.pbc = pbc
        self.Bds = Bds

    @classmethod
    def from_file(cls, data_path_file, cutoff, allow_neighbor_limit,
                  n_neighbor_limit, pbc):
        if os.path.exists(data_path_file):
            with open(data_path_file, 'r') as rf:
                lines = rf.readlines()
                n_atoms = len(lines) - 9
                atom_type = list()
                atom_coords = list()
                Bds = [list(map(float, lines[5].strip().split())),
                       list(map(float, lines[6].strip().split())),
                       list(map(float, lines[7].strip().split()))]
                i = 0
                for line in lines:
                    if i == 3:
                        n_atoms = int(line.strip())
                    # print("i:{}, line:{}".format(i, line))
                    if i > 8:
                        column_values = line.strip().split()
                        atom_type.append(int(column_values[1]))
                        atom_coords.append([np.float128(column_values[2]),
                                            np.float128(column_values[3]),
                                            np.float128(column_values[4])])
                    i += 1
        else:
            raise FileNotFoundError("File {} not found".format(data_path_file))
        return cls(n_atoms, atom_type, cutoff, atom_coords,
                   allow_neighbor_limit, n_neighbor_limit, pbc, Bds)

    def fit(self, X):
        pass

    def transform(self, X):
        n_neighbor_list = np.zeros(self.n_atoms, dtype=np.float128)
        neighbor_lists = np.zeros((self.n_atoms, self.n_neighbor_limit),
                                  dtype=np.float128)
        neighbor_edge_lists = np.zeros((self.n_atoms, self.n_neighbor_limit),
                                       dtype=np.float128)
        neighbor_area_lists = np.zeros((self.n_atoms, self.n_neighbor_limit),
                                       dtype=np.float128)
        neighbor_vol_lists = np.zeros((self.n_atoms, self.n_neighbor_limit),
                                      dtype=np.float128)
        neighbor_distance_lists = np.zeros(
            (self.n_atoms, self.n_neighbor_limit), dtype=np.float128)

        n_neighbor_max = 0
        n_edge_max = 0

        # print(voronoi_nn.voronoi.__doc__)

        n_neighbor_list, neighbor_lists, \
        neighbor_area_lists, neighbor_vol_lists, neighbor_distance_lists, \
        neighbor_edge_lists, n_neighbor_max, n_edge_max = \
            voronoi_nn.voronoi(self.atom_type, self.atom_coords,
                               self.cutoff, self.allow_neighbor_limit,
                               small_face_thres,
                               self.pbc, self.Bds, n_neighbor_list,
                               neighbor_lists, neighbor_area_lists,
                               neighbor_vol_lists, neighbor_distance_lists,
                               neighbor_edge_lists, n_neighbor_max, n_edge_max,
                               n_atoms=self.n_atoms,
                               n_neighbor_limit=self.n_neighbor_limit)

        if write_path is not None:
            check_output_path(write_path)
            with open(os.path.join(write_path,
                                   'n_neighbor_max_and_n_edge_max_voro_nn.txt'),
                      'w+') as wf:
                wf.write('n_neighbor_max is : {}\n'.format(n_neighbor_max))
                wf.write('n_edge_max is : {}\n'.format(n_edge_max))

            concat_array = np.append(np.array([n_neighbor_list]).T,
                                     neighbor_lists, axis=1)
            concat_array = np.append(concat_array,
                                     neighbor_area_lists, axis=1)
            concat_array = np.append(concat_array,
                                     neighbor_vol_lists, axis=1)
            concat_array = np.append(concat_array,
                                     neighbor_distance_lists, axis=1)
            concat_array = np.append(concat_array,
                                     neighbor_edge_lists, axis=1)
            result_df = pd.DataFrame(concat_array, index=range(self.n_atoms),
                                     columns=columns
                                     )
            result_df.to_csv(os.path.join(write_path,
                                          'neighbor_list_merge_voro_nn_edit.csv'))

        return n_neighbor_list, neighbor_lists, neighbor_edge_lists, \
               neighbor_area_lists, neighbor_vol_lists, \
               neighbor_distance_lists, n_neighbor_max, n_edge_max

    def get_feature_names(self):
        columns = ['n_neighbors'] + \
                  ['neighbor_id_{}'.format(i) for i in range(self.n_neighbor_limit)] +\
                  ['neighbor_area_{}'.format(i) for i in range(self.n_neighbor_limit)] + \
                  ['neighbor_vol_{}'.format(i) for i in range(self.n_neighbor_limit)] + \
                  ['neighbor_distance_{}'.format(i) for i in range(self.n_neighbor_limit)] + \
                  ['neighbor_edge_{}'.format(i) for i in range(self.n_neighbor_limit)]
        return columns

