import os
import pickle
from collections import defaultdict
from math import pi
import numpy as np
import pandas as pd

from scipy.spatial.qhull import ConvexHull
from amlearn.featurize.base import BaseSRO, load_radii
from amlearn.utils.data import read_imd, read_lammps_dump
from amlearn.utils.packing import solid_angle, \
    triangular_angle, calc_stats, triangle_area, tetra_volume
from amlearn.utils.verbose import VerboseReporter

from amlearn.amlearn.utils.data import calc_neighbor_coords

try:
    from amlearn.featurize.featurizers.src import voronoi_stats, boop
except Exception:
    print("import fortran file voronoi_stats error!\n")

module_dir = os.path.dirname(os.path.abspath(__file__))


class PackingOfSite(object):
    def __init__(self, pbc, Bds, atom_type, coords, neighbors_type,
                 neighbors_coords, radii=None, radius_type="miracle_radius"):
        self.pbc = pbc
        self.Bds = Bds
        self.atom_type = atom_type
        self.coords = coords
        self.neighbors_type = neighbors_type
        self.neighbors_coords = neighbors_coords
        self.radii = load_radii() if radii is None else radii
        self.radius_type = radius_type

    def nn_coords(self):
        nn_coords_ = [calc_neighbor_coords(self.coords, neighbor_coords,
                                           self.Bds, self.pbc)
                      for neighbor_coords in self.neighbors_coords]
        return nn_coords_

    def convex_hull(self):
        if not hasattr(self, 'convex_hull_'):
            self.convex_hull_ = ConvexHull(self.nn_coords())
        return self.convex_hull_

    def calculate_hull_facet_angles(self):
        triangular_angle_lists_ = list()
        # get convex surface indices
        convex_surfaces_indices = self.convex_hull().simplices
        # get nn_coords
        nn_coords = self.nn_coords()
        # set leaf node indices for find leaf site coords
        leaf_indices_list = [(1, 2), (0, 2), (0, 1)]

        for convex_surface_indices in convex_surfaces_indices:
            triangular_angle_list_ = list()
            for idx, leaf_indices in zip(convex_surface_indices,
                                         leaf_indices_list):
                triangular_angle_ = triangular_angle(nn_coords[idx],
                    nn_coords[convex_surface_indices[leaf_indices[0]]],
                    nn_coords[convex_surface_indices[leaf_indices[1]]])
                triangular_angle_list_.append(triangular_angle_)
            triangular_angle_lists_.append(triangular_angle_list_)

        self.triangular_angle_lists_ = triangular_angle_lists_
        return self.triangular_angle_lists_

    def calculate_hull_tetra_angles(self):
        solid_angle_lists_ = list()
        # get convex surface indices
        convex_surfaces_indices = self.convex_hull().simplices
        # get nn_coords
        nn_coords = self.nn_coords()
        # set leaf node indices for find leaf site coords
        leaf_indices_list = [(1, 2), (0, 2), (0, 1)]

        for convex_surface_indices in convex_surfaces_indices:
            solid_angle_list_ = list()
            for idx, leaf_indices in zip(convex_surface_indices,
                                         leaf_indices_list):
                solid_angle_ = solid_angle(
                    nn_coords[idx], self.coords,
                    nn_coords[convex_surface_indices[leaf_indices[0]]],
                    nn_coords[convex_surface_indices[leaf_indices[1]]])
                solid_angle_list_.append(solid_angle_)
            solid_angle_lists_.append(solid_angle_list_)

        self.solid_angle_lists_ = solid_angle_lists_
        return self.solid_angle_lists_

    def analyze_hull_facet_interstice(self):
        area_list = list()
        area_interstice_list = list()

        # get nn_coords
        nn_coords = np.array(self.nn_coords())

        # iter convex surface (neighbor indices)
        for convex_surface_indices, triangular_angle_list in \
                zip(self.convex_hull().simplices, self.triangular_angle_lists_):
            packed_area = 0
            surface_coords = nn_coords[convex_surface_indices]

            # calculate neighbors' packed_area
            for idx, tri_angle in zip(convex_surface_indices,
                                      triangular_angle_list):
                r = self.radii[str(self.neighbors_type[idx])][self.radius_type]
                packed_area += tri_angle / 2 * pow(r, 2)

            area = triangle_area(*surface_coords)
            area_list.append(area)
            ################################################################
            # Please note that this is percent, not interstice area!!!!!!
            area_interstice_list.append(1 - packed_area/area)

        self.area_list_ = area_list
        self.area_interstice_list_ = area_interstice_list

    def analyze_hull_tetra_interstice(self):
        volume_list = list()
        volume_interstice_list = list()

        # get nn_coords
        nn_coords = np.array(self.nn_coords())

        # iter convex surface (neighbor indices)
        for convex_surface_indices, solid_angle_list in \
                zip(self.convex_hull().simplices, self.solid_angle_lists_):
            packed_volume = 0
            surface_coords = nn_coords[convex_surface_indices]

            # calculate neighbors' packed_volume
            for idx, sol_angle in zip(convex_surface_indices,
                                      solid_angle_list):
                if sol_angle == 0:
                    continue
                r = self.radii[str(self.neighbors_type[idx])][self.radius_type]
                packed_volume += sol_angle / 3 * pow(r, 3)

            # add center's packed_volume
            center_solid_angle = solid_angle(self.coords, *surface_coords)
            center_r = self.radii[str(self.atom_type)][self.radius_type]
            packed_volume += center_solid_angle / 3 * pow(center_r, 3)

            volume = tetra_volume(self.coords, *surface_coords)
            volume_list.append(volume)
            volume_interstice_list.append(1 - packed_volume/volume)

        self.volume_list_ = volume_list
        self.volume_interstice_list_ = volume_interstice_list

    def combine_neighbor_solid_angles(self):
        if not hasattr(self, 'neighbors_solid_angle_'):
            # init neighbors_solid_angle list
            neighbors_solid_angle = [0] * len(self.neighbors_type)

            # iter convex surface (neighbor indices)
            for convex_surface_indices, solid_angle_list in \
                    zip(self.convex_hull().simplices, self.solid_angle_lists_):
                for idx, solid_angle_ in zip(convex_surface_indices, solid_angle_list):
                    neighbors_solid_angle[idx] += solid_angle_
            self.neighbors_solid_angle_ = neighbors_solid_angle
        return self.neighbors_solid_angle_

    def cluster_packed_volume(self):
        """
        Calculate the cluster volume that is packed with atoms, including the
        volume of center atoms plus the volume cones (from solid angle) of
        all the neighbors.
        Args:
            radii: list or dict, default: None.
                If list, index is type, value is radial,
                If dict, key is type, value is radial.
                If None, read type radial from '../radii.json'
        Returns:
            packed_volume
        """
        packed_volume = 4/3 * pi * pow(self.radii[str(self.atom_type)]
                                       [self.radius_type], 3)
        for neighbor_type, solid_angle_ in zip(self.neighbors_type,
                                              self.neighbors_solid_angle_):
            if solid_angle_ == 0:
                continue
            packed_volume += solid_angle_ * 1/3 * pow(self.radii[neighbor_type]
                                                      [self.radius_type], 3)
        return packed_volume

    def cluster_packing_efficiency(self):
        return self.cluster_packed_volume() / self.convex_hull().volume

    def nn_type_dict(self):
        nn_type_dict_ = defaultdict(int)
        for neighbor_type in self.neighbors_type:
            nn_type_dict_[neighbor_type] += 1
        return nn_type_dict_

    def atomic_packing_efficiency(self):
        ideal_ratio_ = {3: 0.154701, 4: 0.224745, 5: 0.361654, 6: 0.414214,
                        7: 0.518145, 8: 0.616517, 9: 0.709914, 10: 0.798907,
                        11: 0.884003, 12: 0.902113, 13: 0.976006, 14: 1.04733,
                        15: 1.11632, 16: 1.18318, 17: 1.2481, 18: 1.31123,
                        19: 1.37271, 20: 1.43267, 21: 1.49119, 22: 1.5484,
                        23: 1.60436, 24: 1.65915}

        r = 0
        for t, n in self.nn_type_dict().items():
            r += self.radii[str(t)][self.radius_type] * n
        r = r / len(self.neighbors_type)
        return self.radii[str(self.atom_type)][self.radius_type] / r - \
               ideal_ratio_[len(self.neighbors_type)]


class DistanceInterstice(BaseSRO):
    def __init__(self, context=None, type_col='type',
                 types_atomic_number_list=None,
                 n_neighbor_limit=80, dependency="voro", tmp_save=True,
                 radii=None, radius_type="miracle_radius",
                 verbose=0, **nn_kwargs):
        super(DistanceInterstice, self).__init__(
            tmp_save=tmp_save, context=context,
            dependency=dependency, **nn_kwargs)
        self.types_atomic_number_list = types_atomic_number_list
        self.n_neighbor_limit = n_neighbor_limit
        self.voro_depend_cols = ['n_neighbors_voro'] + \
                                ['neighbor_id_{}_voro'.format(idx)
                                 for idx in range(n_neighbor_limit)]
        self.dist_depend_cols = ['n_neighbors_dist'] + \
                                ['neighbor_id_{}_dist'.format(idx)
                                 for idx in range(n_neighbor_limit)]
        self.radii = load_radii() if radii is None else radii
        self.radius_type = radius_type
        self.type_col = type_col
        self.verbose = verbose
        self.area_list = list()
        self.area_interstice_list = list()
        self.volume_list = list()
        self.volume_interstice_list = list()

    def transform(self, X, lammp_df=None, Bds=None,
                  lammp_path=None):

        neighbor_id_cols = \
            ['neighbor_id_{}_{}'.format(idx, self.dependency_name)
             for idx in range(self.n_neighbor_limit)]

        neighbor_dist_cols = \
            ['neighbor_dist_{}_{}'.format(idx, self.dependency_name)
             for idx in range(self.n_neighbor_limit)]

        if lammp_path is not None and os.path.exists(lammp_path):
            lammp_df, Bds = read_lammps_dump(lammp_path)
        if self.types_atomic_number_list is not None:
            lammp_df[self.type_col] = lammp_df[self.type_col].apply(
                lambda x: self.types_atomic_number_list[x-1])

        # define print verbose
        if self.verbose > 0:
            vr = VerboseReporter(self.context, total_stage=1,
                                 verbose=1, max_verbose_mod=10000)
            vr.init(total_epoch=len(X), start_epoch=0,
                    init_msg='Calculating DistanceInterstice features.',
                    epoch_name='Atoms', stage=1)

        feature_lists = list()
        integrated_df = X.join(self.dependence_df)
        for idx, row in integrated_df.iterrows():
            neighbors_distance_interstice_list = list()
            for neighbor_id, neighbor_dist in zip(row[neighbor_id_cols], row[neighbor_dist_cols]):
                if neighbor_id > 0:
                    neighbors_distance_interstice_list.append(
                        neighbor_dist / (
                            self.radii[integrated_df.loc[idx][self.type_col]] +
                            self.radii[integrated_df.loc[neighbor_id][self.type_col]]))
                else:
                    continue
            feature_lists.append(neighbors_distance_interstice_list)
        distance_interstice_df = pd.DataFrame(feature_lists,
                                              index=X.index,
                                              columns=self.get_feature_names())
        if self.tmp_save:
            self.context.save_featurizer_as_dataframe(
                output_df=distance_interstice_df,
                name='{}_{}_{}_distance_interstice'.format(self.category,
                                                   self.dependency_name,
                                                   self.radius_type))

        return distance_interstice_df

    def get_feature_names(self):
        feature_names = \
            ['neighbor_distance_interstice_{}_{}'.format(idx, self.dependency_name)
             for idx in range(self.n_neighbor_limit)]
        return feature_names

    @property
    def double_dependency(self):
        return False


class VolumeAreaInterstice(BaseSRO):
    def __init__(self, pbc, context=None,
                 coords_cols=None, type_col='type',
                 types_atomic_number_list=None,
                 n_neighbor_limit=80, dependency="voro", tmp_save=True,
                 radii=None, radius_type="miracle_radius",
                 calc_volume_area='all', verbose=0, **nn_kwargs):
        super(VolumeAreaInterstice, self).__init__(
            tmp_save=tmp_save, context=context,
            dependency=dependency, **nn_kwargs)
        self.pbc = pbc
        self.coords_cols = coords_cols \
            if coords_cols is not None else ['x', 'y', 'z']
        self.type_col = type_col
        self.types_atomic_number_list = types_atomic_number_list
        self.n_neighbor_limit = n_neighbor_limit
        self.voro_depend_cols = ['n_neighbors_voro'] + \
                                ['neighbor_id_{}_voro'.format(idx)
                                 for idx in range(n_neighbor_limit)]
        self.dist_depend_cols = None
        self.radii = load_radii() if radii is None else radii
        self.radius_type = radius_type
        self.calc_volume_area = calc_volume_area
        self.verbose = verbose
        self.area_list = list()
        self.area_interstice_list = list()
        self.volume_list = list()
        self.volume_interstice_list = list()

    def transform(self, X, Bds=None, lammp_path=None):
        neighbor_id_cols = \
            ['neighbor_id_{}_{}'.format(idx, self.dependency_name)
             for idx in range(self.n_neighbor_limit)]

        if lammp_path is not None and os.path.exists(lammp_path):
            X, Bds = read_lammps_dump(lammp_path)
        if self.types_atomic_number_list is not None:
            X[self.type_col] = X[self.type_col].apply(
                lambda x: self.types_atomic_number_list[x-1])

        # define print verbose
        if self.verbose > 0:
            vr = VerboseReporter(self.context, total_stage=1,
                                 verbose=1, max_verbose_mod=10000)
            vr.init(total_epoch=len(X), start_epoch=0,
                    init_msg='Calculating VolumeAreaInterstice features.',
                    epoch_name='Atoms', stage=1)

        integrated_df = X.join(self.dependence_df)
        feature_lists = list()
        for idx, row in integrated_df.iterrows():
            neighbors_type = list()
            neighbors_coords = list()
            for x in row[neighbor_id_cols]:
                if x > 0:
                    neighbors_type.append(integrated_df.loc[x][self.type_col])
                    neighbors_coords.append(integrated_df.loc[x][self.coords_cols])
                else:
                    continue
            pos_ = PackingOfSite(self.pbc, Bds,
                                 row[self.type_col], row[self.coords_cols],
                                 neighbors_type, neighbors_coords,
                                 radii=self.radii, radius_type=self.radius_type)
            if len(neighbors_type) < 4:
                feature_lists.append([0] * len(self.get_feature_names()))
            else:
                if self.calc_packing_efficiency:
                    feature_list = \
                        [pos_.atomic_packing_efficiency(),
                         pos_.glass_packing_efficiency()]

                if self.calc_volume_area == 'volume' or \
                        self.calc_volume_area == 'all':
                    pos_.analyze_hull_tetra_interstice()
                    volume_interstice_list = pos_.volume_interstice_list_
                    volume_list = pos_.volume_list_
                    volume_total = pos_.convex_hull().volume
                    volume_interstice_original_array = np.array(volume_interstice_list)*np.array(volume_list)
                    center_volume = 4/3 * pi * \
                        pow(pos_.radii[str(pos_.atom_type)][pos_.radius_type], 3)

                    # fractional volume_interstices in relative to the tetrahedra volume
                    feature_list.extend(calc_stats(volume_interstice_list))

                    # original volume_interstices (in the units of volume)
                    feature_list.extend(calc_stats(volume_interstice_original_array))

                    # fractional volume_interstices in relative to the entire volume
                    feature_list.extend(calc_stats(volume_interstice_original_array/volume_total*len(volume_list)))

                    # fractional volume_interstices in relative to the center atom volume
                    feature_list.extend(calc_stats(volume_interstice_original_array/center_volume))

                    self.volume_interstice_list.append(pos_.volume_interstice_list_)
                    self.volume_list.append(pos_.volume_list_)

                if self.calc_volume_area == 'area' or \
                        self.calc_volume_area == 'all':
                    pos_.analyze_hull_facet_interstice()
                    area_interstice_list = pos_.area_interstice_list_
                    area_list = pos_.area_list_
                    area_total = pos_.convex_hull().area
                    area_interstice_original_array = np.array(area_interstice_list)*np.array(area_list)
                    center_slice_area = pi * \
                        pow(pos_.radii[str(pos_.atom_type)][pos_.radius_type], 2)

                    # fractional area_interstices in relative to the tetrahedra area
                    feature_list.extend(calc_stats(area_interstice_list))

                    # original area_interstices (in the units of area)
                    feature_list.extend(calc_stats(area_interstice_original_array))

                    # fractional area_interstices in relative to the entire area
                    feature_list.extend(calc_stats(area_interstice_original_array/area_total*len(area_list)))

                    # fractional area_interstices in relative to the center atom volume
                    feature_list.extend(calc_stats(area_interstice_original_array/center_slice_area))

                    self.area_interstice_list.append(
                        pos_.area_interstice_list_)
                    self.area_list.append(pos_.area_list_)
                feature_lists.append(feature_list)

            if self.verbose > 0:
                vr.update(idx - 1)

        volume_area_interstice_df = pd.DataFrame(
            feature_lists, index=X.index, columns=self.get_feature_names())

        if self.tmp_save:
            self.context.save_featurizer_as_dataframe(
                output_df=volume_area_interstice_df,
                name='{}_{}_{}_volume_area_interstice'.format(
                    self.category, self.dependency_name, self.radius_type))

        return volume_area_interstice_df

    def get_feature_names(self):
        feature_names = list()
        feature_prefixs = list()

        stats = ['sum', 'mean', 'std', 'min', 'max']
        if self.calc_volume_area == 'volume' or self.calc_volume_area == 'all':
            feature_prefixs += ['fractional_volume_interstice_tetrahedra',
                                # 'packing_surface_area',
                                "volume_interstice",
                                "fractional_volume_interstice_tetrahedra_avg",
                                "fractional_volume_interstice_center_v"]
        if self.calc_volume_area == 'area' or self.calc_volume_area == 'all':
            feature_prefixs += ['fractional_area_interstice_triangle',
                                "area_interstice",
                                "fractional_area_interstice_triangle_avg",
                                "fractional_area_interstice_center_slice_a"]
        feature_names += ['{} {} {}'.format(feature_prefix, stat,
                                            self.dependency_name)
                          for feature_prefix in feature_prefixs
                          for stat in stats]
        return feature_names


class ClusterPackingEfficiency(BaseSRO):
    """
    Yang, L. et al. Atomic-Scale Mechanisms of the Glass-Forming Ability
    in Metallic Glasses. Phys. Rev. Lett. 109, 105502 (2012)
    """
    def __init__(self, pbc, context=None,
                 coords_cols=None, type_col='type',
                 types_atomic_number_list=None,
                 n_neighbor_limit=80, dependency="voro", tmp_save=True,
                 radii=None, radius_type="miracle_radius",
                 verbose=0, **nn_kwargs):
        super(ClusterPackingEfficiency, self).__init__(
            tmp_save=tmp_save, context=context,
            dependency=dependency, **nn_kwargs)
        self.pbc = pbc
        self.coords_cols = coords_cols \
            if coords_cols is not None else ['x', 'y', 'z']
        self.type_col = type_col
        self.types_atomic_number_list = types_atomic_number_list
        self.n_neighbor_limit = n_neighbor_limit
        self.voro_depend_cols = ['n_neighbors_voro'] + \
                                ['neighbor_id_{}_voro'.format(idx)
                                 for idx in range(n_neighbor_limit)]
        self.dist_depend_cols = None
        self.radii = load_radii() if radii is None else radii
        self.radius_type = radius_type
        self.verbose = verbose
        self.area_list = list()
        self.area_interstice_list = list()
        self.volume_list = list()
        self.volume_interstice_list = list()

    def transform(self, X, Bds=None, lammp_path=None):
        neighbor_id_cols = \
            ['neighbor_id_{}_{}'.format(idx, self.dependency_name)
             for idx in range(self.n_neighbor_limit)]

        if lammp_path is not None and os.path.exists(lammp_path):
            X, Bds = read_lammps_dump(lammp_path)
        if self.types_atomic_number_list is not None:
            X[self.type_col] = X[self.type_col].apply(
                lambda x: self.types_atomic_number_list[x-1])

        # define print verbose
        if self.verbose > 0:
            vr = VerboseReporter(self.context, total_stage=1,
                                 verbose=1, max_verbose_mod=10000)
            vr.init(total_epoch=len(X), start_epoch=0,
                    init_msg='Calculating VolumeAreaInterstice features.',
                    epoch_name='Atoms', stage=1)

        integrated_df = X.join(self.dependence_df)
        feature_lists = list()
        for idx, row in integrated_df.iterrows():
            neighbors_type = list()
            neighbors_coords = list()
            for x in row[neighbor_id_cols]:
                if x > 0:
                    neighbors_type.append(integrated_df.loc[x][self.type_col])
                    neighbors_coords.append(integrated_df.loc[x][self.coords_cols])
                else:
                    continue
            pos_ = PackingOfSite(self.pbc, Bds,
                                 row[self.type_col], row[self.coords_cols],
                                 neighbors_type, neighbors_coords,
                                 radii=self.radii, radius_type=self.radius_type)
            if len(neighbors_type) < 4:
                feature_lists.append([0] * len(self.get_feature_names()))
            else:
                feature_lists.append([pos_.atomic_packing_efficiency()])

        atomic_packing_efficiency_df = pd.DataFrame(
            feature_lists, index=X.index, columns=self.get_feature_names())

        if self.tmp_save:
            self.context.save_featurizer_as_dataframe(
                output_df=atomic_packing_efficiency_df,
                name='{}_{}_{}_atomic_packing_efficiency'.format(
                    self.category, self.dependency_name, self.radius_type))

        return atomic_packing_efficiency_df

    def get_feature_names(self):
        feature_names = ['{}_atomic_packing_efficiency {}'.format(
            self.radius_type.replace("_radius", ""), self.dependency_name)]
        return feature_names


class AtomicPackingEfficiency(BaseSRO):
    """
    Laws, K. J., Miracle, D. B. & Ferry, M. A predictive structural model for
    bulk metallic glasses. Nat. Commun. 6, 8123 (2015).
    """
    def __init__(self, pbc, context=None,
                 coords_cols=None, type_col='type',
                 types_atomic_number_list=None,
                 n_neighbor_limit=80, dependency="voro", tmp_save=True,
                 radii=None, radius_type="miracle_radius",
                 verbose=0, **nn_kwargs):
        super(AtomicPackingEfficiency, self).__init__(
            tmp_save=tmp_save, context=context,
            dependency=dependency, **nn_kwargs)
        self.pbc = pbc
        self.coords_cols = coords_cols \
            if coords_cols is not None else ['x', 'y', 'z']
        self.type_col = type_col
        self.types_atomic_number_list = types_atomic_number_list
        self.n_neighbor_limit = n_neighbor_limit
        self.voro_depend_cols = ['n_neighbors_voro'] + \
                                ['neighbor_id_{}_voro'.format(idx)
                                 for idx in range(n_neighbor_limit)]
        self.dist_depend_cols = None
        self.radii = load_radii() if radii is None else radii
        self.radius_type = radius_type
        self.verbose = verbose
        self.area_list = list()
        self.area_interstice_list = list()
        self.volume_list = list()
        self.volume_interstice_list = list()

    def transform(self, X, Bds=None, lammp_path=None):
        neighbor_id_cols = \
            ['neighbor_id_{}_{}'.format(idx, self.dependency_name)
             for idx in range(self.n_neighbor_limit)]

        if lammp_path is not None and os.path.exists(lammp_path):
            X, Bds = read_lammps_dump(lammp_path)
        if self.types_atomic_number_list is not None:
            X[self.type_col] = X[self.type_col].apply(
                lambda x: self.types_atomic_number_list[x-1])

        # define print verbose
        if self.verbose > 0:
            vr = VerboseReporter(self.context, total_stage=1,
                                 verbose=1, max_verbose_mod=10000)
            vr.init(total_epoch=len(X), start_epoch=0,
                    init_msg='Calculating VolumeAreaInterstice features.',
                    epoch_name='Atoms', stage=1)

        integrated_df = X.join(self.dependence_df)
        feature_lists = list()
        for idx, row in integrated_df.iterrows():
            neighbors_type = list()
            neighbors_coords = list()
            for x in row[neighbor_id_cols]:
                if x > 0:
                    neighbors_type.append(integrated_df.loc[x][self.type_col])
                    neighbors_coords.append(integrated_df.loc[x][self.coords_cols])
                else:
                    continue
            pos_ = PackingOfSite(self.pbc, Bds,
                                 row[self.type_col], row[self.coords_cols],
                                 neighbors_type, neighbors_coords,
                                 radii=self.radii, radius_type=self.radius_type)
            if len(neighbors_type) < 4:
                feature_lists.append([0] * len(self.get_feature_names()))
            else:
                feature_lists.append([pos_.glass_packing_efficiency()])

        glass_packing_efficiency_df = pd.DataFrame(
            feature_lists, index=X.index, columns=self.get_feature_names())

        if self.tmp_save:
            self.context.save_featurizer_as_dataframe(
                output_df=glass_packing_efficiency_df,
                name='{}_{}_{}_glass_packing_efficiency'.format(
                    self.category, self.dependency_name, self.radius_type))

        return glass_packing_efficiency_df

    def get_feature_names(self):
        feature_names = ['{}_glass_packing_efficiency {}'.format(
            self.radius_type.replace("_radius", ""), self.dependency_name)]
        return feature_names


class CN(BaseSRO):
    def __init__(self, dependency="voro", tmp_save=True,
                 context=None, remain_stat=False, **nn_kwargs):
        """

        Args:
            dependency: (object or string) default: "voro"
                if object, it can be "VoroNN()" or "DistanceNN()",
                if string, it can be "voro" or "distance"
        """
        super(CN, self).__init__(tmp_save=tmp_save,
                                 context=context,
                                 dependency=dependency,
                                 remain_stat=remain_stat,
                                 **nn_kwargs)
        self.voro_depend_cols = ['n_neighbors_voro']
        self.dist_depend_cols = ['n_neighbors_dist']

    def transform(self, X=None):
        integrated_df = X.join(self.dependence_df)
        cn_list = np.zeros(len(integrated_df))

        neighbor_col = 'n_neighbors_{}'.format(self.dependency_name)
        cn_list = \
            voronoi_stats.cn_voro(cn_list, integrated_df[neighbor_col].values,
                                  n_atoms=len(integrated_df))
        cn_list_df = pd.DataFrame(cn_list,
                                  index=integrated_df.index,
                                  columns=self.get_feature_names())

        if self.tmp_save:
            name = '{}_{}_cn'.format(self.category, self.dependency_name)
            self.context.save_featurizer_as_dataframe(output_df=cn_list_df,
                                                      name=name)

        return cn_list_df

    def get_feature_names(self):
        feature_names = ['CN {}'.format(self.dependency_name)]
        return feature_names


class VoroIndex(BaseSRO):
    def __init__(self, n_neighbor_limit=80,
                 include_beyond_edge_max=True, dependency="voro",
                 tmp_save=True, context=None, remain_stat=False,
                 edge_min=3, edge_max=7, **nn_kwargs):
        """

        Args:
            dependency: (object or string) default: "voro"
                if object, it can be "VoroNN()" or "DistanceNN()",
                if string, it can be "voro" or "distance"
        """
        super(VoroIndex, self).__init__(tmp_save=tmp_save,
                                        context=context,
                                        dependency=dependency,
                                        remain_stat=remain_stat,
                                        **nn_kwargs)
        self.n_neighbor_limit = n_neighbor_limit
        self.include_beyond_edge_max = include_beyond_edge_max
        self.edge_min = edge_min
        self.edge_max = edge_max
        self.voro_depend_cols = ['n_neighbors_voro'] + \
                                ['neighbor_edge_{}_voro'.format(edge)
                                 for edge in range(edge_min, edge_max + 1)]
        self.dist_depend_cols = None

    def transform(self, X=None):
        integrated_df = X.join(self.dependence_df)
        n_atoms = len(integrated_df)
        columns = integrated_df.columns
        edge_cols = [col for col in columns if col.startswith('neighbor_edge_')]
        edge_num = self.edge_max - self.edge_min + 1

        voronoi_index_list = np.zeros((n_atoms, edge_num))

        voro_index_list = \
            voronoi_stats.voronoi_index(voronoi_index_list,
                                        integrated_df['n_neighbors_voro'].values,
                                        integrated_df[edge_cols].values,
                                        self.edge_min, self.edge_max,
                                        self.include_beyond_edge_max,
                                        n_atoms=n_atoms,
                                        n_neighbor_limit=self.n_neighbor_limit)

        voro_index_df = pd.DataFrame(voro_index_list,
                                     index=integrated_df.index,
                                     columns=self.get_feature_names())
        if self.tmp_save:
            self.context.save_featurizer_as_dataframe(output_df=voro_index_df,
                                                      name='{}_voro_index'.format(self.category))

        return voro_index_df

    def get_feature_names(self):
        feature_names = ['Voronoi idx{} voro'.format(edge)
                         for edge in range(self.edge_min,
                                           self.edge_max + 1)]
        return feature_names


class CharacterMotif(BaseSRO):
    def __init__(self, n_neighbor_limit=80,
                 include_beyond_edge_max=True, dependency="voro",
                 edge_min=3, target_voro_idx=None, frank_kasper=1,
                 tmp_save=True, context=None, **nn_kwargs):
        super(CharacterMotif, self).__init__(tmp_save=tmp_save,
                                             context=context,
                                             dependency=dependency,
                                             **nn_kwargs)
        self.n_neighbor_limit = n_neighbor_limit
        self.include_beyond_edge_max = include_beyond_edge_max
        if target_voro_idx is None:
            self.target_voro_idx = np.array([[0, 0, 12, 0, 0],
                                             [0, 0, 12, 4, 0]],
                                            dtype=np.float128)
        self.frank_kasper = frank_kasper
        self.voro_depend_cols = ['Voronoi idx5 voro']
        self.dist_depend_cols = None
        self.edge_min = edge_min

    def fit(self, X=None, dependence_df=None):
        dependence_df = dependence_df if dependence_df is not None else X
        self._dependency = self.check_dependency(dependence_df)

        # This class is only dependent on 'Voronoi idx*' col, so if dataframe
        # has this col, this class don't need calculate it again.
        if self._dependency is None:
            voro_index = \
                VoroIndex(n_neighbor_limit=self.n_neighbor_limit,
                          include_beyond_edge_max=self.include_beyond_edge_max,
                          dependency=self.dependency, tmp_save=False,
                          context=self.context)
            self.dependence_df = voro_index.fit_transform(dependence_df)
        return self

    def transform(self, X=None):
        integrated_df = X.join(self.dependence_df)
        n_atoms = len(integrated_df)
        columns = integrated_df.columns
        voro_index_cols = [col for col in columns
                           if col.startswith("Voronoi idx")]

        motif_one_hot = np.zeros((n_atoms,
                                  len(self.target_voro_idx) + self.frank_kasper))

        motif_one_hot = \
            voronoi_stats.character_motif(motif_one_hot,
                                          integrated_df[voro_index_cols].values,
                                          self.edge_min, self.target_voro_idx,
                                          self.frank_kasper, n_atoms=n_atoms)
        motif_one_hot_array = np.array(motif_one_hot)
        is_120_124 = motif_one_hot_array[:, 0] | motif_one_hot_array[:, 1]
        # print(motif_one_hot_array.shape)
        # print(is_120_124.shape)
        motif_one_hot_array = np.append(motif_one_hot_array,
                                        np.array([is_120_124]).T, axis=1)
        character_motif_df = pd.DataFrame(motif_one_hot_array,
                                          index=integrated_df.index,
                                          columns=self.get_feature_names())

        if self.tmp_save:
            self.context.save_featurizer_as_dataframe(
                output_df=character_motif_df, name='{}_character_motif'.format(self.category))

        return character_motif_df

    def get_feature_names(self):
        feature_names = ['is <0,0,12,0,0> voro', 'is <0,0,12,4,0> voro'] + \
                        ["_".join(map(str, v)) + " voro"
                         for v in self.target_voro_idx[2:]] + \
                        ['is polytetrahedral voro', 'is <0,0,12,0/4,0> voro']
        return feature_names


class IFoldSymmetry(BaseSRO):
    def __init__(self, n_neighbor_limit=80,
                 include_beyond_edge_max=True,
                 atoms_df=None, dependency="voro",
                 tmp_save=True, context=None, remain_stat=False,
                 edge_min=3, edge_max=7, **nn_kwargs):
        super(IFoldSymmetry, self).__init__(tmp_save=tmp_save,
                                            context=context,
                                            dependency=dependency,
                                            atoms_df=atoms_df,
                                            remain_stat=remain_stat,
                                            **nn_kwargs)
        self.n_neighbor_limit = n_neighbor_limit
        self.include_beyond_edge_max = include_beyond_edge_max
        self.edge_min = edge_min
        self.edge_max = edge_max
        self.voro_depend_cols = ['n_neighbors_voro'] + \
                                ['neighbor_edge_{}_voro'.format(edge)
                                 for edge in range(edge_min, edge_max + 1)]
        self.dist_depend_cols = None

    def transform(self, X=None):
        integrated_df = X.join(self.dependence_df)
        n_atoms = len(integrated_df)
        columns = integrated_df.columns
        edge_cols = [col for col in columns if col.startswith('neighbor_edge_')]
        edge_num = self.edge_max - self.edge_min + 1
        i_symm_list = np.zeros((n_atoms, edge_num))

        i_symm_list = \
            voronoi_stats.i_fold_symmetry(i_symm_list,
                                          integrated_df['n_neighbors_voro'].values,
                                          integrated_df[edge_cols].values,
                                          self.edge_min, self.edge_max,
                                          self.include_beyond_edge_max,
                                          n_atoms=n_atoms,
                                          n_neighbor_limit=
                                          self.n_neighbor_limit)

        i_symm_df = pd.DataFrame(i_symm_list,
                                 index=integrated_df.index,
                                 columns=self.get_feature_names())
        if self.tmp_save:
            self.context.save_featurizer_as_dataframe(output_df=i_symm_df,
                                                      name='{}_i_fold_symmetry'.format(self.category))

        return i_symm_df

    def get_feature_names(self):
        feature_names = ['{}-fold symm idx voro'.format(edge)
                         for edge in range(self.edge_min, self.edge_max+1)]
        return feature_names


class AreaWtIFoldSymmetry(BaseSRO):
    def __init__(self, n_neighbor_limit=80,
                 include_beyond_edge_max=True,
                 atoms_df=None, dependency="voro",
                 tmp_save=True, context=None, remain_stat=False,
                 edge_min=3, edge_max=7, **nn_kwargs):
        super(AreaWtIFoldSymmetry, self).__init__(tmp_save=tmp_save,
                                                  context=context,
                                                  dependency=dependency,
                                                  atoms_df=atoms_df,
                                                  remain_stat=remain_stat,
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
        self.dist_depend_cols = None

    def transform(self, X=None):
        integrated_df = X.join(self.dependence_df)
        n_atoms = len(integrated_df)
        columns = integrated_df.columns
        edge_cols = [col for col in columns if
                     col.startswith('neighbor_edge_')]
        area_cols = [col for col in columns if
                     col.startswith('neighbor_area_')]
        edge_num = self.edge_max - self.edge_min + 1
        area_wt_i_symm_list = np.zeros((n_atoms, edge_num))

        area_wt_i_symm_list = \
            voronoi_stats.area_wt_i_fold_symmetry(area_wt_i_symm_list,
                                                  integrated_df['n_neighbors_voro'].values,
                                                  integrated_df[edge_cols].values,
                                                  integrated_df[area_cols].values.astype(
                                                      np.float128),
                                                  self.edge_min,
                                                  self.edge_max,
                                                  self.include_beyond_edge_max,
                                                  n_atoms=n_atoms,
                                                  n_neighbor_limit=
                                                  self.n_neighbor_limit)

        area_wt_i_symm_df = pd.DataFrame(area_wt_i_symm_list,
                                         index=integrated_df.index,
                                         columns=self.get_feature_names())
        if self.tmp_save:
            self.context.save_featurizer_as_dataframe(
                output_df=area_wt_i_symm_df, name='{}_area_wt_i_fold_symmetry'.format(self.category))

        return area_wt_i_symm_df

    def get_feature_names(self):
        feature_names = ['Area_wt {}-fold symm idx voro'.format(edge)
                         for edge in range(self.edge_min, self.edge_max + 1)]
        return feature_names


class VolWtIFoldSymmetry(BaseSRO):
    def __init__(self, n_neighbor_limit=80,
                 include_beyond_edge_max=True,
                 atoms_df=None, dependency="voro",
                 tmp_save=True, context=None, remain_stat=False,
                 edge_min=3, edge_max=7, **nn_kwargs):
        super(VolWtIFoldSymmetry, self).__init__(tmp_save=tmp_save,
                                            context=context,
                                            dependency=dependency,
                                            atoms_df=atoms_df,
                                            remain_stat=remain_stat,
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
        self.dist_depend_cols = None

    def transform(self, X=None):
        integrated_df = X.join(self.dependence_df)
        n_atoms = len(integrated_df)
        columns = integrated_df.columns
        edge_cols = [col for col in columns if
                     col.startswith('neighbor_edge_')]
        vol_cols = [col for col in columns if
                     col.startswith('neighbor_vol_')]
        edge_num = self.edge_max - self.edge_min + 1
        vol_wt_i_symm_list = np.zeros((n_atoms, edge_num))
        vol_wt_i_symm_list = \
            voronoi_stats.vol_wt_i_fold_symmetry(vol_wt_i_symm_list,
                                                 integrated_df['n_neighbors_voro'].values,
                                                 integrated_df[edge_cols].values,
                                                 integrated_df[vol_cols].values.astype(
                                                     np.float128),
                                                 self.edge_min,
                                                 self.edge_max,
                                                 self.include_beyond_edge_max,
                                                 n_atoms=n_atoms,
                                                 n_neighbor_limit=
                                                 self.n_neighbor_limit)

        vol_wt_i_symm_df = pd.DataFrame(vol_wt_i_symm_list,
                                         index=integrated_df.index,
                                         columns=self.get_feature_names())
        if self.tmp_save:
            self.context.save_featurizer_as_dataframe(
                output_df=vol_wt_i_symm_df, name='{}_vol_wt_i_fold_symmetry'.format(self.category))

        return vol_wt_i_symm_df

    def get_feature_names(self):
        feature_names = ['Vol_wt {}-fold symm idx voro'.format(edge)
                         for edge in range(self.edge_min, self.edge_max + 1)]
        return feature_names


class VoroAreaStats(BaseSRO):
    def __init__(self, n_neighbor_limit=80,
                 atoms_df=None, dependency="voro",
                 tmp_save=True, context=None, remain_stat=False, **nn_kwargs):
        super(VoroAreaStats, self).__init__(tmp_save=tmp_save,
                                            context=context,
                                            dependency=dependency,
                                            atoms_df=atoms_df,
                                            remain_stat=remain_stat,
                                            **nn_kwargs)
        self.n_neighbor_limit = n_neighbor_limit
        self.voro_depend_cols = ['n_neighbors_voro'] + \
                                ['neighbor_area_5_voro']
        self.stats = ['mean', 'std', 'min', 'max']
        self.dist_depend_cols = None

    def transform(self, X=None):
        integrated_df = X.join(self.dependence_df)
        n_atoms = len(integrated_df)
        columns = integrated_df.columns
        area_cols = [col for col in columns if
                     col.startswith('neighbor_area_')]
        area_stats = np.zeros((n_atoms, len(self.stats) + 1))

        area_stats = \
            voronoi_stats.voronoi_area_stats(area_stats,
                                             integrated_df['n_neighbors_voro'].values,
                                             integrated_df[area_cols].values.astype(
                                                 np.float128),
                                             n_atoms=n_atoms,
                                             n_neighbor_limit=
                                             self.n_neighbor_limit)

        area_stats_df = pd.DataFrame(area_stats, index=integrated_df.index,
                                     columns=self.get_feature_names())
        if self.tmp_save:
            self.context.save_featurizer_as_dataframe(
                output_df=area_stats_df, name='{}_voronoi_area_stats'.format(self.category))

        return area_stats_df

    def get_feature_names(self):
        feature_names = ['Voronoi area voro'] + \
                        ['Facet area {} voro'.format(stat)
                         for stat in self.stats]
        return feature_names


class VoroAreaStatsSeparate(BaseSRO):
    def __init__(self, n_neighbor_limit=80, include_beyond_edge_max=True,
                 atoms_df=None, dependency="voro", edge_min=3, edge_max=7,
                 tmp_save=True, context=None, remain_stat=False, **nn_kwargs):
        super(VoroAreaStatsSeparate, self).__init__(tmp_save=tmp_save,
                                                    context=context,
                                                    dependency=dependency,
                                                    atoms_df=atoms_df,
                                                    remain_stat=remain_stat,
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
        self.dist_depend_cols = None

    def transform(self, X=None):
        integrated_df = X.join(self.dependence_df)
        n_atoms = len(integrated_df)
        columns = integrated_df.columns
        edge_cols = [col for col in columns if
                     col.startswith('neighbor_edge_')]
        area_cols = [col for col in columns if
                     col.startswith('neighbor_area_')]
        area_stats_separate = np.zeros((n_atoms,
                                        self.edge_num * len(self.stats)))

        area_stats_separate = \
            voronoi_stats.voronoi_area_stats_separate(
                area_stats_separate, integrated_df['n_neighbors_voro'].values,
                integrated_df[edge_cols].values,
                integrated_df[area_cols].values.astype(np.float128),
                self.edge_min, self.edge_max,
                self.include_beyond_edge_max,
                n_atoms=n_atoms,
                n_neighbor_limit=self.n_neighbor_limit)

        area_stats_separate_df = pd.DataFrame(area_stats_separate, index=integrated_df.index,
                                     columns=self.get_feature_names())
        if self.tmp_save:
            self.context.save_featurizer_as_dataframe(
                output_df=area_stats_separate_df,
                name='{}_voro_area_stats_separate'.format(self.category))

        return area_stats_separate_df

    def get_feature_names(self):
        feature_names = ['{}-edged area {} voro'.format(edge, stat)
                         for edge in range(self.edge_min, self.edge_max + 1)
                         for stat in self.stats]
        return feature_names


class VoroVolStats(BaseSRO):
    def __init__(self, n_neighbor_limit=80,
                 atoms_df=None, dependency="voro",
                 tmp_save=True, context=None, remain_stat=False, **nn_kwargs):
        super(VoroVolStats, self).__init__(tmp_save=tmp_save,
                                            context=context,
                                            dependency=dependency,
                                            atoms_df=atoms_df,
                                            remain_stat=remain_stat,
                                            **nn_kwargs)
        self.n_neighbor_limit = n_neighbor_limit
        self.voro_depend_cols = ['n_neighbors_voro'] + \
                                ['neighbor_vol_5_voro']
        self.stats = ['mean', 'std', 'min', 'max']
        self.dist_depend_cols = None

    def transform(self, X=None):
        integrated_df = X.join(self.dependence_df)
        n_atoms = len(integrated_df)
        columns = integrated_df.columns
        vol_cols = [col for col in columns if
                     col.startswith('neighbor_vol_')]
        vol_stats = np.zeros((n_atoms, len(self.stats) + 1))

        vol_stats = \
            voronoi_stats.voronoi_vol_stats(vol_stats,
                                            integrated_df['n_neighbors_voro'].values,
                                            integrated_df[vol_cols].values.astype(
                                                np.float128),
                                            n_atoms=n_atoms,
                                            n_neighbor_limit=
                                            self.n_neighbor_limit)

        vol_stats_df = pd.DataFrame(vol_stats, index=integrated_df.index,
                                    columns=self.get_feature_names())
        if self.tmp_save:
            self.context.save_featurizer_as_dataframe(
                output_df=vol_stats_df, name='{}_voronoi_vol_stats'.format(self.category))

        return vol_stats_df

    def get_feature_names(self):
        feature_names = ['Voronoi vol voro'] + \
                        ['Sub-polyhedra vol {} voro'.format(stat)
                         for stat in self.stats]
        return feature_names


class VoroVolStatsSeparate(BaseSRO):
    def __init__(self, n_neighbor_limit=80, include_beyond_edge_max=True,
                 atoms_df=None, dependency="voro", edge_min=3, edge_max=7,
                 tmp_save=True, context=None, remain_stat=False, **nn_kwargs):
        super(VoroVolStatsSeparate, self).__init__(tmp_save=tmp_save,
                                            context=context,
                                            dependency=dependency,
                                            atoms_df=atoms_df,
                                            remain_stat=remain_stat,
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
        self.dist_depend_cols = None

    def transform(self, X=None):
        integrated_df = X.join(self.dependence_df)
        n_atoms = len(integrated_df)
        columns = integrated_df.columns
        edge_cols = [col for col in columns if col.startswith('neighbor_edge_')]
        vol_cols = [col for col in columns if col.startswith('neighbor_vol_')]
        vol_stats_separate = np.zeros((n_atoms,
                                       self.edge_num * len(self.stats)))

        vol_stats_separate = \
            voronoi_stats.voronoi_vol_stats_separate(
                vol_stats_separate, integrated_df['n_neighbors_voro'].values,
                integrated_df[edge_cols].values,
                integrated_df[vol_cols].values.astype(np.float128),
                self.edge_min, self.edge_max,
                self.include_beyond_edge_max,
                n_atoms=n_atoms,
                n_neighbor_limit=self.n_neighbor_limit)

        vol_stats_separate_df = pd.DataFrame(vol_stats_separate,
                                             index=integrated_df.index,
                                             columns=self.get_feature_names())
        if self.tmp_save:
            self.context.save_featurizer_as_dataframe(
                output_df=vol_stats_separate_df, name='{}_voro_vol_stats_separate'.format(self.category))

        return vol_stats_separate_df

    def get_feature_names(self):
        feature_names = ['{}-edged vol {} voro'.format(edge, stat)
                         for edge in range(self.edge_min, self.edge_max + 1)
                         for stat in self.stats]
        return feature_names


class DistStats(BaseSRO):
    def __init__(self, dist_type='distance', n_neighbor_limit=80,
                 atoms_df=None, dependency="voro",
                 tmp_save=True, context=None, remain_stat=False, **nn_kwargs):
        super(DistStats, self).__init__(tmp_save=tmp_save,
                                        context=context,
                                        dependency=dependency,
                                        atoms_df=atoms_df,
                                        remain_stat=remain_stat,
                                        **nn_kwargs)
        self.dist_type = dist_type
        self.n_neighbor_limit = n_neighbor_limit
        self.voro_depend_cols = ['n_neighbors_voro'] + \
                                ['neighbor_{}_5_voro'.format(dist_type)]
        self.stats = ['sum', 'mean', 'std', 'min', 'max']
        self.dist_depend_cols = ['n_neighbors_dist'] + \
                                 ['neighbor_{}_5_dist'.format(dist_type)]

    def transform(self, X=None):
        integrated_df = X.join(self.dependence_df)
        n_atoms = len(integrated_df)
        columns = integrated_df.columns
        dist_cols = [col for col in columns if
                     col.startswith('neighbor_{}_'.format(self.dist_type))]
        dist_stats = np.zeros((n_atoms, len(self.stats)))

        dist_stats = \
            voronoi_stats.voronoi_distance_stats(dist_stats,
                                             integrated_df['n_neighbors_{}'.format(
                                                 self.dependency_name)].values,
                                             integrated_df[dist_cols].values,
                                             n_atoms=n_atoms,
                                             n_neighbor_limit=
                                             self.n_neighbor_limit)
        dist_stats_df = pd.DataFrame(dist_stats, index=integrated_df.index,
                                     columns=self.get_feature_names())
        if self.tmp_save:
            self.context.save_featurizer_as_dataframe(
                output_df=dist_stats_df,
                name='{}_{}_{}_stats'.format(self.category, self.dependency_name, self.dist_type))

        return dist_stats_df

    def get_feature_names(self):
        feature_names = ['{} {} {}'.format(self.dist_type, stat,
                                           self.dependency_name)
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
                 remain_stat=False, **nn_kwargs):
        super(BOOP, self).__init__(tmp_save=tmp_save,
                                   context=context,
                                   dependency=dependency,
                                   atoms_df=atoms_df,
                                   remain_stat=remain_stat,
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
        self.dist_depend_cols = ['n_neighbors_dist'] + \
                                 ['neighbor_id_{}_dist'.format(idx)
                                  for idx in range(n_neighbor_limit)]
        self.bq_tags = ['4', '6', '8', '10']

    def transform(self, X=None):
        integrated_df = X.join(self.dependence_df)
        n_atoms = len(integrated_df)
        neighbor_col = ['n_neighbors_{}'.format(self.dependency_name)]
        id_cols = ['neighbor_id_{}_{}'.format(idx, self.dependency_name)
                   for idx in range(self.n_neighbor_limit)]

        Ql = np.zeros((n_atoms, 4), dtype=np.float128)
        Wlbar = np.zeros((n_atoms, 4), dtype=np.float128)
        coarse_Ql = np.zeros((n_atoms, 4), dtype=np.float128)
        coarse_Wlbar = np.zeros((n_atoms, 4), dtype=np.float128)
        Ql, Wlbar, coarse_Ql, coarse_Wlbar = \
            boop.calculate_boop(
                self.atom_coords.astype(np.float128),
                self.pbc, np.array(self.Bds, dtype=np.float128),
                integrated_df[neighbor_col].values,
                integrated_df[id_cols].values,
                self.low_order, self.higher_order, self.coarse_lower_order,
                self.coarse_higher_order, Ql, Wlbar, coarse_Ql, coarse_Wlbar,
                n_atoms=n_atoms, n_neighbor_limit=self.n_neighbor_limit)
        concat_array = np.append(Ql, Wlbar, axis=1)
        concat_array = np.append(concat_array, coarse_Ql, axis=1)
        concat_array = np.append(concat_array, coarse_Wlbar, axis=1)

        boop_df = pd.DataFrame(concat_array, index=integrated_df.index,
                               columns=self.get_feature_names())
        if self.tmp_save:
            self.context.save_featurizer_as_dataframe(
                output_df=boop_df, name='{}_boop_{}'.format(self.category, self.dependency_name))

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
