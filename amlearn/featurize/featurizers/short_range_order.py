import os
from abc import ABCMeta
from collections import defaultdict
from functools import lru_cache
from math import pi
import numpy as np
import pandas as pd
import six
from copy import copy
from amlearn.featurize.featurizers.nearest_neighbor import VoroNN, DistanceNN

from scipy.spatial.qhull import ConvexHull
from amlearn.featurize.base import load_radii, BaseFeaturize
from amlearn.utils.data import read_imd, read_lammps_dump, get_isometric_lists, list_like
from amlearn.utils.packing import solid_angle, \
    triangular_angle, calc_stats, triangle_area, tetra_volume
from amlearn.utils.verbose import VerboseReporter

from amlearn.utils.data import calc_neighbor_coords

try:
    from amlearn.featurize.featurizers.src import voronoi_stats, boop
except Exception:
    print("import fortran file voronoi_stats/boop error!\n")

module_dir = os.path.dirname(os.path.abspath(__file__))


class PackingOfSite(object):
    def __init__(self, pbc, Bds, atom_type, coords, neighbors_type,
                 neighbors_coords, radii=None, radius_type="miracle_radius"):
        self.pbc = pbc
        self.Bds = Bds
        self.atom_type = atom_type
        self.coords = coords.astype(float)
        self.neighbors_type = neighbors_type
        self.neighbors_coords = neighbors_coords
        self.radii = load_radii() if radii is None else radii
        self.radius_type = radius_type

    def nn_coords(self):
        if not hasattr(self, 'nn_coords_'):
            self.nn_coords_ = [calc_neighbor_coords(self.coords,
                                                    neighbor_coords,
                                                    self.Bds, self.pbc)
                               for neighbor_coords in self.neighbors_coords]
        return self.nn_coords_

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
        self.calculate_hull_facet_angles()

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
        self.calculate_hull_tetra_angles()
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
            # surface_coords = surface_coords.astype(float)
            # self.coords = self.coords.astype(float)
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
            self.calculate_hull_tetra_angles()

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
        self.combine_neighbor_solid_angles()
        packed_volume = 4/3 * pi * pow(self.radii[str(self.atom_type)]
                                       [self.radius_type], 3)
        for neighbor_type, solid_angle_ in zip(self.neighbors_type,
                                               self.neighbors_solid_angle_):
            if solid_angle_ == 0:
                continue
            packed_volume += solid_angle_ * 1/3 * pow(
                self.radii[str(int(neighbor_type))][self.radius_type], 3)
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


@lru_cache(maxsize=5)
def get_nn_instance(dependent_name, backend, **nn_kwargs):
    """
    Get Nearest Neighbor instance, for most SRO depends on the same Nearest
    Neighbor instance, we cache the most recently used Nearest Neighbor
    instance by lru_cache.

    Args:
        dependent_name (str): "voro"/"voronoi" or "dist"/"distance".
        backend (Backend): Amlearn Backend object, to prepare amlearn needed
            paths and define the common amlearn's load/save method.
        nn_kwargs: Nearest Neighbor class's keyword arguments.

    Returns:
        dependent_class (object): Nearest Neighbor instance.

    """
    if dependent_name == "voro":
        dependent_class = VoroNN(backend=backend, **nn_kwargs)
    elif dependent_name == "dist":
        dependent_class = DistanceNN(backend=backend, **nn_kwargs)
    else:
        raise ValueError('dependent name {} is unknown, Possible values '
                         'are {}'.format(dependent_name,
                                         '[voro, voronoi, '
                                         'dist, distance]'))
    return dependent_class


class BaseSRO(six.with_metaclass(ABCMeta, BaseFeaturize)):
    """
    Base class of Short Range Order(SRO) Featurizer, most SRO Featurizer
    depends on the output of the Nearest Neighbor class, so this base class
    implements dependency checking. For most SRO depends on the same Nearest
    Neighbor instance, we cache the most recently used Nearest Neighbor
    instance by lru_cache.

    Args:
        save (Boolean): save file or not.
        backend (object): Amlearn Backend object, to prepare amlearn needed
            paths and define the common amlearn's load/save method.
        dependent_class (object or str):
            if object, it can be "VoroNN()" or "DistanceNN()";
            if str, it can be "voro"/"voronoi" or "dist"/"distance"
        nn_kwargs: Nearest Neighbor class's keyword arguments.
    """

    def __init__(self, save=True, backend=None, dependent_class=None,
                 verbose=1, output_path=None, **nn_kwargs):
        super(BaseSRO, self).__init__(save=save,
                                      verbose=verbose,
                                      backend=backend,
                                      output_path=output_path)
        self.calculated_X = None
        if dependent_class is None:
            self.dependent_class_ = None
            self.dependent_name_ = 'voro'
        elif isinstance(dependent_class, type):
            self.dependent_class_ = dependent_class
            self.dependent_name_ = dependent_class.__class__.__name__.lower()[:4]
        elif isinstance(dependent_class, str):
            self.dependent_name_ = dependent_class[:4]
            self.dependent_class_ = get_nn_instance(self.dependent_name_,
                                                    backend, **nn_kwargs)
        else:
            raise ValueError(
                'dependent_class {} is unknown, Possible values are {} or '
                'voro/dist object.'.format(dependent_class,
                                           '[voro, voronoi, dist, distance]'))
        self.neighbor_num_col = \
            'neighbor_num_{}'.format(self.dependent_name_)
        self.neighbor_ids_col = \
            'neighbor_ids_{}'.format(self.dependent_name_)
        self.neighbor_dists_col = \
            'neighbor_dists_{}'.format(self.dependent_name_)
        self.neighbor_areas_col = \
            'neighbor_areas_{}'.format(self.dependent_name_)
        self.neighbor_vols_col = \
            'neighbor_vols_{}'.format(self.dependent_name_)
        self.neighbor_edges_col = \
            'neighbor_edges_{}'.format(self.dependent_name_)

    def fit(self, X=None):
        self.dependent_class_ = self.check_dependency(X)
        if self.dependent_class_:
            self.backend.context.logger_.info(
                "Input X don't have it's dependent columns, "
                "now calculate it automatically")
            self.calculated_X = self.dependent_class_.fit_transform(X)
        return self

    @property
    def category(self):
        return 'sro'


class BaseInterstice(six.with_metaclass(ABCMeta, BaseSRO)):
    def __init__(self, backend=None, dependent_class="voro", type_col='type',
                 type_to_atomic_number_list=None, neighbor_num_limit=80,
                 save=True, radii=None, radius_type="miracle_radius",
                 verbose=1, output_path=None, **nn_kwargs):
        """
        Args:
            lammps_df (DataFrame): Constructed from the output of lammps, which
                common columns is ['type', 'x', 'y', 'z'...] columns.
            Bds (list like): X, y, z boundaries.
            lammps_path (DataFrame): If lammps_df is None, then we automatically
                construct the DataFrame from lammps output path.

        """
        super(BaseInterstice, self).__init__(
            save=save, backend=backend, dependent_class=dependent_class,
            output_path=output_path, **nn_kwargs)
        self.type_col = type_col
        self.type_to_atomic_number_list = type_to_atomic_number_list
        self.neighbor_num_limit = neighbor_num_limit
        self.radii = load_radii() if radii is None else radii
        self.radius_type = radius_type
        self.verbose = verbose

    def fit(self, X=None, lammps_df=None, Bds=None, lammps_path=None):
        """
        Args:
            X (DataFrame): X can be a DataFrame which composed of partial
                columns of Nearest Neighbor class's output; or X can be the
                input of Nearest Neighbor class, which should contains
                ['type', 'x', 'y', 'z'...] columns, we will automatic call
                Nearest Neighbor class to calculate X's output by self.fit()
                method, then feed it as input to this transform() method.

        Returns:
            self (object): Interstice or Packing instance.
        """
        if lammps_df is None and lammps_path is not None \
                and os.path.exists(lammps_path):
            self.lammps_df, self.Bds = read_lammps_dump(lammps_path)
        else:
            self.lammps_df = copy(lammps_df)
            self.Bds = Bds
        if self.type_to_atomic_number_list is not None:
            self.lammps_df[self.type_col] = self.lammps_df[self.type_col].apply(
                lambda x: self.type_to_atomic_number_list[x-1])

        self.dependent_class_ = self.check_dependency(X)
        if self.dependent_class_:
            self.calculated_X = self.dependent_class_.fit_transform(X)
            self.calculated_X = self.calculated_X.join(self.lammps_df)
        return self

    @property
    def category(self):
        return 'interstice_sro'


class DistanceInterstice(BaseInterstice):
    def __init__(self, backend=None, dependent_class="voro", type_col='type',
                 type_to_atomic_number_list=None, neighbor_num_limit=80, 
                 save=True, radii=None, radius_type="miracle_radius", 
                 verbose=1, output_path=None, output_file_prefix=None,
                 **nn_kwargs):
        super(DistanceInterstice, self).__init__(
            save=save, backend=backend,
            dependent_class=dependent_class, type_col=type_col,
            type_to_atomic_number_list=type_to_atomic_number_list,
            neighbor_num_limit=neighbor_num_limit,
            radii=radii, radius_type = radius_type,
            verbose = verbose, output_path=output_path, **nn_kwargs)
        self.output_file_prefix = output_file_prefix \
            if output_file_prefix is not None \
            else 'feature_{}_{}_{}_distance_interstice'.format(self.category,
                                                       self.dependent_name_,
                                                       self.radius_type)
        self.dependent_cols_ = [self.neighbor_num_col,
                                self.neighbor_ids_col,
                                self.neighbor_dists_col]

    def transform(self, X):
        """
        Args:
            X (DataFrame): X can be a DataFrame which composed of partial
                columns of Nearest Neighbor class's output; or X can be the
                input of Nearest Neighbor class, which should contains
                ['type', 'x', 'y', 'z'...] columns, we will automatic call
                Nearest Neighbor class to calculate X's output by self.fit()
                method, then feed it as input to this transform() method.

        Returns:
            dist_interstice_df (DataFrame): Distance interstice DataFrame, which
                index is same as X's index, columns is
                [neighbor_dists_interstice_voro] or
                [neighbor_dists_interstice_dist] dependent on dependent_class.
        """
        X = X.join(self.lammps_df) \
            if self.calculated_X is None else self.calculated_X

        # define print verbose
        if self.verbose > 0:
            vr = VerboseReporter(self.backend, total_stage=1,
                                 verbose=self.verbose, max_verbose_mod=10000)
            vr.init(total_epoch=len(X), start_epoch=0,
                    init_msg='Calculating DistanceInterstice features.',
                    epoch_name='Atoms', stage=1)

        feature_lists = list()
        for idx, row in X.iterrows():
            neighbor_dist_interstice_list = list()
            for neighbor_id, neighbor_dist in zip(row[self.neighbor_ids_col],
                                                  row[self.neighbor_dists_col]):
                if neighbor_id > 0:
                    neighbor_dist_interstice_list.append(neighbor_dist / (
                            self.radii[str(int(X.loc[idx][self.type_col]))][
                                self.radius_type] +
                            self.radii[
                                str(int(X.loc[neighbor_id][self.type_col]))][
                                self.radius_type]))
                else:
                    continue
            feature_lists.append(calc_stats(neighbor_dist_interstice_list))
            if self.verbose > 0:
                vr.update(idx - 1)

        dist_interstice_df = \
            pd.DataFrame(feature_lists, columns=self.get_feature_names(),
                         index=X.index)
        if self.save:
            self.backend.save_featurizer_as_dataframe(
                output_df=dist_interstice_df, name=self.output_file_prefix)
        return dist_interstice_df

    def get_feature_names(self):
        stats = ['sum', 'mean', 'std', 'min', 'max']
        return ['neighbor_dists_interstice_{}_{}'.format(
            stat, self.dependent_name_) for stat in stats]


class VolumeAreaInterstice(BaseInterstice):
    def __init__(self, pbc=None, backend=None, dependent_class="voro",
                 coords_cols=None, type_col='type',
                 type_to_atomic_number_list=None,
                 neighbor_num_limit=80, save=True,
                 radii=None, radius_type="miracle_radius",
                 calc_volume_area='all', verbose=1,
                 volume_types=None, area_types=None,
                 output_path=None, output_file_prefix=None,
                 **nn_kwargs):
        """
        Args:
            volume_types (list like): Can be one or several of the arrays
                ["volume_interstice",
                 "fractional_volume_interstice_tetrahedra",
                 "fractional_volume_interstice_tetrahedra_avg",
                 "fractional_volume_interstice_center_v"];
                 default is : ["fractional_volume_interstice_tetrahedra"]

            area_types (list like): Can be one or several of the arrays
                ["area_interstice",
                "fractional_area_interstice_triangle",
                "fractional_area_interstice_triangle_avg",
                "fractional_area_interstice_center_slice_a"]
                default is : ["fractional_area_interstice_triangle"]

        """
        assert dependent_class == "voro" or dependent_class == "voronoi"
        super(VolumeAreaInterstice, self).__init__(
            save=save, backend=backend,
            dependent_class=dependent_class, type_col=type_col,
            type_to_atomic_number_list=type_to_atomic_number_list,
            neighbor_num_limit=neighbor_num_limit,
            radii=radii, radius_type = radius_type,
            verbose = verbose, output_path=output_path, **nn_kwargs)
        self.pbc = pbc if pbc is not None else [1, 1, 1]
        self.calc_volume_area = calc_volume_area
        self.coords_cols = coords_cols \
            if coords_cols is not None else ['x', 'y', 'z']
        self.area_list = list()
        self.area_interstice_list = list()
        self.volume_list = list()
        self.volume_interstice_list = list()
        self.dependent_cols_ = [self.neighbor_num_col, self.neighbor_ids_col]
        self.volume_types = \
            volume_types if isinstance(volume_types, list_like()) \
                else [volume_types] if volume_types is not None \
                else ['fractional_volume_interstice_tetrahedra']
        self.area_types = \
            area_types if isinstance(area_types, list_like()) \
                else [area_types] if area_types is not None \
                else ['fractional_area_interstice_triangle']
        self.output_file_prefix = output_file_prefix if output_file_prefix is not None \
            else 'feature_{}_{}_{}_volume_area_interstice'.format(
                    self.category, self.dependent_name_, self.radius_type)

    def transform(self, X):
        """
        Args:
            X (DataFrame): X can be a DataFrame which composed of partial
                columns of Nearest Neighbor class's output; or X can be the
                input of Nearest Neighbor class, which should contains
                ['type', 'x', 'y', 'z'...] columns, we will automatic call
                Nearest Neighbor class to calculate X's output by self.fit()
                method, then feed it as input to this transform() method.

        Returns:
            volume_area_interstice_df (DataFrame): Volume/Area interstice
                DataFrame, which index is same as X's index, see
                get_feature_names() method for column names.
        """

        X = X.join(self.lammps_df) \
            if self.calculated_X is None else self.calculated_X

        # define print verbose
        if self.verbose > 0:
            vr = VerboseReporter(self.backend, total_stage=1,
                                 verbose=self.verbose, max_verbose_mod=10000)
            vr.init(total_epoch=len(X), start_epoch=0,
                    init_msg='Calculating VolumeAreaInterstice features.',
                    epoch_name='Atoms', stage=1)

        feature_lists = list()
        for idx, row in X.iterrows():
            neighbor_type = list()
            neighbor_coords = list()
            for neighbor_id in row[self.neighbor_ids_col]:
                if neighbor_id > 0:
                    neighbor_type.append(X.loc[neighbor_id][self.type_col])
                    neighbor_coords.append(
                        X.loc[neighbor_id][self.coords_cols].astype(float))
                else:
                    continue
            pos_ = PackingOfSite(self.pbc, self.Bds, row[self.type_col],
                                 row[self.coords_cols].values.astype(float),
                                 neighbor_type, neighbor_coords,
                                 radii=self.radii, radius_type=self.radius_type)
            if len(neighbor_type) < 4:
                feature_lists.append([0] * len(self.get_feature_names()))
            else:
                feature_list = list()
                if self.calc_volume_area == 'volume' or \
                        self.calc_volume_area == 'all':
                    pos_.analyze_hull_tetra_interstice()
                    volume_interstice_list = pos_.volume_interstice_list_
                    volume_list = pos_.volume_list_
                    volume_total = pos_.convex_hull().volume
                    volume_interstice_original_array = \
                        np.array(volume_interstice_list)*np.array(volume_list)
                    center_volume = 4/3 * pi * pow(
                        pos_.radii[str(pos_.atom_type)][pos_.radius_type], 3)

                    for volume_type in self.volume_types:
                        # fractional volume_interstices in relative to the
                        # tetrahedra volume
                        if volume_type == \
                                "fractional_volume_interstice_tetrahedra":
                            feature_list.extend(
                                calc_stats(volume_interstice_list))
                        # original volume_interstices (in the units of volume)
                        elif volume_type == "volume_interstice":
                            feature_list.extend(
                                calc_stats(volume_interstice_original_array))
                        # fractional volume_interstices in relative to the
                        # entire volume
                        elif volume_type == \
                                "fractional_volume_interstice_tetrahedra_avg":
                            feature_list.extend(
                                calc_stats(volume_interstice_original_array /
                                           volume_total * len(volume_list)))
                        # fractional volume_interstices in relative to the
                        # center atom volume
                        elif volume_type == \
                                "fractional_volume_interstice_center_v":
                            feature_list.extend(
                                calc_stats(volume_interstice_original_array /
                                           center_volume))

                if self.calc_volume_area == 'area' or \
                        self.calc_volume_area == 'all':
                    pos_.analyze_hull_facet_interstice()
                    area_interstice_list = pos_.area_interstice_list_
                    area_list = pos_.area_list_
                    area_total = pos_.convex_hull().area
                    area_interstice_original_array = \
                        np.array(area_interstice_list) * np.array(area_list)
                    center_slice_area = pi * pow(
                        pos_.radii[str(pos_.atom_type)][pos_.radius_type], 2)

                    for area_type in self.area_types:
                        # fractional area_interstices in relative to the
                        # tetrahedra area
                        if area_type == "fractional_area_interstice_triangle":
                            feature_list.extend(
                                calc_stats(area_interstice_list))
                        # original area_interstices (in the units of area)
                        if area_type == "area_interstice":
                            feature_list.extend(
                                calc_stats(area_interstice_original_array))
                        # fractional area_interstices in relative to the entire area
                        if area_type == \
                                "fractional_area_interstice_triangle_avg":
                            feature_list.extend(
                                calc_stats(area_interstice_original_array /
                                           area_total * len(area_list)))
                        # fractional area_interstices in relative to the center
                        # atom volume
                        if area_type == \
                                "fractional_area_interstice_center_slice_a":
                            feature_list.extend(
                                calc_stats(area_interstice_original_array /
                                           center_slice_area))
                feature_lists.append(feature_list)

            if self.verbose > 0:
                vr.update(idx - 1)

        volume_area_interstice_df = pd.DataFrame(
            feature_lists, index=X.index, columns=self.get_feature_names())

        if self.save:
            self.backend.save_featurizer_as_dataframe(
                output_df=volume_area_interstice_df,
                name=self.output_file_prefix)

        return volume_area_interstice_df

    def get_feature_names(self):
        feature_names = list()
        feature_prefixs = list()

        stats = ['sum', 'mean', 'std', 'min', 'max']
        if self.calc_volume_area == 'volume' or self.calc_volume_area == 'all':
            feature_prefixs += self.volume_types
        if self.calc_volume_area == 'area' or self.calc_volume_area == 'all':
            feature_prefixs += self.area_types
        feature_names += ['{}_{}_{}'.format(feature_prefix, stat,
                                            self.dependent_name_)
                          for feature_prefix in feature_prefixs
                          for stat in stats]
        return feature_names


class ClusterPackingEfficiency(BaseInterstice):
    """
    Yang, L. et al. Atomic-Scale Mechanisms of the Glass-Forming Ability
    in Metallic Glasses. Phys. Rev. Lett. 109, 105502 (2012).
    The authors also term this metric as "Atomic Packing Efficiency" in the
    original paper. Here we name it as "Cluster Packing Efficiency" to
    distinguish this with that proposed in Laws, K. J. et al. Nat. Commun.
    6, 8123 (2015).
    """
    def __init__(self, pbc=None, backend=None, dependent_class="voro",
                 coords_cols=None, type_col='type',
                 type_to_atomic_number_list=None,
                 neighbor_num_limit=80, save=True,
                 radii=None, radius_type="miracle_radius",
                 verbose=1, output_path=None, output_file_prefix=None,
                 **nn_kwargs):
        assert dependent_class == "voro" or dependent_class == "voronoi"
        super(ClusterPackingEfficiency, self).__init__(
            save=save, backend=backend,
            dependent_class=dependent_class, type_col=type_col,
            type_to_atomic_number_list=type_to_atomic_number_list,
            neighbor_num_limit=neighbor_num_limit, radii=radii,
            radius_type = radius_type, verbose = verbose,
            output_path=output_path, **nn_kwargs)
        self.pbc = pbc if pbc is not None else [1, 1, 1]
        self.coords_cols = coords_cols \
            if coords_cols is not None else ['x', 'y', 'z']
        self.dependent_cols_ = [self.neighbor_num_col, self.neighbor_ids_col]
        self.output_file_prefix = output_file_prefix \
            if output_file_prefix is not None \
            else 'feature_{}_{}_{}_cluster_packing_efficiency'.format(
                    self.category, self.dependent_name_, self.radius_type)

    def transform(self, X):
        """
        Args:
            X (DataFrame): X can be a DataFrame which composed of partial
                columns of Nearest Neighbor class's output; or X can be the
                input of Nearest Neighbor class, which should contains
                ['type', 'x', 'y', 'z'...] columns, we will automatic call
                Nearest Neighbor class to calculate X's output by self.fit()
                method, then feed it as input to this transform() method.

        Returns:
            cluster_packing_efficiency_df (DataFrame): Cluster Packing
                Efficiency_df DataFrame, which index is same as X's index,
                see get_feature_names() method for column names.
        """
        X = X.join(self.lammps_df) \
            if self.calculated_X is None else self.calculated_X

        # define print verbose
        if self.verbose > 0:
            vr = VerboseReporter(self.backend, total_stage=1,
                                 verbose=self.verbose, max_verbose_mod=10000)
            vr.init(total_epoch=len(X), start_epoch=0,
                    init_msg='Calculating Cluster Packing Efficiency features.',
                    epoch_name='Atoms', stage=1)

        feature_lists = list()
        for idx, row in X.iterrows():
            neighbor_type = list()
            neighbor_coords = list()
            for neighbor_id in row[self.neighbor_ids_col]:
                if neighbor_id > 0:
                    neighbor_type.append(X.loc[neighbor_id][self.type_col])
                    neighbor_coords.append(X.loc[neighbor_id][self.coords_cols])
                else:
                    continue
            pos_ = PackingOfSite(self.pbc, self.Bds,
                                 row[self.type_col], row[self.coords_cols],
                                 neighbor_type, neighbor_coords,
                                 radii=self.radii, radius_type=self.radius_type)
            if len(neighbor_type) < 4:
                feature_lists.append([0] * len(self.get_feature_names()))
            else:
                feature_lists.append([pos_.cluster_packing_efficiency()])

            if self.verbose > 0:
                vr.update(idx - 1)

        cluster_packing_efficiency_df = pd.DataFrame(
            feature_lists, index=X.index, columns=self.get_feature_names())

        if self.save:
            self.backend.save_featurizer_as_dataframe(
                output_df=cluster_packing_efficiency_df,
                name=self.output_file_prefix)

        return cluster_packing_efficiency_df

    def get_feature_names(self):
        feature_names = ['{}_cluster_packing_efficiency_{}'.format(
            self.radius_type.replace("_radius", ""), self.dependent_name_)]
        return feature_names


class AtomicPackingEfficiency(BaseInterstice):
    """
    Laws, K. J., Miracle, D. B. & Ferry, M. A predictive structural model for
    bulk metallic glasses. Nat. Commun. 6, 8123 (2015).
    """
    def __init__(self, pbc=None, backend=None, dependent_class="voro",
                 coords_cols=None, type_col='type',
                 type_to_atomic_number_list=None,
                 neighbor_num_limit=80,  save=True,
                 radii=None, radius_type="miracle_radius",
                 verbose=1, output_path=None, output_file_prefix=None,
                 **nn_kwargs):
        assert dependent_class == "voro" or dependent_class == "voronoi"
        super(AtomicPackingEfficiency, self).__init__(
            save=save, backend=backend,
            dependent_class=dependent_class, type_col=type_col,
            type_to_atomic_number_list=type_to_atomic_number_list,
            neighbor_num_limit=neighbor_num_limit, radii=radii,
            radius_type = radius_type, verbose = verbose,
            output_path=output_path, **nn_kwargs)
        self.pbc = pbc if pbc is not None else [1, 1, 1]
        self.coords_cols = coords_cols \
            if coords_cols is not None else ['x', 'y', 'z']
        self.dependent_cols_ = [self.neighbor_num_col, self.neighbor_ids_col]
        self.output_file_prefix = output_file_prefix \
            if output_file_prefix is not None \
            else 'feature_{}_{}_{}_atomic_packing_efficiency'.format(
                    self.category, self.dependent_name_, self.radius_type)

    def transform(self, X):
        """
        Args:
            X (DataFrame): X can be a DataFrame which composed of partial
                columns of Nearest Neighbor class's output; or X can be the
                input of Nearest Neighbor class, which should contains
                ['type', 'x', 'y', 'z'...] columns, we will automatic call
                Nearest Neighbor class to calculate X's output by self.fit()
                method, then feed it as input to this transform() method.

        Returns:
            atomic_packing_efficiency_df (DataFrame): Atomic Packing Efficiency
                DataFrame, which index is same as X's index, see
                get_feature_names() method for column names.

        """
        X = X.join(self.lammps_df) \
            if self.calculated_X is None else self.calculated_X

        # define print verbose
        if self.verbose > 0:
            vr = VerboseReporter(self.backend, total_stage=1,
                                 verbose=self.verbose, max_verbose_mod=10000)
            vr.init(total_epoch=len(X), start_epoch=0,
                    init_msg='Calculating Atomic Packing Efficiency features.',
                    epoch_name='Atoms', stage=1)

        feature_lists = list()
        for idx, row in X.iterrows():
            neighbor_type = list()
            neighbor_coords = list()
            for neighbor_id in row[self.neighbor_ids_col]:
                if neighbor_id > 0:
                    neighbor_type.append(X.loc[neighbor_id][self.type_col])
                    neighbor_coords.append(X.loc[neighbor_id][self.coords_cols])
                else:
                    continue
            pos_ = PackingOfSite(self.pbc, self.Bds,
                                 row[self.type_col], row[self.coords_cols],
                                 neighbor_type, neighbor_coords,
                                 radii=self.radii, radius_type=self.radius_type)
            if len(neighbor_type) < 4:
                feature_lists.append([0] * len(self.get_feature_names()))
            else:
                feature_lists.append([pos_.atomic_packing_efficiency()])

            if self.verbose > 0:
                vr.update(idx - 1)

        atomic_packing_efficiency_df = pd.DataFrame(
            feature_lists, index=X.index, columns=self.get_feature_names())

        if self.save:
            self.backend.save_featurizer_as_dataframe(
                output_df=atomic_packing_efficiency_df,
                name=self.output_file_prefix)

        return atomic_packing_efficiency_df

    def get_feature_names(self):
        feature_names = ['{}_atomic_packing_efficiency_{}'.format(
            self.radius_type.replace("_radius", ""), self.dependent_name_)]
        return feature_names


class CN(BaseSRO):
    def __init__(self, backend=None, dependent_class="voro", save=True,
                 output_path=None, output_file_prefix=None, **nn_kwargs):
        super(CN, self).__init__(save=save,
                                 backend=backend,
                                 dependent_class=dependent_class,
                                 output_path=output_path,
                                 **nn_kwargs)
        self.dependent_cols_ = [self.neighbor_num_col]
        self.output_file_prefix = output_file_prefix \
            if output_file_prefix is not None \
            else 'feature_{}_{}_cn'.format(self.category, self.dependent_name_)

    def transform(self, X=None):
        X = X if self.calculated_X is None else self.calculated_X
        cn_df = pd.DataFrame(X[self.dependent_cols_].values,
                             index=X.index, columns=self.get_feature_names())

        if self.save:
            self.backend.save_featurizer_as_dataframe(
                output_df=cn_df, name=self.output_file_prefix)

        return cn_df

    def get_feature_names(self):
        feature_names = ['CN {}'.format(self.dependent_name_)]
        return feature_names


class VoroIndex(BaseSRO):
    def __init__(self, backend=None, dependent_class="voro",
                 neighbor_num_limit=80, include_beyond_edge_max=True,
                 save=True, edge_min=3, edge_max=7, output_path=None,
                 output_file_prefix=None, **nn_kwargs):
        assert dependent_class == "voro" or dependent_class == "voronoi"
        super(VoroIndex, self).__init__(save=save,
                                        backend=backend,
                                        dependent_class=dependent_class,
                                        output_path=output_path,
                                        **nn_kwargs)
        self.neighbor_num_limit = neighbor_num_limit
        self.include_beyond_edge_max = include_beyond_edge_max
        self.edge_min = edge_min
        self.edge_max = edge_max
        self.dependent_cols_ = [self.neighbor_num_col, self.neighbor_edges_col]
        self.output_file_prefix = output_file_prefix \
            if output_file_prefix is not None \
            else 'feature_{}_voronoi_index'.format(self.category)

    def transform(self, X=None):
        X = X if self.calculated_X is None else self.calculated_X
        edge_num = self.edge_max - self.edge_min + 1
        edge_lists = get_isometric_lists(X[self.neighbor_edges_col].values,
                                         limit_width=self.neighbor_num_limit)

        voronoi_index_list = np.zeros((len(X), edge_num))
        voro_index_list = voronoi_stats.voronoi_index(
            voronoi_index_list, X[self.neighbor_num_col].values, edge_lists,
            self.edge_min, self.edge_max, self.include_beyond_edge_max,
            n_atoms=len(X), neighbor_num_limit=self.neighbor_num_limit)

        voro_index_df = pd.DataFrame(voro_index_list, index=X.index,
                                     columns=self.get_feature_names())
        if self.save:
            self.backend.save_featurizer_as_dataframe(
                output_df=voro_index_df, name=self.output_file_prefix)

        return voro_index_df

    def get_feature_names(self):
        return ['Voronoi_indices_voro']


class CharacterMotif(BaseSRO):
    def __init__(self, backend=None, dependent_class="voro",
                 neighbor_num_limit=80, include_beyond_edge_max=True,
                 edge_min=3, target_voro_idx=None, frank_kasper=1,
                 save=True, output_path=None, output_file_prefix=None,
                 **nn_kwargs):
        assert dependent_class == "voro" or dependent_class == "voronoi"
        super(CharacterMotif, self).__init__(save=save,
                                             backend=backend,
                                             dependent_class=dependent_class,
                                             output_path=output_path,
                                             **nn_kwargs)
        self.neighbor_num_limit = neighbor_num_limit
        self.include_beyond_edge_max = include_beyond_edge_max
        if target_voro_idx is None:
            self.target_voro_idx = np.array([[0, 0, 12, 0, 0],
                                             [0, 0, 12, 4, 0]],
                                            dtype=np.longdouble)
        self.frank_kasper = frank_kasper
        self.edge_min = edge_min
        self.dependent_cols_ = ['Voronoi_indices_voro']
        self.output_file_prefix = output_file_prefix \
            if output_file_prefix is not None \
            else 'feature_{}_character_motif'.format(self.category)

    def fit(self, X=None):
        self.dependent_class_ = self.check_dependency(X)
        # This class is only dependent on 'Voronoi_indices_voro' col, so if
        # X don't have this col, this method will calculate it automatically.
        if self.dependent_class_ is None:
            voro_index = \
                VoroIndex(neighbor_num_limit=self.neighbor_num_limit,
                          include_beyond_edge_max=self.include_beyond_edge_max,
                          dependent_class=self.dependent_class, save=False,
                          backend=self.backend)
            self.calculated_X = voro_index.fit_transform(X)
        return self

    def transform(self, X=None):
        X = X if self.calculated_X is None else self.calculated_X
        voro_idx_lists = get_isometric_lists(
            X[self.dependent_cols_].values, limit_width=self.neighbor_num_limit)

        motif_one_hot = np.zeros((len(X),
                                  len(self.target_voro_idx) + self.frank_kasper))
        motif_one_hot = \
            voronoi_stats.character_motif(motif_one_hot, voro_idx_lists,
                                          self.edge_min, self.target_voro_idx,
                                          self.frank_kasper, n_atoms=len(X))
        motif_one_hot_array = np.array(motif_one_hot)
        is_120_124 = motif_one_hot_array[:, 0] | motif_one_hot_array[:, 1]
        motif_one_hot_array = np.append(motif_one_hot_array,
                                        np.array([is_120_124]).T, axis=1)
        character_motif_df = pd.DataFrame(motif_one_hot_array, index=X.index,
                                          columns=self.get_feature_names())

        if self.save:
            self.backend.save_featurizer_as_dataframe(
                output_df=character_motif_df, name=self.output_file_prefix)

        return character_motif_df

    def get_feature_names(self):
        feature_names = ['is <0,0,12,0,0> voro', 'is <0,0,12,4,0> voro'] + \
                        ["_".join(map(str, v)) + " voro"
                         for v in self.target_voro_idx[2:]] + \
                        ['is polytetrahedral voro', 'is <0,0,12,0/4,0> voro']
        return feature_names


class IFoldSymmetry(BaseSRO):
    def __init__(self, backend=None, dependent_class="voro",
                 neighbor_num_limit=80, include_beyond_edge_max=True,
                 edge_min=3, edge_max=7, save=True, **nn_kwargs):
        assert dependent_class == "voro" or dependent_class == "voronoi"
        super(IFoldSymmetry, self).__init__(save=save,
                                            backend=backend,
                                            dependent_class=dependent_class,
                                            **nn_kwargs)
        self.neighbor_num_limit = neighbor_num_limit
        self.include_beyond_edge_max = include_beyond_edge_max
        self.edge_min = edge_min
        self.edge_max = edge_max
        self.dependent_cols_ = [self.neighbor_num_col, self.neighbor_edges_col]

    def transform(self, X=None):
        X = X if self.calculated_X is None else self.calculated_X
        edge_num = self.edge_max - self.edge_min + 1
        edge_lists = get_isometric_lists(X[self.neighbor_edges_col].values,
                                         limit_width=self.neighbor_num_limit)

        i_symm_list = np.zeros((len(X), edge_num))
        i_symm_list = voronoi_stats.i_fold_symmetry(
            i_symm_list,  X[self.neighbor_num_col].values, edge_lists,
            self.edge_min, self.edge_max, self.include_beyond_edge_max,
            n_atoms=len(X), neighbor_num_limit=self.neighbor_num_limit)

        i_symm_df = pd.DataFrame(i_symm_list, index=X.index,
                                 columns=self.get_feature_names())
        if self.save:
            self.backend.save_featurizer_as_dataframe(
                output_df=i_symm_df,
                name='{}_i_fold_symmetry'.format(self.category))

        return i_symm_df

    def get_feature_names(self):
        feature_names = ['{}-fold symm idx voro'.format(edge)
                         for edge in range(self.edge_min, self.edge_max+1)]
        return feature_names


class AreaWtIFoldSymmetry(BaseSRO):
    def __init__(self, backend=None, dependent_class="voro",
                 neighbor_num_limit=80, include_beyond_edge_max=True,
                 edge_min=3, edge_max=7, save=True, **nn_kwargs):
        assert dependent_class == "voro" or dependent_class == "voronoi"
        super(AreaWtIFoldSymmetry, self).__init__(
            save=save, backend=backend,
            dependent_class=dependent_class, **nn_kwargs)
        self.neighbor_num_limit = neighbor_num_limit
        self.include_beyond_edge_max = include_beyond_edge_max
        self.edge_min = edge_min
        self.edge_max = edge_max
        self.dependent_cols_ = [self.neighbor_num_col,
                                self.neighbor_edges_col,
                                self.neighbor_areas_col]

    def transform(self, X=None):
        X = X if self.calculated_X is None else self.calculated_X
        edge_lists = get_isometric_lists(X[self.neighbor_edges_col].values,
                                         limit_width=self.neighbor_num_limit)
        area_lists = get_isometric_lists(
            X[self.neighbor_areas_col].values,
            limit_width=self.neighbor_num_limit).astype(np.longdouble)
        edge_num = self.edge_max - self.edge_min + 1

        area_wt_i_symm_list = np.zeros((len(X), edge_num))
        area_wt_i_symm_list = voronoi_stats.area_wt_i_fold_symmetry(
            area_wt_i_symm_list, X[self.neighbor_num_col].values,
            edge_lists, area_lists, self.edge_min, self.edge_max,
            self.include_beyond_edge_max, n_atoms=len(X),
            neighbor_num_limit=self.neighbor_num_limit)

        area_wt_i_symm_df = \
            pd.DataFrame(area_wt_i_symm_list, index=X.index,
                         columns=self.get_feature_names())
        if self.save:
            self.backend.save_featurizer_as_dataframe(
                output_df=area_wt_i_symm_df,
                name='{}_area_wt_i_fold_symmetry'.format(self.category))
        return area_wt_i_symm_df

    def get_feature_names(self):
        feature_names = ['Area_wt {}-fold symm idx voro'.format(edge)
                         for edge in range(self.edge_min, self.edge_max + 1)]
        return feature_names


class VolWtIFoldSymmetry(BaseSRO):
    def __init__(self, backend=None, dependent_class="voro",
                 neighbor_num_limit=80, include_beyond_edge_max=True,
                 edge_min=3, edge_max=7, save=True, **nn_kwargs):
        assert dependent_class == "voro" or dependent_class == "voronoi"
        super(VolWtIFoldSymmetry, self).__init__(
            save=save, backend=backend,
            dependent_class=dependent_class, **nn_kwargs)
        self.neighbor_num_limit = neighbor_num_limit
        self.include_beyond_edge_max = include_beyond_edge_max
        self.edge_min = edge_min
        self.edge_max = edge_max
        self.dependent_cols_ = [self.neighbor_num_col,
                                self.neighbor_edges_col,
                                self.neighbor_vols_col]

    def transform(self, X=None):
        X = X if self.calculated_X is None else self.calculated_X
        edge_lists = get_isometric_lists(X[self.neighbor_edges_col].values,
                                         limit_width=self.neighbor_num_limit)
        vol_lists = get_isometric_lists(
            X[self.neighbor_vols_col].values,
            limit_width=self.neighbor_num_limit).astype(np.longdouble)

        edge_num = self.edge_max - self.edge_min + 1
        vol_wt_i_symm_list = np.zeros((len(X), edge_num))
        vol_wt_i_symm_list = \
            voronoi_stats.vol_wt_i_fold_symmetry(
                vol_wt_i_symm_list, X[self.neighbor_num_col].values, edge_lists,
                vol_lists, self.edge_min, self.edge_max,
                self.include_beyond_edge_max, n_atoms=len(X),
                neighbor_num_limit=self.neighbor_num_limit)

        vol_wt_i_symm_df = pd.DataFrame(vol_wt_i_symm_list, index=X.index,
                                        columns=self.get_feature_names())
        if self.save:
            self.backend.save_featurizer_as_dataframe(
                output_df=vol_wt_i_symm_df,
                name='{}_vol_wt_i_fold_symmetry'.format(self.category))

        return vol_wt_i_symm_df

    def get_feature_names(self):
        feature_names = ['Vol_wt {}-fold symm idx voro'.format(edge)
                         for edge in range(self.edge_min, self.edge_max + 1)]
        return feature_names


class VoroAreaStats(BaseSRO):
    def __init__(self, backend=None, dependent_class="voro",
                 neighbor_num_limit=80, save=True, **nn_kwargs):
        super(VoroAreaStats, self).__init__(save=save,
                                            backend=backend,
                                            dependent_class=dependent_class,
                                            **nn_kwargs)
        self.neighbor_num_limit = neighbor_num_limit
        self.voro_depend_cols = ['neighbor_num_voro'] + \
                                ['neighbor_area_5_voro']
        self.stats = ['mean', 'std', 'min', 'max']
        self.dist_depend_cols = None

    def transform(self, X=None):
        X = X if self.calculated_X is None else self.calculated_X
        n_atoms = len(X)
        columns = X.columns
        area_cols = [col for col in columns if
                     col.startswith('neighbor_area_')]
        area_stats = np.zeros((n_atoms, len(self.stats) + 1))

        area_stats = \
            voronoi_stats.voronoi_area_stats(area_stats,
                                             X['neighbor_num_voro'].values,
                                             X[area_cols].values.astype(
                                                 np.longdouble),
                                             n_atoms=n_atoms,
                                             neighbor_num_limit=
                                             self.neighbor_num_limit)

        area_stats_df = pd.DataFrame(area_stats, index=X.index,
                                     columns=self.get_feature_names())
        if self.save:
            self.backend.save_featurizer_as_dataframe(
                output_df=area_stats_df, name='{}_voronoi_area_stats'.format(self.category))

        return area_stats_df

    def get_feature_names(self):
        feature_names = ['Voronoi area voro'] + \
                        ['Facet area {} voro'.format(stat)
                         for stat in self.stats]
        return feature_names


class VoroAreaStatsSeparate(BaseSRO):
    def __init__(self, neighbor_num_limit=80, include_beyond_edge_max=True,
                 atoms_df=None, dependent_class="voro", edge_min=3, edge_max=7,
                 save=True, backend=None, remain_stat=False, **nn_kwargs):
        super(VoroAreaStatsSeparate, self).__init__(save=save,
                                                    backend=backend,
                                                    dependent_class=dependent_class,
                                                    atoms_df=atoms_df,
                                                    remain_stat=remain_stat,
                                                    **nn_kwargs)
        self.neighbor_num_limit = neighbor_num_limit
        self.voro_depend_cols = ['neighbor_num_voro'] + \
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
        X = X if self.calculated_X is None else self.calculated_X
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
                area_stats_separate, X['neighbor_num_voro'].values,
                X[edge_cols].values,
                X[area_cols].values.astype(np.longdouble),
                self.edge_min, self.edge_max,
                self.include_beyond_edge_max,
                n_atoms=n_atoms,
                neighbor_num_limit=self.neighbor_num_limit)

        area_stats_separate_df = pd.DataFrame(area_stats_separate, index=X.index,
                                     columns=self.get_feature_names())
        if self.save:
            self.backend.save_featurizer_as_dataframe(
                output_df=area_stats_separate_df,
                name='{}_voro_area_stats_separate'.format(self.category))

        return area_stats_separate_df

    def get_feature_names(self):
        feature_names = ['{}-edged area {} voro'.format(edge, stat)
                         for edge in range(self.edge_min, self.edge_max + 1)
                         for stat in self.stats]
        return feature_names


class VoroVolStats(BaseSRO):
    def __init__(self, neighbor_num_limit=80,
                 atoms_df=None, dependent_class="voro",
                 save=True, backend=None, remain_stat=False, **nn_kwargs):
        super(VoroVolStats, self).__init__(save=save,
                                            backend=backend,
                                            dependent_class=dependent_class,
                                            atoms_df=atoms_df,
                                            remain_stat=remain_stat,
                                            **nn_kwargs)
        self.neighbor_num_limit = neighbor_num_limit
        self.voro_depend_cols = ['neighbor_num_voro'] + \
                                ['neighbor_vol_5_voro']
        self.stats = ['mean', 'std', 'min', 'max']
        self.dist_depend_cols = None

    def transform(self, X=None):
        X = X if self.calculated_X is None else self.calculated_X
        n_atoms = len(X)
        columns = X.columns
        vol_cols = [col for col in columns if
                     col.startswith('neighbor_vol_')]
        vol_stats = np.zeros((n_atoms, len(self.stats) + 1))

        vol_stats = \
            voronoi_stats.voronoi_vol_stats(vol_stats,
                                            X['neighbor_num_voro'].values,
                                            X[vol_cols].values.astype(
                                                np.longdouble),
                                            n_atoms=n_atoms,
                                            neighbor_num_limit=
                                            self.neighbor_num_limit)

        vol_stats_df = pd.DataFrame(vol_stats, index=X.index,
                                    columns=self.get_feature_names())
        if self.save:
            self.backend.save_featurizer_as_dataframe(
                output_df=vol_stats_df, name='{}_voronoi_vol_stats'.format(self.category))

        return vol_stats_df

    def get_feature_names(self):
        feature_names = ['Voronoi vol voro'] + \
                        ['Sub-polyhedra vol {} voro'.format(stat)
                         for stat in self.stats]
        return feature_names


class VoroVolStatsSeparate(BaseSRO):
    def __init__(self, neighbor_num_limit=80, include_beyond_edge_max=True,
                 atoms_df=None, dependent_class="voro", edge_min=3, edge_max=7,
                 save=True, backend=None, remain_stat=False, **nn_kwargs):
        super(VoroVolStatsSeparate, self).__init__(save=save,
                                            backend=backend,
                                            dependent_class=dependent_class,
                                            atoms_df=atoms_df,
                                            remain_stat=remain_stat,
                                            **nn_kwargs)
        self.neighbor_num_limit = neighbor_num_limit
        self.voro_depend_cols = ['neighbor_num_voro'] + \
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
        X = X if self.calculated_X is None else self.calculated_X
        n_atoms = len(X)
        columns = X.columns
        edge_cols = [col for col in columns if col.startswith('neighbor_edge_')]
        vol_cols = [col for col in columns if col.startswith('neighbor_vol_')]
        vol_stats_separate = np.zeros((n_atoms,
                                       self.edge_num * len(self.stats)))

        vol_stats_separate = \
            voronoi_stats.voronoi_vol_stats_separate(
                vol_stats_separate, X['neighbor_num_voro'].values,
                X[edge_cols].values,
                X[vol_cols].values.astype(np.longdouble),
                self.edge_min, self.edge_max,
                self.include_beyond_edge_max,
                n_atoms=n_atoms,
                neighbor_num_limit=self.neighbor_num_limit)

        vol_stats_separate_df = pd.DataFrame(vol_stats_separate,
                                             index=X.index,
                                             columns=self.get_feature_names())
        if self.save:
            self.backend.save_featurizer_as_dataframe(
                output_df=vol_stats_separate_df, name='{}_voro_vol_stats_separate'.format(self.category))

        return vol_stats_separate_df

    def get_feature_names(self):
        feature_names = ['{}-edged vol {} voro'.format(edge, stat)
                         for edge in range(self.edge_min, self.edge_max + 1)
                         for stat in self.stats]
        return feature_names


class DistStats(BaseSRO):
    def __init__(self, dist_type='distance', neighbor_num_limit=80,
                 atoms_df=None, dependent_class="voro",
                 save=True, backend=None, remain_stat=False, **nn_kwargs):
        super(DistStats, self).__init__(save=save,
                                        backend=backend,
                                        dependent_class=dependent_class,
                                        atoms_df=atoms_df,
                                        remain_stat=remain_stat,
                                        **nn_kwargs)
        self.dist_type = dist_type
        self.neighbor_num_limit = neighbor_num_limit
        self.voro_depend_cols = ['neighbor_num_voro'] + \
                                ['neighbor_{}_5_voro'.format(dist_type)]
        self.stats = ['sum', 'mean', 'std', 'min', 'max']
        self.dist_depend_cols = ['n_neighbors_dist'] + \
                                 ['neighbor_{}_5_dist'.format(dist_type)]

    def transform(self, X=None):
        X = X if self.calculated_X is None else self.calculated_X
        n_atoms = len(X)
        columns = X.columns
        dist_cols = [col for col in columns if
                     col.startswith('neighbor_{}_'.format(self.dist_type))]
        dist_stats = np.zeros((n_atoms, len(self.stats)))

        dist_stats = \
            voronoi_stats.voronoi_distance_stats(dist_stats,
                                             X['n_neighbors_{}'.format(
                                                 self.dependent_name_)].values,
                                             X[dist_cols].values,
                                             n_atoms=n_atoms,
                                             neighbor_num_limit=
                                             self.neighbor_num_limit)
        dist_stats_df = pd.DataFrame(dist_stats, index=X.index,
                                     columns=self.get_feature_names())
        if self.save:
            self.backend.save_featurizer_as_dataframe(
                output_df=dist_stats_df,
                name='{}_{}_{}_stats'.format(self.category, self.dependent_name_, self.dist_type))

        return dist_stats_df

    def get_feature_names(self):
        feature_names = ['{} {} {}'.format(self.dist_type, stat,
                                           self.dependent_name_)
                         for stat in self.stats]
        return feature_names

    @property
    def double_dependency(self):
        return False


class BOOP(BaseSRO):
    def __init__(self, coords_path=None, atom_coords=None, Bds=None, pbc=None,
                 low_order=1, higher_order=1, coarse_lower_order=1,
                 coarse_higher_order=1, neighbor_num_limit=80, atoms_df=None,
                 dependent_class="voro", save=True, backend=None,
                 remain_stat=False, **nn_kwargs):
        super(BOOP, self).__init__(save=save,
                                   backend=backend,
                                   dependent_class=dependent_class,
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
        self.neighbor_num_limit = neighbor_num_limit
        self.voro_depend_cols = ['neighbor_num_voro'] + \
                                ['neighbor_id_{}_voro'.format(idx)
                                 for idx in range(neighbor_num_limit)]
        self.dist_depend_cols = ['n_neighbors_dist'] + \
                                 ['neighbor_id_{}_dist'.format(idx)
                                  for idx in range(neighbor_num_limit)]
        self.bq_tags = ['4', '6', '8', '10']

    def transform(self, X=None):
        X = X if self.calculated_X is None else self.calculated_X
        n_atoms = len(X)
        neighbor_col = ['n_neighbors_{}'.format(self.dependent_name_)]
        id_cols = ['neighbor_id_{}_{}'.format(idx, self.dependent_name_)
                   for idx in range(self.neighbor_num_limit)]

        Ql = np.zeros((n_atoms, 4), dtype=np.longdouble)
        Wlbar = np.zeros((n_atoms, 4), dtype=np.longdouble)
        coarse_Ql = np.zeros((n_atoms, 4), dtype=np.longdouble)
        coarse_Wlbar = np.zeros((n_atoms, 4), dtype=np.longdouble)
        Ql, Wlbar, coarse_Ql, coarse_Wlbar = \
            boop.calculate_boop(
                self.atom_coords.astype(np.longdouble),
                self.pbc, np.array(self.Bds, dtype=np.longdouble),
                X[neighbor_col].values,
                X[id_cols].values,
                self.low_order, self.higher_order, self.coarse_lower_order,
                self.coarse_higher_order, Ql, Wlbar, coarse_Ql, coarse_Wlbar,
                n_atoms=n_atoms, neighbor_num_limit=self.neighbor_num_limit)
        concat_array = np.append(Ql, Wlbar, axis=1)
        concat_array = np.append(concat_array, coarse_Ql, axis=1)
        concat_array = np.append(concat_array, coarse_Wlbar, axis=1)

        boop_df = pd.DataFrame(concat_array, index=X.index,
                               columns=self.get_feature_names())
        if self.save:
            self.backend.save_featurizer_as_dataframe(
                output_df=boop_df, name='{}_boop_{}'.format(self.category, self.dependent_name_))

        return boop_df

    def get_feature_names(self):
        feature_names = ['q_{} {}'.format(num, self.dependent_name_)
                         for num in self.bq_tags] + \
                        ['w_{} {}'.format(num, self.dependent_name_)
                         for num in self.bq_tags] + \
                        ['Coarse-grained q_{} {}'.format(num,
                                                         self.dependent_name_)
                         for num in self.bq_tags] + \
                        ['Coarse-grained w_{} {}'.format(num,
                                                         self.dependent_name_)
                         for num in self.bq_tags]
        return feature_names

    @property
    def double_dependency(self):
        return False

