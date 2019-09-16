import os
import numpy as np
import pandas as pd
from math import atan2, pi


def list_like():
    return (list, np.ndarray, tuple, pd.Series)


def read_lammps_dump(data_path):
    if os.path.exists(data_path):
        with open(data_path, 'r') as rf:
            lines = rf.readlines()
        cols = list()
        Bds = [list(map(float, lines[5].strip().split())),
               list(map(float, lines[6].strip().split())),
               list(map(float, lines[7].strip().split()))]
        column_names = lines[8].strip().split()[2:]
        for idx, line in enumerate(lines):
            if idx > 8:
                column_values = [eval(x) for x in line.strip().split()]
                cols.append(column_values)
        df = pd.DataFrame(cols, columns=column_names)
        df.index = list(df["id"])
        return df, Bds
    else:
        raise FileNotFoundError("File {} not found".format(data_path))


def read_imd(dirs, bds=None, atom_names=None):
    if atom_names is None:
        atom_names = range(1, 50)
    cell_coords = []
    id_list = []
    types = []
    coords = []
    i = 0
    with open(dirs, 'r') as file_object:
        file_lines = file_object.readlines()
        Bds = [list(map(float, file_lines[5].strip().split())),
               list(map(float, file_lines[6].strip().split())),
               list(map(float, file_lines[7].strip().split()))]

        for line in file_lines:
            i += 1
            if i == 6:
                cell_coords.append(
                    [float(line.strip().split()[0]) * 2, 0, 0])
            elif i == 7:
                cell_coords.append(
                    [0, float(line.strip().split()[0]) * 2, 0])
            elif i == 8:
                cell_coords.append(
                    [0, 0, float(line.strip().split()[0]) * 2])
            elif i > 9:
                data_line = line.strip().split()
                validate = True
                if bds is not None:
                    if isinstance(bds, (list, tuple, np.ndarray)):
                        for index in range(2):
                            data_coord = float(data_line[index + 2])
                            filter_coord = bds[index]
                            if bds[index] is not None:
                                if data_coord < filter_coord[0] \
                                        or data_coord > filter_coord[1]:
                                    validate = False
                                    break
                    else:
                        raise TypeError("Please make sure bds is list or None!")
                if validate:
                    id_list.append(int(data_line[0]))
                    types.append(atom_names[int(data_line[1]) - 1])
                    coords.append([float(data_line[2]), float(data_line[3]),
                                   float(data_line[4])])
        return cell_coords, types, coords, Bds, id_list


def get_valid_lists(raw_lists, valid_num=None, invalid_value=0):
    valid_lists = list() if valid_num is None else [valid_num]
    for raw_list in raw_lists:
        if valid_num is not None:
            valid_lists.append(raw_list[:valid_num])
        else:
            valid_lists.append([x for x in raw_list if x != invalid_value])
    return valid_lists


def get_isometric_lists(raw_lists, limit_width=80, fill_value=0):
    if not isinstance(raw_lists, list_like()):
        raise TypeError('raw_lists should be list like, as list, '
                        'np.ndarray, tuple...')

    isometric_lists = list()
    for raw_list in raw_lists:
        if len(raw_list) >= limit_width:
            raw_list = raw_list[:limit_width]
        else:
            raw_list = \
                list(raw_list) + [fill_value] * (limit_width - len(raw_list))
        isometric_lists.append(raw_list)
    return np.array(isometric_lists)


def calc_neighbor_coords(center_coords, neighbor_coords, bds, pbc):
    if list(pbc) == [0, 0, 0]:
        return neighbor_coords

    bds = np.array(bds)
    dims = bds[:, 1] - bds[:, 0]
    pbc_neighbor_coords = list()
    for center_coord, neighbor_coord, pbc_dim, dim in \
            zip(center_coords, neighbor_coords, pbc, dims):
        if pbc_dim == 0:
            pbc_neighbor_coords.append(neighbor_coord)
        else:
            dist = neighbor_coord - center_coord
            if dist > 0.5 * dim:
                neighbor_coord = neighbor_coord - dim
            elif dist < -0.5 * dim:
                neighbor_coord = neighbor_coord + dim
            pbc_neighbor_coords.append(neighbor_coord)
    return np.array(pbc_neighbor_coords)


# return radian
def calc_plane_angle(center, va, vb):
    a = va - center
    b = vb - center

    cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.arccos(cosine_angle)
    # return np.degrees(np.arccos(cosine_angle))


# return radian
def calc_solid_angle(center, va, vb, vc):
    """
    Args:
        center: coordinates of center atom to calculate the solid angle
        va/vb/vc: coordinates of 3 vertices to form the tetrahedron
    Returns:
        solid_angle in rad
    """
    a = va - center
    b = vb - center
    c = vc - center
    len_a = np.linalg.norm(a)
    len_b = np.linalg.norm(b)
    len_c = np.linalg.norm(c)
    triple_product = abs(np.dot(a, np.cross(b, c)))
    divisor = len_a * len_b * len_c + np.dot(a, b) * len_c + \
              np.dot(a, c) * len_b + np.dot(b, c) * len_a
    solid_angle_ = 2 * atan2(triple_product, divisor)
    if divisor <= 0:
        solid_angle_ += pi
    return solid_angle_


def volume_tetra(va, vb, vc, vd):
    """
    Calculate the volume of a tetrahedron, given the four vertices of vt1,
    vt2, vt3 and vt4.
    Args:
        vt1 (array-like): coordinates of vertex 1.
        vt2 (array-like): coordinates of vertex 2.
        vt3 (array-like): coordinates of vertex 3.
        vt4 (array-like): coordinates of vertex 4.
    Returns:
        (float): volume of the tetrahedron.
    """
    volume_tetra = np.abs(np.dot((va - vd),
                              np.cross((vb - vd), (vc - vd))))/6
    return volume_tetra


def area_triangle(va, vb, vc):
    """
    Calculate the volume of a tetrahedron, given the four vertices of vt1,
    vt2, vt3 and vt4.
    Args:
        vt1 (array-like): coordinates of vertex 1.
        vt2 (array-like): coordinates of vertex 2.
        vt3 (array-like): coordinates of vertex 3.
    Returns:
        (float): volume of the tetrahedron.
    """
    a = np.linalg.norm(np.array(va) - np.array(vb))
    b = np.linalg.norm(np.array(va) - np.array(vc))
    c = np.linalg.norm(np.array(vb) - np.array(vc))
    s = (a + b + c) / 2
    return (s * (s - a) * (s - b) * (s - c)) ** 0.5


def calc_stats(prop_list):
    prop_array = np.array(prop_list)
    prop_sum = sum(prop_array)
    prop_mean = np.mean(prop_array)
    prop_std = np.std(prop_array)
    prop_min = min(prop_array)
    prop_max = max(prop_array)
    stats_list = [prop_sum, prop_mean, prop_std, prop_min, prop_max]
    return stats_list
