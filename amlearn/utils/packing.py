import os
import json
import numpy as np
from math import atan2

__author__ = "Qi Wang"
__email__ = "qiwang.mse@gmail.com"

module_dir = os.path.dirname(os.path.abspath( __file__))

def load_radii():
    """Get Periodic Table of Elements dict.

    Returns:
        PTE_dict_ (dict): The Periodic Table of Elements dict, key is atomic id,
            value is dict which contains 'symbol', 'covalent_radius' and
            'miracle_radius'.

    """
    with open(os.path.join(module_dir, 'PTE.json'), 'r') as rf:
        PTE_dict_ = json.load(rf)
    return PTE_dict_


def pbc_image_nn_coords(center_coords, neighbor_coords, bds, pbc):
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


def triangular_angle(center, va, vb):
    a = va - center
    b = vb - center
    cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.arccos(cosine_angle)


def triangle_area(va, vb, vc):
    """
    Calculate the volume of a triangle, given the three vertices of va, vb, vc.
    Args:
        va/vb/vc (array-like): coordinates of vertex 1, 2, 3.
    Returns:
        (float): area of the triangle.
    """
    triangle_area = 0.5 * np.linalg.norm(np.cross(np.array(va) - np.array(vc),
                                                  np.array(vb) - np.array(vc)))
    return triangle_area


def solid_angle(center, va, vb, vc):
    """
    Args:
        center: coordinates of center atom to calculate the solid angle
        va/vb/vc: coordinates of 3 vertices to form the tetrahedron
    Returns:
        (float): solid_angle in rad
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
    return solid_angle_


def tetra_volume(va, vb, vc, vd):
    """
    Calculate the volume of a tetrahedron, given the four vertices of va,
    vb, vc and vd.
    Args:
        va/vb/vc/vd (array-like): coordinates of vertex 1, 2, 3, 4.
    Returns:
        (float): volume of the tetrahedron.
    """
    return np.abs(np.dot((va - vd),
                         np.cross((vb - vd), (vc - vd))))/6


def calc_stats(prop_list, stat_ops=None):
    prop_array = np.array(prop_list)
    stats_list = list()
    stat_ops = ['sum', 'mean', 'std', 'min', 'max'] if stat_ops is None \
        else stat_ops
    for stat_op in stat_ops:

        if stat_op == 'sum':
            prop_stat = sum(prop_array)
        elif stat_op == 'mean':
            prop_stat = np.mean(prop_array)
        elif stat_op == 'std':
            prop_stat = np.std(prop_array)
        elif stat_op == 'min':
            prop_stat = min(prop_array)
        elif stat_op == 'max':
            prop_stat = max(prop_array)
        else:
            raise ValueError('Now statistical operators only support: sum, '
                             'mean, std, min, max.')
        stats_list.append(prop_stat)
    return stats_list

