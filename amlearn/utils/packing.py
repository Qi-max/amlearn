from math import atan2, pi
import numpy as np


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
    # return np.degrees(np.arccos(cosine_angle))


def triangle_area(va, vb, vc):
    """
    Calculate the volume of a triangle, given the three vertices of va, vb, vc.
    Args:
        va/vb/vc (array-like): coordinates of vertex 1, 2, 3.
    Returns:
        (float): volume of the tetrahedron.
    """
    a = np.linalg.norm(np.array(va) - np.array(vb))
    b = np.linalg.norm(np.array(va) - np.array(vc))
    c = np.linalg.norm(np.array(vb) - np.array(vc))
    s = (a + b + c) / 2
    return (s * (s - a) * (s - b) * (s - c)) ** 0.5


def solid_angle(center, va, vb, vc):
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


def calc_stats(prop_list):
    prop_array = np.array(prop_list)
    prop_sum = sum(prop_array)
    prop_mean = np.mean(prop_array)
    prop_std = np.std(prop_array)
    prop_min = min(prop_array)
    prop_max = max(prop_array)
    stats_list = [prop_sum, prop_mean, prop_std, prop_min, prop_max]
    return stats_list

