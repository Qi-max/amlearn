from __future__ import division
import os
import sys
import six
import operator
import numpy as np
import pandas as pd


# featurizer check
from amlearn.utils.data import list_like


def check_featurizer_X(X, atoms_df):
    X = atoms_df if X is None else X
    if X is None:
        raise ValueError("Either of X and atoms_df should not be None.")
    return X


def check_dependency(depend_cls, df_cols,
                     voro_depend_cols=None, dist_denpend_cols=None):
    if depend_cls.__class__.__name__ == "VoroNN" \
         and set(voro_depend_cols).issubset(set(df_cols)):
        return True

    elif depend_cls.__class__.__name__ == "DistanceNN" \
            and set(dist_denpend_cols).issubset(set(df_cols)):
        return True

    else:
        return False


def check_neighbor_col(neighbor_col):
    valid_cols = ['n_neighbors_voro', 'n_neighbors_dist']
    if neighbor_col == "all":
        neighbor_col = ['n_neighbors_voro', 'n_neighbors_dist']
    elif isinstance(neighbor_col, list_like):
        neighbor_col = neighbor_col
    else:
        neighbor_col = [neighbor_col]

    if not set(neighbor_col).issubset(valid_cols):
        raise ValueError("neighbor_col {} is unknown. "
                         "Possible values are: {}".format(neighbor_col,
                                                          valid_cols))
    return neighbor_col


# common check
def is_abstract(c):
    if not(hasattr(c, '__abstractmethods__')):
        return False
    if not len(c.__abstractmethods__):
        return False
    return True

def check_file_name(file_name):
    if isinstance(file_name, six.string_types):
        file_name = file_name.replace('.', 'p')
    return file_name


def check_output_path(output_path, msg="output path"):
    if os.path.exists(output_path):
        print("{}: {} already exists!".format(msg, output_path))
    else:
        os.makedirs(output_path)
        print("create {}: {} successful!".format(msg, output_path))


def check_list_equal(list_1, list_2):
    if len(list_1) != len(list_2):
        return False
    if np.array_equal(sorted(list_1), sorted(list_2)):
        return True
    else:
        return False


def check_list_contain(list_1, list_2):
    return set(list_1) >= set(list_2)
    # if set(list_1) >= set(list_2)
    #     return True
    # else:
    #     return False


def check_dict_key_contain(d, key_list):
    return set(list(d.keys())) >= set(key_list)


def atom_mapping(df, number_column="number",
                 includes=None, excludes=None):
    if not hasattr(df, number_column):
        raise ValueError("Please make sure the dataframe has number column")
    if includes is not None:
        df = df[(df[number_column].isin(includes))]
    if excludes is not None:
        df = df[~(df[number_column].isin(excludes))]
    return df


def check_within(x, criterion, criterion_type="range"):
    """
    Check if x is "within" the criterion of a specific criterion_type,
    the returned bool(s) can be useful in slicing.
    Args:
        x: can be a number, an array or a column of a dataframe,
           even a dataframe is supported, at least in the form
        criterion: eg. [3, 5] or [None, 3] or [1, 3, 6, 7]
        criterion_type: eg. "range" or "range" or "value", corresponding to
                        the criterion
    Returns: a bool or an array/series/dataframe of bools

    """
    if criterion_type is "range":
        try:
            range_lw = criterion[0] if criterion[0] is not None \
                else -sys.float_info.max
            range_hi = criterion[1] if criterion[1] is not None \
                else sys.float_info.max
        except Exception:
            raise ValueError("Please input a two-element list "
                             "if you want a range!")
        # return range_lw <= x <= range_hi # Series cannot be written like this
        return (x > range_lw) & (x < range_hi)
    elif criterion_type is "value":
        criterion = criterion if not isinstance(criterion, six.string_types) \
            else [criterion]
        return x in criterion
    else:
        raise RuntimeError("Criterion_type {} is not supported yet."
                           "Please use range or value".
                           format(criterion_type))
