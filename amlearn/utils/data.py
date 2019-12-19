import os
import numpy as np
import pandas as pd

__author__ = "Qi Wang"
__email__ = "qiwang.mse@gmail.com"


def list_like():
    return (list, np.ndarray, tuple, pd.Series)


def read_lammps_dump(data_path):
    if os.path.exists(data_path):
        with open(data_path, 'r') as rf:
            lines = rf.readlines()
        cols = list()
        bds = [list(map(float, lines[5].strip().split())),
               list(map(float, lines[6].strip().split())),
               list(map(float, lines[7].strip().split()))]
        column_names = lines[8].strip().split()[2:]
        for idx, line in enumerate(lines):
            if idx > 8:
                column_values = [eval(x) for x in line.strip().split()]
                cols.append(column_values)
        df = pd.DataFrame(cols, columns=column_names)
        df.index = list(df["id"])
        return df, bds
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
        bds = [list(map(float, file_lines[5].strip().split())),
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
        return cell_coords, types, coords, bds, id_list


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
