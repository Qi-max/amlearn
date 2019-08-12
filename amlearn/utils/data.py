import os
import numpy as np
import pandas as pd


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
