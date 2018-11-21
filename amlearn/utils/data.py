import os
import numpy as np
import pandas as pd


def list_like():
    return (list, np.ndarray, tuple, pd.Series)


def read_imd(dirs, bds, atom_names):
    cell_coords = []
    types = []
    coords = []
    i = 0
    with open(dirs, 'r') as file_object:
        file_lines = file_object.readlines()
        print("file length is: {}".format(len(file_lines)))
        for line in file_lines:
            # if i > 100:
            #     continue
            i += 1
            if i == 6:
                cell_coords.append(
                    [float(line.strip().split()[0] ) *2, 0, 0])
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
                        raise TypeError("Please make sure bds is list!")
                if validate:
                    types.append(atom_names[int(data_line[1] ) -1])
                    coords.append([float(data_line[2]), float(data_line[3]),
                                   float(data_line[4])])
    return cell_coords, types, coords