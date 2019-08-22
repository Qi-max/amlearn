import os
import gc
import ast
import pandas as pd
import numpy as np
from . import fractal
import matplotlib.pyplot as plt
from amlearn.utils.data import read_lammps_dump
from amlearn.utils.check import check_output_path

"""
This is an example script of fractal analysis, based on the Fortran source 
codes in ./fractal.f90. Please make sure to compile the Fortran code using f2py
before running this script. 

Compiling command example:
f2py -c fractal.f90 -m fractal
"""

system = ["Cu65Zr35", "qr_5plus10^10"]

lammps_file = "xxx/dump.lmp"
structure, bds = read_lammps_dump(lammps_file)
print(structure)

n_atoms = len(structure)
atom_type = np.zeros(n_atoms, dtype=int)
atom_coords = structure[["x", "y", "z"]].iloc[0:n_atoms].values
pbc = np.array([0, 1, 0])

cutoff = 30.0
bin = 0.2

output_path = "xxx"
check_output_path(output_path)

prediction_file = "xx"
df = pd.read_csv(os.path.join(prediction_file), index_col="number")
qs_col = "QS_predict"

structure[qs_col] = df[qs_col]

qs_higher_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
qs_lower_thresholds = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
                       0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

for lower_threshold in qs_lower_thresholds:
    structure[qs_col + "_lower_than_{:.2f}".format(lower_threshold)] = \
        df[qs_col].apply(lambda x: 1 if x < lower_threshold else 0)

for higher_threshold in qs_higher_thresholds:
    structure[qs_col + "_higher_than_{:.2f}".format(higher_threshold)] = \
        df[qs_col].apply(lambda x: 1 if x > higher_threshold else 0)

cols = [qs_col + "_lower_than_{:.2f}".format(lower_threshold)
        for lower_threshold in qs_lower_thresholds] + \
       [qs_col + "_higher_than_{:.2f}".format(higher_threshold)
        for higher_threshold in qs_higher_thresholds]

# here 777 represents all atoms, which is for compatibility with Fortran.
center_types = np.array([777, 1, 2])

for prob_col in cols:
    selected_atoms = structure[structure[prob_col] == 1]
    # n_atoms = 20000
    n_atoms = len(selected_atoms)

    # check if the fractal file is already calculated
    output_file = os.path.join(output_path, "fractal_{}_{}_bin_{}_all_{}_col_{}.csv".format(system[0], system[1], bin, n_atoms, prob_col))
    if os.path.exists(output_file) and os.path.isfile(output_file):
        print("skip", prob_col)
        continue

    print("Fractal calculation started, "
          "the number of selected atoms:", prob_col, n_atoms)

    atom_type = selected_atoms[["type"]].iloc[0:n_atoms].values
    atom_coords = selected_atoms[["x", "y", "z"]].iloc[0:n_atoms].values

    fractal_distwise = np.zeros((int(cutoff / bin), len(center_types)),
                                    dtype=np.float64)
    fractal_accumulative = np.zeros((int(cutoff / bin), len(center_types)),
                                    dtype=np.float64)

    fractal_distwise, fractal_accumulative = fractal.fractal_intense(
        atom_type=atom_type, atom_coords=atom_coords, pbc=pbc, bds=bds,
        cutoff=cutoff, bin=bin, bin_num=int(cutoff / bin),
        center_types=center_types,
        fractal_distwise=fractal_distwise,
        fractal_accumulative=fractal_accumulative)

    fractal_distwise_df = pd.DataFrame(index=np.arange(0, cutoff, bin))
    fractal_accumulative_df = pd.DataFrame(index=np.arange(0, cutoff, bin))

    for idx, col in enumerate(center_types):
        fractal_distwise_df[prob_col + "_" + str(col) + "_distw"] = np.array(
            fractal_distwise)[:, idx]
        fractal_accumulative_df[prob_col + "_" + str(col) + "_acc"] = np.array(
            fractal_accumulative)[:, idx]

    plt.bar(np.arange(0, cutoff, bin), height=fractal_accumulative[:, 0])
    plt.bar(np.arange(0, cutoff, bin), height=fractal_distwise[:, 0])

    if n_atoms == len(selected_atoms):
        pd.concat([fractal_distwise_df, fractal_accumulative_df],
                  axis=1).to_csv(os.path.join(output_path, "fractal_{}_{}_bin_{}_all_{}_col_{}.csv".format(
                    system[0],
                    system[1],
                    bin,
                    n_atoms, prob_col)))
    else:
        pd.concat([fractal_distwise_df, fractal_accumulative_df],
                  axis=1).to_csv(os.path.join(output_path, "fractal_{}_{}_bin_{}_test_{}_col_{}.csv".format(
                    system[0],
                    system[1],
                    bin,
                    n_atoms, prob_col)))
    del [selected_atoms, fractal_distwise_df, fractal_accumulative_df]
    gc.collect()


def distwise_stats_to_gr(fractal_df, atom_num, volume, distwise_col_end="_distw"):
    # the default index of fractal_df is the center distance of each bin
    sphere_shell_vol = 4 * np.pi * (np.array(fractal_df.index)) ** 2 * bin

    for col in fractal.columns:
        if col.endswith(distwise_col_end):
            # not atom_num**2, we already divide atom_num in calculating _distw,
            fractal_df[col + "_vol_norm"] = \
                fractal_df[col] / sphere_shell_vol * volume / atom_num

    return fractal_df
