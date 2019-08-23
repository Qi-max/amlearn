import os
import gc
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from amlearn.utils.data import read_lammps_dump
from amlearn.utils.check import check_output_path

"""
This is an example script of deriving features for each atom, based on the 
Fortran source codes in amlearn/featurize/featurizers/src/ and classes in 
short/medium_range_order. Please make sure to compile the Fortran code using
f2py before running this script. 
"""

system = ["Cu65Zr35", "qr_5plus10^10"]

lammps_file = "xxx/dump.lmp"
structure, bds = read_lammps_dump(lammps_file)
print(structure)

n_atoms = len(structure)
atom_type = np.zeros(n_atoms, dtype=int)
atom_coords = structure[["x", "y", "z"]].iloc[0:n_atoms].values
pbc = np.array([0, 1, 0])

