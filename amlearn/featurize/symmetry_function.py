import numpy as np
import pandas as pd
from functools import reduce
from itertools import combinations_with_replacement
from amlearn.featurize.base import create_featurizer_backend
from amlearn.utils.packing import load_radii
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from amlearn.featurize.src import bp_symmfunc
except Exception:
    print("import fortran file bp_symmfunc error!\n")

__author__ = "Qi Wang"
__email__ = "qiwang.mse@gmail.com"


class BPRadialFunction(BaseEstimator, TransformerMixin):
    def __init__(self, bds, atom_type_symbols, pbc=None,
                 delta_r=0.2, n_r=50, cutoff=6.5,
                 id_col='id', type_col='type', coords_cols=None,
                 backend=None, verbose=1, save=True, output_path=None,
                 output_file_prefix='feature_bp_radial_function',
                 print_freq=1000):
        self.bds = bds
        self.pbc = np.array([1, 1, 1]) if pbc is None else pbc
        self.atom_type_symbols = atom_type_symbols
        self.delta_r = delta_r
        self.n_r = n_r
        self.cutoff = cutoff
        self.id_col = id_col
        self.type_col = type_col
        self.coords_cols = ["x", "y", "z"] if coords_cols is None \
            else coords_cols
        self.save = save
        self.verbose = verbose
        self.backend = backend if backend is not None \
            else create_featurizer_backend(output_path=output_path)
        self.output_file_prefix = output_file_prefix
        self.print_freq = print_freq

    @classmethod
    def default_from_system(cls, bds, atom_type_symbols, ref_atom_number,
                     delta_r=0.1, n_r=50, cutoff=None, pbc=None,
                     sigma_AA=None, radii=None, radius_type="miracle_radius",
                     id_col='id', type_col='type', coords_cols=None,
                     backend=None, verbose=1, save=True, output_path=None,
                     output_file_prefix='feature_bp_radial_function',
                     print_freq=1000):
        radii = load_radii() if radii is None else radii
        if sigma_AA is None:
            sigma_AA = \
                radii[str(ref_atom_number)][radius_type] * 2
        delta_r = sigma_AA * delta_r
        cutoff = (2.5 * sigma_AA) if cutoff is None else cutoff

        return cls(bds=bds, atom_type_symbols=atom_type_symbols, pbc=pbc,
                   delta_r=delta_r, n_r=n_r, cutoff=cutoff,
                   id_col=id_col, type_col=type_col, coords_cols=coords_cols,
                   backend=backend, verbose=verbose, save=save,
                   output_path=output_path,
                   output_file_prefix=output_file_prefix,
                   print_freq=print_freq)

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def transform(self, X):
        n_atoms = len(X)
        atom_ids = X[[self.id_col]].values
        atom_types = X[[self.type_col]].values
        atom_coords = X[self.coords_cols].values

        radial_funcs = np.zeros(
            (n_atoms, self.n_r * len(self.atom_type_symbols)), dtype=np.float)

        radial_funcs = bp_symmfunc.bp_radial(
            center_atom_ids=atom_ids, center_atom_coords=atom_coords,
            atom_ids=atom_ids, atom_types=atom_types,
            atom_type_symbols=self.atom_type_symbols,
            atom_coords=atom_coords, pbc=self.pbc, bds=self.bds,
            cutoff=self.cutoff, delta_r=self.delta_r, n_r=self.n_r,
            radial_funcs=radial_funcs, print_freq=self.print_freq)

        radial_funcs_df = pd.DataFrame(radial_funcs,
                                       index=atom_ids.transpose().tolist()[0],
                                       columns=self.get_feature_names())

        if self.save:
            self.backend.save_featurizer_as_dataframe(
                output_df=radial_funcs_df, name=self.output_file_prefix)

        return radial_funcs_df

    def get_feature_names(self):
        return reduce(list.__add__,
                      ([["{}_{:.3f}".format(str(t), i)
                         for i in np.arange(0, self.n_r) * self.delta_r]
                        for t in self.atom_type_symbols]))

class BPAngularFunction(BaseEstimator, TransformerMixin):
    def __init__(self, bds, atom_type_symbols, ksaais, lambdas,
                 zetas, pbc=None, cutoff=6.5,
                 id_col='id', type_col='type', coords_cols=None,
                 backend=None, verbose=1, save=True, output_path=None,
                 output_file_prefix='feature_bp_angular_function',
                 print_freq=1000):
        self.bds = bds
        self.atom_type_symbols = atom_type_symbols
        self.ksaais = ksaais
        self.lambdas = lambdas
        self.zetas = zetas
        self.pbc = np.array([1, 1, 1]) if pbc is None else pbc
        self.cutoff = cutoff
        self.id_col = id_col
        self.type_col = type_col
        self.coords_cols = ["x", "y", "z"] if coords_cols is None \
            else coords_cols
        self.save = save
        self.verbose = verbose
        self.backend = backend if backend is not None \
            else create_featurizer_backend(output_path=output_path)
        self.output_file_prefix = output_file_prefix
        self.print_freq = print_freq

    @classmethod
    def default_from_system(cls, ref_atom_number, atom_type_symbols, ksaais,
                 lambdas, zetas, bds, cutoff=None, pbc=None, sigma_AA=None,
                 radii=None, radius_type="miracle_radius",
                 id_col='id', type_col='type', coords_cols=None,
                 backend=None, verbose=1, save=True, output_path=None,
                 output_file_prefix='feature_bp_angular_function',
                 print_freq=1000):
        radii = load_radii() if radii is None else radii
        sigma_AA = sigma_AA if sigma_AA is not None else \
            radii[str(ref_atom_number)][radius_type] * 2
        ksaais = ksaais * sigma_AA  # in this case, ksaais are in the unit of sigma_AA
        cutoff = (2.5 * sigma_AA) if cutoff is None else cutoff

        return cls(bds=bds, atom_type_symbols=atom_type_symbols,
                   ksaais=ksaais, lambdas=lambdas, zetas=zetas,
                   pbc=pbc, cutoff=cutoff,
                   id_col=id_col, type_col=type_col, coords_cols=coords_cols,
                   backend=backend, verbose=verbose, save=save,
                   output_path=output_path,
                   output_file_prefix=output_file_prefix,
                   print_freq=print_freq)

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def transform(self, X):
        n_atoms = len(X)
        n_atom_types = len(self.atom_type_symbols)
        atom_ids = X[[self.id_col]].values
        atom_types = X[[self.type_col]].values
        atom_coords = X[self.coords_cols].values

        angular_funcs = \
            np.zeros((n_atoms, int(n_atom_types * (n_atom_types + 1) /
                                   2 * len(self.ksaais))),
                     dtype=np.float)

        angular_funcs = bp_symmfunc.bp_angular(
            center_atom_ids=atom_ids, center_atom_coords=atom_coords,
            atom_ids=atom_ids, atom_types=atom_types,
            atom_type_symbols=self.atom_type_symbols,
            atom_coords=atom_coords, pbc=self.pbc, bds=self.bds,
            ksaais=self.ksaais, lambdas=self.lambdas, zetas=self.zetas,
            cutoff=self.cutoff, angular_funcs=angular_funcs,
            print_freq=self.print_freq)

        angular_funcs_df = pd.DataFrame(angular_funcs,
                                        index=atom_ids.transpose().tolist()[0],
                                        columns=self.get_feature_names())

        if self.save:
            self.backend.save_featurizer_as_dataframe(
                output_df=angular_funcs_df, name=self.output_file_prefix)

        return angular_funcs_df

    def get_feature_names(self):
        return reduce(list.__add__,
                      ([["{}_{}_{:.3f}_{:.3f}_{:.3f}".format(
                          str(t1), str(t2), i, j, k)
                         for i, j, k in zip(self.ksaais,
                                            self.lambdas, self.zetas)]
                        for t1, t2 in combinations_with_replacement(
                              self.atom_type_symbols, 2)]))
