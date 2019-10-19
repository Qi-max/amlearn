import numpy as np
import pandas as pd
from amlearn.featurize.base import create_featurizer_backend, load_radii
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from amlearn.featurize.src import bp_symmfunc
except Exception:
    print("import fortran file bp_symmfunc error!\n")

__author__ = "Qi Wang"
__email__ = "qiwang.mse@gmail.com"


class BPRadialFunction(BaseEstimator, TransformerMixin):
    def __init__(self, ref_atom_number, atom_type_symbols, bds,
                 delta_r=0.1, n_r=50, cutoff=None, pbc=None,
                 sigma_AA=None, radii=None, radius_type="miracle_radius",
                 id_col='id', type_col='type', coords_cols=None,
                 backend=None, verbose=1, save=True, output_path=None,
                 output_file_prefix='feature_bp_radial_function'):
        self.ref_atom_number = ref_atom_number
        self.atom_type_symbols = atom_type_symbols
        self.bds = bds
        self.delta_r = delta_r
        self.n_r = n_r
        self.pbc = np.array([1, 1, 1]) if pbc is None else pbc
        self.radii = load_radii() if radii is None else radii
        self.radius_type = radius_type
        self.id_col = id_col
        self.type_col = type_col
        self.coords_cols = ["x", "y", "z"] if coords_cols is None \
            else coords_cols
        self.save = save
        self.verbose = verbose
        self.backend = backend if backend is not None \
            else create_featurizer_backend(output_path=output_path)
        self.output_file_prefix = output_file_prefix

        # general setting
        if sigma_AA is None:
            sigma_AA = \
                self.radii[str(self.ref_atom_number)][self.radius_type] * 2
        self.delta_r = sigma_AA * self.delta_r
        self.cutoff = (2.5 * sigma_AA) if cutoff is None else cutoff

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
            radial_funcs=radial_funcs)

        radial_funcs_df = pd.DataFrame(radial_funcs,
                                       index=atom_ids.transpose().tolist()[0],
                                       columns=self.get_feature_names())

        if self.save:
            self.backend.save_featurizer_as_dataframe(
                output_df=radial_funcs_df, name=self.output_file_prefix)

        return radial_funcs_df

    def get_feature_names(self):
        return ["A_{:.3f}".format(i)
                for i in np.arange(0, self.cutoff_r, self.delta_r)] + \
               ["B_{:.3f}".format(i)
                for i in np.arange(0, self.cutoff_r, self.delta_r)]


class BPAngularFunction(BaseEstimator, TransformerMixin):
    def __init__(self, ref_atom_number, atom_type_symbols, ksaais, lambdas,
                 zetas, bds, cutoff=None, pbc=None, sigma_AA=None,
                 radii=None, radius_type="miracle_radius",
                 id_col='id', type_col='type', coords_cols=None,
                 backend=None, verbose=1, save=True, output_path=None,
                 output_file_prefix='feature_bp_angular_function'):
        self.ref_atom_number = ref_atom_number
        self.atom_type_symbols = atom_type_symbols
        self.radii = load_radii() if radii is None else radii
        self.radius_type = radius_type
        self.sigma_AA = sigma_AA if sigma_AA is not None else \
            self.radii[str(self.ref_atom_number)][self.radius_type] * 2
        self.ksaais = ksaais * self.sigma_AA
        self.lambdas = lambdas
        self.zetas = zetas
        self.bds = bds
        self.pbc = np.array([1, 1, 1]) if pbc is None else pbc
        self.id_col = id_col
        self.type_col = type_col
        self.coords_cols = ["x", "y", "z"] if coords_cols is None \
            else coords_cols
        self.save = save
        self.verbose = verbose
        self.backend = backend if backend is not None \
            else create_featurizer_backend(output_path=output_path)
        self.output_file_prefix = output_file_prefix
        self.cutoff = (2.5 * self.sigma_AA) if cutoff is None else cutoff


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
            cutoff=self.cutoff, angular_funcs=angular_funcs)

        angular_funcs_df = pd.DataFrame(angular_funcs,
                                        index=atom_ids.transpose().tolist()[0],
                                        columns=self.get_feature_names())

        if self.save:
            self.backend.save_featurizer_as_dataframe(
                output_df=angular_funcs_df, name=self.output_file_prefix)

        return angular_funcs_df

    def get_feature_names(self):
        return ["AA_{:.3f}_{:.3f}_{:.3f}".format(i, j, k)
                for i, j, k in zip(self.ksaais / self.sigma_AA,
                                   self.lambdas, self.zetas)] + \
               ["AB_{:.3f}_{:.3f}_{:.3f}".format(i, j, k)
                for i, j, k in zip(self.ksaais / self.sigma_AA,
                                   self.lambdas, self.zetas)] + \
               ["BB_{:.3f}_{:.3f}_{:.3f}".format(i, j, k)
                for i, j, k in zip(self.ksaais / self.sigma_AA,
                                   self.lambdas, self.zetas)]
