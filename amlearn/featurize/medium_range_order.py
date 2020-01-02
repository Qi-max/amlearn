import warnings
import numpy as np
import pandas as pd
from amlearn.featurize.base import BaseFeaturize
from amlearn.utils.check import check_neighbor_col
from amlearn.utils.data import get_isometric_lists
from amlearn.utils.verbose import VerboseReporter

try:
    from amlearn.featurize.src import mro_stats, voronoi_stats
except Exception:
    print("import fortran file mro_stats/voronoi_stats error!\n")

__author__ = "Qi Wang"
__email__ = "qiwang.mse@gmail.com"


class MRO(BaseFeaturize):
    def __init__(self, backend=None, neighbor_num_limit=80, stat_ops="all",
                 stats_types=None, stats_names=None, neighbor_cols="all",
                 calc_features="all", save=True, output_path=None,
                 output_file_prefix='feature_mro'):
        super(MRO, self).__init__(save=save,
                                  backend=backend,
                                  output_path=output_path)
        if stats_types is None:
            self.stat_ops = stat_ops if stat_ops != 'all' \
                else ['sum', 'mean', 'std', 'min', 'max', 'diff']
            self.stats_types = [1 if 'sum' in self.stat_ops else 0,
                                1 if 'mean' in self.stat_ops else 0,
                                1 if 'std' in self.stat_ops else 0,
                                1 if 'min' in self.stat_ops else 0,
                                1 if 'max' in self.stat_ops else 0,
                                1 if 'diff' in self.stat_ops else 0]
        else:
            self.stats_types = stats_types if stats_types != "all" \
                else [1, 1, 1, 1, 1, 1]

        self.neighbor_num_limit = neighbor_num_limit
        self.neighbor_cols = check_neighbor_col(neighbor_cols)
        self.calc_features = calc_features
        self.stats_names = stats_names if stats_names is not None else \
            ['MRO_sum', 'MRO_mean', 'MRO_std', 'MRO_min', 'MRO_max', 'MRO_diff']
        self.output_file_prefix = output_file_prefix
        self.neighbor_ids_col = 'neighbor_ids_{}'

        # calced_sysmm_percent used in get_feature_names() method, it tags
        # whether 'Avg i-fold symm idx' features have been calculated.
        self.calced_sysmm_percent = False

    def fit(self, X, dependent_df):
        self.calc_features = list(X.columns) \
            if self.calc_features == "all" else self.calc_features
        if not set(self.calc_features).issubset(set(list(X.columns))):
            raise ValueError(
                "self.calc_features have unknown features: {}. Supported "
                "features are: {}".format(
                    set(list(X.columns)) - set(self.calc_features),
                    list(X.columns)))

        self.dependent_df = dependent_df
        self.calced_neighbor_cols = list()

        for neighbor_col in self.neighbor_cols:
            if neighbor_col not in list(self.dependent_df.columns):
                if self.save:
                    self.backend.logger.warning(
                        "neighbor column {} is not in dependent DataFrame. So "
                        "ignore this neighbor type calculation and continue "
                        "to calculate next neighbor type.".format(neighbor_col))
                else:
                    warnings.warn(
                        "neighbor column {} is not in dependent DataFrame. So "
                        "ignore this neighbor type calculation and continue "
                        "to calculate next neighbor type.".format(neighbor_col))

                continue
            self.calced_neighbor_cols.append(neighbor_col)
        if not self.calced_neighbor_cols:
            raise ValueError(
                "dependent_df don't have any neighbor_col, please make sure {} "
                "is in dependent_df columns.".format(self.neighbor_cols))
        return self

    def transform(self, X):
        feature_lists = None
        n_atoms = len(X)

        # define print verbose
        if self.verbose > 0 and self.save:
            vr = VerboseReporter(self.backend, total_stage=1,
                                 verbose=self.verbose, max_verbose_mod=10000)
            vr.init(total_epoch=len(self.calced_neighbor_cols) *
                                len(self.calc_features), start_epoch=0,
                    init_msg='Calculating Medium Range Order features, '
                             'statistics from {} features.'.format(
                        len(self.calced_neighbor_cols) *
                        len(self.calc_features)),
                    epoch_name='Feature', stage=1)
            vr_idx = 0

        for neighbor_col in self.calced_neighbor_cols:
            neighbor_tag = neighbor_col.split('_')[-1]

            neighbor_num_list = self.dependent_df[neighbor_col].values
            neighbor_ids_lists = get_isometric_lists(
                self.dependent_df[
                    self.neighbor_ids_col.format(neighbor_tag)].values,
                limit_width=self.neighbor_num_limit)
            for feature in self.calc_features:
                if not feature.endswith(neighbor_tag):
                    continue
                mro_feature = np.zeros(
                    (len(X), sum(self.stats_types)))
                mro_feature = mro_stats.sro_to_mro(
                    X[feature].values,
                    neighbor_num_list, neighbor_ids_lists,
                    self.stats_types, mro_feature, n_atoms=n_atoms,
                    n_neighbor_limit=self.neighbor_num_limit,
                    sum_stats_types=sum(self.stats_types))
                feature_lists = np.append(feature_lists, mro_feature, axis=1) \
                    if feature_lists is not None else mro_feature
                if self.verbose > 0 and self.save:
                    vr_idx += 1
                    vr.update(vr_idx)

        result_df = pd.DataFrame(feature_lists, index=X.index,
                                 columns=self.get_common_names())

        voro_mean_cols = [col for col in result_df.columns
                          if col.startswith('MRO_mean_Voronoi_idx_')]

        if voro_mean_cols:
            self.calced_sysmm_percent = True
            self.idx_list = [col.split('_')[4] for col in voro_mean_cols]
            percent_list = \
                line_percent(value_list=result_df[voro_mean_cols].values)
            percent_df = pd.DataFrame(percent_list, index=X.index,
                                      columns=self.get_symm_percent_names())
            result_df = result_df.join(percent_df)

        if self.save:
            self.backend.save_featurizer_as_dataframe(
                output_df=result_df, name=self.output_file_prefix)

        return result_df

    def get_common_names(self):
        feature_names = list()
        for neighbor_col in self.calced_neighbor_cols:
            neighbor_tag = neighbor_col.split('_')[-1]
            feature_names += \
                ["{}_{}".format(stats_name, feature)
                 for feature in self.calc_features if feature.endswith(neighbor_tag)
                 for stats_name, stats_type in zip(self.stats_names,
                                                   self.stats_types)
                 if stats_type == 1]
        return feature_names

    def get_symm_percent_names(self):
        feature_names = ['MRO_Avg_{}_fold_symm_idx'.format(edge)
                         for edge in self.idx_list]
        return feature_names

    def get_feature_names(self):
        feature_names = self.get_common_names()
        if self.calced_sysmm_percent:
            feature_names += self.get_symm_percent_names()
        return feature_names

    @property
    def category(self):
        return 'mro'


def line_percent(value_list):
    percent_list = np.zeros(value_list.shape)

    percent_list = \
        voronoi_stats.line_percent(percent_list, value_list)
    return percent_list
