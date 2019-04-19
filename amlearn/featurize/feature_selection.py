import time
from amlearn.utils.logging import setup_logger, get_logger
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold


class FeatureSelection(BaseEstimator, TransformerMixin):

    def __init__(self, feature_selection=None, fs_params=None, logger_path=None):
        """

        Args:
            feature_selection:
            fs_params:
            logger_path:

            TODO: More logger params could be customized.

        """
        self.feature_selection = \
            feature_selection if feature_selection is not None \
                else [('variance_check',
                       VarianceThreshold(threshold=(.8 * (1 - .8))))]

        self.fs_params = fs_params if fs_params is not None else {}
        setup_logger(logger_path=logger_path)
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("Initialize feature selection.")

    def fit_transform(self, X):
        start_time = time.time()
        fs_pl = Pipeline(
            self.feature_selection).set_params(**self.fs_params)
        X = fs_pl.fit_transform(X)
        self.logger.info("Feature selection finish in {:.4f} seconds.".format(
            time.time() - start_time))
        return X
