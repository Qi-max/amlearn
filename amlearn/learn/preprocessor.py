import numpy as np
import time
import pandas as pd
from amlearn.learn.base import BasePreprocessor
from amlearn.utils.logging import get_logger, setup_logger
from imblearn.utils.testing import all_estimators
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class Preprocessor(BasePreprocessor):

    def __init__(self, feature_selector=None, scaler=None,
                 preprocessor_params=None, test_split=None, logger_file=None):
        """

        Args:
            scaler:
            preprocessor_params:
            test_split: use train test split, test percent
        """
        self.feature_selector = \
            feature_selector if feature_selector is not None else \
                [('variance_check',
                  VarianceThreshold(threshold=(.8 * (1 - .8))))]

        self.scaler = scaler

        self.preprocessor_params = preprocessor_params \
            if preprocessor_params is not None else {}

        self.test_split = test_split
        setup_logger(logger_file=logger_file)
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("Initialize preprocessor.")

    def fit_transform(self, X, y=None):
        """

        Args:
            X: DataFrame
            y: DataFrame

        Returns:
            X_train_val, y_train_val, X_test, y_test

        """
        start_time = time.time()
        fs_pl = Pipeline(self.feature_selector).set_params(
            **self.preprocessor_params)
        self.logger.info("Feature Selector pipeline is \n\t{}.".format(fs_pl))

        feature_names = np.array(X.columns)
        X_index = X.index

        X = fs_pl.fit_transform(X)

        for fs_name, fs_obj in fs_pl.named_steps.items():
            deleted_features = feature_names[~(fs_obj.get_support())]
            if deleted_features:
                self.logger.info(
                    "Feature Selector {} delete features {}.".format(
                        fs_name, deleted_features))
                feature_names = feature_names[fs_obj.get_support()]
        self.feature_names_ = feature_names

        self.logger.info("Feature Selector pipeline finish in {:.4f} seconds.".format(
            time.time() - start_time))

        if self.scaler is not None:
            scaler_pl = Pipeline(self.scaler).set_params(**self.preprocessor_params)
            self.logger.info("Scaler pipeline is \n\t{}.".format(scaler_pl))

        if self.test_split is not None:
            self.logger.info("Start train_validation test split, "
                             "kwargs is {}".format(self.test_split))
            split_start = time.time()
            X_train_val, X_test, y_train_val, y_test = \
                train_test_split(X, y, **self.test_split)
            self.logger.info("Train_validation test split finish in {:.4f} "
                             "seconds".format(time.time() - split_start))
            if self.scaler is not None:
                scaler_start = time.time()
                X_train_val = scaler_pl.fit_transform(X_train_val)
                X_test = scaler_pl.transform(X_test)
                self.logger.info("Scaler pipeline finish in {:.4f} seconds.".format(
                    time.time() - scaler_start))

                self.logger.info("Preprocessor finish in {:.4f} seconds.".format(
                    time.time() - start_time))
            X_train_val = pd.DataFrame(X_train_val, columns=feature_names)
            X_test = pd.DataFrame(X_test, columns=feature_names)
            return X_train_val, y_train_val, X_test, y_test
        else:
            if self.scaler is not None:
                scaler_start = time.time()
                X = scaler_pl.fit_transform(X)
                self.logger.info("Scaler pipeline finish in {:.4f} seconds.".format(
                    time.time() - scaler_start))

            self.logger.info("Preprocessor finish in {:.4f} seconds.".format(
                time.time() - start_time))
            X = pd.DataFrame(X, columns=feature_names, index=X_index)
            return X , y, None, None

    def get_feature_names(self):
        msg = ("This %(name)s instance is not fitted yet. Call 'fit_transform' "
               "with appropriate arguments before using this method.")

        if not hasattr(self, 'feature_names_'):
            raise NotFittedError(msg % {'name': type(self).__name__})
        return self.feature_names_


class ImblearnPreprocessor(BasePreprocessor):

    @property
    def valid_components(self):
        if not hasattr(self, "valid_components_"):
            self.valid_components_ = np.array(all_estimators())

        return self.valid_components_

    def fit(self, X, y, imblearn_method, imblearn_kwargs):
        return self.fit_transform(X=X, y=y,
                                  imblearn_method=imblearn_method,
                                  imblearn_kwargs=imblearn_kwargs)

    def fit_transform(self, X, y, imblearn_method, imblearn_kwargs):
        imblearn_methods = dict(self.valid_components)

        if isinstance(imblearn_method, str):
            if imblearn_method in imblearn_methods.keys():
                imblearn_method = imblearn_methods[imblearn_method]
            else:
                raise ValueError('imblearn_method {} is unknown,Possible values'
                                 ' are {}'.format(imblearn_method,
                                                  imblearn_methods.keys()))

        elif callable(imblearn_method):
            if imblearn_method.__name__ not in imblearn_methods.keys():
                raise ValueError('imblearn_method {} is unknown,Possible values'
                                 ' are {}'.format(imblearn_method,
                                                  imblearn_methods.keys()))
        else:
            raise ValueError('imblearn_method {} is unknown,Possible values'
                             ' are {}'.format(imblearn_method,
                                              imblearn_methods.keys()))

        imbalanced_sampling = imblearn_method

        if imblearn_kwargs:
            X, y = imbalanced_sampling(**imblearn_kwargs).fit_sample(X, y)
        else:
            X, y = imbalanced_sampling().fit_sample(X, y)

        return X, y