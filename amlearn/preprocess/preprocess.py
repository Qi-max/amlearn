import os
import numpy as np
import time
from amlearn.preprocess.base import BasePreprocess
from amlearn.utils.logging import get_logger, setup_logger
from imblearn.utils.testing import all_estimators
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class DataPreprocess(BasePreprocess):

    def __init__(self, scaler=None, preprocess_params=None, test_split=None,
                 logger_path=None):
        """

        Args:
            scaler:
            preprocess_params:
            test_split: use train test split, test percent
        """
        self.scaler = scaler if scaler is not None \
            else [('standard_scaler', StandardScaler())]

        self.preprocess_params = preprocess_params \
            if preprocess_params is not None else {}
        self.test_split = test_split
        setup_logger(logger_path=logger_path)
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("Initialize preprocess.")

    def fit_transform(self, X, y=None):
        self.logger.info("Setup preprocess pipeline.")
        start_time = time.time()
        scaler_pl = Pipeline(self.scaler).set_params(**self.preprocess_params)
        self.logger.info("Preprocess pipeline is \n{}.".format(scaler_pl))

        if self.test_split is not None:
            self.logger.info("First start train_validation test split, "
                             "kwargs is {}".format(self.test_split))
            split_start = time.time()
            X_train_val, X_test, y_tarin_val, y_test = \
                train_test_split(X, y, **self.test_split)
            self.logger.info("Train_validation test split finish in {:.4f} "
                             "seconds".format(time.time() - split_start))

            X_train_val = scaler_pl.fit_transform(X_train_val)
            X_test = scaler_pl.transform(X_test)
            self.logger.info("Preprocess finish in {:.4f} seconds.".format(
                time.time() - start_time))
            return X_train_val, y_tarin_val, X_test, y_test
        else:
            X = scaler_pl.fit_transform(X)
            self.logger.info("Preprocess finish in {:.4f} seconds.".format(
                time.time() - start_time))
            return X, y, None, None


class ImblearnPreprocess(BasePreprocess):

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