"""Use scikit-learn regressor algorithm to regress data.

This module contains imblearn method to deal with the imbalanced problems and
scikit-learn regressor algorithms to regress data and cross_validate method
to evaluate estimator's performance.

"""
import os
import time
import joblib
import numpy as np
from collections import OrderedDict
from amlearn.learn.base import AmBaseLearn
from amlearn.learn.sklearn_patch import calc_scores, cross_validate
from amlearn.utils.check import appropriate_kwargs
from amlearn.utils.data import list_like
from amlearn.utils.directory import write_file, create_path
from sklearn.base import RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
try:
    from sklearn.utils import all_estimators
except:
    from sklearn.utils.testing import all_estimators
from sklearn.utils.validation import check_is_fitted

__author__ = "Qi Wang"
__email__ = "qiwang.mse@gmail.com"


class AmRegressor(AmBaseLearn):
    """Base regressor class of amlearn.

    Args:
        backend: Backend object (default: None)
            MLBackend object which defined output_path, environment
            configuration, save_predictions, and so on.
            If default, use default MLBackend object.
        regressor: object, string or class (default: None)
            Sklearn regressor object.
            If default, use the default_regressor which defined in
            backend environment configuration.
        regressor_params: dict (default: None)
            The regressor parameters.
        #TODO: Define decimals and seed later.
        decimals: int (default: None)
            Output decimals.
        seed: int (default: 1)
            Random seed.

    """
    def __init__(self, backend=None, output_path='tmp', regressor=None,
                 regressor_params=None, decimals=None, seed=1):
        super().__init__(backend, output_path, decimals=decimals, seed=seed)

        # Get all supported regressors
        regressors_dict = dict(self.valid_components)

        # Set regressor object
        if regressor is None:
            regressor_name = self.default_regressor
        elif isinstance(regressor, RegressorMixin):
            regressor_name = regressor.__class__.__name__
        elif callable(regressor):
            regressor_name = regressor.__name__
        elif isinstance(regressor, str):
            regressor_name = regressor
        else:
            regressor_name = None

        if regressor_name not in regressors_dict.keys():
            raise ValueError('regressor {} is unknown, Possible values '
                             'are {}'.format(regressor,
                                             regressors_dict.keys()))

        if regressor_params is None:
            regressor_params = {}

        self.regressor = \
            regressor if isinstance(regressor, RegressorMixin) \
            else regressors_dict[regressor_name](**regressor_params)
        # self.regressor = regressor

        self.backend.logger.info(
            'Initialize regression, regressor is : \n\t{}'.format(
                self.regressor))

    def fit(self, X, y, val_size=0.3, scoring=None, random_state=None,
            cv_num=1, cv_params=None, save_model=True, save_score=True,
            save_prediction=True, prediction_types='dataframe',
            save_train_val_idx=True, save_feature_importances=True,
            **fit_params):
        """Fit the amlearn regressor model.

        Args:
            X (list like): The data to fit.
            y (list like): The target variable to try to predict in the case of
                supervised learning.
            val_size (float, int or None): If float, should be between 0.0 and
                1.0 and represent the proportion of the dataset to include in
                the validation split. If int, represents the absolute number of
                validation samples. If None, the value is set to the complement
                of the train size. By default, the value is set to 0.3.
            random_state (int, RandomState instance or None): If int,
                random_state is the seed used by the random number generator;
                If RandomState instance, random_state is the random number
                generator; If None, the random number generator is the
                RandomState instance used by `np.random`.
            scoring (string, callable or None): A string (see model evaluation
                documentation) or a scorer callable object / function with
                signature ``scorer(estimator, X, y)``.
            cv_num (int): Determines the cross-validation splitting strategy.
                If cv_num is 1, not use cross_validation, just fit the model
                by train_test_split.
            cv_params (dict): Cross-validation parameters.
            save_train_val_idx (boolean): Whether to save the train
                validation indexes.
            save_model (boolean): Whether to save the model.
            save_prediction (boolean): Whether to save the prediction from each
                cross validation model.
            prediction_types (str): It effect when save_prediction is True.
                The Optional parameters are: {"npy", "txt", "dataframe"}.
            **fit_params: regressor fit parameters.

        Returns:
            self
        """
        self.scoring = scoring
        self.feature_names_ = X.columns
        self._fit_cv(
            X, y, random_state=random_state, scoring=scoring,
            cv_params=cv_params, cv_num=cv_num, val_size=val_size,
            save_model=save_model, save_score=save_score,
            save_prediction=save_prediction, prediction_types=prediction_types,
            save_feature_importances=save_feature_importances,
            save_train_val_idx=save_train_val_idx, **fit_params)
        return self

    def _fit(self, X, y, regressor, val_size=0.3, random_state=None,
             scoring=None, return_train_score=True, **fit_params):
        result_dict = dict()
        indices = range(np.array(X).shape[0])

        if 'train_idx' in fit_params.keys() and 'val_idx' in fit_params.keys():
            train_idx = fit_params['train_idx']
            val_idx = fit_params['val_idx']
            X_train = X.loc[train_idx]
            X_val = X.loc[val_idx]
            y_train = y.loc[train_idx]
            y_val = y.loc[val_idx]
        else:
            X_train, X_val, y_train, y_val, train_idx, val_idx = \
                train_test_split(X, y, indices,
                                 test_size=val_size, random_state=random_state)

        train_idx = list(map(int, train_idx))
        val_idx = list(map(int, val_idx))

        np.random.seed(random_state)
        regressor_params = appropriate_kwargs(fit_params, regressor.fit)
        regressor = regressor.fit(X_train, y_train, **regressor_params)
        result_dict['estimators'] = [regressor]
        result_dict['indices'] = [[train_idx, val_idx]]

        val_scores, scorers = calc_scores(X=X_val, y=y_val,
                                          estimator=regressor,
                                          scoring=scoring)

        if return_train_score:
            train_scores, _ = calc_scores(X=X_train, y=y_train,
                                          estimator=regressor,
                                          scoring=scoring)
        for name in scorers:
            result_dict['test_%s' % name] = [val_scores[name]]
            if return_train_score:
                key = 'train_%s' % name
                result_dict[key] = [train_scores[name]]
                if return_train_score == 'warn':
                    message = (
                        'You are accessing a training score ({!r}), '
                        'which will not be available by default '
                        'any more in 0.21. If you need training scores, '
                        'please set return_train_score=True').format(key)
                    # warn on key access
                    result_dict.add_warning(key, message, FutureWarning)
        return result_dict, scorers

    def _fit_cv(self, X, y, val_size=0.3, random_state=None, scoring=None,
                cv_num=1, cv_params=None, save_train_val_idx=True,
                save_model=True, save_score=True, save_prediction=True,
                prediction_types='dataframe', save_feature_importances=True,
                **fit_params):

        # If user's cv_params contains 'cv_num' parameter, use the max value
        # between function parameter 'cv_num' and cv_params's 'cv_num'.
        self.backend.logger.info('Start Cross Validation.')
        cv_start_time = time.time()

        if cv_params is None:
            cv_params = {}

        if 'cv_num' in cv_params.keys():
            cv_num = max(cv_num, cv_params['cv_num'])
            cv_params.pop('cv_num')

        if 'scoring' in cv_params.keys():
            cv_params.pop('scoring')

        return_train_score = cv_params.get('return_train_score', True)
        if cv_num > 1:
            if random_state is False:
                pass
            else:
                np.random.seed(random_state)
            results, scorers = \
                cross_validate(estimator=self.regressor, scoring=scoring,
                               fit_params=fit_params, X=X, y=y, cv=cv_num,
                               **cv_params)
        else:
            results, scorers = self._fit(
                X, y, self.regressor, val_size=val_size,
                return_train_score=return_train_score,
                random_state=random_state, scoring=scoring, **fit_params)
            cv_num = 1

        # TODO: now if scorers list length is more than 1, score_name only can
        #  be the first of them.
        self.score_name = self.score_name if hasattr(self, 'score_name') \
            else list(scorers.keys())[0]
        self.best_score_, (self.best_model_, self.best_model_tag_)= \
            max(zip(results['test_{}'.format(self.score_name)],
                    zip(results['estimators'],
                        [''] if cv_num == 1 else
                        ["cv_{}".format(i) for i in range(cv_num)])),
                key=lambda x: x[0])

        self.backend.logger.info(
            "\tCV regression finish in {:.4f} seconds.".format(
                time.time() - cv_start_time))
        if save_score:
            write_file(
                os.path.join(self.backend.output_path, 'mean_scores.txt'),
                '{}\n{}\n{}'.format(
                    ','.join(['dataset'] + list(scorers.keys())),
                    ','.join(['test'] +
                             [str(np.mean(results['test_{}'.format(
                                 score_name)]))
                              for score_name in scorers.keys()]),
                    ','.join(['train'] +
                             [str(np.mean(results['train_{}'.format(
                                 score_name)]))
                              for score_name in scorers.keys()])
                    if return_train_score else -1))

        for cv_idx in range(cv_num):
            cv_tag = "cv_{}".format(cv_idx)
            cv_output_path = os.path.join(self.backend.output_path, cv_tag)
            create_path(cv_output_path, merge=True)

            if save_score:
                write_file(os.path.join(cv_output_path, 'scores.txt'),
                           '{}\n{}\n{}'.format(
                               ','.join(['dataset'] + list(scorers.keys())),
                               ','.join(['test'] +
                                        [str(results['test_{}'.format(
                                            score_name)][cv_idx])
                                         for score_name in scorers.keys()]),
                               ','.join(['train'] +
                                        [str(results['train_{}'.format(
                                            score_name)][cv_idx])
                                         for score_name in scorers.keys()])
                               if return_train_score else -1))

            score_model = results['estimators'][cv_idx]
            if save_model:
                self.backend.save_model(score_model, cv_tag)
            if save_feature_importances:
                self.backend.save_json(
                    self.feature_importances_dict(score_model), cv_tag,
                    name='feature_importances')
            if save_train_val_idx:
                train_idx = results['indices'][cv_idx][0]
                val_idx = results['indices'][cv_idx][1]
                write_file(os.path.join(cv_output_path, 'train_idx.txt'),
                           "\n".join(list(map(str, train_idx))))

                write_file(os.path.join(cv_output_path, 'val_idx.txt'),
                           "\n".join(list(map(str, val_idx))))
            if save_prediction:
                predictions = \
                    score_model.predict(X.iloc[results['indices'][cv_idx][1]])
                targets_and_predictions = \
                    np.array(list(zip(y.iloc[results['indices'][cv_idx][1]],
                                      predictions)))

                if not isinstance(prediction_types, list_like()):
                    prediction_types = [prediction_types]
                for predict_type in prediction_types:
                    if predict_type in self.backend.valid_predictions_type:
                        instance = getattr(self.backend,
                                           'save_predictions_as_{}'.format(
                                               predict_type))
                        instance(targets_and_predictions, cv_tag)
                    else:
                        raise ValueError(
                            'predict_type {} is unknown, '
                            'Possible values are {}'.format(
                                predict_type,
                                self.backend.valid_predictions_type))
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.best_model_.predict(X)

    def calc_score(self, X, y, scoring=None):
        check_is_fitted(self)
        scores, _ = \
            calc_scores(X=X, y=y, estimator=self.best_model_,
                        scoring=scoring if scoring is not None
                        else self.scoring)
        return scores

    def save_best_model(self):
        check_is_fitted(self)
        model_file = \
            os.path.join(self.backend.output_path,
                         'best_model_{}.pkl'.format(self.best_model_tag_))
        joblib.dump(self.best_model_, model_file)

    @property
    def best_model(self):
        check_is_fitted(self)
        return self.best_model_

    def feature_importances_(self, model=None):
        check_is_fitted(self)
        return self.best_model_.feature_importances_ if model is None \
            else model.feature_importances_

    def feature_importances_dict(self, model=None):
        check_is_fitted(self)
        feature_importances_dict_ = \
            sorted(zip(self.get_feature_names(),
                       self.feature_importances_(model)),
                   key=lambda x: x[1], reverse=True)
        return OrderedDict(feature_importances_dict_)

    def get_feature_names(self):
        msg = ("This %(name)s instance is not fitted yet. Call 'fit_transform' "
               "with appropriate arguments before using this method.")

        if not hasattr(self, 'feature_names_'):
            raise NotFittedError(msg % {'name': type(self).__name__})
        return self.feature_names_

    @property
    def default_regressor(self):
        """Find the default regressor by reading the default environment file.

        Returns:
            default_regressor: RegressorMixin object
                Default regressor.

        """
        self.default_regressor_r = self.backend.def_env["default_regressor"]
        return self.default_regressor_r

    @property
    def valid_components(self):
        """Find all supported regressors.

        Returns:
            valid_components: numpy.array([[regressor name, object], ...])
                Valid regressors
        """
        if not hasattr(self, "valid_components_r"):
            regressors = np.array([est for est in all_estimators() if
                                   issubclass(est[1], RegressorMixin)])

            self.valid_components_r = regressors
        return self.valid_components_r
