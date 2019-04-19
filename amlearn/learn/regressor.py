"""Use scikit-learn regressor algorithm to regress data.

This module contains imblearn method to deal with the imbalanced problems and
scikit-learn regressor algorithms to regress data and cross_validate method
to evaluate estimator's performance.

"""
import os
import numpy as np

import time
from amlearn.learn.base import AmBaseLearn
from amlearn.learn.sklearn_patch import cross_validate
from amlearn.learn.sklearn_patch import calc_scores
from amlearn.utils.backend import BackendContext, MLBackend
from amlearn.utils.data import list_like
from amlearn.utils.directory import write_file, create_path
from sklearn.base import RegressorMixin
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import all_estimators
from sklearn.utils.validation import check_is_fitted


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
    def __init__(self, backend=None, regressor=None, regressor_params=None,
                 decimals=None, seed=1):
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
        self.regressor = regressor if isinstance(regressor, RegressorMixin) \
            else regressors_dict[regressor_name](**regressor_params)

        # Set backend object
        if backend is None:
            backend_context = BackendContext(merge_path=True)
            backend = MLBackend(backend_context)

        super().__init__(backend, decimals=decimals, seed=seed)
        self.backend.logger.info(
            'Initialize regression, regressor is : \n\t{}'.format(
                self.regressor))

    def fit(self, X, y, random_state=None, scoring=None,
            cv_num=1, cv_params=None, val_size=0.3,
            save_model=True, save_prediction=True,
            save_train_val_idx=True, prediction_types='dataframe', **fit_params):
        """Fit the amlearn regressor model.

        Args:
            X: array-like
                The data to fit. Can be for example a list, or an array.
            y: array-like, optional, default: None
                The target variable to try to predict in the case of
                supervised learning.
            random_state: int, RandomState instance or None, optional (default=0)
                If int, random_state is the seed used by the random number generator;
                If RandomState instance, random_state is the random number generator;
                If None, the random number generator is the RandomState instance used
                by `np.random`.
            scoring: string, callable or None, optional, default: None
                A string (see model evaluation documentation) or
                a scorer callable object / function with signature
                ``scorer(estimator, X, y)``.
            cv_num: int. (default: 1)
                Determines the cross-validation splitting strategy.
                If cv_num is 1, not use cross_validation, just fit the
                model by train_test_split.
            cv_params: dict. (default: {})
                Cross-validation parameters.
            val_size: float, int or None, optional (default=0.3)
                If float, should be between 0.0 and 1.0 and represent the
                proportion of the dataset to include in the validation split.
                If int, represents the absolute number of validation samples.
                If None, the value is set to the complement of the train size.
                By default, the value is set to 0.3.
            save_model: boolean (default: True)
                Whether to save the model.
            save_prediction: boolean (default: True)
                Whether to save the prediction from each cross validation model.
            save_train_val_idx: boolean (default: True)
                Whether to save the train validation indexes.
            prediction_types: str (default: dataframe)
                It effect when save_prediction is True.
                The Optional parameters are: {"npy", "txt", "dataframe"}.
            **fit_params:
                regressor fit parameters.

        Returns:
            self
        """
        self.scoring = scoring
        self._fit_cv(
            X, y, random_state=random_state, scoring=scoring,
            cv_params=cv_params, cv_num=cv_num, val_size=val_size,
            save_model=save_model, save_prediction=save_prediction,
            prediction_types=prediction_types,
            save_train_val_idx=save_train_val_idx, **fit_params)
        return self

    def _fit(self, X, y, regressor, val_size=0.3, random_state=None,
             scoring=None, return_train_score=False, **fit_params):
        ret = dict()
        indices = range(np.array(X).shape[0])
        X_train, X_val, y_train, y_val, train_idx, val_idx = \
            train_test_split(X, y, indices,
                             test_size=val_size, random_state=random_state)

        train_idx = list(map(int, train_idx))
        val_idx = list(map(int, val_idx))

        np.random.seed(random_state)
        regressor = regressor.fit(X_train, y_train, **fit_params)
        ret['models'] = [regressor]
        ret['indexs'] = [[train_idx, val_idx]]

        val_scores, scorers = calc_scores(X=X_val, y=y_val,
                                          estimator=regressor,
                                          scoring=scoring)

        if return_train_score:
            train_scores, _ = calc_scores(X=X_train, y=y_train,
                                          estimator=regressor,
                                          scoring=scoring)
        for name in scorers:
            ret['test_%s' % name] = [val_scores[name]]
            if return_train_score:
                key = 'train_%s' % name
                ret[key] = [train_scores[name]]
                if return_train_score == 'warn':
                    message = (
                        'You are accessing a training score ({!r}), '
                        'which will not be available by default '
                        'any more in 0.21. If you need training scores, '
                        'please set return_train_score=True').format(key)
                    # warn on key access
                    ret.add_warning(key, message, FutureWarning)
        return ret, scorers

    def _fit_cv(self, X, y, random_state=None, scoring=None,
                cv_num=1, cv_params=None, val_size=0.3,
                save_model=True, save_prediction=True, prediction_types='dataframe',
                save_train_val_idx=True, **fit_params):

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

        if cv_num > 1:
            np.random.seed(random_state)
            results, scorers = \
                cross_validate(estimator=self.regressor, scoring=scoring,
                               fit_params=fit_params, X=X, y=y, cv=cv_num,
                               **cv_params)
        else:
            results, scorers = self._fit(
                X, y, self.regressor,
                val_size=val_size, random_state=random_state,
                return_train_score=cv_params.get('return_train_score', False),
                scoring=scoring, **fit_params)
            cv_num = 1

        # TODO: now if scoring is more than one, score_name only can be the first of them.
        self.score_name = self.score_name if hasattr(self, 'score_name') \
            else list(scorers.keys())[0]
        print(results['models'],
                        [''] if cv_num == 1 else
                        ["cv_{}".format(i) for i in range(cv_num)])
        self.best_score_, (self.best_model_, self.best_model_tag_)= \
            max(zip(results['test_{}'.format(self.score_name)],
                    zip(results['models'],
                        [''] if cv_num == 1 else
                        ["cv_{}".format(i) for i in range(cv_num)])),
                key=lambda x: x[0])

        # get temporary path, if self._tmp_path exist get it (most create by
        # imblearn), else get self.backend.tmp_path
        tmp_path = self._tmp_path \
            if hasattr(self, '_tmp_path') else self.backend.tmp_path

        self.backend.logger.info(
            "\tCV regression finish in {:.4f} seconds. "
            "It's best {} score is {:.4f}".format(
                time.time() - cv_start_time,
                self.score_name, self.best_score_))

        if save_model or save_train_val_idx or save_prediction:
            for cv_idx in range(cv_num):
                tmp_path_cv = os.path.join(tmp_path, "cv_{}".format(cv_idx))
                create_path(tmp_path_cv)
                score_model = results['models'][cv_idx]
                if save_model:
                    self.backend.save_model(score_model, self.seed)
                if save_train_val_idx:
                    train_idx = results['indexs'][cv_idx][0]
                    val_idx = results['indexs'][cv_idx][1]
                    write_file( os.path.join(tmp_path_cv, 'train_idx.txt'),
                                "\n".join(list(map(str, train_idx))))

                    write_file( os.path.join(tmp_path_cv, 'val_idx.txt'),
                                "\n".join(list(map(str, val_idx))))

                if save_prediction:
                    predictions = score_model.predict(X)
                    if not isinstance(prediction_types, list_like()):
                        prediction_types = [prediction_types]
                    for predict_type in prediction_types:
                        if predict_type in self.backend.valid_predictions_type:
                            getattr(self.backend,
                                    'save_predictions_as_{}'.format(
                                        predict_type))(predictions, self.seed)
                        else:
                            raise ValueError(
                                'predict_type {} is unknown, '
                                'Possible values are {}'.format(
                                    predict_type,
                                    self.backend.valid_predictions_type))
        self.backend.tmp_persistence(tmp_path)
        return self

    def predict(self, X):
        check_is_fitted(self, 'best_model_')
        return self.best_model_.predict(X)

    def calc_score(self, X, y, scoring=None):
        check_is_fitted(self, 'best_model_')
        scores, _ = \
            calc_scores(X=X, y=y, estimator=self.best_model_,
                        scoring=scoring if scoring is not None
                        else self.scoring)
        return scores

    def save_best_model(self):
        check_is_fitted(self, 'best_model_')
        model_file = \
            os.path.join(self.backend.output_path,
                         'best_model_{}.pkl'.format(self.best_model_tag_))
        joblib.dump(self.best_model_, model_file)

    @property
    def best_model(self):
        check_is_fitted(self, 'best_model_')
        return self.best_model_

    @property
    def feature_importances_(self):
        check_is_fitted(self, 'best_model_')
        return self.best_model_.feature_importances_

    @property
    def default_regressor(self):
        """Find the default regressor by reading the default environment file.

        Returns:
            default_regressor: RegressorMixin object
                Default regressor.

        """
        self.default_regressor_ = self.backend.def_env["default_regressor"]
        return self.default_regressor_

    @property
    def valid_components(self):
        """Find all supported regressors.

        Returns:
            valid_components: numpy.array([[regressor name, object], ...])
                Valid regressors
        """
        if not hasattr(self, "valid_components_"):
            regressors = np.array([est for est in all_estimators() if
                                   issubclass(est[1], RegressorMixin)])

            self.valid_components_ = regressors
        return self.valid_components_
