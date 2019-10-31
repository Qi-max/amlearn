"""Use scikit-learn classifier algorithm to classify data.

This module contains imblearn method to deal with the imbalanced problems and
scikit-learn classifier algorithms to classify data and cross_validate method
to evaluate estimator's performance.

"""
import os
from collections import OrderedDict

import numpy as np
from copy import copy

import time
from amlearn.learn.base import AmBaseLearn
from amlearn.learn.sklearn_patch import cross_validate
from amlearn.learn.sklearn_patch import calc_scores
from amlearn.utils.backend import check_path_while_saving
from amlearn.utils.check import appropriate_kwargs
from amlearn.utils.data import list_like
from amlearn.utils.directory import write_file, create_path
from sklearn.base import ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import all_estimators
from sklearn.utils.validation import check_is_fitted

from amlearn.learn.preprocessor import ImblearnPreprocessor

__author__ = "Qi Wang"
__email__ = "qiwang.mse@gmail.com"


class AmClassifier(AmBaseLearn):
    """Base Classifier class of amlearn.

    Args:
        backend: Backend object (default: None)
            MLBackend object which defined output_path, environment
            configuration, save_predictions, and so on.
            If default, use default MLBackend object.
        classifier: object, string or class (default: None)
            Sklearn classifier object.
            If default, use the default_classifier which defined in backend
            environment configuration.
        classifier_params: dict (default: None)
            The classifier parameters.
        random_state: int, RandomState instance or None, optional (default=0)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.
            #TODO: Support dict, define different random_state to different method.
        decimals: int (default: None)
            Output decimals.
            #TODO: Define decimals and seed later.
        seed: int (default: 1)
            Random seed

    """
    def __init__(self, backend=None, output_path='tmp', classifier=None,
                 classifier_params=None, decimals=None, seed=1):
        super().__init__(backend, output_path, decimals=decimals, seed=seed)

        # Get all supported classifiers
        classifiers_dict = dict(self.valid_components)

        # Set classifier object
        if classifier is None:
            classifier_name = self.default_classifier
        elif isinstance(classifier, ClassifierMixin):
            classifier_name = classifier.__class__.__name__
        elif callable(classifier):
            classifier_name = classifier.__name__
        elif isinstance(classifier, str):
            classifier_name = classifier
        else:
            classifier_name = None

        if classifier_name not in classifiers_dict.keys():
            raise ValueError('Classifier {} is unknown, Possible values '
                             'are {}'.format(classifier,
                                             classifiers_dict.keys()))
        if classifier_params is None:
            classifier_params = {}

        classifier_class = classifier if isinstance(classifier, ClassifierMixin) \
            else classifiers_dict[classifier_name]

        classifier_params = appropriate_kwargs(classifier_params, classifier_class)
        self.classifier = classifier_class(**classifier_params)

        self.backend.logger.info(
            'Initialize classification, classifier is : \n\t{}'.format(
                self.classifier))
        self.imblearn_tag = ''

    def fit(self, X, y, val_size=0.3, scoring=None, random_state=None,
            cv_num=5, cv_params=None, imblearn=True, imblearn_method=None,
            imblearn_params=None, save_model=True, save_score=True,
            save_prediction=True, prediction_types='dataframe',
            save_train_val_idx=True, save_feature_importances=True,
            **fit_params):
        """Fit the amlearn classifier model.

        Args:
            X: DataFrame
                The data to fit. Can be for example a list, or an array.
            y: array-like, optional, default: None
                The target variable to try to predict in the case of
                supervised learning.
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
            imblearn: boolean. (default: False)
                Use imblearn or not.
                If the dataset is unbalanced, it's recommended to set imblearn
                to True, it will automatically sample the dataset into a
                balanced dataset.
            imblearn_method: str or callable (default: None)
                Imblearn method name or Imblearn method object.
            imblearn_params: dict (default: None)
                The parameters dict of imblearn method.
            save_model: boolean (default: True)
                Whether to save the model.
            save_prediction: boolean (default: True)
                Whether to save the prediction from each cross validation model.
            save_train_val_idx: boolean (default: True)
                Whether to save the train validation indices.
            prediction_types: str (default: dataframe)
                It effect when save_prediction is True.
                The Optional parameters are: {"npy", "txt", "dataframe"}.
            **fit_params:
                Classifier fit parameters.

        Returns:
            self
        """
        self.scoring = scoring
        self.imblearn = imblearn
        self.feature_names_ = X.columns
        if imblearn:
            self._fit_imblearn(
                X, y, random_state=random_state, scoring=scoring,
                cv_num=cv_num, cv_params=cv_params, val_size=val_size,
                imblearn_method=imblearn_method,
                imblearn_params=imblearn_params, save_model=save_model,
                save_prediction=save_prediction,
                prediction_types=prediction_types,
                save_feature_importances=save_feature_importances,
                save_train_val_idx=save_train_val_idx,
                **fit_params)
        else:
            self._fit_cv(
                X, y, random_state=random_state, scoring=scoring,
                cv_params=cv_params, cv_num=cv_num, val_size=val_size,
                save_model=save_model, save_score=save_score,
                save_prediction=save_prediction,
                prediction_types=prediction_types,
                save_feature_importances=save_feature_importances,
                save_train_val_idx=save_train_val_idx, **fit_params)
        return self

    def _fit(self, X, y, classifier, val_size=0.3, random_state=None,
             scoring=None, return_train_score=True, **fit_params):
        ret = dict()
        indices = range(np.array(X).shape[0])
        X_train, X_val, y_train, y_val, train_idx, val_idx = \
            train_test_split(X, y, indices, test_size=val_size,
                             random_state=random_state)

        train_idx = list(map(int, train_idx))
        val_idx = list(map(int, val_idx))

        classifier = classifier.fit(X_train, y_train, **fit_params)
        ret['estimators'] = [classifier]
        ret['indices'] = [[train_idx, val_idx]]

        val_scores, scorers = calc_scores(X=X_val, y=y_val,
                                          estimator=classifier,
                                          scoring=scoring)

        if return_train_score:
            train_scores, _ = calc_scores(X=X_train, y=y_train,
                                          estimator=classifier,
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
                save_model=False, save_score=True, save_prediction=False,
                prediction_types='dataframe', save_feature_importances=True,
                save_train_val_idx=False, **fit_params):

        # If user's cv_params contains 'cv_num' parameter, use the max value
        # between function parameter 'cv_num' and cv_params's 'cv_num'.
        if not self.imblearn:
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
            np.random.seed(random_state)
            results, scorers = \
                cross_validate(estimator=self.classifier, scoring=scoring,
                               fit_params=fit_params, X=X, y=y, cv=cv_num,
                               **cv_params)
        else:
            results, scorers = self._fit(
                X, y, self.classifier, val_size=val_size,
                return_train_score=return_train_score,
                random_state=random_state, scoring=scoring, **fit_params)

            cv_num = 1

        # TODO: now if scoring is more than one, score_name only can be the first of them.
        self.score_name = self.score_name if hasattr(self, 'score_name') \
            else list(scorers.keys())[0]
        self.best_score_, (self.best_model_, self.best_model_tag_)= \
            max(zip(results['test_{}'.format(self.score_name)],
                    zip(results['estimators'],
                        [''] if cv_num == 1 else
                        ["cv_{}".format(i) for i in range(cv_num)])),
                key=lambda x: x[0])

        # get temporary path, if self.tmp_path_ exist get it (most create by
        # imblearn), else get self.backend.tmp_path
        # tmp_path = self.tmp_path_ \
        #     if hasattr(self, 'tmp_path_') else self.backend.tmp_path

        if not self.imblearn:
            self.backend.logger.info(
                "\tCV classification finish in {:.4f} seconds. "
                "It's best {} score is {:.4f}".format(
                    time.time() - cv_start_time,
                    self.score_name, self.best_score_))

        if save_model or save_score or save_train_val_idx or save_prediction \
                or save_feature_importances:
            imblearn_output_path = \
                os.path.join(self.backend.output_path, self.imblearn_tag)
            create_path(imblearn_output_path)
            if save_score:
                write_file(
                    os.path.join(imblearn_output_path, 'mean_scores.txt'),
                    '{}\n{}\n{}'.format(
                        ','.join(list(scorers.keys())),
                        ','.join([
                            str(np.mean(results['test_{}'.format(score_name)]))
                            for score_name in scorers.keys()]),
                        ','.join([
                            str(np.mean(results['train_{}'.format(score_name)]))
                            for score_name in scorers.keys()])
                        if return_train_score else -1))

            check_path_while_saving(self.backend.tmp_path)
            check_path_while_saving(self.backend.output_path)
            for cv_idx in range(cv_num):
                sub_path = os.path.join(self.imblearn_tag,
                                        "cv_{}".format(cv_idx))
                cv_output_path = \
                    os.path.join(self.backend.output_path, sub_path)
                create_path(cv_output_path)

                if save_score:
                    write_file(os.path.join(cv_output_path, 'scores.txt'),
                               '{}\n{}\n{}'.format(
                                   ','.join(list(scorers.keys())),
                                   ','.join([str(results['test_{}'.format(
                                       score_name)][cv_idx])
                                             for score_name in scorers.keys()]),
                                   ','.join([str(results['train_{}'.format(
                                       score_name)][cv_idx])
                                             for score_name in scorers.keys()])
                                   if return_train_score else -1))

                score_model = results['estimators'][cv_idx]
                if save_model:
                    self.backend.save_model(score_model, sub_path,
                                            seed=self.seed)
                if save_feature_importances:
                    self.backend.save_json(self.feature_importances_dict,
                                           sub_path, name='feature_importances')
                if save_train_val_idx:
                    train_idx = results['indices'][cv_idx][0]
                    val_idx = results['indices'][cv_idx][1]
                    write_file(os.path.join(cv_output_path, 'train_idx.txt'),
                                "\n".join(list(map(str, train_idx))))

                    write_file(os.path.join(cv_output_path, 'val_idx.txt'),
                                "\n".join(list(map(str, val_idx))))

                if save_prediction:
                    predictions = score_model.predict(X)
                    if not isinstance(prediction_types, list_like()):
                        prediction_types = [prediction_types]
                    for predict_type in prediction_types:
                        if predict_type in self.backend.valid_predictions_type:
                            getattr(self.backend,
                                    'save_predictions_as_{}'.format(
                                        predict_type))(predictions, sub_path,
                                                       seed=self.seed)
                        else:
                            raise ValueError(
                                'predict_type {} is unknown, '
                                'Possible values are {}'.format(
                                    predict_type,
                                    self.backend.valid_predictions_type))
        return self

    def _fit_imblearn(self, X, y, random_state=None, scoring=None,
                      imblearn_method=None, imblearn_params=None,
                      cv_num=1, cv_params=None, val_size=0.3,
                      save_model=True, save_score=True, save_prediction=True,
                      prediction_types='dataframe', save_train_val_idx=True,
                      save_feature_importances=True, **fit_params):
        self.backend.logger.info('Start Imblearn.')
        imblearn_start_time = time.time()
        imblearn = ImblearnPreprocessor()
        if imblearn_method is None:
            imblearn_method = 'EasyEnsemble'
        if imblearn_params is None:
            imblearn_params = {"random_state": random_state, "n_subsets": 3}
        if 'random_state' not in imblearn_params:
            imblearn_params['random_state'] = random_state
        X, y = imblearn.fit(X, y, imblearn_method, imblearn_params)

        score_model_list = list()
        # get the imblearn n_subsets num from X shape.
        if len(X.shape) == 2:
            n_subsets = 1
            X = [X]
            y = [y]
        elif len(X.shape) == 3:
            n_subsets = X.shape[0]
        else:
            raise ValueError("imblearn result error!")

        self.backend.logger.info(
            '\tData imblearn finished in {:.4f} seconds.'.format(
                time.time() - imblearn_start_time))

        for imblearn_idx in range(n_subsets):
            self.backend.logger.info('Start imblearn_{} classification.'.format(
                imblearn_idx))
            start_time = time.time()
            X_imb = np.array(copy(X))[imblearn_idx, :, :]
            y_imb = np.array(copy(y))[imblearn_idx, :]
            self.imblearn_tag = 'imblearn_{}'.format(imblearn_idx)
            self._fit_cv(
                X=X_imb, y=y_imb, random_state=random_state, scoring=scoring,
                cv_params=cv_params, cv_num=cv_num, val_size=val_size,
                save_model=save_model, save_score=save_score,
                save_prediction=save_prediction,
                prediction_types=prediction_types,
                save_feature_importances=save_feature_importances,
                save_train_val_idx=save_train_val_idx, **fit_params)
            score_model_list.append(
                (self.best_score_,
                 (self.best_model_,
                  "imblearn_{}_{}".format(imblearn_idx, self.best_model_tag_))))
            self.backend.logger.info(
                "\tImblearn_{} classification finish in {:.4f} seconds. "
                "It's best {} score is {:.4f}".format(
                    imblearn_idx, time.time() - start_time,
                    self.score_name, self.best_score_))

        self.best_score_, (self.best_model_, self.best_model_tag_) = \
            max(score_model_list, key=lambda x: x[0])
        self.backend.logger.info('Whole classification finish in {:.4f} '
                                 'seconds. The best validation {} score '
                                 'is {:.4f}'.format(
            time.time() - imblearn_start_time,
            self.score_name, self.best_score_))

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

    def predict_proba(self, X):
        check_is_fitted(self, 'best_model_')
        return self.best_model_.predict_proba(X)

    def predict_log_proba(self, X):
        check_is_fitted(self, 'best_model_')
        return self.best_model_.predict_log_proba(X)

    def save_best_model(self):
        check_is_fitted(self, 'best_model_')
        check_path_while_saving(self.backend.tmp_path)
        check_path_while_saving(self.backend.output_path)
        model_file = \
            os.path.join(self.backend.output_path,
                         'best_model_{}.pkl'.format(self.best_model_tag_))
        joblib.dump(self.best_model_, model_file)
        self.backend.logger.info(
            'Finish saving best model to is :{}'.format(model_file))

    @property
    def best_model(self):
        check_is_fitted(self, 'best_model_')
        return self.best_model_

    @property
    def best_score(self):
        check_is_fitted(self, 'best_model_')
        return self.best_score_

    @property
    def feature_importances_(self):
        check_is_fitted(self, 'best_model_')
        return self.best_model_.feature_importances_

    @property
    def feature_importances_dict(self):
        check_is_fitted(self, 'best_model_')
        feature_importances_dict_ = \
            sorted(zip(self.get_feature_names(),
                       self.best_model_.feature_importances_),
                   key=lambda x: x[1], reverse=True)
        return OrderedDict(feature_importances_dict_)

    def get_feature_names(self):
        msg = ("This %(name)s instance is not fitted yet. Call 'fit_transform' "
               "with appropriate arguments before using this method.")

        if not hasattr(self, 'feature_names_'):
            raise NotFittedError(msg % {'name': type(self).__name__})
        return self.feature_names_

    @property
    def default_classifier(self):
        """Find the default classifier by reading the default environment file.

        Returns:
            default_classifier: ClassifierMixin object
                Default classifier.

        """
        self.default_classifier_ = self.backend.def_env["default_classifier"]
        return self.default_classifier_

    @property
    def valid_components(self):
        """Find all supported classifiers.

        Returns:
            valid_components: numpy.array([[classifier name, object], ...])
                Valid classifiers
        """
        if not hasattr(self, "valid_components_"):
            classifiers = np.array([est for est in all_estimators() if
                                    issubclass(est[1], ClassifierMixin)])

            self.valid_components_ = classifiers
        return self.valid_components_
