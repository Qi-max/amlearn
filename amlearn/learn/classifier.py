"""Use scikit-learn classifier algorithm to classify data.

This module contains imblearn method to deal with the imbalanced problems and
scikit-learn classifier algorithms to classify data and cross_validate method
to evaluate estimator's performance.

"""
import os
import numpy as np
from copy import copy

from amlearn.learn.base import AmBaseLearn
from amlearn.learn.sklearn_patch import cross_validate
from amlearn.learn.sklearn_patch import calc_scores
from amlearn.preprocess.imblearn_preprocess import ImblearnPreprocess
from amlearn.utils.check import check_output_path
from amlearn.utils.data import list_like
from amlearn.utils.directory import write_file
from sklearn.base import ClassifierMixin
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import all_estimators
from sklearn.utils.validation import check_is_fitted


class AmClassifier(AmBaseLearn):
    """Base Classifier class of Amlearn

    Args:
        backend:
        classifier: classifier from sklearn.
        imblearn:
        decimals:
        seed:

    """
    def __init__(self, backend, classifier=None, classifier_params=None,
                 decimals=None, seed=1):
        classifiers_dict = dict(self.valid_components)
        if classifier is None:
            classifier_name = self.default_classifier
        elif isinstance(classifier, ClassifierMixin):
            classifier_name = classifier.__class__.__name__
        elif isinstance(classifier, type):
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
        self.classifier = classifier if isinstance(classifier, ClassifierMixin) \
            else classifiers_dict[classifier_name](**classifier_params)
        super().__init__(backend, decimals=decimals, seed=seed)

    def fit(self, X, y, random_state=None,
            cv_num=1, cv_params=None, test_size=0.3,
            imblearn=False, imblearn_method=None, imblearn_params=None,
            save_model=True, save_predict=True, save_train_test_idx=True,
            predict_types='dataframe'):

        if imblearn:
            self._fit_imblearn(
                X, y, random_state=random_state,
                cv_num=cv_num, cv_params=cv_params, test_size=test_size,
                imblearn_method=imblearn_method,
                imblearn_params=imblearn_params, save_model=save_model,
                save_predict=save_predict,
                save_train_test_idx=save_train_test_idx,
                predict_types=predict_types)

        else:
            self._fit_cv(
                X, y, random_state=random_state,
                cv_params=cv_params, cv_num=cv_num, test_size=test_size,
                save_model=save_model, save_predict=save_predict,
                predict_types=predict_types,
                save_train_test_idx=save_train_test_idx)

    def _fit(self, X, y, classifier, test_size=0.3,
             random_state=None, scoring=None, return_train_score=False):
        ret = dict()
        indices = range(np.array(X).shape[1])
        X_train, X_test, y_train, y_test, train_idx, test_idx = \
            train_test_split(X, y, indices,
                             test_size=test_size, random_state=random_state)

        train_idx = list(map(int, train_idx))
        test_idx = list(map(int, test_idx))

        classifier = classifier.fit(X_train, y_train)
        ret['models'] = classifier
        ret['indexs'] = [train_idx, test_idx]

        test_scores, scorers = calc_scores(X=X_test, y=y_test,
                                           estimator=classifier,
                                           scoring=scoring)

        if return_train_score:
            train_scores, _ = calc_scores(X=X_test, y=y_test,
                                          estimator=classifier,
                                          scoring=scoring)
        for name in scorers:
            ret['test_%s' % name] = np.array(test_scores[name])
            if return_train_score:
                key = 'train_%s' % name
                ret[key] = np.array(train_scores[name])
                if return_train_score == 'warn':
                    message = (
                        'You are accessing a training score ({!r}), '
                        'which will not be available by default '
                        'any more in 0.21. If you need training scores, '
                        'please set return_train_score=True').format(key)
                    # warn on key access
                    ret.add_warning(key, message, FutureWarning)
        return ret

    def _fit_cv(self, X, y, random_state=None,
                cv_num=1, cv_params=None, test_size=0.3,
                save_model=True, save_predict=True, predict_types='dataframe',
                save_train_test_idx=True):

        # If user's cv_params contains 'cv_num' parameter, use the max value
        # between function parameter 'cv_num' and cv_params's 'cv_num'.
        if cv_params is None:
            cv_params = {}

        if 'cv_num' in cv_params.keys():
            cv_num = max(cv_num, cv_params['cv_num'])
            cv_params.pop('cv_num')

        if cv_num > 1:
            np.random.seed(random_state)
            scores = cross_validate(estimator=self.classifier,
                                    X=X, y=y, cv=cv_num, **cv_params)

        else:
            scores = self._fit(
                X, y, self.classifier,
                test_size=test_size, random_state=random_state,
                return_train_score=cv_params.get('return_train_score', False),
                scoring=cv_params.get('scoring', None))

            cv_num = 1

        self.best_score_, (self.best_model_, self.best_model_tag_)= \
            max(zip(scores['test_score'],
                    zip(scores['models'],
                        [''] if cv_num == 1 else
                        ["cv_{}".format(i) for i in range(cv_num)])),
                key=lambda x: x[0])

        # get temporary path, if self._tmp_path exist get it (most create by
        # imblearn), else get self.backend.tmp_path
        tmp_path = self._tmp_path \
            if hasattr(self, '_tmp_path') else self.backend.tmp_path

        if save_model or save_train_test_idx or save_predict:
            for cv_idx in range(cv_num):
                print(cv_idx)
                print(tmp_path)
                print("cv_{}".format(cv_idx))
                tmp_path_cv = os.path.join(tmp_path, "cv_{}".format(cv_idx))
                check_output_path(tmp_path_cv)
                score_model = scores['models'][cv_idx]
                if save_model:
                    self.backend.save_model(score_model, self.seed)
                if save_train_test_idx:
                    train_idx = scores['indexs'][cv_idx][0]
                    test_idx = scores['indexs'][cv_idx][1]
                    write_file( os.path.join(tmp_path_cv, 'train_idx.txt'),
                                "\n".join(list(map(str, train_idx))))

                    write_file( os.path.join(tmp_path_cv, 'test_idx.txt'),
                                "\n".join(list(map(str, test_idx))))

                if save_predict:
                    predictions = score_model.predict(X)
                    if not isinstance(predict_types, list_like()):
                        predict_types = [predict_types]
                    for predict_type in predict_types:
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

    def _fit_imblearn(self, X, y, random_state=None,
                      imblearn_method=None, imblearn_params=None,
                      cv_num=1, cv_params=None, test_size=0.3,
                      save_model=True, save_predict=True,
                      save_train_test_idx=True,
                      predict_types='dataframe', classifier_params=None):
        imblearn = ImblearnPreprocess()
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

        for imblearn_idx in range(n_subsets):
            X_imb = np.array(copy(X))[imblearn_idx, :, :]
            y_imb = np.array(copy(y))[imblearn_idx, :]
            self._tmp_path = os.path.join(self.backend.tmp_path,
                                          'imblearn_{}'.format(imblearn_idx))
            self._fit_cv(
                X=X_imb, y=y_imb, random_state=random_state,
                cv_params=cv_params, cv_num=cv_num, test_size=test_size,
                save_model=save_model, save_predict=save_predict,
                predict_types=predict_types,
                save_train_test_idx=save_train_test_idx)
            score_model_list.append(
                (self.best_score_,
                 (self.best_model_,
                  "imblearn_{}_{}".format(imblearn_idx, self.best_model_tag_))))
        print(score_model_list)
        self.best_score_, (self.best_model_, self.best_model_tag_) = \
            max(score_model_list, key=lambda x: x[0])
        return self

    def predict(self, X):
        check_is_fitted(self, 'best_model_')
        return self.best_model_.predict(X)

    def score(self, X):
        check_is_fitted(self, 'best_model_')
        pass

    def predict_proba(self, X):
        check_is_fitted(self, 'best_model_')
        return self.best_model_.predict_proba(X)

    def predict_log_proba(self, X):
        check_is_fitted(self, 'best_model_')
        return self.best_model_.predict_log_proba(X)

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
    def default_classifier(self):
        """Read the default environment file to find the default comprehensive
           classifier at present.


        Returns:

        _default_classifier: object
            Returns default classifier

        """
        self.default_classifier_ = self.backend.def_env["default_classifier"]
        return self.default_classifier_

    @property
    def valid_components(self):
        """
        Returns:
        valid_components: numpy.array([[classifier name, object], ...])
                          Returns valid classifiers
        """
        if not hasattr(self, "valid_components_"):
            classifiers = np.array([est for est in all_estimators() if
                                    issubclass(est[1], ClassifierMixin)])

            self.valid_components_ = classifiers
        return self.valid_components_
