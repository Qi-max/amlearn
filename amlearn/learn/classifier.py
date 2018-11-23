"""Use scikit-learn classifier algorithm to classify data.

This module contains imblearn method to deal with the imbalanced problems and
scikit-learn classifier algorithms to classify data and cross_validate method
to evaluate estimator's performance.

"""
import os
import numpy as np
from copy import copy
from amlearn.learn.base_learn import AmBaseLearn
from amlearn.learn.sklearn_patch import cross_validate
from amlearn.learn.sklearn_patch import calc_scores
from amlearn.preprocess.imblearn_preprocess import ImblearnPreprocess
from amlearn.utils.data import list_like
from amlearn.utils.directory import write_file
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import all_estimators


class AmClassifier(AmBaseLearn):
    """Base Classifier class of Amlearn

    Args:
        backend:
        classifier: classifier from sklearn.
        imblearn:
        decimals:
        seed:

    """
    def __init__(self, backend, classifier=None, imblearn=False,
                 decimals=None, seed=1):
        classifiers_dict = dict(self.valid_components)
        if classifier is None:
            classifier = self.best_classifier
        if isinstance(classifier, type):
            classifier = classifier.__name__
        if classifier not in classifiers_dict.keys():
            raise ValueError('Classifier {} is unknown, Possible values '
                             'are {}'.format(classifier,
                                             classifiers_dict.keys()))
        self.classifier = classifiers_dict[classifier]
        super().__init__(backend, decimals=decimals, seed=seed)
        self.imblearn = imblearn

    def fit(self, X, y, random_state=None,
            cv_num=1, cv_kwargs=None, test_size=0.3,
            imblearn=False, imblearn_method=None, imblearn_kwargs=None,
            save_model=True, save_predict=True, save_train_test_idx=True,
            predict_types='dataframe', **classifier_kwargs):

        if imblearn:
            self._fit_imblearn(
                X, y, random_state=random_state,
                cv_num=cv_num, cv_kwargs=cv_kwargs, test_size=test_size,
                imblearn_method=imblearn_method,
                imblearn_kwargs=imblearn_kwargs, save_model=save_model,
                save_predict=save_predict,
                save_train_test_idx=save_train_test_idx,
                predict_types=predict_types, **classifier_kwargs)

        else:
            self._fit_cv(
                X, y, random_state=random_state,
                cv_kwargs=cv_kwargs, cv_num=cv_num, test_size=test_size,
                save_model=save_model, save_predict=save_predict,
                predict_types=predict_types,
                save_train_test_idx=save_train_test_idx,
                **classifier_kwargs)

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
                cv_num=1, cv_kwargs=None, test_size=0.3,
                save_model=True, save_predict=True, predict_types='dataframe',
                save_train_test_idx=True, **classifier_kwargs):

        if hasattr(self.classifier, 'random_state'):
            classifier = self.classifier(random_state=random_state,
                                         **classifier_kwargs)
        else:
            classifier = self.classifier(**classifier_kwargs)

        # If user's cv_kwargs contains 'cv_num' parameter, use the max value
        # between function parameter 'cv_num' and cv_kwargs's 'cv_num'.
        if 'cv_num' in cv_kwargs.keys():
            cv_num = max(cv_num, cv_kwargs['cv_num'])
            cv_kwargs.pop('cv_num')

        if cv_num > 1:
            np.random.seed(random_state)
            scores = cross_validate(estimator=classifier, X=X, y=y, cv=cv_num,
                                    **cv_kwargs)

        else:
            scores = self._fit(
                X, y, classifier,
                test_size=test_size,
                random_state=random_state,
                return_train_score=cv_kwargs.get('return_train_score', False),
                scoring=cv_kwargs.get('scoring', None))

            cv_num = 1

        # get temporary path, if self._tmp_path exist get it (most create by
        # imblearn), else get self.backend.tmp_path
        tmp_path = self._tmp_path \
            if hasattr(self, '_tmp_path') else self.backend.tmp_path

        for cv_idx in range(cv_num):
            self.backend.tmp_path = os.path.join(tmp_path,
                                                 "cv_{}".format(cv_idx))
            score_model = scores['models'][cv_idx]
            if save_model:
                self.backend.save_model(score_model, self.seed)
            if save_train_test_idx:
                train_idx = scores['indexs'][cv_idx][0]
                test_idx = scores['indexs'][cv_idx][1]
                write_file(
                    os.path.join(self.backend.tmp_path,
                                 'train_idx.txt'),
                    "\n".join(list(map(str, train_idx))))

                write_file(
                    os.path.join(self.backend.tmp_path,
                                 'test_idx.txt'),
                    "\n".join(list(map(str, test_idx))))

            if save_predict:
                predictions = score_model.predict(X)
                if not isinstance(predict_types, list_like):
                    predict_types = [predict_types]
                for predict_type in predict_types:
                    if predict_type in self.backend.valid_predictions_type:
                        self.backend.eval('save_predictions_as_{}'.format(
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
                      imblearn_method=None, imblearn_kwargs=None,
                      cv_num=1, cv_kwargs=None, test_size=0.3,
                      save_model=True, save_predict=True,
                      save_train_test_idx=True,
                      predict_types='dataframe', **classifier_kwargs):
        imblearn = ImblearnPreprocess()
        if 'random_state' not in imblearn_kwargs:
            imblearn_kwargs['random_state'] = random_state
        X, y = imblearn.fit(X, y, imblearn_method, imblearn_kwargs)

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
                cv_kwargs=cv_kwargs, cv_num=cv_num, test_size=test_size,
                save_model=save_model, save_predict=save_predict,
                predict_types=predict_types,
                save_train_test_idx=save_train_test_idx,
                **classifier_kwargs)
        return self

    def predict(self, X, y):

        return

    @property
    def best_classifier(self):
        """Read the default environment file to find the best comprehensive
           classifier at present.


        Returns:

        _best_classifier: object
            Returns best classifier

        """
        self._best_classifier = self.backend.def_env["best_classifier"]
        return self._best_classifier

    @property
    def valid_components(self):
        """
        Returns:
        valid_components: numpy.array([[classifier name, object], ...])
                          Returns valid classifiers
        """
        if not hasattr(self, "_valid_components"):
            classifiers = np.array([est for est in all_estimators() if
                                    issubclass(est[1], ClassifierMixin)])

            self._valid_components = classifiers
        return self._valid_components
