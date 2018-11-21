"""Use scikit-learn classifier algorithm to classify data.

This module contains imblearn method to deal with the imbalanced problems and
scikit-learn classifier algorithms to classify data and cross_validate method
to evaluate estimator's performance.

"""
import os
import numpy as np
from amlearn.learn.base_learn import AmBaseLearn
from amlearn.learn.sklearn_patch import cross_validate
from amlearn.utils.data import list_like
from amlearn.utils.directory import write_file
from sklearn.base import ClassifierMixin
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

    def fit_imblearn(self):

        return

    def fit(self, X, y, cv_kwargs, cv_num=1,
            save_model=True, save_predict=True, predict_types='dataframe',
            save_train_test_idx=True,
            **classifier_kwargs):
        classifier = self.classifier(**classifier_kwargs)

        # If user's cv_kwargs contains 'cv_num' parameter, use the max value
        # between function parameter 'cv_num' and cv_kwargs's 'cv_num'.
        if 'cv_num' in cv_kwargs.keys():
            cv_num = max(cv_num, cv_kwargs['cv_num'])
            cv_kwargs.pop('cv_num')

        if cv_num > 1:
            scores = cross_validate(estimator=classifier, X=X, y=y,
                                    **cv_kwargs)

        else:
            # TODO: add result to scores
            cv_num = 1
            pass

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

    def predict(self, X, y):

        return
