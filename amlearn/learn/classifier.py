"""Use scikit-learn classifier algorithm to classify data.

This module contains imblearn method to deal with the imbalanced problems and
scikit-learn classifier algorithms to classify data and cross_validate method
to evaluate estimator's performance.

"""
import numpy as np
from amlearn.learn.base_learn import AmBaseLearn
from amlearn.learn.sklearn_patch import cross_validate
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
        self._best_classifier = self.backend.def_env["best_classifier"]
        return self._best_classifier

    @property
    def valid_components(self):
        """
        Returns: valid classifiers
        """
        if not hasattr(self, "_valid_components"):
            classifiers = np.array([est for est in all_estimators() if
                                    issubclass(est[1], ClassifierMixin)])

            self._valid_components = classifiers
        return self._valid_components

    def fit_imblearn(self):

        return

    def fit(self, X, y, cv_kwargs, cv_num=1, **classifier_kwargs):
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
            pass
        return self

    def predict(self, X, y):

        return
