import numpy as np
from amlearn.learn.base import AmBaseLearn
from sklearn.base import ClassifierMixin
from sklearn.utils.testing import all_estimators


class AmClassifier(AmBaseLearn):
    def __init__(self, backend, classifier=None, imblearn=False,
                 decimals=None, seed=1):
        classifier_strs = self.valid_components[:, 0]
        if classifier is None:
            classifier = self.best_classifier
        if isinstance(classifier, type):
            classifier = classifier.__name__
        if classifier not in classifier_strs:
            raise ValueError('Classifier {} is unknown, Possible values '
                             'are {}'.format(classifier, classifier_strs))

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

    def fit(self, X, y):

        return

    def predict(self, X, y):

        return
