import numpy as np
from amlearn.preprocess.base_preprocess import BasePreprocess
from imblearn.utils.testing import all_estimators


class ImblearnPreprocess(BasePreprocess):

    @property
    def valid_components(self):
        if not hasattr(self, "valid_components"):
            self._valid_components = np.array(all_estimators())

        return self._valid_components

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

        elif imblearn_method.__name__ not in imblearn_methods.keys():
            raise ValueError('imblearn_method {} is unknown,Possible values'
                             ' are {}'.format(imblearn_method,
                                              imblearn_methods.keys()))
        imbalanced_sampling = imblearn_method

        if imblearn_kwargs:
            X, y = imbalanced_sampling(**imblearn_kwargs).fit_sample(X, y)
        else:
            X, y = imbalanced_sampling().fit_sample(X, y)

        return X, y