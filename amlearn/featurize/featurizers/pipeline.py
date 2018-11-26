import inspect
import os
import pkgutil
from operator import itemgetter
from amlearn.featurize.featurizers.base import BaseFeaturize
from amlearn.utils.check import check_featurizer_X, is_abstract
from sklearn.base import BaseEstimator
from sklearn.pipeline import FeatureUnion, Pipeline

module_dir = os.path.dirname(os.path.abspath(__file__))


def all_featurizers():
    all_classes = []
    for importer, modname, ispkg in pkgutil.walk_packages(
            path=[module_dir], prefix='amlearn.featurize.featurizers.',
            onerror=lambda x: None):
        if ".tests." in modname or ".sro_mro." in modname:
            continue
        module = __import__(modname, fromlist="dummy")
        classes = inspect.getmembers(module, inspect.isclass)
        all_classes.extend(classes)

    all_classes = set(all_classes)

    estimators = [c for c in all_classes
                  if (issubclass(c[1], BaseFeaturize) and
                      c[0] != 'BaseFeaturize')]

    # get rid of abstract base classes
    estimators = [c for c in estimators if not is_abstract(c[1])]

    return sorted(set(estimators), key=itemgetter(0))


class MultiFeaturizer(BaseFeaturize):
    def __init__(self, featurizers="all",
                 atoms_df=None, tmp_save=True, context=None):
        super(MultiFeaturizer, self).__init__(
            tmp_save=tmp_save, context=context, atoms_df=atoms_df)
        self.featurizers = featurizers

    def fit(self, X=None):
        featurizer_dict = dict()
        for featurizer in all_featurizers():
            instance = featurizer[1]()
            if instance.category in featurizer_dict.keys():
                featurizers = featurizer_dict[instance.category]
                featurizers.append(instance)
            else:
                featurizer_dict[instance.category] = [instance]
        self.featurizer_dict = featurizer_dict
        return self

    def transform(self, X=None):
        pipeline_list = []
        category_list = self.featurizer_dict.keys()
        if 'voro_and_dist' in category_list:
            pipeline_list.append(
                ('voro_and_dist',
                 FeatureUnion(((instance.__class__.__name__, instance)
                               for instance in
                               self.featurizer_dict['voro_and_dist']))))
        if 'sro' in category_list:
            pipeline_list.append(
                ('sro',
                 FeatureUnion(((instance.__class__.__name__, instance)
                               for instance in
                               self.featurizer_dict['sro']))))
        if 'mro' in category_list:
            pipeline_list.append(
                ('mro', self.featurizer_dict['mro'][0]))
        pipeline_list.append(('final', FinalEstimator()))

        X = check_featurizer_X(X=X, atoms_df=self.atoms_df)
        pipe = Pipeline(pipeline_list)
        pipe.fit(X)
        X = pipe.named_steps['final'].data
        return X

    def get_feature_names(self):
        pass


class FinalEstimator(BaseEstimator):

    def fit(self, X):
        self._data = X
        return self

    @property
    def data(self):
        return self._data